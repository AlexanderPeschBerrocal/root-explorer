#!/usr/bin/env python3
"""
ROOT Explorer
=============

A general-purpose Streamlit app for interactive plotting and event inspection 
of HEP ROOT TTrees in the web browser.

The app is intentionally analysis-agnostic. It works with generic TTrees from 
NanoAOD, DAOD, and small custom ROOT files, as long as uproot can read the selected branches.

Main features
-------------
- Open a local ROOT file from a path or via Streamlit upload.
- Browse TTrees and branches, including branch type and collection grouping.
- Plot direct branches, object multiplicities, leading/subleading elements, and
  common derived quantities such as r, phi, eta, delta-phi, delta-R, ratios,
  sums and differences.
- Make 1D histograms, 2D histograms, 2D scatter plots, and interactive 3D
  Plotly scatter plots.
- Apply multiple simultaneous range cuts using any supported variable expression.
- Inspect a single event and rank events by multiplicity, sum, mean, max, min,
  or threshold counts.
- Export static plots as PNG and/or PDF, and interactive 3D plots as HTML.
- Load GDML detector geometry from a path or upload and overlay transparent detector meshes on 3D scatter plots.

Install
-------
    micromamba install -c conda-forge streamlit uproot awkward numpy matplotlib pandas plotly

Run
---
    streamlit run root_explorer.py -- path/to/file.root

or simply:
    streamlit run root_explorer.py

Notes
-----
This app is designed as an exploratory viewer, not as a replacement for a final
analysis workflow. For very large files, start with a limited entry range and
increase it only after the desired plot configuration is stable.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import math
import struct
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import uproot
from matplotlib.colors import LogNorm

APP_TITLE = "ROOT Explorer"
DEFAULT_MAX_ENTRIES = 20_000
DEFAULT_MAX_POINTS = 200_000

st.set_page_config(page_title=APP_TITLE, layout="wide")


@dataclass(frozen=True)
class BranchInfo:
    """Small summary of one TTree branch."""

    name: str
    typename: str
    interpretation: str
    collection: str
    field: str


@dataclass(frozen=True)
class VariableSpec:
    """A user-selected variable definition."""

    mode: str
    branch: Optional[str] = None
    index: int = 0
    branch_a: Optional[str] = None
    branch_b: Optional[str] = None
    branch_c: Optional[str] = None
    branch_d: Optional[str] = None
    label: str = ""




# -----------------------------------------------------------------------------
# GDML detector geometry utilities
# -----------------------------------------------------------------------------

Vec3 = Tuple[float, float, float]
Mat4 = List[List[float]]
Triangle = Tuple[Vec3, Vec3, Vec3]

LEN_UNITS = {
    None: 1.0,
    "": 1.0,
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "um": 0.001,
    "micrometer": 0.001,
    "nm": 1e-6,
}
ANGLE_UNITS = {
    None: math.pi / 180.0,
    "": math.pi / 180.0,
    "deg": math.pi / 180.0,
    "degree": math.pi / 180.0,
    "rad": 1.0,
    "radian": 1.0,
}

QUALITY_SEGMENTS = {
    "fast": 16,
    "medium": 32,
    "high": 64,
}

DEFAULT_MATERIAL_COLORS = {
    "G4_Si": (84, 199, 110, 210),
    "G4_Fe": (155, 159, 166, 220),
    "Material_ECAL": (54, 132, 245, 190),
    "Material_HCAL": (245, 166, 35, 190),
    "Galactic": (255, 255, 255, 16),
    "G4_AIR": (180, 210, 255, 32),
}


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_by_tag(el: ET.Element, tag: str) -> Optional[ET.Element]:
    for ch in list(el):
        if strip_namespace(ch.tag) == tag:
            return ch
    return None


def children_by_tag(el: ET.Element, tag: str) -> List[ET.Element]:
    return [ch for ch in list(el) if strip_namespace(ch.tag) == tag]


def as_float(attrs: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(attrs.get(key, default))
    except (TypeError, ValueError):
        return default


def length_value(attrs: Dict[str, str], key: str, default: float = 0.0) -> float:
    return as_float(attrs, key, default) * LEN_UNITS.get(attrs.get("lunit") or attrs.get("unit"), 1.0)


def angle_value(attrs: Dict[str, str], key: str, default_deg: float = 0.0) -> float:
    return as_float(attrs, key, default_deg) * ANGLE_UNITS.get(attrs.get("aunit") or attrs.get("unit"), math.pi / 180.0)


def identity() -> Mat4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matmul(a: Mat4, b: Mat4) -> Mat4:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        ai = a[i]
        for k in range(4):
            aik = ai[k]
            if aik:
                bk = b[k]
                out[i][0] += aik * bk[0]
                out[i][1] += aik * bk[1]
                out[i][2] += aik * bk[2]
                out[i][3] += aik * bk[3]
    return out


def translate(x: float, y: float, z: float) -> Mat4:
    m = identity()
    m[0][3], m[1][3], m[2][3] = x, y, z
    return m


def rotate_x(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]


def rotate_y(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]


def rotate_z(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def euler_xyz(rx: float, ry: float, rz: float) -> Mat4:
    # Apply local X, then local Y, then local Z. Most detector GDML placements use only z.
    return matmul(rotate_z(rz), matmul(rotate_y(ry), rotate_x(rx)))


def transform_point(m: Mat4, p: Vec3) -> Vec3:
    x, y, z = p
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
        m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
        m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
    )


def transform_mesh(tris: Iterable[Triangle], m: Mat4) -> List[Triangle]:
    return [(transform_point(m, a), transform_point(m, b), transform_point(m, c)) for a, b, c in tris]


def parse_position(el: Optional[ET.Element]) -> Vec3:
    if el is None:
        return (0.0, 0.0, 0.0)
    attrs = el.attrib
    unit = LEN_UNITS.get(attrs.get("unit"), 1.0)
    return (as_float(attrs, "x") * unit, as_float(attrs, "y") * unit, as_float(attrs, "z") * unit)


def parse_rotation(el: Optional[ET.Element]) -> Vec3:
    if el is None:
        return (0.0, 0.0, 0.0)
    attrs = el.attrib
    unit = ANGLE_UNITS.get(attrs.get("unit"), math.pi / 180.0)
    return (as_float(attrs, "x") * unit, as_float(attrs, "y") * unit, as_float(attrs, "z") * unit)


def placement_matrix(physvol: ET.Element) -> Mat4:
    pos = parse_position(child_by_tag(physvol, "position"))
    rot = parse_rotation(child_by_tag(physvol, "rotation"))
    return matmul(translate(*pos), euler_xyz(*rot))


def add_tri(tris: List[Triangle], a: Vec3, b: Vec3, c: Vec3) -> None:
    # Skip degenerate triangles; they confuse transparency and picking.
    ax, ay, az = a; bx, by, bz = b; cx, cy, cz = c
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    if nx * nx + ny * ny + nz * nz > 1e-18:
        tris.append((a, b, c))


def add_quad(tris: List[Triangle], a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> None:
    add_tri(tris, a, b, c)
    add_tri(tris, a, c, d)


def angular_segments(delta: float, base_segments: int) -> int:
    full = 2 * math.pi
    frac = min(1.0, max(0.0, abs(delta) / full))
    if frac > 0.98:
        return base_segments
    return max(1, int(math.ceil(base_segments * frac)))


def ring_point(r: float, phi: float, z: float) -> Vec3:
    return (r * math.cos(phi), r * math.sin(phi), z)


def make_tube_like(
    rmin_low: float,
    rmax_low: float,
    rmin_high: float,
    rmax_high: float,
    zlen: float,
    start: float,
    delta: float,
    seg_base: int,
    cut_low: Optional[Vec3] = None,
    cut_high: Optional[Vec3] = None,
) -> List[Triangle]:
    tris: List[Triangle] = []
    if zlen <= 0 or max(rmax_low, rmax_high) <= 0:
        return tris
    full = abs(delta) >= 2 * math.pi * 0.999
    nseg = angular_segments(delta, seg_base)
    phis = [start + delta * i / nseg for i in range(nseg + 1)]
    if full:
        # Avoid duplicating exactly 2pi angle with tiny drift.
        phis[-1] = start + (2 * math.pi if delta >= 0 else -2 * math.pi)

    z0_default = -zlen / 2.0
    z1_default = zlen / 2.0

    def z_low(x: float, y: float) -> float:
        if not cut_low:
            return z0_default
        nx, ny, nz = cut_low
        if abs(nz) < 1e-12:
            return z0_default
        return z0_default - (nx * x + ny * y) / nz

    def z_high(x: float, y: float) -> float:
        if not cut_high:
            return z1_default
        nx, ny, nz = cut_high
        if abs(nz) < 1e-12:
            return z1_default
        return z1_default - (nx * x + ny * y) / nz

    low_outer: List[Vec3] = []
    high_outer: List[Vec3] = []
    low_inner: List[Vec3] = []
    high_inner: List[Vec3] = []

    for phi in phis:
        x0, y0 = rmax_low * math.cos(phi), rmax_low * math.sin(phi)
        x1, y1 = rmax_high * math.cos(phi), rmax_high * math.sin(phi)
        low_outer.append((x0, y0, z_low(x0, y0)))
        high_outer.append((x1, y1, z_high(x1, y1)))
        xi0, yi0 = rmin_low * math.cos(phi), rmin_low * math.sin(phi)
        xi1, yi1 = rmin_high * math.cos(phi), rmin_high * math.sin(phi)
        low_inner.append((xi0, yi0, z_low(xi0, yi0)))
        high_inner.append((xi1, yi1, z_high(xi1, yi1)))

    for i in range(nseg):
        j = i + 1
        add_quad(tris, low_outer[i], low_outer[j], high_outer[j], high_outer[i])
        if max(rmin_low, rmin_high) > 1e-9:
            # reverse winding on the inner surface
            add_quad(tris, low_inner[j], low_inner[i], high_inner[i], high_inner[j])
        else:
            # If there is no hole, cap with triangles to the axis.
            pass
        # lower and upper caps
        if max(rmin_low, rmin_high) > 1e-9:
            add_quad(tris, low_inner[i], low_inner[j], low_outer[j], low_outer[i])
            add_quad(tris, high_inner[j], high_inner[i], high_outer[i], high_outer[j])
        else:
            center_low = (0.0, 0.0, z0_default)
            center_high = (0.0, 0.0, z1_default)
            add_tri(tris, center_low, low_outer[j], low_outer[i])
            add_tri(tris, center_high, high_outer[i], high_outer[j])

    if not full:
        # radial side faces at the beginning and end of the phi sector
        add_quad(tris, low_inner[0], low_outer[0], high_outer[0], high_inner[0])
        add_quad(tris, low_outer[-1], low_inner[-1], high_inner[-1], high_outer[-1])

    return tris


def make_box(attrs: Dict[str, str]) -> List[Triangle]:
    x = length_value(attrs, "x") / 2.0
    y = length_value(attrs, "y") / 2.0
    z = length_value(attrs, "z") / 2.0
    v = [
        (-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
        (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z),
    ]
    faces = [
        (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0),
    ]
    tris: List[Triangle] = []
    for a, b, c, d in faces:
        add_quad(tris, v[a], v[b], v[c], v[d])
    return tris


def make_tube(attrs: Dict[str, str], seg_base: int) -> List[Triangle]:
    rmin = length_value(attrs, "rmin")
    rmax = length_value(attrs, "rmax")
    z = length_value(attrs, "z")
    start = angle_value(attrs, "startphi", 0.0)
    delta = angle_value(attrs, "deltaphi", 360.0)
    return make_tube_like(rmin, rmax, rmin, rmax, z, start, delta, seg_base)


def make_cut_tube(attrs: Dict[str, str], seg_base: int) -> List[Triangle]:
    rmin = length_value(attrs, "rmin")
    rmax = length_value(attrs, "rmax")
    z = length_value(attrs, "z")
    start = angle_value(attrs, "startphi", 0.0)
    delta = angle_value(attrs, "deltaphi", 360.0)
    low = (as_float(attrs, "lowX"), as_float(attrs, "lowY"), as_float(attrs, "lowZ", -1.0))
    high = (as_float(attrs, "highX"), as_float(attrs, "highY"), as_float(attrs, "highZ", 1.0))
    return make_tube_like(rmin, rmax, rmin, rmax, z, start, delta, seg_base, low, high)


def make_cone(attrs: Dict[str, str], seg_base: int) -> List[Triangle]:
    z = length_value(attrs, "z")
    start = angle_value(attrs, "startphi", 0.0)
    delta = angle_value(attrs, "deltaphi", 360.0)
    return make_tube_like(
        length_value(attrs, "rmin1"), length_value(attrs, "rmax1"),
        length_value(attrs, "rmin2"), length_value(attrs, "rmax2"),
        z, start, delta, seg_base,
    )


@dataclass
class SolidDef:
    name: str
    tag: str
    element: ET.Element
    seq: int


@dataclass
class VolumeDef:
    name: str
    material: str
    solid_ref: str
    solid_seq_hint: int
    element: ET.Element
    seq: int


@dataclass
class ObjectInfo:
    id: int
    name: str
    path: str
    volume: str
    volume_seq: int
    material: str
    solid: str
    solid_type: str
    first_vertex: int
    vertex_count: int
    bbox: List[float]


@dataclass
class SceneBuilder:
    gdml_path: Path
    quality: str = "medium"
    include_containers: bool = False
    include_world: bool = False
    max_objects: int = 100000
    max_vertices: int = 1500000
    warn_limit: int = 100
    root: ET.Element = field(init=False)
    solids_by_name: Dict[str, List[SolidDef]] = field(default_factory=dict)
    volumes_by_name: Dict[str, List[VolumeDef]] = field(default_factory=dict)
    world_name: str = ""
    warnings: List[str] = field(default_factory=list)
    warning_counts: Dict[str, int] = field(default_factory=dict)
    positions: List[float] = field(default_factory=list)
    colors: bytearray = field(default_factory=bytearray)
    ids: List[int] = field(default_factory=list)
    objects: List[ObjectInfo] = field(default_factory=list)
    material_counts: Dict[str, int] = field(default_factory=dict)
    solid_type_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_volume_names: Dict[str, int] = field(default_factory=dict)
    duplicate_solid_names: Dict[str, int] = field(default_factory=dict)
    bbox_min: List[float] = field(default_factory=lambda: [float("inf"), float("inf"), float("inf")])
    bbox_max: List[float] = field(default_factory=lambda: [float("-inf"), float("-inf"), float("-inf")])
    mesh_cache: Dict[Tuple[str, int, str], Tuple[List[Triangle], str, str]] = field(default_factory=dict)
    stop_reason: str = ""

    def __post_init__(self) -> None:
        self.root = ET.parse(self.gdml_path).getroot()
        self.seg_base = QUALITY_SEGMENTS[self.quality]

    def warn(self, msg: str) -> None:
        self.warning_counts[msg] = self.warning_counts.get(msg, 0) + 1
        if self.warning_counts[msg] == 1 and len(self.warnings) < self.warn_limit:
            self.warnings.append(msg)

    def parse(self) -> None:
        solids_el = child_by_tag(self.root, "solids")
        if solids_el is None:
            raise ValueError("No <solids> section found")
        solid_counts: Dict[str, int] = {}
        for el in list(solids_el):
            name = el.attrib.get("name", f"anonymous_{len(solid_counts)}")
            seq = solid_counts.get(name, 0)
            solid_counts[name] = seq + 1
            self.solids_by_name.setdefault(name, []).append(SolidDef(name, strip_namespace(el.tag), el, seq))
        self.duplicate_solid_names = {k: len(v) for k, v in self.solids_by_name.items() if len(v) > 1}

        structure_el = child_by_tag(self.root, "structure")
        if structure_el is None:
            raise ValueError("No <structure> section found")
        volume_counts: Dict[str, int] = {}
        solid_use_counter: Dict[str, int] = {}
        for el in children_by_tag(structure_el, "volume"):
            name = el.attrib.get("name", f"volume_{len(volume_counts)}")
            seq = volume_counts.get(name, 0)
            volume_counts[name] = seq + 1
            matref = child_by_tag(el, "materialref")
            solidref = child_by_tag(el, "solidref")
            mat = matref.attrib.get("ref", "") if matref is not None else ""
            solid = solidref.attrib.get("ref", "") if solidref is not None else ""
            hint = solid_use_counter.get(solid, 0)
            solid_use_counter[solid] = hint + 1
            self.volumes_by_name.setdefault(name, []).append(VolumeDef(name, mat, solid, hint, el, seq))
        self.duplicate_volume_names = {k: len(v) for k, v in self.volumes_by_name.items() if len(v) > 1}

        setup_el = child_by_tag(self.root, "setup")
        if setup_el is not None:
            world_el = child_by_tag(setup_el, "world")
            if world_el is not None:
                self.world_name = world_el.attrib.get("ref", "")
        if not self.world_name:
            # Fall back to the last volume if no setup exists.
            self.world_name = next(reversed(self.volumes_by_name.keys()))

    def select_solid(self, name: str, hint: int = 0) -> Optional[SolidDef]:
        defs = self.solids_by_name.get(name)
        if not defs:
            self.warn(f"Missing solid reference: {name}")
            return None
        idx = hint if 0 <= hint < len(defs) else min(hint % len(defs), len(defs) - 1)
        return defs[idx]

    def select_volume(self, name: str, occurrence: int = 0, total_occurrences: int = 1) -> List[VolumeDef]:
        defs = self.volumes_by_name.get(name)
        if not defs:
            self.warn(f"Missing volume reference: {name}")
            return []
        n = len(defs)
        if n == 1:
            return [defs[0]]
        if total_occurrences >= n and total_occurrences % n == 0:
            block = max(1, total_occurrences // n)
            idx = min(n - 1, occurrence // block)
            return [defs[idx]]
        # If the relationship is not clean, cycle through duplicated definitions. This is safer
        # than exploding all alternatives for large segmented detectors.
        idx = occurrence % n
        return [defs[idx]]

    def solid_mesh(self, solid_ref: str, hint: int = 0) -> Tuple[List[Triangle], str, str]:
        key = (solid_ref, hint, self.quality)
        if key in self.mesh_cache:
            return self.mesh_cache[key]
        solid = self.select_solid(solid_ref, hint)
        if solid is None:
            self.mesh_cache[key] = ([], "missing", solid_ref)
            return self.mesh_cache[key]
        tag = solid.tag
        attrs = solid.element.attrib
        if tag == "box":
            tris = make_box(attrs)
        elif tag == "tube":
            tris = make_tube(attrs, self.seg_base)
        elif tag == "cutTube":
            tris = make_cut_tube(attrs, self.seg_base)
        elif tag == "cone":
            tris = make_cone(attrs, self.seg_base)
        elif tag in {"subtraction", "union", "intersection"}:
            first = child_by_tag(solid.element, "first")
            first_ref = first.attrib.get("ref") if first is not None else ""
            tris, subtype, _ = self.solid_mesh(first_ref, solid.seq)
            # Apply the optional transform attached to the first boolean operand if present.
            # For subtraction, this visualizer approximates the result by the minuend mesh.
            if tag != "union":
                self.warn(
                    f"Boolean {tag!r} for solid name {solid.name!r} is approximated by its first operand."
                )
            else:
                second = child_by_tag(solid.element, "second")
                second_ref = second.attrib.get("ref") if second is not None else ""
                tris2, _, _ = self.solid_mesh(second_ref, solid.seq)
                pos = parse_position(child_by_tag(solid.element, "position"))
                rot = parse_rotation(child_by_tag(solid.element, "rotation"))
                tris = tris + transform_mesh(tris2, matmul(translate(*pos), euler_xyz(*rot)))
            tag = f"{tag}/{subtype}"
        else:
            tris = []
            self.warn(f"Unsupported solid type {tag!r} for {solid.name}#{solid.seq}")
        self.mesh_cache[key] = (tris, tag, solid.name)
        return self.mesh_cache[key]

    def traverse(self, volume: VolumeDef, matrix: Mat4, path: str, depth: int = 0) -> None:
        physvols = children_by_tag(volume.element, "physvol")
        render_this = True
        if volume.name == self.world_name and not self.include_world:
            render_this = False
        if physvols and not self.include_containers:
            # GDML container volumes often use air/vacuum and obscure the leaf detector volumes.
            render_this = False
        if volume.material in {"Galactic", "G4_Galactic"} and not self.include_containers:
            render_this = False

        if render_this:
            self.add_volume_mesh(volume, matrix, path)

        if len(self.objects) >= self.max_objects or len(self.positions) // 3 >= self.max_vertices:
            return

        ref_totals: Dict[str, int] = {}
        for pv in physvols:
            vr = child_by_tag(pv, "volumeref")
            if vr is not None:
                ref = vr.attrib.get("ref", "")
                ref_totals[ref] = ref_totals.get(ref, 0) + 1
        ref_seen: Dict[str, int] = {}
        for pv in physvols:
            vr = child_by_tag(pv, "volumeref")
            if vr is None:
                continue
            ref = vr.attrib.get("ref", "")
            occ = ref_seen.get(ref, 0)
            ref_seen[ref] = occ + 1
            child_matrix = matmul(matrix, placement_matrix(pv))
            child_defs = self.select_volume(ref, occ, ref_totals.get(ref, 1))
            pv_name = pv.attrib.get("name", ref)
            for child in child_defs:
                child_path = f"{path}/{pv_name}:{child.name}#{child.seq}"
                self.traverse(child, child_matrix, child_path, depth + 1)
                if len(self.objects) >= self.max_objects or len(self.positions) // 3 >= self.max_vertices:
                    self.stop_reason = f"Stopped at {len(self.objects)} objects / {len(self.positions)//3} vertices. Increase --max-objects or --max-vertices for more."
                    return

    def add_volume_mesh(self, volume: VolumeDef, matrix: Mat4, path: str) -> None:
        if len(self.objects) >= self.max_objects or len(self.positions) // 3 >= self.max_vertices:
            return
        tris, solid_type, solid_name = self.solid_mesh(volume.solid_ref, volume.solid_seq_hint)
        if not tris:
            return
        obj_id = len(self.objects)
        rgba = material_color(volume.material)
        first_vertex = len(self.positions) // 3
        local_min = [float("inf"), float("inf"), float("inf")]
        local_max = [float("-inf"), float("-inf"), float("-inf")]
        for tri in tris:
            for p in tri:
                tp = transform_point(matrix, p)
                for k in range(3):
                    if tp[k] < local_min[k]: local_min[k] = tp[k]
                    if tp[k] > local_max[k]: local_max[k] = tp[k]
                    if tp[k] < self.bbox_min[k]: self.bbox_min[k] = tp[k]
                    if tp[k] > self.bbox_max[k]: self.bbox_max[k] = tp[k]
                self.positions.extend(tp)
                self.colors.extend(rgba)
                self.ids.append(obj_id)
        vertex_count = len(self.positions) // 3 - first_vertex
        if vertex_count == 0:
            return
        self.objects.append(ObjectInfo(
            id=obj_id,
            name=path.rsplit("/", 1)[-1],
            path=path,
            volume=volume.name,
            volume_seq=volume.seq,
            material=volume.material,
            solid=solid_name,
            solid_type=solid_type,
            first_vertex=first_vertex,
            vertex_count=vertex_count,
            bbox=[*local_min, *local_max],
        ))
        self.material_counts[volume.material] = self.material_counts.get(volume.material, 0) + 1
        self.solid_type_counts[solid_type] = self.solid_type_counts.get(solid_type, 0) + 1

    def build(self) -> Dict[str, object]:
        self.parse()
        world_defs = self.volumes_by_name.get(self.world_name)
        if not world_defs:
            raise ValueError(f"World volume {self.world_name!r} was not found")
        self.traverse(world_defs[0], identity(), f"{self.world_name}:0")
        if self.bbox_min[0] == float("inf"):
            self.bbox_min = [-1.0, -1.0, -1.0]
            self.bbox_max = [1.0, 1.0, 1.0]
        return self.scene_dict()

    def scene_dict(self) -> Dict[str, object]:
        pos_bytes = struct.pack(f"<{len(self.positions)}f", *self.positions) if self.positions else b""
        id_bytes = struct.pack(f"<{len(self.ids)}I", *self.ids) if self.ids else b""
        objects_data = [obj.__dict__ for obj in self.objects]
        warnings = []
        for msg in self.warnings:
            n = self.warning_counts.get(msg, 1)
            warnings.append(f"{msg} (x{n})" if n > 1 else msg)
        if self.stop_reason:
            warnings.append(self.stop_reason)
        return {
            "meta": {
                "source": self.gdml_path.name,
                "quality": self.quality,
                "world": self.world_name,
                "objects": len(self.objects),
                "vertices": len(self.positions) // 3,
                "triangles": len(self.positions) // 9,
                "materials": self.material_counts,
                "solidTypes": self.solid_type_counts,
                "duplicateVolumeNames": self.duplicate_volume_names,
                "duplicateSolidNames": self.duplicate_solid_names,
                "bbox": [*self.bbox_min, *self.bbox_max],
                "warnings": warnings,
            },
            "positions": base64.b64encode(pos_bytes).decode("ascii"),
            "colors": base64.b64encode(bytes(self.colors)).decode("ascii"),
            "ids": base64.b64encode(id_bytes).decode("ascii"),
            "objects": objects_data,
        }


def material_color(name: str) -> Tuple[int, int, int, int]:
    if name in DEFAULT_MATERIAL_COLORS:
        return DEFAULT_MATERIAL_COLORS[name]
    # Deterministic pleasant-ish color from the name.
    h = 2166136261
    for ch in name:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    r = 80 + (h & 0x7F)
    g = 80 + ((h >> 8) & 0x7F)
    b = 80 + ((h >> 16) & 0x7F)
    return (r, g, b, 190)


# -----------------------------------------------------------------------------
# ROOT and branch utilities
# -----------------------------------------------------------------------------


def parse_cli_rootfile() -> Optional[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("rootfile", nargs="?")
    args, _ = parser.parse_known_args()
    return args.rootfile


@st.cache_resource(show_spinner=False)
def open_root_file(path: str):
    return uproot.open(path)


def is_ttree(obj: Any) -> bool:
    return isinstance(obj, uproot.behaviors.TTree.TTree)


def walk_ttrees(root_file: Any) -> list[str]:
    """Return all readable TTree paths in a ROOT file, including nested folders."""

    names: list[str] = []
    for key, obj in root_file.items(recursive=True):
        try:
            if is_ttree(obj):
                names.append(key.split(";")[0])
        except Exception:
            continue
    return sorted(set(names))


def split_collection(branch_name: str) -> tuple[str, str]:
    """Infer a HEP-style collection and field from a branch name.

    Examples
    --------
    Muon_pt      -> (Muon, pt)
    Electron.eta -> (Electron, eta)
    eventNumber  -> (event-level, eventNumber)
    """

    if "." in branch_name:
        head, tail = branch_name.split(".", 1)
        return head or "event-level", tail or branch_name
    if "_" in branch_name:
        head, tail = branch_name.split("_", 1)
        if head and tail:
            return head, tail
    return "event-level", branch_name


@st.cache_data(show_spinner=False)
def get_branch_info(path: str, tree_name: str) -> list[BranchInfo]:
    f = uproot.open(path)
    t = f[tree_name]
    info: list[BranchInfo] = []
    for name in t.keys():
        try:
            branch = t[name]
            typename = str(getattr(branch, "typename", "unknown"))
            interpretation = str(getattr(branch, "interpretation", "unknown"))
        except Exception:
            typename = "unknown"
            interpretation = "unknown"
        collection, field = split_collection(name)
        info.append(BranchInfo(name, typename, interpretation, collection, field))
    return info


@st.cache_data(show_spinner=True, max_entries=64)
def read_branch(path: str, tree_name: str, branch: str, entry_start: int, entry_stop: Optional[int]):
    f = uproot.open(path)
    t = f[tree_name]
    return t[branch].array(library="ak", entry_start=entry_start, entry_stop=entry_stop)


@st.cache_data(show_spinner=True, max_entries=128)
def read_event_branch(path: str, tree_name: str, branch: str, event_index: int):
    f = uproot.open(path)
    t = f[tree_name]
    arr = t[branch].array(library="ak", entry_start=event_index, entry_stop=event_index + 1)
    return arr[0]


def is_probably_numeric_typename(typename: str) -> bool:
    lower = typename.lower()
    return any(token in lower for token in ["int", "float", "double", "bool", "char", "short", "long"])


def numeric_branches(branch_info: list[BranchInfo]) -> list[str]:
    numeric = [b.name for b in branch_info if is_probably_numeric_typename(b.typename)]
    return numeric or [b.name for b in branch_info]


def branch_choices(branches: list[str], query: str = "") -> list[str]:
    if not query.strip():
        return branches
    q = query.casefold()
    return [b for b in branches if q in b.casefold()]


def choose_default(options: Sequence[str], preferred: Sequence[str]) -> int:
    for item in preferred:
        if item in options:
            return list(options).index(item)
    for item in preferred:
        for i, option in enumerate(options):
            if item.casefold() in option.casefold():
                return i
    return 0


# -----------------------------------------------------------------------------
# Array conversion and derived variables
# -----------------------------------------------------------------------------


def to_numpy_flat(arr: Any) -> np.ndarray:
    try:
        return ak.to_numpy(ak.flatten(arr, axis=None))
    except Exception:
        try:
            return np.asarray(arr)
        except Exception:
            return np.array([])


def event_to_numpy(arr: Any) -> np.ndarray:
    try:
        return ak.to_numpy(ak.flatten(arr, axis=None))
    except Exception:
        try:
            return np.asarray(arr)
        except Exception:
            return np.array([])


def safe_float_array(arr: Any) -> np.ndarray:
    out = np.asarray(arr)
    try:
        return out.astype(float, copy=False)
    except Exception:
        return out


class ArrayShapeMismatchError(ValueError):
    """Raised when plot inputs cannot be aligned without changing their meaning."""


def align_arrays(
    arrays: Sequence[np.ndarray],
    *,
    names: Optional[Sequence[str]] = None,
    context: str = "arrays",
) -> list[np.ndarray]:
    """Validate that flat arrays have identical shapes; never truncate values."""

    normalized: list[np.ndarray] = []
    for arr in arrays:
        values = np.asarray(arr)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1:
            raise ArrayShapeMismatchError(
                f"Cannot align {context}: expected flat one-dimensional arrays, "
                f"but received shape {values.shape}."
            )
        normalized.append(values)

    if not normalized:
        return []

    labels = list(names) if names is not None else [f"array {i + 1}" for i in range(len(normalized))]
    if len(labels) != len(normalized):
        raise ValueError("The number of array names must match the number of arrays.")

    reference_shape = normalized[0].shape
    mismatches = [
        f"{label} has shape {values.shape}"
        for label, values in zip(labels, normalized)
        if values.shape != reference_shape
    ]
    if mismatches:
        reference = f"{labels[0]} has shape {reference_shape}"
        raise ArrayShapeMismatchError(
            f"Cannot align {context}: {reference}; " + "; ".join(mismatches) + ". "
            "No values were truncated. Choose variables with matching shapes or use "
            "a per-event reduction such as multiplicity, sum, mean, max, min, or nth."
        )

    return normalized


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    aligned = align_arrays(arrays)
    if not aligned:
        return np.array([], dtype=bool)
    n = len(aligned[0])
    mask = np.ones(n, dtype=bool)
    for arr in aligned:
        try:
            mask &= np.isfinite(arr.astype(float, copy=False))
        except Exception:
            # Non-numeric arrays are left unchanged; downstream plotting may fail
            # with a useful error.
            pass
    return mask


def apply_finite_filter(arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    aligned = align_arrays(arrays)
    mask = finite_mask(*aligned)
    return [a[mask] for a in aligned]


def multiplicity_per_event(arr: Any) -> np.ndarray:
    try:
        return ak.to_numpy(ak.num(arr, axis=1)).astype(float)
    except Exception:
        # Scalar event-level branch: multiplicity is one value per event.
        try:
            return np.ones(len(arr), dtype=float)
        except Exception:
            return np.array([], dtype=float)


def nth_per_event(arr: Any, index: int) -> np.ndarray:
    try:
        selected = ak.pad_none(arr, index + 1, axis=1)[:, index]
        return ak.to_numpy(ak.fill_none(selected, np.nan)).astype(float)
    except Exception:
        flat = to_numpy_flat(arr)
        return safe_float_array(flat)


def reduce_per_event(arr: Any, mode: str) -> np.ndarray:
    try:
        values = ak.values_astype(arr, np.float64)
        if mode == "sum":
            out = ak.sum(values, axis=1)
        elif mode == "mean":
            out = ak.mean(values, axis=1)
        elif mode == "max":
            out = ak.max(values, axis=1, mask_identity=False)
        elif mode == "min":
            out = ak.min(values, axis=1, mask_identity=False)
        else:
            raise ValueError(mode)
        return ak.to_numpy(out).astype(float)
    except Exception:
        return safe_float_array(to_numpy_flat(arr))


def delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))


def read_for_scope(path: str, tree_name: str, branch: str, entry_start: int, entry_stop: Optional[int], event_index: Optional[int]):
    if event_index is None:
        return read_branch(path, tree_name, branch, entry_start, entry_stop)
    return read_event_branch(path, tree_name, branch, event_index)


def branch_values(path: str, tree_name: str, branch: str, entry_start: int, entry_stop: Optional[int], event_index: Optional[int]) -> np.ndarray:
    arr = read_for_scope(path, tree_name, branch, entry_start, entry_stop, event_index)
    return event_to_numpy(arr) if event_index is not None else to_numpy_flat(arr)


def variable_reference_branch(spec: VariableSpec) -> Optional[str]:
    if spec.mode in {"branch", "multiplicity", "nth", "sum", "mean", "max", "min"}:
        return spec.branch
    return spec.branch_a


def variable_is_per_event(spec: VariableSpec) -> bool:
    return spec.mode in {"multiplicity", "nth", "sum", "mean", "max", "min"}


def array_length(arr: Any) -> int:
    values = np.asarray(arr)
    if values.ndim == 0:
        return 1
    return len(values)


def raw_counts_per_event(arr: Any, event_index: Optional[int]) -> np.ndarray:
    """Return the number of scalar values contributed by each event."""

    if event_index is not None:
        if value_is_scalar(arr):
            return np.ones(1, dtype=int)
        return np.asarray([array_length(event_to_numpy(arr))], dtype=int)

    try:
        values = ak.Array(arr)
        while ak.to_layout(values).purelist_depth > 2:
            # Flatten nested object substructure while preserving the outer event axis.
            values = ak.flatten(values, axis=2)
        return ak.to_numpy(ak.num(values, axis=1)).astype(int)
    except Exception:
        try:
            return np.ones(array_length(arr), dtype=int)
        except Exception:
            return np.array([], dtype=int)


def raw_list_shape_levels(arr: Any) -> list[np.ndarray]:
    """Return list lengths at every jagged depth without discarding boundaries."""

    try:
        values = ak.Array(arr)
        depth = ak.to_layout(values).purelist_depth
        return [
            ak.to_numpy(ak.flatten(ak.num(values, axis=axis), axis=None)).astype(int)
            for axis in range(1, depth)
        ]
    except Exception:
        return []


def validate_raw_event_shapes(
    arrays: Sequence[Any],
    names: Sequence[str],
    *,
    entry_start: int,
    event_index: Optional[int],
    context: str,
) -> None:
    """Validate scalar counts and nested list boundaries before flattening."""

    validate_event_counts(
        [raw_counts_per_event(arr, event_index) for arr in arrays],
        names,
        entry_start=entry_start,
        event_index=event_index,
        context=context,
    )
    if len(arrays) <= 1:
        return

    levels = [raw_list_shape_levels(arr) for arr in arrays]
    reference = levels[0]
    for name, candidate in zip(names[1:], levels[1:]):
        if len(candidate) != len(reference):
            raise ArrayShapeMismatchError(
                f"Cannot align {context}: {names[0]!r} and {name!r} have different "
                "nested list depths. Flattening them would discard collection boundaries. "
                "No plot was produced."
            )
        for depth, (reference_counts, candidate_counts) in enumerate(
            zip(reference, candidate), start=1
        ):
            if (
                reference_counts.shape != candidate_counts.shape
                or not np.array_equal(reference_counts, candidate_counts)
            ):
                raise ArrayShapeMismatchError(
                    f"Cannot align {context}: {names[0]!r} and {name!r} have different "
                    f"list boundaries at nested depth {depth}. No values were flattened or "
                    "truncated; choose branches with matching collection structure."
                )


def variable_spec_name(spec: VariableSpec) -> str:
    """Return a compact human-readable name for shape diagnostics."""

    if spec.label:
        return spec.label
    if spec.mode == "branch":
        return str(spec.branch or "branch")
    if spec.mode in {"multiplicity", "nth", "sum", "mean", "max", "min"}:
        return f"{spec.mode}({spec.branch or '?'})"
    branches = [spec.branch_a, spec.branch_b, spec.branch_c, spec.branch_d]
    return f"{spec.mode}({','.join(str(branch) for branch in branches if branch)})"


def validate_event_counts(
    counts: Sequence[np.ndarray],
    names: Sequence[str],
    *,
    entry_start: int,
    event_index: Optional[int],
    context: str,
) -> None:
    """Require identical per-event multiplicities before flattening arrays."""

    if len(counts) <= 1:
        return
    if len(counts) != len(names):
        raise ValueError("The number of count arrays and names must match.")

    normalized = [np.asarray(values, dtype=int).reshape(-1) for values in counts]
    reference = normalized[0]
    for name, candidate in zip(names[1:], normalized[1:]):
        if candidate.shape != reference.shape:
            raise ArrayShapeMismatchError(
                f"Cannot align {context}: {names[0]!r} covers {len(reference)} event(s), "
                f"but {name!r} covers {len(candidate)} event(s). No plot was produced."
            )
        mismatches = np.flatnonzero(candidate != reference)
        if len(mismatches):
            local_index = int(mismatches[0])
            absolute_index = int(event_index) if event_index is not None else int(entry_start) + local_index
            raise ArrayShapeMismatchError(
                f"Cannot align {context}: in event {absolute_index}, {names[0]!r} contributes "
                f"{int(reference[local_index])} value(s), while {name!r} contributes "
                f"{int(candidate[local_index])}. Flattening these branches would mix event "
                "boundaries. No plot was produced; choose matching collections or use "
                "per-event reductions."
            )


def variable_counts_per_event(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: Optional[int],
    event_index: Optional[int],
) -> np.ndarray:
    """Return how many plotted values each event contributes for a variable."""

    if variable_is_per_event(spec):
        ref_branch = variable_reference_branch(spec)
        if not ref_branch:
            return np.array([], dtype=int)
        arr = read_for_scope(path, tree_name, ref_branch, entry_start, entry_stop, event_index)
        return np.ones(len(raw_counts_per_event(arr, event_index)), dtype=int)

    ref_branch = variable_reference_branch(spec)
    if not ref_branch:
        return np.array([], dtype=int)

    arr = read_for_scope(path, tree_name, ref_branch, entry_start, entry_stop, event_index)
    return raw_counts_per_event(arr, event_index)


def value_is_scalar(arr: Any) -> bool:
    try:
        return np.asarray(arr).ndim == 0
    except Exception:
        return False


def weight_is_per_event_array(arr: Any, event_index: Optional[int]) -> bool:
    if event_index is not None:
        return value_is_scalar(arr)
    try:
        ak.num(arr, axis=1)
        return False
    except Exception:
        return True


def histogram_weights_for_values(
    path: str,
    tree_name: str,
    weight_branch: str,
    reference_spec: VariableSpec,
    value_count: int,
    entry_start: int,
    entry_stop: Optional[int],
    event_index: Optional[int],
) -> np.ndarray:
    """Read histogram weights and broadcast per-event weights over object values."""

    raw_weights = read_for_scope(path, tree_name, weight_branch, entry_start, entry_stop, event_index)
    weights = safe_float_array(event_to_numpy(raw_weights) if event_index is not None else to_numpy_flat(raw_weights))
    reference_counts = variable_counts_per_event(
        reference_spec, path, tree_name, entry_start, entry_stop, event_index
    )
    reference_name = variable_spec_name(reference_spec)
    expected_values = int(np.sum(reference_counts))
    if expected_values != value_count:
        raise ArrayShapeMismatchError(
            f"Cannot align weights with {reference_name!r}: its event structure predicts "
            f"{expected_values} value(s), but evaluation produced {value_count}. No plot was produced."
        )

    if weight_is_per_event_array(raw_weights, event_index):
        if array_length(weights) == len(reference_counts):
            return np.repeat(np.asarray(weights).reshape(-1), reference_counts)
        if array_length(weights) == 1:
            return np.repeat(np.asarray(weights).reshape(-1), value_count)
        raise ArrayShapeMismatchError(
            f"Cannot align weight branch {weight_branch!r} with {reference_name!r}: "
            f"there are {array_length(weights)} event weight(s) for {len(reference_counts)} "
            "selected event(s). No plot was produced."
        )

    weight_counts = raw_counts_per_event(raw_weights, event_index)
    validate_event_counts(
        [reference_counts, weight_counts],
        [reference_name, weight_branch],
        entry_start=entry_start,
        event_index=event_index,
        context="object-level histogram weights",
    )
    reference_branch = variable_reference_branch(reference_spec)
    if reference_branch:
        raw_reference = read_for_scope(
            path, tree_name, reference_branch, entry_start, entry_stop, event_index
        )
        validate_raw_event_shapes(
            [raw_reference, raw_weights],
            [reference_name, weight_branch],
            entry_start=entry_start,
            event_index=event_index,
            context="object-level histogram weights",
        )
    validated_weights = align_arrays(
        [weights],
        names=[weight_branch],
        context="object-level histogram weights",
    )[0]
    if len(validated_weights) != value_count:
        raise ArrayShapeMismatchError(
            f"Cannot align weight branch {weight_branch!r} with {reference_name!r}: "
            f"it produced {len(validated_weights)} value(s), while the reference variable "
            f"produced {value_count}. No plot was produced."
        )
    return validated_weights


def variable_is_event_level_array(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: Optional[int],
    event_index: Optional[int],
) -> bool:
    if variable_is_per_event(spec):
        return True

    ref_branch = variable_reference_branch(spec)
    if not ref_branch:
        return False

    arr = read_for_scope(path, tree_name, ref_branch, entry_start, entry_stop, event_index)
    if event_index is not None:
        return value_is_scalar(arr)

    try:
        ak.num(arr, axis=1)
        return False
    except Exception:
        return True


def validate_variable_event_shapes(
    specs: Sequence[VariableSpec],
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: Optional[int],
    event_index: Optional[int],
    *,
    context: str,
) -> None:
    """Require plot-axis variables to contribute equally within every event."""

    selected = [spec for spec in specs if spec is not None]
    names = [variable_spec_name(spec) for spec in selected]
    counts = [
        variable_counts_per_event(spec, path, tree_name, entry_start, entry_stop, event_index)
        for spec in selected
    ]
    validate_event_counts(
        counts,
        names,
        entry_start=entry_start,
        event_index=event_index,
        context=context,
    )

    object_specs = [spec for spec in selected if not variable_is_per_event(spec)]
    if len(object_specs) > 1:
        reference_branches = [variable_reference_branch(spec) for spec in object_specs]
        if all(reference_branches):
            raw_arrays = [
                read_for_scope(
                    path, tree_name, str(branch), entry_start, entry_stop, event_index
                )
                for branch in reference_branches
            ]
            validate_raw_event_shapes(
                raw_arrays,
                [variable_spec_name(spec) for spec in object_specs],
                entry_start=entry_start,
                event_index=event_index,
                context=context,
            )


def read_aligned_branch_values(
    branches: Sequence[str],
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: Optional[int],
    event_index: Optional[int],
    *,
    context: str,
) -> list[np.ndarray]:
    """Read related branches and validate event multiplicities before flattening."""

    raw_arrays = [
        read_for_scope(path, tree_name, branch, entry_start, entry_stop, event_index)
        for branch in branches
    ]
    validate_raw_event_shapes(
        raw_arrays,
        list(branches),
        entry_start=entry_start,
        event_index=event_index,
        context=context,
    )
    values = [
        safe_float_array(event_to_numpy(arr) if event_index is not None else to_numpy_flat(arr))
        for arr in raw_arrays
    ]
    return align_arrays(values, names=list(branches), context=context)


def values_for_reference_shape(
    values: np.ndarray,
    reference_spec: VariableSpec,
    value_count: int,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: Optional[int],
    event_index: Optional[int],
    *,
    values_are_per_event: bool = False,
    values_name: str = "secondary variable",
) -> np.ndarray:
    """Broadcast event-level values to match flattened object-level plot values."""

    counts = variable_counts_per_event(reference_spec, path, tree_name, entry_start, entry_stop, event_index)
    reference_name = variable_spec_name(reference_spec)
    expected_values = int(np.sum(counts))
    if expected_values != value_count:
        raise ArrayShapeMismatchError(
            f"Cannot align values with {reference_name!r}: its event structure predicts "
            f"{expected_values} value(s), but evaluation produced {value_count}. No plot was produced."
        )

    if values_are_per_event and len(counts) and array_length(values) == len(counts):
        return np.repeat(np.asarray(values).reshape(-1), counts)

    if array_length(values) == value_count:
        return align_arrays(
            [values],
            names=[values_name],
            context="plot values",
        )[0]

    if array_length(values) == 1 and value_count != 1 and values_are_per_event:
        return np.repeat(np.asarray(values).reshape(-1), value_count)

    raise ArrayShapeMismatchError(
        f"Cannot align {values_name!r} with {reference_name!r}: it produced "
        f"{array_length(values)} value(s), while the reference variable produced "
        f"{value_count}. No plot was produced."
    )


def variable_values_for_values(
    spec: VariableSpec,
    reference_spec: VariableSpec,
    value_count: int,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: Optional[int],
    event_index: Optional[int],
) -> tuple[np.ndarray, str]:
    values, label = evaluate_variable(spec, path, tree_name, entry_start, entry_stop, event_index)
    values_are_per_event = variable_is_event_level_array(
        spec, path, tree_name, entry_start, entry_stop, event_index
    )
    if not values_are_per_event:
        validate_event_counts(
            [
                variable_counts_per_event(reference_spec, path, tree_name, entry_start, entry_stop, event_index),
                variable_counts_per_event(spec, path, tree_name, entry_start, entry_stop, event_index),
            ],
            [variable_spec_name(reference_spec), label],
            entry_start=entry_start,
            event_index=event_index,
            context="object-level plot variables",
        )
        reference_branch = variable_reference_branch(reference_spec)
        value_branch = variable_reference_branch(spec)
        if reference_branch and value_branch:
            validate_raw_event_shapes(
                [
                    read_for_scope(
                        path, tree_name, reference_branch, entry_start, entry_stop, event_index
                    ),
                    read_for_scope(
                        path, tree_name, value_branch, entry_start, entry_stop, event_index
                    ),
                ],
                [variable_spec_name(reference_spec), label],
                entry_start=entry_start,
                event_index=event_index,
                context="object-level plot variables",
            )
    values = values_for_reference_shape(
        values,
        reference_spec,
        value_count,
        path,
        tree_name,
        entry_start,
        entry_stop,
        event_index,
        values_are_per_event=values_are_per_event,
        values_name=label,
    )
    return values, label


def cut_values_for_values(
    spec: VariableSpec,
    reference_spec: VariableSpec,
    value_count: int,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: Optional[int],
    event_index: Optional[int],
) -> tuple[np.ndarray, str]:
    return variable_values_for_values(
        spec,
        reference_spec,
        value_count,
        path,
        tree_name,
        entry_start,
        entry_stop,
        event_index,
    )


def evaluate_variable(spec: VariableSpec, path: str, tree_name: str, entry_start: int,
                      entry_stop: Optional[int], event_index: Optional[int] = None) -> tuple[np.ndarray, str]:
    """Evaluate a variable specification for either a global range or one event."""

    mode = spec.mode

    if mode == "branch":
        if not spec.branch:
            raise ValueError("No branch selected.")
        return safe_float_array(branch_values(path, tree_name, spec.branch, entry_start, entry_stop, event_index)), spec.branch

    if mode == "multiplicity":
        if not spec.branch:
            raise ValueError("No branch selected.")
        arr = read_for_scope(path, tree_name, spec.branch, entry_start, entry_stop, event_index)
        if event_index is None:
            return multiplicity_per_event(arr), f"N({spec.branch})"
        return np.array([len(event_to_numpy(arr))], dtype=float), f"N({spec.branch})"

    if mode == "nth":
        if not spec.branch:
            raise ValueError("No branch selected.")
        arr = read_for_scope(path, tree_name, spec.branch, entry_start, entry_stop, event_index)
        if event_index is None:
            return nth_per_event(arr, spec.index), f"{spec.branch}[{spec.index}]"
        values = event_to_numpy(arr)
        value = values[spec.index] if len(values) > spec.index else np.nan
        return np.asarray([value], dtype=float), f"{spec.branch}[{spec.index}]"

    if mode in {"sum", "mean", "max", "min"}:
        if not spec.branch:
            raise ValueError("No branch selected.")
        arr = read_for_scope(path, tree_name, spec.branch, entry_start, entry_stop, event_index)
        if event_index is None:
            return reduce_per_event(arr, mode), f"{mode}({spec.branch})"
        values = safe_float_array(event_to_numpy(arr))
        if len(values) == 0:
            empty_value = 0.0 if mode == "sum" else np.nan
            return np.array([empty_value], dtype=float), f"{mode}({spec.branch})"
        func = {"sum": np.nansum, "mean": np.nanmean, "max": np.nanmax, "min": np.nanmin}[mode]
        return np.array([func(values)], dtype=float), f"{mode}({spec.branch})"

    if mode in {"r_xy", "phi_xy", "eta_xyz", "ratio", "difference", "sum_ab", "product", "delta_phi", "delta_r"}:
        if not spec.branch_a or not spec.branch_b:
            raise ValueError("This derived variable needs at least two branches.")

        branches = [spec.branch_a, spec.branch_b]
        if mode == "eta_xyz":
            if not spec.branch_c:
                raise ValueError("eta needs x, y, and z branches.")
            branches.append(spec.branch_c)
        elif mode == "delta_r":
            if not spec.branch_c or not spec.branch_d:
                raise ValueError("delta-R needs eta1, phi1, eta2, and phi2 branches.")
            branches.extend([spec.branch_c, spec.branch_d])

        aligned_values = read_aligned_branch_values(
            branches,
            path,
            tree_name,
            entry_start,
            entry_stop,
            event_index,
            context=f"derived variable {mode!r}",
        )
        a, b = aligned_values[:2]

        if mode == "r_xy":
            return np.sqrt(a * a + b * b), f"r({spec.branch_a},{spec.branch_b})"
        if mode == "phi_xy":
            return np.arctan2(b, a), f"phi({spec.branch_a},{spec.branch_b})"
        if mode == "ratio":
            out = np.full(len(a), np.nan, dtype=float)
            mask = b != 0
            out[mask] = a[mask] / b[mask]
            return out, f"{spec.branch_a}/{spec.branch_b}"
        if mode == "difference":
            return a - b, f"{spec.branch_a}-{spec.branch_b}"
        if mode == "sum_ab":
            return a + b, f"{spec.branch_a}+{spec.branch_b}"
        if mode == "product":
            return a * b, f"{spec.branch_a}*{spec.branch_b}"
        if mode == "delta_phi":
            return delta_phi(a, b), f"Δφ({spec.branch_a},{spec.branch_b})"
        if mode == "eta_xyz":
            c = aligned_values[2]
            r = np.sqrt(a * a + b * b)
            out = np.full(len(r), np.nan, dtype=float)
            mask = np.isfinite(r) & np.isfinite(c) & (r > 0)
            out[mask] = np.arcsinh(c[mask] / r[mask])
            return out, f"eta({spec.branch_a},{spec.branch_b},{spec.branch_c})"
        if mode == "delta_r":
            c, d = aligned_values[2:4]
            return np.sqrt((a - c) ** 2 + delta_phi(b, d) ** 2), f"ΔR({spec.branch_a},{spec.branch_b},{spec.branch_c},{spec.branch_d})"

    raise ValueError(f"Unsupported variable mode: {mode}")


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def downsample_arrays(arrays: Sequence[np.ndarray], max_points: int, seed: int = 12345) -> list[np.ndarray]:
    arrays = align_arrays(arrays)
    if not arrays:
        return []
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(max_points), replace=False)
    return [a[idx] for a in arrays]


def apply_range_cuts(
    arrays: Sequence[np.ndarray],
    cuts: Sequence[tuple[np.ndarray, Optional[float], Optional[float]]],
) -> list[np.ndarray]:
    """Apply all range cuts with one combined mask.

    Every cut is evaluated on the same original read range/event. Building the
    mask once avoids the subtle bug where later cuts could be aligned against
    already-filtered plot arrays instead of the original entries/objects.
    """

    if not cuts:
        return align_arrays(arrays)

    cut_arrays = [cut_var for cut_var, _, _ in cuts]
    aligned = align_arrays([*arrays, *cut_arrays])
    if not aligned:
        return []

    values = aligned[:len(arrays)]
    aligned_cuts = aligned[len(arrays):]
    n = len(values[0]) if values else len(aligned_cuts[0])
    mask = np.ones(n, dtype=bool)

    for cut, (_, lo, hi) in zip(aligned_cuts, cuts):
        cut_values = cut.astype(float, copy=False)
        mask &= np.isfinite(cut_values)
        if lo is not None:
            mask &= cut_values >= lo
        if hi is not None:
            mask &= cut_values <= hi

    return [v[mask] for v in values]


def apply_range_cut(arrays: Sequence[np.ndarray], cut_var: Optional[np.ndarray], lo: Optional[float], hi: Optional[float]) -> list[np.ndarray]:
    # Kept as a small compatibility wrapper for any local code that imports this helper.
    if cut_var is None:
        return align_arrays(arrays)
    return apply_range_cuts(arrays, [(cut_var, lo, hi)])


def parse_optional_float(text: str) -> Optional[float]:
    stripped = str(text).strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def describe_array(name: str, arr: np.ndarray) -> dict[str, Any]:
    if len(arr) == 0:
        return {"variable": name, "n": 0, "min": np.nan, "median": np.nan, "mean": np.nan, "max": np.nan}
    values = arr.astype(float, copy=False)
    return {
        "variable": name,
        "n": int(len(values)),
        "min": float(np.nanmin(values)),
        "median": float(np.nanmedian(values)),
        "mean": float(np.nanmean(values)),
        "max": float(np.nanmax(values)),
    }


def safe_hist2d(ax, x: np.ndarray, y: np.ndarray, bins_x: int, bins_y: int, log_scale: bool, weights: Optional[np.ndarray] = None):
    h, xedges, yedges = np.histogram2d(x, y, bins=(bins_x, bins_y), weights=weights)
    positive = h[np.isfinite(h) & (h > 0)]
    norm = None
    if log_scale and len(positive) > 1 and np.nanmax(positive) > np.nanmin(positive):
        norm = LogNorm(vmin=float(np.nanmin(positive)), vmax=float(np.nanmax(positive)))
    return ax.pcolormesh(xedges, yedges, h.T, shading="auto", norm=norm)


def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("_") or "plot"


def save_matplotlib_figure(fig, outdir: Path, name: str, formats: Sequence[str]) -> list[Path]:
    selected = [fmt for fmt in formats if fmt in {"png", "pdf"}]
    if not selected:
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    base = safe_filename(name)
    saved_paths: list[Path] = []
    for fmt in selected:
        path = outdir / f"{base}.{fmt}"
        if fmt == "png":
            fig.savefig(path, dpi=180, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        saved_paths.append(path)
    return saved_paths


def save_plotly_figure(fig, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{safe_filename(name)}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def matplotlib_figure_bytes(fig, file_format: str) -> bytes:
    """Render a Matplotlib figure for a browser download."""
    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {"format": file_format, "bbox_inches": "tight"}
    if file_format == "png":
        save_kwargs["dpi"] = 180
    fig.savefig(buffer, **save_kwargs)
    return buffer.getvalue()


def plotly_figure_html(fig) -> bytes:
    """Render a Plotly figure as HTML for a browser download."""
    return fig.to_html(include_plotlyjs="cdn").encode("utf-8")


def build_plotly_3d(x, y, z, x_label, y_label, z_label, *, color=None, color_label=None,
                    title="", marker_size=2.5, opacity=0.65, colorscale="Viridis",
                    reversescale=False, aspectmode="data", color_min=None, color_max=None):
    marker = dict(size=marker_size, opacity=opacity, colorscale=colorscale, reversescale=reversescale)
    if color is not None:
        marker.update(color=color, colorbar=dict(title=color_label or "color"))
        if color_min is not None:
            marker["cmin"] = color_min
        if color_max is not None:
            marker["cmax"] = color_max
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=marker)])
    fig.update_traces(
        hovertemplate=(
            f"{x_label}: %{{x:.5g}}<br>{y_label}: %{{y:.5g}}<br>{z_label}: %{{z:.5g}}<br>"
            + (f"{color_label}: %{{marker.color:.5g}}<br>" if color is not None else "")
            + "<extra></extra>"
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label, aspectmode=aspectmode),
        margin=dict(l=0, r=0, b=0, t=42),
    )
    return fig


# -----------------------------------------------------------------------------
# Event ranking
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=True, max_entries=32)
def compute_event_metric(path: str, tree_name: str, branch: str, mode: str, entry_start: int,
                         entry_stop: int, threshold: Optional[float], abs_value: bool):
    arr = read_branch(path, tree_name, branch, entry_start, entry_stop)
    if mode == "multiplicity":
        values = multiplicity_per_event(arr)
    else:
        vals = ak.values_astype(arr, np.float64)
        if abs_value:
            vals = abs(vals)
        try:
            ak.num(vals, axis=1)
            is_jagged = True
        except Exception:
            is_jagged = False

        if is_jagged:
            if mode == "sum":
                values = ak.to_numpy(ak.sum(vals, axis=1)).astype(float)
            elif mode == "mean":
                values = ak.to_numpy(ak.mean(vals, axis=1)).astype(float)
            elif mode == "max":
                values = ak.to_numpy(ak.max(vals, axis=1, mask_identity=False)).astype(float)
            elif mode == "min":
                values = ak.to_numpy(ak.min(vals, axis=1, mask_identity=False)).astype(float)
            elif mode == "count_above":
                values = ak.to_numpy(ak.sum(vals > (threshold or 0.0), axis=1)).astype(float)
            elif mode == "count_below":
                values = ak.to_numpy(ak.sum(vals < (threshold or 0.0), axis=1)).astype(float)
            else:
                raise ValueError(f"Unknown metric: {mode}")
        else:
            flat = safe_float_array(to_numpy_flat(vals))
            if mode in {"sum", "mean", "max", "min"}:
                values = flat
            elif mode == "count_above":
                values = (flat > (threshold or 0.0)).astype(float)
            elif mode == "count_below":
                values = (flat < (threshold or 0.0)).astype(float)
            else:
                raise ValueError(f"Unknown metric: {mode}")
    values = np.asarray(values, dtype=float)
    values[~np.isfinite(values)] = np.nan
    return np.arange(entry_start, entry_start + len(values), dtype=int), values


def metric_dataframe(event_indices: np.ndarray, values: np.ndarray, descending: bool, max_rows: int) -> pd.DataFrame:
    mask = np.isfinite(values)
    df = pd.DataFrame({"event": event_indices[mask].astype(int), "metric": values[mask]})
    return df.sort_values("metric", ascending=not descending).head(max_rows).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Streamlit widgets
# -----------------------------------------------------------------------------


def variable_selector(prefix: str, branches: list[str], preferred: Sequence[str], *, allow_event_reductions: bool = True) -> VariableSpec:
    modes = {
        "Direct branch values": "branch",
        "Multiplicity per event": "multiplicity",
        "nth object per event": "nth",
        "Sum per event": "sum",
        "Mean per event": "mean",
        "Max per event": "max",
        "Min per event": "min",
        "r = sqrt(x² + y²)": "r_xy",
        "phi = atan2(y, x)": "phi_xy",
        "eta = asinh(z / sqrt(x²+y²))": "eta_xyz",
        "A / B": "ratio",
        "A - B": "difference",
        "A + B": "sum_ab",
        "A * B": "product",
        "delta phi": "delta_phi",
        "delta R": "delta_r",
    }
    if not allow_event_reductions:
        for key in ["Multiplicity per event", "nth object per event", "Sum per event", "Mean per event", "Max per event", "Min per event"]:
            modes.pop(key, None)

    mode_label = st.selectbox(f"{prefix}: variable type", list(modes.keys()), key=f"{prefix}_mode")
    mode = modes[mode_label]
    options = branches

    if mode in {"branch", "multiplicity", "nth", "sum", "mean", "max", "min"}:
        branch = st.selectbox(f"{prefix}: branch", options, index=choose_default(options, preferred), key=f"{prefix}_branch")
        index = 0
        if mode == "nth":
            index = st.number_input(f"{prefix}: object index", min_value=0, max_value=1000, value=0, step=1, key=f"{prefix}_index")
        return VariableSpec(mode=mode, branch=branch, index=int(index))

    if mode in {"r_xy", "phi_xy", "ratio", "difference", "sum_ab", "product", "delta_phi"}:
        a = st.selectbox(f"{prefix}: A / x / phi1", options, index=choose_default(options, preferred), key=f"{prefix}_a")
        b = st.selectbox(f"{prefix}: B / y / phi2", options, index=min(1, len(options) - 1), key=f"{prefix}_b")
        return VariableSpec(mode=mode, branch_a=a, branch_b=b)

    if mode == "eta_xyz":
        a = st.selectbox(f"{prefix}: x", options, index=choose_default(options, ["x", "px"]), key=f"{prefix}_x")
        b = st.selectbox(f"{prefix}: y", options, index=choose_default(options, ["y", "py"]), key=f"{prefix}_y")
        c = st.selectbox(f"{prefix}: z", options, index=choose_default(options, ["z", "pz"]), key=f"{prefix}_z")
        return VariableSpec(mode=mode, branch_a=a, branch_b=b, branch_c=c)

    if mode == "delta_r":
        a = st.selectbox(f"{prefix}: eta 1", options, index=choose_default(options, ["eta"]), key=f"{prefix}_eta1")
        b = st.selectbox(f"{prefix}: phi 1", options, index=choose_default(options, ["phi"]), key=f"{prefix}_phi1")
        c = st.selectbox(f"{prefix}: eta 2", options, index=choose_default(options, ["eta"]), key=f"{prefix}_eta2")
        d = st.selectbox(f"{prefix}: phi 2", options, index=choose_default(options, ["phi"]), key=f"{prefix}_phi2")
        return VariableSpec(mode=mode, branch_a=a, branch_b=b, branch_c=c, branch_d=d)

    raise RuntimeError("Unreachable variable selector state.")


PLOT_TYPES = ["1D histogram", "2D histogram", "2D scatter", "3D scatter interactive"]
COLOR_SCALES = ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "Jet"]


def needs_y_variable(plot_type: str) -> bool:
    return plot_type in {"2D histogram", "2D scatter", "3D scatter interactive"}


def needs_z_variable(plot_type: str) -> bool:
    return plot_type == "3D scatter interactive"


def supports_point_color(plot_type: str) -> bool:
    return plot_type in {"2D scatter", "3D scatter interactive"}


def supports_histogram_weights(plot_type: str) -> bool:
    return plot_type in {"1D histogram", "2D histogram"}


def histogram_weight_selector(prefix: str, plot_type: str, branches: list[str]) -> Optional[str]:
    if not supports_histogram_weights(plot_type):
        return None
    st.markdown("### Histogram weights")
    use_weights = st.checkbox("Use weights", value=False, key=f"{prefix}_use_weights")
    if not use_weights:
        return None
    return st.selectbox(
        "weight branch",
        branches,
        index=choose_default(branches, ["weight", "genWeight", "eventWeight"]),
        key=f"{prefix}_weight_branch",
    )


def range_cut_controls(prefix: str, branches: list[str]) -> tuple[list[VariableSpec], list[tuple[Optional[float], Optional[float]]]]:
    st.markdown("### Cuts")
    n_cuts = st.number_input("Number of range cuts", min_value=0, max_value=20, value=0, step=1, key=f"{prefix}_n_cuts")
    cut_specs: list[VariableSpec] = []
    cut_bounds: list[tuple[Optional[float], Optional[float]]] = []
    for i in range(int(n_cuts)):
        with st.expander(f"Cut {i + 1}", expanded=i < 3):
            cut_specs.append(variable_selector(f"{prefix}_cut_{i}", branches, ["pt", "eta", "mass"], allow_event_reductions=False))
            lo = st.text_input(f"cut {i + 1} min", value="", key=f"{prefix}_cut_{i}_min")
            hi = st.text_input(f"cut {i + 1} max", value="", key=f"{prefix}_cut_{i}_max")
            lo_value = parse_optional_float(lo)
            hi_value = parse_optional_float(hi)
            if str(lo).strip() and lo_value is None:
                st.warning(f"Cut {i + 1} min is not a valid number and will be ignored.")
            if str(hi).strip() and hi_value is None:
                st.warning(f"Cut {i + 1} max is not a valid number and will be ignored.")
            cut_bounds.append((lo_value, hi_value))
    return cut_specs, cut_bounds


def plot_style_controls(
    prefix: str,
    plot_type: str,
    *,
    color_enabled: bool = False,
    default_bins: int = 100,
    default_marker_size_2d: float = 2.0,
    default_marker_size_3d: float = 2.5,
    default_opacity_2d: float = 0.45,
    default_opacity_3d: float = 0.65,
    default_equal_aspect: bool = False,
) -> dict[str, Any]:
    """Show only the style settings relevant to the selected plot type."""

    st.markdown("### Style")
    style: dict[str, Any] = {}

    if plot_type in {"1D histogram", "2D histogram", "2D scatter"}:
        style["fig_width"] = st.slider("figure width", 4.0, 16.0, 8.0, key=f"{prefix}_fig_width")
        style["fig_height"] = st.slider("figure height", 3.0, 12.0, 6.0, key=f"{prefix}_fig_height")

    if plot_type == "1D histogram":
        style["bins_x"] = st.slider("bins", 5, 500, default_bins, key=f"{prefix}_bins")
        style["log_y"] = st.checkbox("log y-axis", value=False, key=f"{prefix}_log_y")
        style["grid"] = st.checkbox("grid", value=True, key=f"{prefix}_grid")

    elif plot_type == "2D histogram":
        style["bins_x"] = st.slider("x bins", 5, 500, default_bins, key=f"{prefix}_bins_x")
        style["bins_y"] = st.slider("y bins", 5, 500, default_bins, key=f"{prefix}_bins_y")
        style["log_z"] = st.checkbox("log color scale", value=True, key=f"{prefix}_log_z")
        style["equal_aspect"] = st.checkbox("equal aspect", value=default_equal_aspect, key=f"{prefix}_equal_aspect")
        style["grid"] = st.checkbox("grid", value=True, key=f"{prefix}_grid")

    elif plot_type == "2D scatter":
        style["marker_size_2d"] = st.slider("marker size", 0.2, 50.0, default_marker_size_2d, key=f"{prefix}_marker_size_2d")
        style["opacity_2d"] = st.slider("opacity", 0.05, 1.0, default_opacity_2d, key=f"{prefix}_opacity_2d")
        style["max_points"] = st.number_input("max scatter points", min_value=1_000, max_value=10_000_000, value=DEFAULT_MAX_POINTS, step=10_000, key=f"{prefix}_max_points_2d")
        if color_enabled:
            style["colorscale"] = st.selectbox("color scale", COLOR_SCALES, key=f"{prefix}_colorscale_2d")
            style["reverse_colorscale"] = st.checkbox("reverse color scale", key=f"{prefix}_reverse_colorscale_2d")
            style["color_min_text"] = st.text_input("color min", value="", key=f"{prefix}_color_min_2d")
            style["color_max_text"] = st.text_input("color max", value="", key=f"{prefix}_color_max_2d")
        style["equal_aspect"] = st.checkbox("equal aspect", value=default_equal_aspect, key=f"{prefix}_equal_aspect")
        style["grid"] = st.checkbox("grid", value=True, key=f"{prefix}_grid")

    elif plot_type == "3D scatter interactive":
        style["marker_size_3d"] = st.slider("marker size", 1.0, 20.0, default_marker_size_3d, key=f"{prefix}_marker_size_3d")
        style["opacity_3d"] = st.slider("opacity", 0.05, 1.0, default_opacity_3d, key=f"{prefix}_opacity_3d")
        style["max_points"] = st.number_input("max scatter points", min_value=1_000, max_value=10_000_000, value=DEFAULT_MAX_POINTS, step=10_000, key=f"{prefix}_max_points_3d")
        style["aspectmode"] = st.selectbox("aspect mode", ["data", "cube", "auto", "manual"], key=f"{prefix}_aspectmode")
        if color_enabled:
            style["colorscale"] = st.selectbox("color scale", COLOR_SCALES, key=f"{prefix}_colorscale_3d")
            style["reverse_colorscale"] = st.checkbox("reverse color scale", key=f"{prefix}_reverse_colorscale_3d")
            style["color_min_text"] = st.text_input("color min", value="", key=f"{prefix}_color_min_3d")
            style["color_max_text"] = st.text_input("color max", value="", key=f"{prefix}_color_max_3d")

    return style


def label_export_controls(prefix: str, plot_type: str, *, default_title: str = "", default_save_name: str = "plot") -> dict[str, Any]:
    st.markdown("### Labels")
    controls = {
        "title": st.text_input("title", value=default_title, key=f"{prefix}_title"),
        "custom_x": st.text_input("x label override", value="", key=f"{prefix}_custom_x"),
        "custom_y": st.text_input("y label override", value="", key=f"{prefix}_custom_y"),
        "custom_z": "",
    }
    if needs_z_variable(plot_type):
        controls["custom_z"] = st.text_input("z label override", value="", key=f"{prefix}_custom_z")

    st.markdown("### Export")
    controls["save_directory"] = st.text_input(
        "save directory",
        value="root_explorer_plots",
        key=f"{prefix}_save_directory",
    )
    controls["save_name"] = st.text_input(
        "save name",
        value=default_save_name,
        key=f"{prefix}_save_name",
    )
    if plot_type != "3D scatter interactive":
        controls["save_png"] = st.checkbox("save PNG", value=True, key=f"{prefix}_save_png")
        controls["save_pdf"] = st.checkbox("save PDF", value=False, key=f"{prefix}_save_pdf")
        if not controls["save_png"] and not controls["save_pdf"]:
            st.warning("Select at least one static export format.")
    return controls


def selected_static_formats(label_controls: dict[str, Any]) -> list[str]:
    formats: list[str] = []
    if label_controls.get("save_png", True):
        formats.append("png")
    if label_controls.get("save_pdf", False):
        formats.append("pdf")
    return formats


def matplotlib_cmap_name(colorscale: str, reverse: bool = False) -> str:
    name = colorscale.lower()
    return f"{name}_r" if reverse else name


def render_static_plot(
    fig,
    ax,
    plot_type: str,
    x: np.ndarray,
    y: Optional[np.ndarray],
    x_label: str,
    y_label: str,
    style: dict[str, Any],
    *,
    title: str,
    weights: Optional[np.ndarray] = None,
    weight_branch: Optional[str] = None,
    color: Optional[np.ndarray] = None,
    color_label: Optional[str] = None,
):
    if plot_type == "1D histogram":
        ax.hist(x, bins=style["bins_x"], weights=weights)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label or ("entries" if weights is None else f"weighted entries ({weight_branch})"))
        if style.get("log_y", False):
            ax.set_yscale("log")

    elif plot_type == "2D histogram":
        if y is None:
            raise ValueError("2D histogram requires a y variable.")
        mesh = safe_hist2d(ax, x, y, style["bins_x"], style["bins_y"], style["log_z"], weights=weights)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(mesh, ax=ax, label="entries" if weights is None else f"sum weights ({weight_branch})")

    elif plot_type == "2D scatter":
        if y is None:
            raise ValueError("2D scatter requires a y variable.")
        if color is not None:
            cmin = parse_optional_float(style.get("color_min_text", ""))
            cmax = parse_optional_float(style.get("color_max_text", ""))
            scatter = ax.scatter(
                x,
                y,
                s=style["marker_size_2d"],
                alpha=style["opacity_2d"],
                c=color,
                cmap=matplotlib_cmap_name(style.get("colorscale", "Viridis"), style.get("reverse_colorscale", False)),
                vmin=cmin,
                vmax=cmax,
            )
            fig.colorbar(scatter, ax=ax, label=color_label or "color")
        else:
            ax.scatter(x, y, s=style["marker_size_2d"], alpha=style["opacity_2d"])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    else:
        raise ValueError(f"Unsupported static plot type: {plot_type}")

    ax.set_title(title or plot_type)
    if style.get("equal_aspect", False):
        ax.set_aspect("equal", adjustable="box")
    if style.get("grid", False):
        ax.grid(alpha=0.3)


def show_cached_plot_result(result_key: str, outdir: Path, *, save_button_key: str, empty_message: str) -> None:
    """Display the most recently generated plot and provide a save button.

    Streamlit buttons are momentary: clicking a save button triggers a rerun where
    the original "Make plot" button is no longer pressed. Keeping the last
    figure in session state lets the save button exist on that rerun.
    """

    result = st.session_state.get(result_key)
    if result is None:
        st.info(empty_message)
        return

    st.caption("Showing the most recently generated plot. Click the make-plot button again to update it after changing settings.")

    summary = result.get("summary")
    if summary is not None:
        st.dataframe(summary, width="stretch")

    preview = result.get("preview")
    if preview is not None and len(preview) > 0:
        st.markdown("#### Data preview")
        st.dataframe(preview, width="stretch")

    fig = result["figure"]
    save_name = result["save_name"]
    download_name = safe_filename(save_name)

    if result["kind"] == "plotly":
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### Export")
        st.caption("Save writes to the cluster; Download sends the file to this browser.")
        save_col, download_col = st.columns(2)
        with save_col:
            if st.button(result.get("save_button_label", "Save HTML"), key=save_button_key, width="stretch"):
                saved_path = save_plotly_figure(fig, outdir, save_name)
                st.success(f"Saved: {saved_path}")
        with download_col:
            st.download_button(
                "Download HTML",
                data=lambda: plotly_figure_html(fig),
                file_name=f"{download_name}.html",
                mime="text/html",
                key=f"{save_button_key}_download_html",
                on_click="ignore",
                width="stretch",
            )
    elif result["kind"] == "matplotlib":
        st.pyplot(fig)
        st.markdown("#### Export")
        st.caption("Save writes to the cluster; Download sends the file to this browser.")
        formats = result.get("save_formats", ["png"])
        export_columns = st.columns(1 + max(len(formats), 1))
        with export_columns[0]:
            if st.button(result.get("save_button_label", "Save plot"), key=save_button_key, width="stretch"):
                saved_paths = save_matplotlib_figure(fig, outdir, save_name, formats)
                if saved_paths:
                    st.success("Saved: " + ", ".join(str(path) for path in saved_paths))
                else:
                    st.warning("No static export format selected. Select PNG, PDF, or both before making the plot.")
        mime_types = {"png": "image/png", "pdf": "application/pdf"}
        for column, file_format in zip(export_columns[1:], formats):
            with column:
                st.download_button(
                    f"Download {file_format.upper()}",
                    data=lambda fmt=file_format: matplotlib_figure_bytes(fig, fmt),
                    file_name=f"{download_name}.{file_format}",
                    mime=mime_types[file_format],
                    key=f"{save_button_key}_download_{file_format}",
                    on_click="ignore",
                    width="stretch",
                )
    else:
        st.error(f"Unknown cached plot kind: {result['kind']}")

def collection_summary(branch_info: list[BranchInfo]) -> pd.DataFrame:
    rows = []
    by_collection: dict[str, list[BranchInfo]] = {}
    for b in branch_info:
        by_collection.setdefault(b.collection, []).append(b)
    for collection, items in sorted(by_collection.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        fields = ", ".join(item.field for item in items[:12])
        if len(items) > 12:
            fields += ", ..."
        rows.append({"collection": collection, "branches": len(items), "example fields": fields})
    return pd.DataFrame(rows)


def store_uploaded_file(uploaded) -> str:
    suffix = Path(uploaded.name).suffix or ".root"
    data = bytes(uploaded.getbuffer())
    digest = hashlib.sha256(data).hexdigest()[:12]
    temp_dir = Path(tempfile.gettempdir()) / "root_explorer_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{safe_filename(Path(uploaded.name).stem)}_{digest}{suffix}"
    path.write_bytes(data)
    return str(path)




# -----------------------------------------------------------------------------
# Geometry scene loading and Plotly overlay helpers
# -----------------------------------------------------------------------------

GEOMETRY_QUALITY_LEVELS = ["fast", "medium", "high"]
DEFAULT_MAX_GEOMETRY_OBJECTS = 50_000
DEFAULT_MAX_GEOMETRY_VERTICES = 1_500_000
DEFAULT_DISPLAY_GEOMETRY_VERTICES = 750_000


@st.cache_data(show_spinner=True, max_entries=8)
def load_geometry_scene(
    path: str,
    quality: str,
    include_containers: bool,
    include_world: bool,
    max_objects: int,
    max_vertices: int,
) -> dict[str, Any]:
    suffix = Path(path).suffix.casefold()
    if suffix != ".gdml":
        raise ValueError(f"Unsupported geometry file extension: {suffix}. Only .gdml files are supported.")
    builder = SceneBuilder(
        Path(path),
        quality=quality,
        include_containers=include_containers,
        include_world=include_world,
        max_objects=max_objects,
        max_vertices=max_vertices,
    )
    return builder.build()


def decode_geometry_scene(scene: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.frombuffer(base64.b64decode(scene.get("positions", "")), dtype="<f4")
    positions = pos.reshape((-1, 3)) if len(pos) else np.empty((0, 3), dtype=np.float32)
    raw_colors = base64.b64decode(scene.get("colors", "")) if scene.get("colors") else b""
    colors = np.frombuffer(raw_colors, dtype=np.uint8).reshape((-1, 4)) if raw_colors else np.empty((0, 4), dtype=np.uint8)
    raw_ids = base64.b64decode(scene.get("ids", "")) if scene.get("ids") else b""
    ids = np.frombuffer(raw_ids, dtype="<u4") if raw_ids else np.zeros(len(positions), dtype=np.uint32)
    return positions, colors, ids


def plotly_color_from_rgba(rgba: Sequence[int], opacity_scale: float = 1.0) -> tuple[str, float]:
    r, g, b = [int(x) for x in rgba[:3]]
    a = int(rgba[3]) if len(rgba) > 3 else 190
    opacity = max(0.0, min(1.0, (a / 255.0) * opacity_scale))
    return f"rgb({r},{g},{b})", opacity


def geometry_materials(scene: Optional[dict[str, Any]]) -> list[str]:
    if not scene:
        return []
    mats = scene.get("meta", {}).get("materials", {})
    if isinstance(mats, dict) and mats:
        return sorted(str(k) for k in mats.keys())
    out = sorted({str(obj.get("material", "geometry")) for obj in scene.get("objects", [])})
    return out or ["geometry"]


def geometry_summary_dataframe(scene: dict[str, Any]) -> pd.DataFrame:
    meta = scene.get("meta", {})
    rows = [
        {"item": "source", "value": meta.get("source", "")},
        {"item": "objects", "value": meta.get("objects", 0)},
        {"item": "triangles", "value": meta.get("triangles", 0)},
        {"item": "vertices", "value": meta.get("vertices", 0)},
        {"item": "world", "value": meta.get("world", "")},
    ]
    bbox = meta.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 6:
        rows.append({"item": "bbox [xmin,ymin,zmin,xmax,ymax,zmax]", "value": ", ".join(f"{x:.4g}" for x in bbox)})
    summary = pd.DataFrame(rows)
    summary["value"] = summary["value"].astype(str)
    return summary


def make_geometry_plotly_traces(
    scene: dict[str, Any],
    *,
    selected_materials: Optional[set[str]] = None,
    opacity_scale: float = 0.30,
    max_vertices: int = DEFAULT_DISPLAY_GEOMETRY_VERTICES,
    scale: float = 1.0,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
    wireframe: bool = False,
    max_wireframe_triangles: int = 20_000,
) -> tuple[list[go.BaseTraceType], dict[str, Any]]:
    positions, colors, _ = decode_geometry_scene(scene)
    objects = scene.get("objects", []) or []
    if len(positions) == 0:
        return [], {"shown_vertices": 0, "shown_objects": 0, "truncated": False}

    if not objects:
        objects = [{
            "id": 0,
            "name": "geometry",
            "material": "geometry",
            "first_vertex": 0,
            "vertex_count": len(positions) - (len(positions) % 3),
        }]

    by_material: dict[str, list[np.ndarray]] = {}
    material_rgba: dict[str, tuple[int, int, int, int]] = {}
    shown_objects = 0
    shown_vertices = 0
    truncated = False
    offset_arr = np.asarray(offset, dtype=float)

    for obj in objects:
        material = str(obj.get("material") or "geometry")
        if selected_materials is not None and material not in selected_materials:
            continue
        first = int(obj.get("first_vertex", 0))
        count = int(obj.get("vertex_count", 0))
        count -= count % 3
        if count <= 0 or first >= len(positions):
            continue
        stop = min(first + count, len(positions))
        stop -= (stop - first) % 3
        if stop <= first:
            continue
        pts = positions[first:stop].astype(float, copy=True)
        if shown_vertices + len(pts) > max_vertices:
            remaining = max_vertices - shown_vertices
            remaining -= remaining % 3
            if remaining <= 0:
                truncated = True
                break
            pts = pts[:remaining]
            truncated = True
        pts = pts * float(scale) + offset_arr
        by_material.setdefault(material, []).append(pts)
        if material not in material_rgba:
            if len(colors) > first:
                material_rgba[material] = tuple(int(v) for v in colors[first])  # type: ignore[assignment]
            else:
                material_rgba[material] = material_color(material)
        shown_objects += 1
        shown_vertices += len(pts)
        if shown_vertices >= max_vertices:
            truncated = True
            break

    traces: list[go.BaseTraceType] = []
    for material, chunks in by_material.items():
        if not chunks:
            continue
        pts = np.vstack(chunks)
        if len(pts) < 3:
            continue
        n_tri = len(pts) // 3
        trim = n_tri * 3
        pts = pts[:trim]
        tri = np.arange(trim, dtype=np.int64).reshape(-1, 3)
        rgb, opacity = plotly_color_from_rgba(material_rgba.get(material, material_color(material)), opacity_scale)
        traces.append(go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
            name=f"geometry: {material}",
            color=rgb,
            opacity=opacity,
            flatshading=True,
            hoverinfo="skip",
            showscale=False,
            legendgroup="geometry",
        ))
        if wireframe:
            tris = pts.reshape(-1, 3, 3)
            if len(tris) > max_wireframe_triangles:
                tris = tris[:max_wireframe_triangles]
            xs: list[float | None] = []
            ys: list[float | None] = []
            zs: list[float | None] = []
            for a, b, c in tris:
                for p, q in ((a, b), (b, c), (c, a)):
                    xs.extend([float(p[0]), float(q[0]), None])
                    ys.extend([float(p[1]), float(q[1]), None])
                    zs.extend([float(p[2]), float(q[2]), None])
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                name=f"wire: {material}",
                line=dict(width=1, color=rgb),
                hoverinfo="skip",
                showlegend=False,
                legendgroup="geometry",
            ))

    return traces, {"shown_vertices": shown_vertices, "shown_objects": shown_objects, "truncated": truncated}


def overlay_geometry_on_plotly(fig: go.Figure, scene: Optional[dict[str, Any]], options: Optional[dict[str, Any]]) -> go.Figure:
    if not scene or not options or not options.get("enabled", False):
        return fig
    selected = options.get("materials")
    selected_set = set(selected) if selected is not None else None
    traces, info = make_geometry_plotly_traces(
        scene,
        selected_materials=selected_set,
        opacity_scale=float(options.get("opacity", 0.30)),
        max_vertices=int(options.get("max_vertices", DEFAULT_DISPLAY_GEOMETRY_VERTICES)),
        scale=float(options.get("scale", 1.0)),
        offset=(float(options.get("offset_x", 0.0)), float(options.get("offset_y", 0.0)), float(options.get("offset_z", 0.0))),
        wireframe=bool(options.get("wireframe", False)),
    )
    if not traces:
        return fig
    new_fig = go.Figure(data=[*traces, *list(fig.data)], layout=fig.layout)
    new_fig.update_layout(legend=dict(itemsizing="constant"))
    base_meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    new_fig.update_layout(meta={**base_meta, "geometry_overlay": info})
    return new_fig


# -----------------------------------------------------------------------------
# App body
# -----------------------------------------------------------------------------


cli_path = parse_cli_rootfile()
st.title(APP_TITLE)
st.caption("A general-purpose Streamlit app for interactive plotting and event inspection of HEP ROOT TTrees in the web browser.")

with st.sidebar:
    st.header("Input")
    source_mode = st.radio("File source", ["Path", "Upload"], horizontal=True)
    root_path = ""
    if source_mode == "Path":
        root_path = st.text_input("ROOT file path", value=cli_path or "")
    else:
        uploaded = st.file_uploader("Upload ROOT file", type=["root"])
        if uploaded is not None:
            root_path = store_uploaded_file(uploaded)
            st.caption(f"Uploaded copy: `{root_path}`")

    if not root_path:
        st.info("Provide a ROOT file to begin.")
        st.stop()

    root_path_obj = Path(root_path)
    if source_mode == "Path" and not root_path_obj.exists() and not str(root_path).startswith(("root://", "http://", "https://")):
        st.error(f"File does not exist: {root_path}")
        st.stop()

    try:
        root_file = open_root_file(root_path)
    except Exception as exc:
        st.error(f"Could not open ROOT file: {exc}")
        st.stop()

    available_trees = walk_ttrees(root_file)
    if not available_trees:
        st.error("No TTrees were found. This app currently focuses on TTree-like event data.")
        st.stop()

    tree_name = st.selectbox("TTree", available_trees, index=0)
    tree = root_file[tree_name]
    n_entries = int(tree.num_entries)
    st.write(f"Entries/events: **{n_entries:,}**")
    if n_entries <= 0:
        st.warning("This TTree has no entries to inspect.")
        st.stop()

    st.subheader("Read range")
    entry_start = st.number_input("entry start", min_value=0, max_value=max(n_entries - 1, 0), value=0, step=1)
    default_stop = min(n_entries, DEFAULT_MAX_ENTRIES)
    entry_stop = st.number_input("entry stop", min_value=1, max_value=n_entries, value=default_stop, step=1)
    if entry_stop <= entry_start:
        st.warning("entry stop must be larger than entry start.")
        st.stop()


    st.subheader("Detector geometry overlay")
    geometry_enabled = st.checkbox("Enable 3D detector overlay", value=False)
    geometry_scene = None
    geometry_options: dict[str, Any] = {"enabled": False}
    if geometry_enabled:
        geometry_source_mode = st.radio("Geometry source", ["Path", "Upload"], horizontal=True, key="geometry_source_mode")
        geometry_path = ""
        if geometry_source_mode == "Path":
            geometry_path = st.text_input("GDML geometry path", value="", key="geometry_path")
        else:
            geometry_uploaded = st.file_uploader("Upload GDML geometry", type=["gdml"], key="geometry_upload")
            if geometry_uploaded is not None:
                geometry_path = store_uploaded_file(geometry_uploaded)
                st.caption(f"Uploaded copy: `{geometry_path}`")

        geometry_quality = st.selectbox("mesh quality", GEOMETRY_QUALITY_LEVELS, index=1)
        include_containers = st.checkbox("include container volumes", value=False, help="Usually leave off.")
        include_world = st.checkbox("include world volume", value=False, help="Usually leave off.")
        geometry_build_max_objects = st.number_input("max geometry objects to parse", min_value=100, max_value=500_000, value=DEFAULT_MAX_GEOMETRY_OBJECTS, step=1_000)
        geometry_build_max_vertices = st.number_input("max geometry vertices to parse", min_value=10_000, max_value=20_000_000, value=DEFAULT_MAX_GEOMETRY_VERTICES, step=100_000)
        geometry_display_max_vertices = st.number_input("max geometry vertices to draw", min_value=10_000, max_value=20_000_000, value=DEFAULT_DISPLAY_GEOMETRY_VERTICES, step=50_000)
        geometry_opacity = st.slider("geometry opacity scale", 0.02, 1.0, 0.30, step=0.02)
        geometry_scale = st.number_input("geometry coordinate scale", value=1.0, format="%.8g", help="Use this if ROOT vertices and GDML use different length units.")
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            geometry_offset_x = st.number_input("geom dx", value=0.0, format="%.8g")
        with gc2:
            geometry_offset_y = st.number_input("geom dy", value=0.0, format="%.8g")
        with gc3:
            geometry_offset_z = st.number_input("geom dz", value=0.0, format="%.8g")
        geometry_wireframe = st.checkbox("add wireframe edges", value=False)

        if geometry_path:
            geometry_path_obj = Path(geometry_path)
            if geometry_path_obj.suffix.casefold() != ".gdml":
                st.error("Only `.gdml` geometry files are supported.")
                geometry_path = ""
            elif geometry_source_mode == "Path" and not geometry_path_obj.exists():
                st.error(f"Geometry file does not exist: {geometry_path}")
                geometry_path = ""

        if geometry_path:
            try:
                geometry_scene = load_geometry_scene(
                    geometry_path,
                    geometry_quality,
                    bool(include_containers),
                    bool(include_world),
                    int(geometry_build_max_objects),
                    int(geometry_build_max_vertices),
                )
                meta = geometry_scene.get("meta", {})
                st.success(f"Loaded geometry: {meta.get('objects', 0):,} objects, {meta.get('triangles', 0):,} triangles")
                materials = geometry_materials(geometry_scene)
                default_materials = materials[: min(len(materials), 30)]
                selected_materials = st.multiselect("materials to draw", materials, default=default_materials)
                geometry_options = {
                    "enabled": True,
                    "materials": selected_materials,
                    "opacity": float(geometry_opacity),
                    "scale": float(geometry_scale),
                    "offset_x": float(geometry_offset_x),
                    "offset_y": float(geometry_offset_y),
                    "offset_z": float(geometry_offset_z),
                    "wireframe": bool(geometry_wireframe),
                    "max_vertices": int(geometry_display_max_vertices),
                }
                warnings = meta.get("warnings") or []
                if warnings:
                    st.warning("Geometry warnings: " + "; ".join(str(w) for w in warnings[:3]))
            except Exception as exc:
                geometry_scene = None
                geometry_options = {"enabled": False}
                st.error(f"Could not load geometry: {exc}")
        else:
            st.info("Provide a `.gdml` detector description by path or upload to overlay it on 3D plots.")

branch_info = get_branch_info(root_path, tree_name)
all_branches = [b.name for b in branch_info]
branches = numeric_branches(branch_info)

current_data_context = (root_path, tree_name, int(entry_start), int(entry_stop))
if st.session_state.get("data_context") != current_data_context:
    for key in ["plot_result", "event_plot_result", "rank_df", "rank_label"]:
        st.session_state.pop(key, None)
    st.session_state["event_index"] = 0
    st.session_state["data_context"] = current_data_context

plot_tab, event_tab, branch_tab, geometry_tab, help_tab = st.tabs(["Plotter", "Event viewer", "Branches", "Geometry", "Help"])

with branch_tab:
    st.subheader("Branch and collection browser")
    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        st.markdown("#### Collections")
        st.dataframe(collection_summary(branch_info), width="stretch", height=540)
    with c2:
        st.markdown("#### Branches")
        query = st.text_input("Filter branches", value="", key="branch_table_filter")
        df = pd.DataFrame([b.__dict__ for b in branch_info])
        if query:
            q = query.casefold()
            df = df[df.apply(lambda row: q in " ".join(map(str, row.values)).casefold(), axis=1)]
        st.dataframe(df, width="stretch", height=540)

with geometry_tab:
    st.subheader("Detector geometry overlay")
    if geometry_scene is None:
        st.info("Enable the 3D detector overlay in the sidebar and provide a GDML geometry file by path or upload.")
    else:
        st.dataframe(geometry_summary_dataframe(geometry_scene), width="stretch")
        meta = geometry_scene.get("meta", {})
        materials = meta.get("materials", {})
        if isinstance(materials, dict) and materials:
            mat_df = pd.DataFrame([{"material": k, "objects": v} for k, v in sorted(materials.items(), key=lambda kv: (-kv[1], kv[0]))])
            st.markdown("#### Materials")
            st.dataframe(mat_df, width="stretch")
        solid_types = meta.get("solidTypes", {})
        if isinstance(solid_types, dict) and solid_types:
            st.markdown("#### Solid types")
            st.dataframe(pd.DataFrame([{"solid type": k, "objects": v} for k, v in sorted(solid_types.items(), key=lambda kv: (-kv[1], kv[0]))]), width="stretch")
        warnings = meta.get("warnings") or []
        if warnings:
            st.markdown("#### Geometry parser warnings")
            st.write("\n".join(f"- {w}" for w in warnings[:50]))
        st.markdown(
            """
The overlay is only drawn for **3D scatter interactive** plots. The GDML geometry is converted to triangle meshes and added as transparent Plotly `Mesh3d` traces underneath your event or global vertex scatter.

Geometry files can be loaded either from a local `.gdml` path or via upload in the sidebar. Use the scale and offset controls when your ROOT coordinates and GDML coordinates use different units or origins.
"""
        )

with help_tab:
    st.subheader("How to use this app")
    st.markdown(
        """
**Basic workflow**
1. Choose a ROOT file, TTree, and read range in the sidebar. Start with a small range, then increase it once the plot setup looks right.
2. Use the **Plotter** tab for histogram or scatter plots over the selected entry range.
3. Use the **Event viewer** tab to inspect one event at a time. The ranking tool can help find events with large multiplicities, sums, extrema, or threshold counts.
4. Pick variables from the branch selectors. Besides direct branches, the app can derive `r`, `phi`, `eta`, ratios, sums, differences, products, `delta phi`, and `delta R`.
5. Click **Make plot** or **Make event plot** after changing settings. The output area keeps showing the most recently generated plot until you make a new one.

**Jagged collections and cuts**
- For object collections, direct branch values are flattened for plotting.
- In the Plotter, **Multiplicity per event**, **nth object per event**, and per-event reductions such as sum, mean, max, and min are useful for turning jagged collections into one value per event.
- Range cuts are combined with logical AND and applied simultaneously before plotting.
- Histogram weights support the common HEP case where one event-level branch such as `genWeight` should weight every plotted object from that event. Object-level weight branches are also supported when they already match the plotted values.

**Detector geometry overlay**
- Enable **3D detector overlay** in the sidebar and provide a `.gdml` file by path or upload.
- Geometry is drawn only on **3D scatter interactive** plots.
- Use the geometry scale and offset controls if the ROOT coordinates and GDML geometry use different units or origins.
- Keep the geometry vertex limits modest when exploring large detector descriptions.

**Saving plots**
- Static matplotlib plots can be saved as PNG, PDF, or both using the export checkboxes.
- Interactive 3D Plotly plots are saved as HTML.
- The save button exports the currently displayed generated plot, not merely the current control settings.

**Performance tips**
- Large entry ranges, many cached branches, high scatter point limits, and detailed GDML meshes can consume a lot of memory.
- Use smaller read ranges while configuring plots.
- For scatter plots, tune **max scatter points** in the Style section.
- For geometry, reduce the parse/draw vertex limits or disable container/world volumes unless you need them.

Good generic starting points:
- Object kinematics: `pt`, `eta`, `phi`, `mass`, `energy`.
- Event quantities: `MET`, `HT`, `nMuon`, `nElectron`, `nJet`, `run`, `event`, `genWeight`.
- Geometry studies: `x`, `y`, `z`, `r`, `phi`, `eta`.

This app is meant for quick inspection and plot prototyping. Keep final selections and publication plots in a reproducible analysis script.
"""
    )

with plot_tab:
    controls, output = st.columns([0.36, 0.64], gap="large")
    with controls:
        st.subheader("Global plot configuration")
        plot_type = st.selectbox("Plot type", PLOT_TYPES, index=0, key="plot_type")

        st.markdown("### Variables")
        x_spec = variable_selector("x", branches, ["pt", "energy", "eta", "x"])
        y_spec = z_spec = color_spec = None
        if needs_y_variable(plot_type):
            y_spec = variable_selector("y", branches, ["eta", "phi", "y"])
        if needs_z_variable(plot_type):
            z_spec = variable_selector("z", branches, ["phi", "z"])

        use_color = False
        if supports_point_color(plot_type):
            use_color = st.checkbox("Color points", value=plot_type == "3D scatter interactive", key="plot_color_points")
            if use_color:
                color_spec = variable_selector("color", branches, ["pt", "energy", "mass", "pdgId", "charge"], allow_event_reductions=False)

        weight_branch = histogram_weight_selector("plot", plot_type, branches)
        cut_specs, cut_bounds = range_cut_controls("plot", branches)
        style = plot_style_controls(
            "plot",
            plot_type,
            color_enabled=use_color,
            default_bins=100,
            default_marker_size_2d=2.0,
            default_marker_size_3d=2.5,
            default_opacity_2d=0.45,
            default_opacity_3d=0.65,
            default_equal_aspect=False,
        )
        label_controls = label_export_controls("plot", plot_type, default_save_name="root_explorer_plot")
        make_plot = st.button(
            "Make plot",
            type="primary",
            key="plot_make",
            shortcut="P",
        )

    with output:
        st.subheader("Output")
        if make_plot:
            try:
                validate_variable_event_shapes(
                    [spec for spec in [x_spec, y_spec, z_spec] if spec is not None],
                    root_path,
                    tree_name,
                    int(entry_start),
                    int(entry_stop),
                    None,
                    context="plot-axis variables",
                )
                arrays: list[np.ndarray] = []
                labels: list[str] = []
                x, x_label = evaluate_variable(x_spec, root_path, tree_name, int(entry_start), int(entry_stop))
                arrays.append(x); labels.append(x_label)
                if y_spec is not None:
                    y, y_label = evaluate_variable(y_spec, root_path, tree_name, int(entry_start), int(entry_stop))
                    arrays.append(y); labels.append(y_label)
                else:
                    y = None; y_label = ""
                if z_spec is not None:
                    z, z_label = evaluate_variable(z_spec, root_path, tree_name, int(entry_start), int(entry_stop))
                    arrays.append(z); labels.append(z_label)
                else:
                    z = None; z_label = ""
                weights = None
                if weight_branch is not None:
                    weights = histogram_weights_for_values(root_path, tree_name, weight_branch, x_spec, len(x), int(entry_start), int(entry_stop), None)
                    arrays.append(weights)
                color = None; color_label = None
                if color_spec is not None:
                    color, color_label = variable_values_for_values(color_spec, x_spec, len(x), root_path, tree_name, int(entry_start), int(entry_stop), None)
                    arrays.append(color)

                cut_definitions: list[tuple[np.ndarray, Optional[float], Optional[float]]] = []
                for spec, bounds in zip(cut_specs, cut_bounds):
                    cut_values, _ = cut_values_for_values(spec, x_spec, len(x), root_path, tree_name, int(entry_start), int(entry_stop), None)
                    cut_definitions.append((cut_values, bounds[0], bounds[1]))
                arrays = apply_range_cuts(arrays, cut_definitions)

                arrays = apply_finite_filter(arrays)
                idx = 0
                x = arrays[idx]; idx += 1
                if y_spec is not None:
                    y = arrays[idx]; idx += 1
                if z_spec is not None:
                    z = arrays[idx]; idx += 1
                if weight_branch is not None:
                    weights = arrays[idx]; idx += 1
                if color_spec is not None:
                    color = arrays[idx]; idx += 1

                if plot_type in {"2D scatter", "3D scatter interactive"}:
                    ds = [x]
                    if y_spec is not None: ds.append(y)
                    if z_spec is not None: ds.append(z)
                    if color is not None: ds.append(color)
                    ds = downsample_arrays(ds, int(style["max_points"]))
                    x = ds[0]
                    if y_spec is not None: y = ds[1]
                    if z_spec is not None: z = ds[2]
                    if color is not None: color = ds[-1]

                rows = [describe_array(labels[0], x)]
                if y_spec is not None: rows.append(describe_array(labels[1], y))
                if z_spec is not None: rows.append(describe_array(labels[2], z))
                if color is not None: rows.append(describe_array(color_label or "color", color))
                summary_df = pd.DataFrame(rows)

                xlab = label_controls["custom_x"] or labels[0]
                ylab = label_controls["custom_y"] or (labels[1] if y_spec is not None else "entries")
                zlab = label_controls["custom_z"] or (labels[2] if z_spec is not None else "")
                title = label_controls["title"] or plot_type
                save_name = label_controls["save_name"]

                if plot_type == "3D scatter interactive":
                    cmin = parse_optional_float(style.get("color_min_text", ""))
                    cmax = parse_optional_float(style.get("color_max_text", ""))
                    fig3d = build_plotly_3d(
                        x, y, z, xlab, ylab, zlab,
                        color=color,
                        color_label=color_label,
                        title=title,
                        marker_size=style["marker_size_3d"],
                        opacity=style["opacity_3d"],
                        colorscale=style.get("colorscale", "Viridis"),
                        reversescale=style.get("reverse_colorscale", False),
                        aspectmode=style["aspectmode"],
                        color_min=cmin,
                        color_max=cmax,
                    )
                    fig3d = overlay_geometry_on_plotly(fig3d, geometry_scene, geometry_options)
                    st.session_state["plot_result"] = {
                        "kind": "plotly",
                        "figure": fig3d,
                        "summary": summary_df,
                        "preview": None,
                        "save_name": save_name,
                        "save_button_label": "Save 3D HTML",
                    }
                else:
                    fig, ax = plt.subplots(figsize=(style["fig_width"], style["fig_height"]))
                    render_static_plot(
                        fig,
                        ax,
                        plot_type,
                        x,
                        y,
                        xlab,
                        ylab,
                        style,
                        title=title,
                        weights=weights,
                        weight_branch=weight_branch,
                        color=color,
                        color_label=color_label,
                    )
                    st.session_state["plot_result"] = {
                        "kind": "matplotlib",
                        "figure": fig,
                        "summary": summary_df,
                        "preview": None,
                        "save_name": save_name,
                        "save_formats": selected_static_formats(label_controls),
                        "save_button_label": "Save selected formats",
                    }
            except ArrayShapeMismatchError as exc:
                st.session_state.pop("plot_result", None)
                st.warning(str(exc))
            except Exception as exc:
                st.session_state.pop("plot_result", None)
                st.error(f"Plot failed: {exc}")
                st.exception(exc)

        show_cached_plot_result(
            "plot_result",
            Path(label_controls["save_directory"]),
            save_button_key="plot_save_cached",
            empty_message="Configure variables on the left, then click **Make plot**.",
        )

with event_tab:
    controls, output = st.columns([0.36, 0.64], gap="large")
    with controls:
        st.subheader("Event inspection")
        if "event_index" not in st.session_state:
            st.session_state["event_index"] = 0
        event_index = st.number_input("event index", min_value=0, max_value=max(n_entries - 1, 0), value=int(st.session_state["event_index"]), step=1)
        st.session_state["event_index"] = int(event_index)

        with st.expander("Rank events", expanded=True):
            metric_branch = st.selectbox("metric branch", branches, index=choose_default(branches, ["pt", "energy", "mass", "x"]), key="metric_branch")
            metric_mode = st.selectbox("metric", ["multiplicity", "sum", "mean", "max", "min", "count_above", "count_below"], key="metric_mode")
            metric_abs = st.checkbox("use absolute values", value=False)
            threshold = None
            if metric_mode in {"count_above", "count_below"}:
                threshold = st.number_input("threshold", value=0.0)
            r1, r2 = st.columns(2)
            with r1:
                metric_start = st.number_input("ranking start", min_value=0, max_value=max(n_entries - 1, 0), value=0)
            with r2:
                metric_stop = st.number_input("ranking stop", min_value=1, max_value=n_entries, value=min(n_entries, DEFAULT_MAX_ENTRIES))
            descending = st.checkbox("largest first", value=True)
            max_rank_rows = st.slider("rows", 5, 200, 10)
            if st.button("Compute ranking"):
                if int(metric_stop) <= int(metric_start):
                    st.error("ranking stop must be larger than ranking start.")
                else:
                    try:
                        ev, vals = compute_event_metric(root_path, tree_name, metric_branch, metric_mode, int(metric_start), int(metric_stop), threshold, metric_abs)
                        st.session_state["rank_df"] = metric_dataframe(ev, vals, descending, int(max_rank_rows))
                        st.session_state["rank_label"] = f"{metric_mode}({metric_branch})"
                    except Exception as exc:
                        st.error(f"Ranking failed: {exc}")
            if "rank_df" in st.session_state:
                st.write(f"Metric: **{st.session_state.get('rank_label', 'metric')}**")
                st.dataframe(st.session_state["rank_df"], width="stretch", height=220)
                events = st.session_state["rank_df"]["event"].astype(int).tolist()
                if events:
                    selected = st.selectbox("jump to event", events, format_func=lambda e: f"event {e}")
                    if st.button("Use selected event"):
                        st.session_state["event_index"] = int(selected)
                        st.rerun()

        st.subheader("Event plot configuration")
        event_plot_type = st.selectbox("Plot type", PLOT_TYPES, index=0, key="event_plot_type")

        st.markdown("### Variables")
        ex_spec = variable_selector("event_x", branches, ["x", "pt", "eta"], allow_event_reductions=False)
        ey_spec = ez_spec = event_color_spec = None
        if needs_y_variable(event_plot_type):
            ey_spec = variable_selector("event_y", branches, ["y", "eta", "phi"], allow_event_reductions=False)
        if needs_z_variable(event_plot_type):
            ez_spec = variable_selector("event_z", branches, ["z", "phi"], allow_event_reductions=False)

        event_use_color = False
        if supports_point_color(event_plot_type):
            event_use_color = st.checkbox("Color points", value=event_plot_type == "3D scatter interactive", key="event_color_points")
            if event_use_color:
                event_color_spec = variable_selector("event_color", branches, ["energy", "pt", "charge", "pdgId"], allow_event_reductions=False)

        event_weight_branch = histogram_weight_selector("event", event_plot_type, branches)
        event_cut_specs, event_cut_bounds = range_cut_controls("event", branches)
        event_style = plot_style_controls(
            "event",
            event_plot_type,
            color_enabled=event_use_color,
            default_bins=50,
            default_marker_size_2d=8.0,
            default_marker_size_3d=4.0,
            default_opacity_2d=0.75,
            default_opacity_3d=0.75,
            default_equal_aspect=False,
        )
        event_label_controls = label_export_controls("event", event_plot_type, default_title="", default_save_name="root_explorer_event_plot")
        make_event_plot = st.button("Make event plot", type="primary", key="event_make", shortcut="Shift+P")

    with output:
        st.subheader(f"Output: event {int(event_index)}")
        if make_event_plot:
            try:
                validate_variable_event_shapes(
                    [spec for spec in [ex_spec, ey_spec, ez_spec] if spec is not None],
                    root_path,
                    tree_name,
                    int(entry_start),
                    int(entry_stop),
                    int(event_index),
                    context="event plot-axis variables",
                )
                arrays: list[np.ndarray] = []
                labels: list[str] = []
                ex, ex_label = evaluate_variable(ex_spec, root_path, tree_name, int(entry_start), int(entry_stop), int(event_index))
                arrays.append(ex); labels.append(ex_label)
                if ey_spec is not None:
                    ey, ey_label = evaluate_variable(ey_spec, root_path, tree_name, int(entry_start), int(entry_stop), int(event_index))
                    arrays.append(ey); labels.append(ey_label)
                else:
                    ey = None; ey_label = ""
                if ez_spec is not None:
                    ez, ez_label = evaluate_variable(ez_spec, root_path, tree_name, int(entry_start), int(entry_stop), int(event_index))
                    arrays.append(ez); labels.append(ez_label)
                else:
                    ez = None; ez_label = ""
                event_weights = None
                if event_weight_branch is not None:
                    event_weights = histogram_weights_for_values(root_path, tree_name, event_weight_branch, ex_spec, array_length(ex), int(entry_start), int(entry_stop), int(event_index))
                    arrays.append(event_weights)
                event_color = None; event_color_label = None
                if event_color_spec is not None:
                    event_color, event_color_label = variable_values_for_values(event_color_spec, ex_spec, array_length(ex), root_path, tree_name, int(entry_start), int(entry_stop), int(event_index))
                    arrays.append(event_color)
                event_cut_definitions: list[tuple[np.ndarray, Optional[float], Optional[float]]] = []
                for spec, bounds in zip(event_cut_specs, event_cut_bounds):
                    cv, _ = cut_values_for_values(spec, ex_spec, array_length(ex), root_path, tree_name, int(entry_start), int(entry_stop), int(event_index))
                    event_cut_definitions.append((cv, bounds[0], bounds[1]))
                arrays = apply_range_cuts(arrays, event_cut_definitions)
                arrays = apply_finite_filter(arrays)
                idx = 0
                ex = arrays[idx]; idx += 1
                if ey_spec is not None: ey = arrays[idx]; idx += 1
                if ez_spec is not None: ez = arrays[idx]; idx += 1
                if event_weight_branch is not None: event_weights = arrays[idx]; idx += 1
                if event_color_spec is not None: event_color = arrays[idx]; idx += 1

                if event_plot_type in {"2D scatter", "3D scatter interactive"}:
                    ds = [ex]
                    if ey_spec is not None: ds.append(ey)
                    if ez_spec is not None: ds.append(ez)
                    if event_color is not None: ds.append(event_color)
                    ds = downsample_arrays(ds, int(event_style["max_points"]))
                    ex = ds[0]
                    if ey_spec is not None: ey = ds[1]
                    if ez_spec is not None: ez = ds[2]
                    if event_color is not None: event_color = ds[-1]

                rows = [describe_array(labels[0], ex)]
                if ey_spec is not None: rows.append(describe_array(labels[1], ey))
                if ez_spec is not None: rows.append(describe_array(labels[2], ez))
                if event_color is not None: rows.append(describe_array(event_color_label or "color", event_color))
                summary_df = pd.DataFrame(rows)

                preview = {"x": ex}
                if ey_spec is not None: preview["y"] = ey
                if ez_spec is not None: preview["z"] = ez
                if event_weights is not None: preview["weights"] = event_weights
                if event_color is not None: preview["color"] = event_color
                nprev = min(len(v) for v in preview.values()) if preview else 0
                preview_df = None
                if nprev:
                    preview_df = pd.DataFrame({k: v[:nprev] for k, v in preview.items()}).head(500)

                exlab = event_label_controls["custom_x"] or labels[0]
                eylab = event_label_controls["custom_y"] or (labels[1] if ey_spec is not None else "entries")
                ezlab = event_label_controls["custom_z"] or (labels[2] if ez_spec is not None else "")
                event_title = event_label_controls["title"] or f"Event {int(event_index)}"
                event_save_name = f"{event_label_controls['save_name']}_{int(event_index)}"

                if event_plot_type == "3D scatter interactive":
                    cmin = parse_optional_float(event_style.get("color_min_text", ""))
                    cmax = parse_optional_float(event_style.get("color_max_text", ""))
                    fig3d = build_plotly_3d(
                        ex,
                        ey,
                        ez,
                        exlab,
                        eylab,
                        ezlab,
                        color=event_color,
                        color_label=event_color_label,
                        title=event_title,
                        marker_size=event_style["marker_size_3d"],
                        opacity=event_style["opacity_3d"],
                        colorscale=event_style.get("colorscale", "Viridis"),
                        reversescale=event_style.get("reverse_colorscale", False),
                        aspectmode=event_style["aspectmode"],
                        color_min=cmin,
                        color_max=cmax,
                    )
                    fig3d = overlay_geometry_on_plotly(fig3d, geometry_scene, geometry_options)
                    st.session_state["event_plot_result"] = {
                        "kind": "plotly",
                        "figure": fig3d,
                        "summary": summary_df,
                        "preview": preview_df,
                        "save_name": event_save_name,
                        "save_button_label": "Save 3D HTML",
                    }
                else:
                    fig, ax = plt.subplots(figsize=(event_style["fig_width"], event_style["fig_height"]))
                    render_static_plot(
                        fig,
                        ax,
                        event_plot_type,
                        ex,
                        ey,
                        exlab,
                        eylab,
                        event_style,
                        title=event_title,
                        weights=event_weights,
                        weight_branch=event_weight_branch,
                        color=event_color,
                        color_label=event_color_label,
                    )
                    st.session_state["event_plot_result"] = {
                        "kind": "matplotlib",
                        "figure": fig,
                        "summary": summary_df,
                        "preview": preview_df,
                        "save_name": event_save_name,
                        "save_formats": selected_static_formats(event_label_controls),
                        "save_button_label": "Save selected formats",
                    }
            except ArrayShapeMismatchError as exc:
                st.session_state.pop("event_plot_result", None)
                st.warning(str(exc))
            except Exception as exc:
                st.session_state.pop("event_plot_result", None)
                st.error(f"Event plot failed: {exc}")
                st.exception(exc)

        show_cached_plot_result(
            "event_plot_result",
            Path(event_label_controls["save_directory"]),
            save_button_key="event_save_cached",
            empty_message="Choose an event and variables, then click **Make event plot**.",
        )
