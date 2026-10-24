"""GDML parsing, geometry primitives, and detector scene construction."""

from __future__ import annotations

import base64
import math
import struct
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

Vec3 = tuple[float, float, float]
Mat4 = list[list[float]]
Triangle = tuple[Vec3, Vec3, Vec3]

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
# Like Geant4's GDML reader, angles without a unit are in radians.
ANGLE_UNITS = {
    None: 1.0,
    "": 1.0,
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


def child_by_tag(el: ET.Element, tag: str) -> ET.Element | None:
    for ch in list(el):
        if strip_namespace(ch.tag) == tag:
            return ch
    return None


def children_by_tag(el: ET.Element, tag: str) -> list[ET.Element]:
    return [ch for ch in list(el) if strip_namespace(ch.tag) == tag]


def as_float(attrs: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(attrs.get(key, default))
    except (TypeError, ValueError):
        return default


def length_value(attrs: dict[str, str], key: str, default: float = 0.0) -> float:
    return as_float(attrs, key, default) * LEN_UNITS.get(
        attrs.get("lunit") or attrs.get("unit"), 1.0
    )


def angle_value(attrs: dict[str, str], key: str, default_deg: float = 0.0) -> float:
    """Return an angle in radians.

    A missing or unparseable attribute, such as an unevaluated expression, uses
    ``default_deg`` degrees.
    """

    try:
        value = float(attrs[key])
    except (KeyError, TypeError, ValueError):
        return math.radians(default_deg)
    return value * ANGLE_UNITS.get(attrs.get("aunit") or attrs.get("unit"), 1.0)


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


def transform_mesh(tris: Iterable[Triangle], m: Mat4) -> list[Triangle]:
    return [
        (transform_point(m, a), transform_point(m, b), transform_point(m, c)) for a, b, c in tris
    ]


def parse_position(el: ET.Element | None) -> Vec3:
    if el is None:
        return (0.0, 0.0, 0.0)
    attrs = el.attrib
    unit = LEN_UNITS.get(attrs.get("unit"), 1.0)
    return (as_float(attrs, "x") * unit, as_float(attrs, "y") * unit, as_float(attrs, "z") * unit)


def parse_rotation(el: ET.Element | None) -> Vec3:
    if el is None:
        return (0.0, 0.0, 0.0)
    attrs = el.attrib
    unit = ANGLE_UNITS.get(attrs.get("unit"), 1.0)
    return (as_float(attrs, "x") * unit, as_float(attrs, "y") * unit, as_float(attrs, "z") * unit)


def transform_element(
    parent: ET.Element, tag: str, definitions: dict[str, ET.Element] | None = None
) -> ET.Element | None:
    """Resolve an inline or named transform; never silently ignore a broken reference."""
    inline = child_by_tag(parent, tag)
    if inline is not None:
        return inline
    reference = child_by_tag(parent, f"{tag}ref")
    if reference is None:
        return None
    name = reference.attrib.get("ref", "")
    if definitions is None or name not in definitions:
        raise ValueError(f"Missing GDML {tag} reference: {name}")
    return definitions[name]


def placement_matrix(
    physvol: ET.Element,
    positions: dict[str, ET.Element] | None = None,
    rotations: dict[str, ET.Element] | None = None,
) -> Mat4:
    """Return the transform that places a daughter volume in its mother.

    Like Geant4's GDML reader (G4GDMLReadStructure::PhysvolRead), the placement
    uses the inverse of the rotation matrix built from the x, y and z angles: a
    physvol rotation describes the rotation of the daughter's frame, so the
    volume itself is turned the opposite way. Boolean operands are different and
    use the rotation as given (see SceneBuilder.solid_mesh).
    """

    pos = parse_position(transform_element(physvol, "position", positions))
    rx, ry, rz = parse_rotation(transform_element(physvol, "rotation", rotations))
    # (Rz Ry Rx)^-1 = Rx(-x) Ry(-y) Rz(-z)
    inverse = matmul(rotate_x(-rx), matmul(rotate_y(-ry), rotate_z(-rz)))
    return matmul(translate(*pos), inverse)


def add_tri(tris: list[Triangle], a: Vec3, b: Vec3, c: Vec3) -> None:
    # Skip degenerate triangles; they confuse transparency and picking.
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    if nx * nx + ny * ny + nz * nz > 1e-18:
        tris.append((a, b, c))


def add_quad(tris: list[Triangle], a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> None:
    add_tri(tris, a, b, c)
    add_tri(tris, a, c, d)


def angular_segments(delta: float, base_segments: int) -> int:
    full = 2 * math.pi
    frac = min(1.0, max(0.0, abs(delta) / full))
    if frac > 0.98:
        return base_segments
    return max(1, math.ceil(base_segments * frac))


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
    cut_low: Vec3 | None = None,
    cut_high: Vec3 | None = None,
) -> list[Triangle]:
    tris: list[Triangle] = []
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

    low_outer: list[Vec3] = []
    high_outer: list[Vec3] = []
    low_inner: list[Vec3] = []
    high_inner: list[Vec3] = []

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


def make_box(attrs: dict[str, str]) -> list[Triangle]:
    x = length_value(attrs, "x") / 2.0
    y = length_value(attrs, "y") / 2.0
    z = length_value(attrs, "z") / 2.0
    v = [
        (-x, -y, -z),
        (x, -y, -z),
        (x, y, -z),
        (-x, y, -z),
        (-x, -y, z),
        (x, -y, z),
        (x, y, z),
        (-x, y, z),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]
    tris: list[Triangle] = []
    for a, b, c, d in faces:
        add_quad(tris, v[a], v[b], v[c], v[d])
    return tris


def make_tube(attrs: dict[str, str], seg_base: int) -> list[Triangle]:
    rmin = length_value(attrs, "rmin")
    rmax = length_value(attrs, "rmax")
    z = length_value(attrs, "z")
    start = angle_value(attrs, "startphi", 0.0)
    delta = angle_value(attrs, "deltaphi", 360.0)
    return make_tube_like(rmin, rmax, rmin, rmax, z, start, delta, seg_base)


def make_cut_tube(attrs: dict[str, str], seg_base: int) -> list[Triangle]:
    rmin = length_value(attrs, "rmin")
    rmax = length_value(attrs, "rmax")
    z = length_value(attrs, "z")
    start = angle_value(attrs, "startphi", 0.0)
    delta = angle_value(attrs, "deltaphi", 360.0)
    low = (as_float(attrs, "lowX"), as_float(attrs, "lowY"), as_float(attrs, "lowZ", -1.0))
    high = (as_float(attrs, "highX"), as_float(attrs, "highY"), as_float(attrs, "highZ", 1.0))
    return make_tube_like(rmin, rmax, rmin, rmax, z, start, delta, seg_base, low, high)


def make_cone(attrs: dict[str, str], seg_base: int) -> list[Triangle]:
    z = length_value(attrs, "z")
    start = angle_value(attrs, "startphi", 0.0)
    delta = angle_value(attrs, "deltaphi", 360.0)
    return make_tube_like(
        length_value(attrs, "rmin1"),
        length_value(attrs, "rmax1"),
        length_value(attrs, "rmin2"),
        length_value(attrs, "rmax2"),
        z,
        start,
        delta,
        seg_base,
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
    bbox: list[float]


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
    solids_by_name: dict[str, list[SolidDef]] = field(default_factory=dict)
    volumes_by_name: dict[str, list[VolumeDef]] = field(default_factory=dict)
    world_name: str = ""
    warnings: list[str] = field(default_factory=list)
    warning_counts: dict[str, int] = field(default_factory=dict)
    positions: list[float] = field(default_factory=list)
    colors: bytearray = field(default_factory=bytearray)
    ids: list[int] = field(default_factory=list)
    objects: list[ObjectInfo] = field(default_factory=list)
    material_counts: dict[str, int] = field(default_factory=dict)
    solid_type_counts: dict[str, int] = field(default_factory=dict)
    duplicate_volume_names: dict[str, int] = field(default_factory=dict)
    duplicate_solid_names: dict[str, int] = field(default_factory=dict)
    bbox_min: list[float] = field(
        default_factory=lambda: [float("inf"), float("inf"), float("inf")]
    )
    bbox_max: list[float] = field(
        default_factory=lambda: [float("-inf"), float("-inf"), float("-inf")]
    )
    mesh_cache: dict[tuple[str, int, str], tuple[list[Triangle], str, str]] = field(
        default_factory=dict
    )
    stop_reason: str = ""
    position_definitions: dict[str, ET.Element] = field(default_factory=dict)
    rotation_definitions: dict[str, ET.Element] = field(default_factory=dict)
    active_solids: set[tuple[str, int, str]] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.root = ET.parse(self.gdml_path).getroot()
        self.seg_base = QUALITY_SEGMENTS[self.quality]

    def warn(self, msg: str) -> None:
        self.warning_counts[msg] = self.warning_counts.get(msg, 0) + 1
        if self.warning_counts[msg] == 1 and len(self.warnings) < self.warn_limit:
            self.warnings.append(msg)

    def parse(self) -> None:
        definitions = child_by_tag(self.root, "define")
        if definitions is not None:
            self.position_definitions = {
                item.attrib["name"]: item
                for item in children_by_tag(definitions, "position")
                if "name" in item.attrib
            }
            self.rotation_definitions = {
                item.attrib["name"]: item
                for item in children_by_tag(definitions, "rotation")
                if "name" in item.attrib
            }
        self.warn_unsupported_constructs()
        solids_el = child_by_tag(self.root, "solids")
        if solids_el is None:
            raise ValueError("No <solids> section found")
        solid_counts: dict[str, int] = {}
        for el in list(solids_el):
            name = el.attrib.get("name", f"anonymous_{len(solid_counts)}")
            seq = solid_counts.get(name, 0)
            solid_counts[name] = seq + 1
            self.solids_by_name.setdefault(name, []).append(
                SolidDef(name, strip_namespace(el.tag), el, seq)
            )
        self.duplicate_solid_names = {
            k: len(v) for k, v in self.solids_by_name.items() if len(v) > 1
        }

        structure_el = child_by_tag(self.root, "structure")
        if structure_el is None:
            raise ValueError("No <structure> section found")
        volume_counts: dict[str, int] = {}
        solid_use_counter: dict[str, int] = {}
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
            self.volumes_by_name.setdefault(name, []).append(
                VolumeDef(name, mat, solid, hint, el, seq)
            )
        self.duplicate_volume_names = {
            k: len(v) for k, v in self.volumes_by_name.items() if len(v) > 1
        }

        setup_el = child_by_tag(self.root, "setup")
        if setup_el is not None:
            world_el = child_by_tag(setup_el, "world")
            if world_el is not None:
                self.world_name = world_el.attrib.get("ref", "")
        if not self.world_name:
            # Fall back to the last volume if no setup exists.
            if not self.volumes_by_name:
                raise ValueError("No supported GDML volumes were found")
            self.world_name = next(reversed(self.volumes_by_name.keys()))

    def warn_unsupported_constructs(self) -> None:
        """Expose omissions and numeric fallbacks instead of implying full fidelity."""
        unsupported = {
            "assembly",
            "replicavol",
            "paramvol",
            "divisionvol",
            "file",
            "scale",
            "scaleref",
            "firstposition",
            "firstpositionref",
            "firstrotation",
            "firstrotationref",
        }
        numeric_tags = {"box", "tube", "cutTube", "cone", "position", "rotation"}
        for element in self.root.iter():
            tag = strip_namespace(element.tag)
            if tag in unsupported:
                self.warn(
                    f"Unsupported GDML construct {tag!r}; geometry may be omitted or simplified."
                )
            if tag not in numeric_tags:
                continue
            for key, value in element.attrib.items():
                if key == "name":
                    continue
                if key in {"unit", "lunit", "aunit"}:
                    units = ANGLE_UNITS if tag == "rotation" or key == "aunit" else LEN_UNITS
                    if value not in units:
                        self.warn(f"Unknown {tag} unit {value!r}; using the default unit.")
                    continue
                try:
                    number = float(value)
                except ValueError:
                    self.warn(
                        f"Unevaluated {tag} attribute {key}={value!r}; using its default value."
                    )
                else:
                    if not math.isfinite(number):
                        raise ValueError(f"GDML {tag} attribute {key} must be finite")

    def select_solid(self, name: str, hint: int = 0) -> SolidDef | None:
        defs = self.solids_by_name.get(name)
        if not defs:
            self.warn(f"Missing solid reference: {name}")
            return None
        idx = hint if 0 <= hint < len(defs) else min(hint % len(defs), len(defs) - 1)
        return defs[idx]

    def select_volume(
        self, name: str, occurrence: int = 0, total_occurrences: int = 1
    ) -> list[VolumeDef]:
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

    def solid_mesh(self, solid_ref: str, hint: int = 0) -> tuple[list[Triangle], str, str]:
        key = (solid_ref, hint, self.quality)
        if key in self.mesh_cache:
            return self.mesh_cache[key]
        if key in self.active_solids:
            raise ValueError(f"Cyclic GDML solid reference: {solid_ref}")
        solid = self.select_solid(solid_ref, hint)
        if solid is None:
            self.mesh_cache[key] = ([], "missing", solid_ref)
            return self.mesh_cache[key]
        tag = solid.tag
        self.active_solids.add(key)
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
            first_ref = first.attrib.get("ref", "") if first is not None else ""
            tris, subtype, _ = self.solid_mesh(first_ref, solid.seq)
            # For subtraction, this visualizer approximates the result by the minuend mesh.
            if tag != "union":
                self.warn(
                    f"Boolean {tag!r} for solid name {solid.name!r} is approximated by its first operand."
                )
            else:
                second = child_by_tag(solid.element, "second")
                second_ref = second.attrib.get("ref", "") if second is not None else ""
                tris2, _, _ = self.solid_mesh(second_ref, solid.seq)
                pos = parse_position(
                    transform_element(solid.element, "position", self.position_definitions)
                )
                rot = parse_rotation(
                    transform_element(solid.element, "rotation", self.rotation_definitions)
                )
                tris = tris + transform_mesh(tris2, matmul(translate(*pos), euler_xyz(*rot)))
            tag = f"{tag}/{subtype}"
        else:
            tris = []
            self.warn(f"Unsupported solid type {tag!r} for {solid.name}#{solid.seq}")
        self.mesh_cache[key] = (tris, tag, solid.name)
        self.active_solids.remove(key)
        return self.mesh_cache[key]

    def traverse(
        self,
        volume: VolumeDef,
        matrix: Mat4,
        path: str,
        depth: int = 0,
        ancestors: frozenset[tuple[str, int]] = frozenset(),
    ) -> None:
        identity_key = (volume.name, volume.seq)
        if identity_key in ancestors:
            raise ValueError(f"Cyclic GDML volume reference: {volume.name}")
        if depth >= 256:
            raise ValueError("GDML volume nesting exceeds the supported depth of 256")
        ancestors = ancestors | {identity_key}
        physvols = children_by_tag(volume.element, "physvol")
        render_this = True
        if volume.name == self.world_name:
            render_this = self.include_world
        elif physvols and not self.include_containers:
            # GDML container volumes often use air/vacuum and obscure the leaf detector volumes.
            render_this = False
        elif volume.material in {"Galactic", "G4_Galactic"} and not self.include_containers:
            render_this = False

        if render_this:
            self.add_volume_mesh(volume, matrix, path)

        if len(self.objects) >= self.max_objects or len(self.positions) // 3 >= self.max_vertices:
            return

        ref_totals: dict[str, int] = {}
        for pv in physvols:
            vr = child_by_tag(pv, "volumeref")
            if vr is not None:
                ref = vr.attrib.get("ref", "")
                ref_totals[ref] = ref_totals.get(ref, 0) + 1
        ref_seen: dict[str, int] = {}
        for pv in physvols:
            vr = child_by_tag(pv, "volumeref")
            if vr is None:
                continue
            ref = vr.attrib.get("ref", "")
            occ = ref_seen.get(ref, 0)
            ref_seen[ref] = occ + 1
            child_matrix = matmul(
                matrix, placement_matrix(pv, self.position_definitions, self.rotation_definitions)
            )
            child_defs = self.select_volume(ref, occ, ref_totals.get(ref, 1))
            pv_name = pv.attrib.get("name", ref)
            for child in child_defs:
                child_path = f"{path}/{pv_name}:{child.name}#{child.seq}"
                self.traverse(child, child_matrix, child_path, depth + 1, ancestors)
                if (
                    len(self.objects) >= self.max_objects
                    or len(self.positions) // 3 >= self.max_vertices
                ):
                    self.stop_reason = f"Stopped at {len(self.objects)} objects / {len(self.positions) // 3} vertices. Increase the geometry parse limits for more."
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
            if len(self.ids) + 3 > self.max_vertices:
                self.stop_reason = f"Stopped at {len(self.ids)} vertices. Increase the geometry vertex parse limit for more."
                break
            for p in tri:
                tp = transform_point(matrix, p)
                for k in range(3):
                    if tp[k] < local_min[k]:
                        local_min[k] = tp[k]
                    if tp[k] > local_max[k]:
                        local_max[k] = tp[k]
                    if tp[k] < self.bbox_min[k]:
                        self.bbox_min[k] = tp[k]
                    if tp[k] > self.bbox_max[k]:
                        self.bbox_max[k] = tp[k]
                self.positions.extend(tp)
                self.colors.extend(rgba)
                self.ids.append(obj_id)
        vertex_count = len(self.positions) // 3 - first_vertex
        if vertex_count == 0:
            return
        self.objects.append(
            ObjectInfo(
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
            )
        )
        self.material_counts[volume.material] = self.material_counts.get(volume.material, 0) + 1
        self.solid_type_counts[solid_type] = self.solid_type_counts.get(solid_type, 0) + 1

    def build(self) -> dict[str, object]:
        self.parse()
        world_defs = self.volumes_by_name.get(self.world_name)
        if not world_defs:
            raise ValueError(f"World volume {self.world_name!r} was not found")
        self.traverse(world_defs[0], identity(), f"{self.world_name}:0")
        if self.bbox_min[0] == float("inf"):
            self.bbox_min = [-1.0, -1.0, -1.0]
            self.bbox_max = [1.0, 1.0, 1.0]
        return self.scene_dict()

    def scene_dict(self) -> dict[str, object]:
        pos_bytes = (
            struct.pack(f"<{len(self.positions)}f", *self.positions) if self.positions else b""
        )
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


def material_color(name: str) -> tuple[int, int, int, int]:
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
