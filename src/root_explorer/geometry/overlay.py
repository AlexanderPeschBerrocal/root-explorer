"""GDML scene loading and Plotly detector-overlay helpers."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..io.source import SourceVersion, source_version
from .gdml import SceneBuilder, material_color

GEOMETRY_QUALITY_LEVELS = ["fast", "medium", "high"]
DEFAULT_MAX_GEOMETRY_OBJECTS = 50_000
DEFAULT_MAX_GEOMETRY_VERTICES = 1_500_000
DEFAULT_DISPLAY_GEOMETRY_VERTICES = 750_000


def load_geometry_scene(
    path: str,
    quality: str,
    include_containers: bool,
    include_world: bool,
    max_objects: int,
    max_vertices: int,
) -> dict[str, Any]:
    """Parse a GDML file into a scene, rebuilding it when the file changes."""

    return _cached_geometry_scene(
        path,
        source_version(path),
        quality,
        include_containers,
        include_world,
        max_objects,
        max_vertices,
    )


@st.cache_data(show_spinner="Loading GDML geometry…", max_entries=8)
def _cached_geometry_scene(
    path: str,
    version: SourceVersion,
    quality: str,
    include_containers: bool,
    include_world: bool,
    max_objects: int,
    max_vertices: int,
) -> dict[str, Any]:
    """Cached implementation; ``version`` only keys the cache (see ``io.source``)."""

    suffix = Path(path).suffix.casefold()
    if suffix != ".gdml":
        raise ValueError(
            f"Unsupported geometry file extension: {suffix}. Only .gdml files are supported."
        )
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
    colors = (
        np.frombuffer(raw_colors, dtype=np.uint8).reshape((-1, 4))
        if raw_colors
        else np.empty((0, 4), dtype=np.uint8)
    )
    raw_ids = base64.b64decode(scene.get("ids", "")) if scene.get("ids") else b""
    ids = (
        np.frombuffer(raw_ids, dtype="<u4")
        if raw_ids
        else np.zeros(len(positions), dtype=np.uint32)
    )
    return positions, colors, ids


def plotly_color_from_rgba(rgba: Sequence[int], opacity_scale: float = 1.0) -> tuple[str, float]:
    r, g, b = [int(x) for x in rgba[:3]]
    a = int(rgba[3]) if len(rgba) > 3 else 190
    opacity = max(0.0, min(1.0, (a / 255.0) * opacity_scale))
    return f"rgb({r},{g},{b})", opacity


def geometry_materials(scene: dict[str, Any] | None) -> list[str]:
    if not scene:
        return []
    mats = scene.get("meta", {}).get("materials", {})
    if isinstance(mats, dict) and mats:
        return sorted(str(k) for k in mats)
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
        rows.append(
            {
                "item": "bbox [xmin,ymin,zmin,xmax,ymax,zmax]",
                "value": ", ".join(f"{x:.4g}" for x in bbox),
            }
        )
    summary = pd.DataFrame(rows)
    summary["value"] = summary["value"].astype(str)
    return summary


def make_geometry_plotly_traces(
    scene: dict[str, Any],
    *,
    selected_materials: set[str] | None = None,
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
        objects = [
            {
                "id": 0,
                "name": "geometry",
                "material": "geometry",
                "first_vertex": 0,
                "vertex_count": len(positions) - (len(positions) % 3),
            }
        ]

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
        rgb, opacity = plotly_color_from_rgba(
            material_rgba.get(material, material_color(material)), opacity_scale
        )
        traces.append(
            go.Mesh3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                i=tri[:, 0],
                j=tri[:, 1],
                k=tri[:, 2],
                name=f"geometry: {material}",
                color=rgb,
                opacity=opacity,
                flatshading=True,
                hoverinfo="skip",
                showscale=False,
                legendgroup="geometry",
            )
        )
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
            traces.append(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    name=f"wire: {material}",
                    line={"width": 1, "color": rgb},
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup="geometry",
                )
            )

    return traces, {
        "shown_vertices": shown_vertices,
        "shown_objects": shown_objects,
        "truncated": truncated,
    }


def overlay_geometry_on_plotly(
    fig: go.Figure, scene: dict[str, Any] | None, options: dict[str, Any] | None
) -> go.Figure:
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
        offset=(
            float(options.get("offset_x", 0.0)),
            float(options.get("offset_y", 0.0)),
            float(options.get("offset_z", 0.0)),
        ),
        wireframe=bool(options.get("wireframe", False)),
    )
    if not traces:
        return fig
    new_fig = go.Figure(data=[*traces, *list(fig.data)], layout=fig.layout)
    new_fig.update_layout(legend={"itemsizing": "constant"})
    base_meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    new_fig.update_layout(meta={**base_meta, "geometry_overlay": info})
    return new_fig
