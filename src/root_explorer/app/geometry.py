"""GDML geometry controls and geometry-tab display."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ..geometry.overlay import (
    DEFAULT_DISPLAY_GEOMETRY_VERTICES,
    DEFAULT_MAX_GEOMETRY_OBJECTS,
    DEFAULT_MAX_GEOMETRY_VERTICES,
    GEOMETRY_QUALITY_LEVELS,
    geometry_materials,
    geometry_summary_dataframe,
    load_geometry_scene,
)
from ..io.source import SourceUnavailableError, source_version
from .controls import slider_or_number_input
from .results import discard_plot_result, store_uploaded_file
from .state import GeometryState


def render_geometry_sidebar() -> GeometryState:
    """Render sidebar geometry controls and load a scene when requested."""

    st.subheader("Detector geometry overlay", anchor=False)
    enabled = st.checkbox("Enable 3D detector overlay", value=False, key="geometry_enabled")
    if not enabled:
        return GeometryState(scene=None, options={"enabled": False})

    source_mode = st.radio(
        "Geometry source", ["Path", "Upload"], horizontal=True, key="geometry_source_mode"
    )
    geometry_path = ""
    if source_mode == "Path":
        geometry_path = st.text_input("GDML geometry path", value="", key="geometry_path")
    else:
        uploaded = st.file_uploader("Upload GDML geometry", type=["gdml"], key="geometry_upload")
        if uploaded is not None:
            geometry_path = store_uploaded_file(uploaded)
            st.caption(f"Uploaded copy: `{geometry_path}`")

    quality = st.selectbox("mesh quality", GEOMETRY_QUALITY_LEVELS, index=1)
    include_containers = st.checkbox(
        "include container volumes", value=False, help="Usually leave off."
    )
    include_world = st.checkbox("include world volume", value=False, help="Usually leave off.")
    max_objects = st.number_input(
        "max geometry objects to parse", 100, 500_000, DEFAULT_MAX_GEOMETRY_OBJECTS, 1_000
    )
    max_vertices = st.number_input(
        "max geometry vertices to parse", 10_000, 20_000_000, DEFAULT_MAX_GEOMETRY_VERTICES, 100_000
    )
    display_max_vertices = st.number_input(
        "max geometry vertices to draw",
        10_000,
        20_000_000,
        DEFAULT_DISPLAY_GEOMETRY_VERTICES,
        50_000,
    )
    opacity = slider_or_number_input(
        "geometry opacity scale", 0.02, 1.0, 0.30, step=0.02, key="geometry_opacity"
    )
    scale = st.number_input(
        "geometry coordinate scale",
        value=1.0,
        format="%.8g",
        help="Use this if ROOT vertices and GDML use different length units.",
    )
    columns = st.columns(3)
    with columns[0]:
        offset_x = st.number_input("geom dx", value=0.0, format="%.8g")
    with columns[1]:
        offset_y = st.number_input("geom dy", value=0.0, format="%.8g")
    with columns[2]:
        offset_z = st.number_input("geom dz", value=0.0, format="%.8g")
    wireframe = st.checkbox("add wireframe edges", value=False)

    if geometry_path:
        path = Path(geometry_path)
        if path.suffix.casefold() != ".gdml":
            st.error("Only `.gdml` geometry files are supported.")
            geometry_path = ""
        elif source_mode == "Path" and not path.exists():
            st.error(f"Geometry file does not exist: {geometry_path}")
            geometry_path = ""

    if not geometry_path:
        st.info(
            "Provide a `.gdml` detector description by path or upload to overlay it on 3D plots."
        )
        return GeometryState(scene=None, options={"enabled": False})

    try:
        version = source_version(geometry_path)
        scene = load_geometry_scene(
            geometry_path,
            quality,
            bool(include_containers),
            bool(include_world),
            int(max_objects),
            int(max_vertices),
        )
    except Exception as exc:
        st.error(f"Could not load geometry: {exc}")
        return GeometryState(scene=None, options={"enabled": False})

    meta = scene.get("meta", {})
    st.success(
        f"Loaded geometry: {meta.get('objects', 0):,} objects, {meta.get('triangles', 0):,} triangles"
    )
    materials = geometry_materials(scene)
    selected_materials = st.multiselect(
        "materials to draw", materials, default=materials[: min(len(materials), 30)]
    )
    warnings = meta.get("warnings") or []
    if warnings:
        st.warning("Geometry warnings: " + "; ".join(str(warning) for warning in warnings[:3]))
    return GeometryState(
        scene=scene,
        source=(geometry_path, version),
        options={
            "enabled": True,
            "quality": quality,
            "include_containers": bool(include_containers),
            "include_world": bool(include_world),
            "parse_max_objects": int(max_objects),
            "parse_max_vertices": int(max_vertices),
            "materials": selected_materials,
            "opacity": float(opacity),
            "scale": float(scale),
            "offset_x": float(offset_x),
            "offset_y": float(offset_y),
            "offset_z": float(offset_z),
            "wireframe": bool(wireframe),
            "max_vertices": int(display_max_vertices),
        },
    )


def invalidate_geometry_results() -> None:
    """Discard retained 3D plots when their GDML source changes or disappears."""

    for key in ("plot_result", "event_plot_result"):
        result = st.session_state.get(key)
        source = result.get("geometry_source") if result else None
        if source is None:
            continue
        path, previous_version = source
        try:
            changed = source_version(path) != previous_version
        except SourceUnavailableError:
            changed = True
        if changed:
            discard_plot_result(key)


def render_geometry_tab(state: GeometryState) -> None:
    """Render the geometry summary and user guidance tab."""

    st.subheader("Detector geometry overlay", anchor=False)
    if state.scene is None:
        st.info(
            "Enable the 3D detector overlay in the sidebar and provide a GDML geometry file by path or upload."
        )
        return
    st.dataframe(geometry_summary_dataframe(state.scene), width="stretch")
    meta = state.scene.get("meta", {})
    for title, key in [("Materials", "materials"), ("Solid types", "solidTypes")]:
        values: Any = meta.get(key, {})
        if isinstance(values, dict) and values:
            st.markdown(f"#### {title}", anchors=False)
            st.dataframe(
                pd.DataFrame(
                    [
                        {title[:-1].casefold(): name, "objects": count}
                        for name, count in sorted(
                            values.items(), key=lambda item: (-item[1], item[0])
                        )
                    ]
                ),
                width="stretch",
            )
    warnings = meta.get("warnings") or []
    if warnings:
        st.markdown("#### Geometry parser warnings", anchors=False)
        st.write("\n".join(f"- {warning}" for warning in warnings[:50]))
    st.markdown(
        """
The overlay is only drawn for **3D scatter interactive** plots. The GDML geometry is converted to triangle meshes and added as transparent Plotly `Mesh3d` traces underneath your event or global vertex scatter.

Geometry files can be loaded either from a local `.gdml` path or via upload in the sidebar. Use the scale and offset controls when your ROOT coordinates and GDML coordinates use different units or origins.
"""
    )
