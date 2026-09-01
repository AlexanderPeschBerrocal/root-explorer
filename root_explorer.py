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

import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

# Keep ``streamlit run root_explorer.py`` working from a checkout while the
# package is being developed. Installed users will resolve the package normally.
_SOURCE_DIR = Path(__file__).resolve().parent / "src"
if _SOURCE_DIR.is_dir():
    sys.path.insert(0, str(_SOURCE_DIR))

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from root_explorer.analysis.arrays import (
    ArrayShapeMismatchError,
    apply_finite_filter,
    array_length,
    branch_values,
    cut_values_for_values,
    delta_phi,
    event_to_numpy,
    finite_mask,
    histogram_weights_for_values,
    multiplicity_per_event,
    nth_per_event,
    raw_counts_per_event,
    raw_list_shape_levels,
    read_aligned_branch_values,
    read_for_scope,
    reduce_per_event,
    safe_float_array,
    to_numpy_flat,
    validate_event_counts,
    validate_raw_event_shapes,
    validate_variable_event_shapes,
    value_is_scalar,
    values_for_reference_shape,
    variable_counts_per_event,
    variable_is_event_level_array,
    variable_is_per_event,
    variable_reference_branch,
    variable_spec_name,
    variable_values_for_values,
    weight_is_per_event_array,
)
from root_explorer.models import BranchInfo, VariableSpec
from root_explorer.io.root import (
    branch_choices,
    choose_default,
    get_branch_info,
    is_probably_numeric_typename,
    is_ttree,
    numeric_branches,
    open_root_file,
    parse_cli_rootfile,
    read_branch,
    read_event_branch,
    split_collection,
    walk_ttrees,
)
from root_explorer.analysis.variables import evaluate_variable
from root_explorer.geometry.overlay import (
    DEFAULT_DISPLAY_GEOMETRY_VERTICES,
    DEFAULT_MAX_GEOMETRY_OBJECTS,
    DEFAULT_MAX_GEOMETRY_VERTICES,
    GEOMETRY_QUALITY_LEVELS,
    geometry_materials,
    geometry_summary_dataframe,
    load_geometry_scene,
    overlay_geometry_on_plotly,
)
from root_explorer.plotting.plots import (
    apply_range_cut,
    apply_range_cuts,
    build_plotly_3d,
    describe_array,
    downsample_arrays,
    matplotlib_figure_bytes,
    render_static_plot,
    parse_optional_float,
    plotly_figure_html,
    safe_filename,
    save_matplotlib_figure,
    save_plotly_figure,
)

APP_TITLE = "ROOT Explorer"
DEFAULT_MAX_ENTRIES = 20_000
DEFAULT_MAX_POINTS = 200_000

st.set_page_config(page_title=APP_TITLE, layout="wide")


# ROOT access, array conversion, shape validation, and variable evaluation are
# implemented in the src/root_explorer package and imported above.

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
