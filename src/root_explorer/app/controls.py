"""Streamlit controls for selecting variables and plot options."""

from __future__ import annotations

from typing import Any, Optional
from collections.abc import Sequence

import streamlit as st

from ..config import DEFAULT_MAX_POINTS
from ..io.root import choose_default
from ..models import VariableSpec
from ..plotting.plots import parse_optional_float


def variable_selector(
    prefix: str,
    branches: list[str],
    preferred: Sequence[str],
    *,
    allow_event_reductions: bool = True,
) -> VariableSpec:
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
        for key in [
            "Multiplicity per event",
            "nth object per event",
            "Sum per event",
            "Mean per event",
            "Max per event",
            "Min per event",
        ]:
            modes.pop(key, None)

    mode_label = st.selectbox(f"{prefix}: variable type", list(modes.keys()), key=f"{prefix}_mode")
    mode = modes[mode_label]
    options = branches

    if mode in {"branch", "multiplicity", "nth", "sum", "mean", "max", "min"}:
        branch = st.selectbox(
            f"{prefix}: branch",
            options,
            index=choose_default(options, preferred),
            key=f"{prefix}_branch",
        )
        index = 0
        if mode == "nth":
            index = st.number_input(
                f"{prefix}: object index",
                min_value=0,
                max_value=1000,
                value=0,
                step=1,
                key=f"{prefix}_index",
            )
        return VariableSpec(mode=mode, branch=branch, index=int(index))

    if mode in {"r_xy", "phi_xy", "ratio", "difference", "sum_ab", "product", "delta_phi"}:
        a = st.selectbox(
            f"{prefix}: A / x / phi1",
            options,
            index=choose_default(options, preferred),
            key=f"{prefix}_a",
        )
        b = st.selectbox(
            f"{prefix}: B / y / phi2", options, index=min(1, len(options) - 1), key=f"{prefix}_b"
        )
        return VariableSpec(mode=mode, branch_a=a, branch_b=b)

    if mode == "eta_xyz":
        a = st.selectbox(
            f"{prefix}: x", options, index=choose_default(options, ["x", "px"]), key=f"{prefix}_x"
        )
        b = st.selectbox(
            f"{prefix}: y", options, index=choose_default(options, ["y", "py"]), key=f"{prefix}_y"
        )
        c = st.selectbox(
            f"{prefix}: z", options, index=choose_default(options, ["z", "pz"]), key=f"{prefix}_z"
        )
        return VariableSpec(mode=mode, branch_a=a, branch_b=b, branch_c=c)

    if mode == "delta_r":
        a = st.selectbox(
            f"{prefix}: eta 1",
            options,
            index=choose_default(options, ["eta"]),
            key=f"{prefix}_eta1",
        )
        b = st.selectbox(
            f"{prefix}: phi 1",
            options,
            index=choose_default(options, ["phi"]),
            key=f"{prefix}_phi1",
        )
        c = st.selectbox(
            f"{prefix}: eta 2",
            options,
            index=choose_default(options, ["eta"]),
            key=f"{prefix}_eta2",
        )
        d = st.selectbox(
            f"{prefix}: phi 2",
            options,
            index=choose_default(options, ["phi"]),
            key=f"{prefix}_phi2",
        )
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


def histogram_weight_selector(prefix: str, plot_type: str, branches: list[str]) -> str | None:
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


def range_cut_controls(
    prefix: str, branches: list[str]
) -> tuple[list[VariableSpec], list[tuple[float | None, float | None]]]:
    st.markdown("### Cuts")
    n_cuts = st.number_input(
        "Number of range cuts", min_value=0, max_value=20, value=0, step=1, key=f"{prefix}_n_cuts"
    )
    cut_specs: list[VariableSpec] = []
    cut_bounds: list[tuple[float | None, float | None]] = []
    for i in range(int(n_cuts)):
        with st.expander(f"Cut {i + 1}", expanded=i < 3):
            cut_specs.append(
                variable_selector(
                    f"{prefix}_cut_{i}",
                    branches,
                    ["pt", "eta", "mass"],
                    allow_event_reductions=False,
                )
            )
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
        style["equal_aspect"] = st.checkbox(
            "equal aspect", value=default_equal_aspect, key=f"{prefix}_equal_aspect"
        )
        style["grid"] = st.checkbox("grid", value=True, key=f"{prefix}_grid")

    elif plot_type == "2D scatter":
        style["marker_size_2d"] = st.slider(
            "marker size", 0.2, 50.0, default_marker_size_2d, key=f"{prefix}_marker_size_2d"
        )
        style["opacity_2d"] = st.slider(
            "opacity", 0.05, 1.0, default_opacity_2d, key=f"{prefix}_opacity_2d"
        )
        style["max_points"] = st.number_input(
            "max scatter points",
            min_value=1_000,
            max_value=10_000_000,
            value=DEFAULT_MAX_POINTS,
            step=10_000,
            key=f"{prefix}_max_points_2d",
        )
        if color_enabled:
            style["colorscale"] = st.selectbox(
                "color scale", COLOR_SCALES, key=f"{prefix}_colorscale_2d"
            )
            style["reverse_colorscale"] = st.checkbox(
                "reverse color scale", key=f"{prefix}_reverse_colorscale_2d"
            )
            style["color_min_text"] = st.text_input(
                "color min", value="", key=f"{prefix}_color_min_2d"
            )
            style["color_max_text"] = st.text_input(
                "color max", value="", key=f"{prefix}_color_max_2d"
            )
        style["equal_aspect"] = st.checkbox(
            "equal aspect", value=default_equal_aspect, key=f"{prefix}_equal_aspect"
        )
        style["grid"] = st.checkbox("grid", value=True, key=f"{prefix}_grid")

    elif plot_type == "3D scatter interactive":
        style["marker_size_3d"] = st.slider(
            "marker size", 1.0, 20.0, default_marker_size_3d, key=f"{prefix}_marker_size_3d"
        )
        style["opacity_3d"] = st.slider(
            "opacity", 0.05, 1.0, default_opacity_3d, key=f"{prefix}_opacity_3d"
        )
        style["max_points"] = st.number_input(
            "max scatter points",
            min_value=1_000,
            max_value=10_000_000,
            value=DEFAULT_MAX_POINTS,
            step=10_000,
            key=f"{prefix}_max_points_3d",
        )
        style["aspectmode"] = st.selectbox(
            "aspect mode", ["data", "cube", "auto", "manual"], key=f"{prefix}_aspectmode"
        )
        if color_enabled:
            style["colorscale"] = st.selectbox(
                "color scale", COLOR_SCALES, key=f"{prefix}_colorscale_3d"
            )
            style["reverse_colorscale"] = st.checkbox(
                "reverse color scale", key=f"{prefix}_reverse_colorscale_3d"
            )
            style["color_min_text"] = st.text_input(
                "color min", value="", key=f"{prefix}_color_min_3d"
            )
            style["color_max_text"] = st.text_input(
                "color max", value="", key=f"{prefix}_color_max_3d"
            )

    return style


def label_export_controls(
    prefix: str, plot_type: str, *, default_title: str = "", default_save_name: str = "plot"
) -> dict[str, Any]:
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
