"""Streamlit controls for selecting variables and plot options."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import streamlit as st

from ..config import DEFAULT_MAX_POINTS
from ..io.root import choose_default
from ..models import VariableSpec
from ..plotting.plots import parse_optional_float


def variable_selector(
    prefix: str,
    fields: list[str],
    preferred: Sequence[str],
    *,
    item_name: str,
    allow_event_reductions: bool = True,
    display_prefix: str | None = None,
) -> VariableSpec:
    display_prefix = display_prefix or prefix
    mode_labels = {
        "branch": f"Direct {item_name} values",
        "multiplicity": "Multiplicity per event",
        "nth": "nth object per event",
        "sum": "Sum per event",
        "mean": "Mean per event",
        "max": "Max per event",
        "min": "Min per event",
        "r_xy": "r = sqrt(x² + y²)",
        "phi_xy": "phi = atan2(y, x)",
        "eta_xyz": "eta = asinh(z / sqrt(x²+y²))",
        "energy_massless": "Energy (massless)",
        "energy_massive": "Energy (with mass)",
        "invariant_mass": "Invariant mass (single particle)",
        "delta_phi": "delta phi",
        "delta_eta": "delta eta",
        "delta_r": "delta R",
        "absolute": "|A|",
        "ratio": "A / B",
        "difference": "A - B",
        "sum_ab": "A + B",
        "product": "A * B",
        "magnitude_3d": "sqrt(A² + B² + C²)",
    }
    if not allow_event_reductions:
        for mode in [
            "multiplicity",
            "nth",
            "sum",
            "mean",
            "max",
            "min",
        ]:
            mode_labels.pop(mode)

    mode = st.selectbox(
        f"{display_prefix}: variable type",
        list(mode_labels),
        format_func=mode_labels.__getitem__,
        key=f"{prefix}_mode",
    )
    options = fields

    if mode in {"branch", "multiplicity", "nth", "sum", "mean", "max", "min"}:
        branch = st.selectbox(
            f"{display_prefix}: {item_name}",
            options,
            index=choose_default(options, preferred),
            key=f"{prefix}_branch",
        )
        index = 0
        if mode == "nth":
            index = st.number_input(
                f"{display_prefix}: object index",
                min_value=0,
                max_value=1000,
                value=0,
                step=1,
                key=f"{prefix}_index",
            )
        return VariableSpec(mode=mode, branch=branch, index=int(index))

    if mode == "absolute":
        a = st.selectbox(
            f"{display_prefix}: A",
            options,
            index=choose_default(options, preferred),
            key=f"{prefix}_a",
        )
        return VariableSpec(mode=mode, branch_a=a)

    if mode in {"r_xy", "phi_xy", "ratio", "difference", "sum_ab", "product"}:
        field_labels = {
            "r_xy": ("x", "y"),
            "phi_xy": ("x", "y"),
            "ratio": ("A", "B"),
            "difference": ("A", "B"),
            "sum_ab": ("A", "B"),
            "product": ("A", "B"),
        }
        label_a, label_b = field_labels[mode]

        a = st.selectbox(
            f"{display_prefix}: {label_a}",
            options,
            index=choose_default(options, preferred),
            key=f"{prefix}_a",
        )
        b = st.selectbox(
            f"{display_prefix}: {label_b}",
            options,
            index=min(1, len(options) - 1),
            key=f"{prefix}_b",
        )
        return VariableSpec(mode=mode, branch_a=a, branch_b=b)

    if mode == "eta_xyz":
        a = st.selectbox(
            f"{display_prefix}: x",
            options,
            index=choose_default(options, ["x", "px"]),
            key=f"{prefix}_x",
        )
        b = st.selectbox(
            f"{display_prefix}: y",
            options,
            index=choose_default(options, ["y", "py"]),
            key=f"{prefix}_y",
        )
        c = st.selectbox(
            f"{display_prefix}: z",
            options,
            index=choose_default(options, ["z", "pz"]),
            key=f"{prefix}_z",
        )
        return VariableSpec(mode=mode, branch_a=a, branch_b=b, branch_c=c)

    if mode == "magnitude_3d":
        a = st.selectbox(
            f"{display_prefix}: A",
            options,
            index=choose_default(options, ["px", "x"]),
            key=f"{prefix}_a",
        )
        b = st.selectbox(
            f"{display_prefix}: B",
            options,
            index=choose_default(options, ["py", "y"]),
            key=f"{prefix}_b",
        )
        c = st.selectbox(
            f"{display_prefix}: C",
            options,
            index=choose_default(options, ["pz", "z"]),
            key=f"{prefix}_c",
        )
        return VariableSpec(mode=mode, branch_a=a, branch_b=b, branch_c=c)

    if mode in {"energy_massless", "energy_massive"}:
        pt = st.selectbox(
            f"{display_prefix}: pT",
            options,
            index=choose_default(options, ["pt"]),
            key=f"{prefix}_pt",
        )
        longitudinal_labels = {"eta": "eta (pseudorapidity)", "y": "y (rapidity)"}
        longitudinal_kind = st.selectbox(
            f"{display_prefix}: longitudinal coordinate",
            list(longitudinal_labels),
            format_func=longitudinal_labels.__getitem__,
            key=f"{prefix}_longitudinal_kind",
        )
        longitudinal = st.selectbox(
            f"{display_prefix}: {longitudinal_kind}",
            options,
            index=choose_default(
                options,
                ["eta"] if longitudinal_kind == "eta" else ["rapidity", "y"],
            ),
            key=f"{prefix}_longitudinal",
        )
        mass = None
        if mode == "energy_massive":
            mass = st.selectbox(
                f"{display_prefix}: mass",
                options,
                index=choose_default(options, ["mass", "m"]),
                key=f"{prefix}_mass",
            )
        return VariableSpec(
            mode=mode,
            branch_a=pt,
            branch_b=longitudinal,
            branch_c=mass,
            longitudinal_kind=longitudinal_kind,
        )

    if mode == "invariant_mass":
        branches = []
        for component, preferred_components in [
            ("E", ["energy", "E"]),
            ("px", ["px"]),
            ("py", ["py"]),
            ("pz", ["pz"]),
        ]:
            branches.append(
                st.selectbox(
                    f"{display_prefix}: {component}",
                    options,
                    index=choose_default(options, preferred_components),
                    key=f"{prefix}_{component.lower()}",
                )
            )
        return VariableSpec(
            mode=mode,
            branch_a=branches[0],
            branch_b=branches[1],
            branch_c=branches[2],
            branch_d=branches[3],
        )

    if mode == "delta_phi":
        a = st.selectbox(
            f"{display_prefix}: phi 1",
            options,
            index=choose_default(options, ["phi"]),
            key=f"{prefix}_phi1",
        )
        b = st.selectbox(
            f"{display_prefix}: phi 2",
            options,
            index=choose_default(options, ["phi"]),
            key=f"{prefix}_phi2",
        )
        return VariableSpec(mode=mode, branch_a=a, branch_b=b)

    if mode == "delta_eta":
        a = st.selectbox(
            f"{display_prefix}: eta 1",
            options,
            index=choose_default(options, ["eta"]),
            key=f"{prefix}_eta1",
        )
        b = st.selectbox(
            f"{display_prefix}: eta 2",
            options,
            index=choose_default(options, ["eta"]),
            key=f"{prefix}_eta2",
        )
        return VariableSpec(mode=mode, branch_a=a, branch_b=b)

    if mode == "delta_r":
        a = st.selectbox(
            f"{display_prefix}: eta 1",
            options,
            index=choose_default(options, ["eta"]),
            key=f"{prefix}_eta1",
        )
        b = st.selectbox(
            f"{display_prefix}: phi 1",
            options,
            index=choose_default(options, ["phi"]),
            key=f"{prefix}_phi1",
        )
        c = st.selectbox(
            f"{display_prefix}: eta 2",
            options,
            index=choose_default(options, ["eta"]),
            key=f"{prefix}_eta2",
        )
        d = st.selectbox(
            f"{display_prefix}: phi 2",
            options,
            index=choose_default(options, ["phi"]),
            key=f"{prefix}_phi2",
        )
        return VariableSpec(mode=mode, branch_a=a, branch_b=b, branch_c=c, branch_d=d)

    raise RuntimeError("Unreachable variable selector state.")


PLOT_TYPES = ["1D histogram", "2D histogram", "2D scatter", "3D scatter interactive"]
COLOR_SCALES = ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Greys", "coolwarm"]


def needs_y_variable(plot_type: str) -> bool:
    return plot_type in {"2D histogram", "2D scatter", "3D scatter interactive"}


def needs_z_variable(plot_type: str) -> bool:
    return plot_type == "3D scatter interactive"


def supports_point_color(plot_type: str) -> bool:
    return plot_type in {"2D scatter", "3D scatter interactive"}


def supports_histogram_weights(plot_type: str) -> bool:
    return plot_type in {"1D histogram", "2D histogram"}


def histogram_weight_selector(
    prefix: str, plot_type: str, fields: list[str], *, item_name: str
) -> str | None:
    if not supports_histogram_weights(plot_type):
        return None
    st.markdown("### Histogram weights")
    use_weights = st.checkbox("Use weights", value=False, key=f"{prefix}_use_weights")
    if not use_weights:
        return None
    return st.selectbox(
        f"weight {item_name}",
        fields,
        index=choose_default(fields, ["weight", "genWeight", "eventWeight"]),
        key=f"{prefix}_weight_branch",
    )


def range_cut_controls(
    prefix: str, fields: list[str], *, item_name: str
) -> tuple[list[VariableSpec], list[tuple[float | None, float | None]]]:
    st.markdown("### Cuts")
    n_cuts = st.number_input(
        "Number of cuts", min_value=0, max_value=20, value=0, step=1, key=f"{prefix}_n_cuts"
    )
    cut_specs: list[VariableSpec] = []
    cut_bounds: list[tuple[float | None, float | None]] = []
    for i in range(int(n_cuts)):
        with st.expander(f"Cut {i + 1}", expanded=i < 3):
            cut_specs.append(
                variable_selector(
                    f"{prefix}_cut_{i}",
                    fields,
                    ["pt", "eta", "mass"],
                    item_name=item_name,
                    allow_event_reductions=False,
                    display_prefix=f"cut_{i + 1}",
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
            "marker opacity", 0.05, 1.0, default_opacity_2d, key=f"{prefix}_opacity_2d"
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
            "marker opacity", 0.05, 1.0, default_opacity_3d, key=f"{prefix}_opacity_3d"
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


def label_controls(prefix: str, plot_type: str, *, default_title: str = "") -> dict[str, Any]:
    controls = {
        "title": st.text_input("title", value=default_title, key=f"{prefix}_title"),
        "custom_x": st.text_input("x label override", value="", key=f"{prefix}_custom_x"),
        "custom_y": st.text_input("y label override", value="", key=f"{prefix}_custom_y"),
        "custom_z": "",
    }
    if needs_z_variable(plot_type):
        controls["custom_z"] = st.text_input("z label override", value="", key=f"{prefix}_custom_z")
    return controls


def export_controls(prefix: str, plot_kind: str, *, default_save_name: str) -> dict[str, Any]:
    """Render all export settings beneath an already-generated plot."""

    st.markdown("### Export")
    controls: dict[str, Any] = {}
    controls["save_directory"] = st.text_input(
        "save directory",
        value="root_explorer_plots",
        key=f"{prefix}_save_directory",
    )
    controls["save_name"] = st.text_input(
        "filename",
        value=default_save_name,
        key=f"{prefix}_save_name",
    )
    if plot_kind == "matplotlib":
        controls["save_png"] = st.checkbox("save PNG", value=True, key=f"{prefix}_save_png")
        controls["save_pdf"] = st.checkbox("save PDF", value=False, key=f"{prefix}_save_pdf")
        if not controls["save_png"] and not controls["save_pdf"]:
            st.warning("Select at least one static export format.")
    return controls


def selected_static_formats(export_settings: dict[str, Any]) -> list[str]:
    formats: list[str] = []
    if export_settings.get("save_png", True):
        formats.append("png")
    if export_settings.get("save_pdf", False):
        formats.append("pdf")
    return formats
