"""Streamlit controls for selecting variables and plot options."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import streamlit as st

from ..config import DEFAULT_MAX_POINTS
from ..io.root import choose_default
from ..models import FieldInfo, VariableSpec
from ..plotting.plots import ContourConfigurationError, parse_contour_levels, parse_optional_float

DEFAULT_STATIC_DISPLAY_DPI = 200
DEFAULT_STATIC_EXPORT_DPI = 300
STATIC_EXPORT_DPI_OPTIONS = (150, 300, 600)


def field_structure_label(field: FieldInfo) -> str:
    """Return the compact structural badge text for one source field."""

    if field.structure == "regular" and field.fixed_shape:
        dimensions = "x".join(str(size) for size in field.fixed_shape)
        return f"fixed {dimensions}"
    return field.structure


def field_option_labels(field_info: Sequence[FieldInfo]) -> dict[str, str]:
    """Return name-first selector labels with structural and dtype badges."""

    labels: dict[str, str] = {}
    for field in field_info:
        badges = [field_structure_label(field)]
        if field.dtype not in {"unknown", "record"}:
            badges.append(field.dtype)
        labels[field.name] = f"{field.name}  " + " ".join(f"[{badge}]" for badge in badges)
    return labels


def variable_selector(
    prefix: str,
    fields: list[str],
    preferred: Sequence[str],
    *,
    item_name: str,
    field_info: Sequence[FieldInfo] = (),
    show_field_details: bool = False,
    allow_event_reductions: bool = True,
    display_prefix: str | None = None,
) -> VariableSpec:
    display_prefix = display_prefix or prefix
    metadata_by_name = {field.name: field for field in field_info}
    option_labels = field_option_labels(field_info) if show_field_details else {}

    def format_field(field_name: str) -> str:
        return option_labels.get(field_name, field_name)

    mode_labels = {
        "branch": f"Direct {item_name} values: A",
        "absolute": "|A|",
        "multiplicity": "Multiplicity per event",
        "nth": "nth object per event",
        "sum": "Sum per event",
        "mean": "Mean per event",
        "max": "Max per event",
        "min": "Min per event",
        "sum_ab": "A + B",
        "difference": "A - B",
        "ratio": "A / B",
        "product": "A * B",
        "magnitude_3d": "sqrt(A² + B² + C²)",
        "r_xy": "r = sqrt(x² + y²)",
        "phi_xy": "phi = atan2(y, x)",
        "eta_xyz": "eta = asinh(z / sqrt(x²+y²))",
        "energy_massless": "Energy (massless)",
        "energy_massive": "Energy (with mass)",
        "invariant_mass": "Invariant mass (single particle)",
        "delta_phi": "delta phi",
        "delta_eta": "delta eta",
        "delta_r": "delta R",
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
            format_func=format_field,
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
        if show_field_details and mode in {"multiplicity", "nth", "sum", "mean", "max", "min"}:
            source = metadata_by_name.get(branch)
            if source is not None:
                st.caption(
                    "Result: one event-level scalar per event from "
                    f"[{field_structure_label(source)}] [{source.dtype}] input."
                )
        return VariableSpec(mode=mode, branch=branch, index=int(index))

    if mode == "absolute":
        a = st.selectbox(
            f"{display_prefix}: A",
            options,
            index=choose_default(options, preferred),
            format_func=format_field,
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
            format_func=format_field,
            key=f"{prefix}_a",
        )
        b = st.selectbox(
            f"{display_prefix}: {label_b}",
            options,
            index=min(1, len(options) - 1),
            format_func=format_field,
            key=f"{prefix}_b",
        )
        return VariableSpec(mode=mode, branch_a=a, branch_b=b)

    if mode == "eta_xyz":
        a = st.selectbox(
            f"{display_prefix}: x",
            options,
            index=choose_default(options, ["x", "px"]),
            format_func=format_field,
            key=f"{prefix}_x",
        )
        b = st.selectbox(
            f"{display_prefix}: y",
            options,
            index=choose_default(options, ["y", "py"]),
            format_func=format_field,
            key=f"{prefix}_y",
        )
        c = st.selectbox(
            f"{display_prefix}: z",
            options,
            index=choose_default(options, ["z", "pz"]),
            format_func=format_field,
            key=f"{prefix}_z",
        )
        return VariableSpec(mode=mode, branch_a=a, branch_b=b, branch_c=c)

    if mode == "magnitude_3d":
        a = st.selectbox(
            f"{display_prefix}: A",
            options,
            index=choose_default(options, ["px", "x"]),
            format_func=format_field,
            key=f"{prefix}_a",
        )
        b = st.selectbox(
            f"{display_prefix}: B",
            options,
            index=choose_default(options, ["py", "y"]),
            format_func=format_field,
            key=f"{prefix}_b",
        )
        c = st.selectbox(
            f"{display_prefix}: C",
            options,
            index=choose_default(options, ["pz", "z"]),
            format_func=format_field,
            key=f"{prefix}_c",
        )
        return VariableSpec(mode=mode, branch_a=a, branch_b=b, branch_c=c)

    if mode in {"energy_massless", "energy_massive"}:
        pt = st.selectbox(
            f"{display_prefix}: pT",
            options,
            index=choose_default(options, ["pt"]),
            format_func=format_field,
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
            format_func=format_field,
            key=f"{prefix}_longitudinal",
        )
        mass = None
        if mode == "energy_massive":
            mass = st.selectbox(
                f"{display_prefix}: mass",
                options,
                index=choose_default(options, ["mass", "m"]),
                format_func=format_field,
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
                    format_func=format_field,
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
            format_func=format_field,
            key=f"{prefix}_phi1",
        )
        b = st.selectbox(
            f"{display_prefix}: phi 2",
            options,
            index=choose_default(options, ["phi"]),
            format_func=format_field,
            key=f"{prefix}_phi2",
        )
        return VariableSpec(mode=mode, branch_a=a, branch_b=b)

    if mode == "delta_eta":
        a = st.selectbox(
            f"{display_prefix}: eta 1",
            options,
            index=choose_default(options, ["eta"]),
            format_func=format_field,
            key=f"{prefix}_eta1",
        )
        b = st.selectbox(
            f"{display_prefix}: eta 2",
            options,
            index=choose_default(options, ["eta"]),
            format_func=format_field,
            key=f"{prefix}_eta2",
        )
        return VariableSpec(mode=mode, branch_a=a, branch_b=b)

    if mode == "delta_r":
        a = st.selectbox(
            f"{display_prefix}: eta 1",
            options,
            index=choose_default(options, ["eta"]),
            format_func=format_field,
            key=f"{prefix}_eta1",
        )
        b = st.selectbox(
            f"{display_prefix}: phi 1",
            options,
            index=choose_default(options, ["phi"]),
            format_func=format_field,
            key=f"{prefix}_phi1",
        )
        c = st.selectbox(
            f"{display_prefix}: eta 2",
            options,
            index=choose_default(options, ["eta"]),
            format_func=format_field,
            key=f"{prefix}_eta2",
        )
        d = st.selectbox(
            f"{display_prefix}: phi 2",
            options,
            index=choose_default(options, ["phi"]),
            format_func=format_field,
            key=f"{prefix}_phi2",
        )
        return VariableSpec(mode=mode, branch_a=a, branch_b=b, branch_c=c, branch_d=d)

    raise RuntimeError("Unreachable variable selector state.")


PLOT_TYPES = ["1D histogram", "2D histogram", "2D scatter", "3D scatter interactive"]
COLOR_SCALES = ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Greys", "coolwarm"]


def color_scale_controls(
    prefix: str,
    suffix: str = "",
    *,
    log_style_key: str | None = None,
    log_widget_key: str | None = None,
) -> dict[str, Any]:
    """Render the colour map, optional log scale and colour-limit controls.

    Shared by 2D histograms, persisted 2D histograms and coloured scatter plots
    so that equivalent colour settings look and behave identically.
    """

    style: dict[str, Any] = {}
    style["colorscale"] = st.selectbox(
        "Colour map", COLOR_SCALES, key=f"{prefix}_colorscale{suffix}"
    )
    style["reverse_colorscale"] = st.checkbox(
        "Reverse colour map", key=f"{prefix}_reverse_colorscale{suffix}"
    )
    if log_style_key is not None:
        style[log_style_key] = st.checkbox(
            "Log colour scale",
            value=False,
            key=log_widget_key or f"{prefix}_{log_style_key}{suffix}",
        )
    min_column, max_column = st.columns(2)
    with min_column:
        style["color_min_text"] = st.text_input(
            "Colour minimum", value="", key=f"{prefix}_color_min{suffix}", placeholder="Automatic"
        )
    with max_column:
        style["color_max_text"] = st.text_input(
            "Colour maximum", value="", key=f"{prefix}_color_max{suffix}", placeholder="Automatic"
        )
    return style


def needs_y_variable(plot_type: str) -> bool:
    return plot_type in {"2D histogram", "2D scatter", "3D scatter interactive"}


def needs_z_variable(plot_type: str) -> bool:
    return plot_type == "3D scatter interactive"


def supports_point_color(plot_type: str) -> bool:
    return plot_type in {"2D scatter", "3D scatter interactive"}


def supports_histogram_weights(plot_type: str) -> bool:
    return plot_type in {"1D histogram", "2D histogram"}


def histogram_weight_selector(
    prefix: str,
    plot_type: str,
    fields: list[str],
    *,
    item_name: str,
    field_info: Sequence[FieldInfo] = (),
    show_field_details: bool = False,
) -> str | None:
    if not supports_histogram_weights(plot_type):
        return None
    use_weights = st.checkbox("Use weights", value=False, key=f"{prefix}_use_weights")
    if not use_weights:
        return None
    option_labels = field_option_labels(field_info) if show_field_details else {}
    return st.selectbox(
        f"Weight {item_name}",
        fields,
        index=choose_default(fields, ["weight", "genWeight", "eventWeight"]),
        format_func=lambda field_name: option_labels.get(field_name, field_name),
        key=f"{prefix}_weight_branch",
    )


def fit_controls(prefix: str, plot_type: str, *, default_use_variances: bool = False):
    """Configure a binned least-squares fit for a 1D histogram."""
    if plot_type != "1D histogram":
        return None
    from ..analysis.fit_expressions import discover_fit_parameters
    from ..analysis.fitting import FIT_MODELS, FitConfig

    st.markdown("**Histogram fit**")
    if not st.checkbox("Enable fit", value=False, key=f"{prefix}_fit_enabled"):
        return None
    model = st.selectbox("Fit model", [*FIT_MODELS, "Custom expression"], key=f"{prefix}_fit_model")
    expression = None
    names = FIT_MODELS.get(model, ())
    if model == "Custom expression":
        expression = st.text_input(
            "Mathematical expression",
            value="A * exp(-0.5 * ((x - mu) / sigma)**2) + c",
            key=f"{prefix}_fit_expression",
            help="Use x, numeric constants, arithmetic, and whitelisted mathematical functions.",
        )
        try:
            names = discover_fit_parameters(expression)
            st.caption("Discovered parameters: " + ", ".join(names))
        except ValueError as exc:
            st.warning(f"Invalid custom fit expression: {exc}")
            names = ()
    use_variances = st.checkbox(
        "Use stored bin variances (weighted least squares)",
        value=default_use_variances,
        key=f"{prefix}_fit_variances",
        help="When disabled, all selected bins receive equal weight. No Poisson model is assumed.",
    )
    range_enabled = st.checkbox("Custom fit range", key=f"{prefix}_fit_range_enabled")
    fit_range = (None, None)
    if range_enabled:
        left, right = st.columns(2)
        with left:
            minimum = st.text_input("Fit minimum", key=f"{prefix}_fit_min", placeholder="Automatic")
        with right:
            maximum = st.text_input("Fit maximum", key=f"{prefix}_fit_max", placeholder="Automatic")

        def parse(value: str) -> float | None:
            if not value.strip():
                return None
            try:
                return float(value)
            except ValueError:
                return float("nan")

        fit_range = (parse(minimum), parse(maximum))

    initial = {}
    bounds = {}
    if model == "Custom expression" or st.checkbox(
        "Override initial parameters and bounds", key=f"{prefix}_fit_override"
    ):
        st.caption("Leave fields blank for generic initial values and default bounds.")
        for name in names:
            cols = st.columns(4)
            cols[0].markdown(f"**{name}**")
            fields = []
            for col, suffix in zip(cols[1:], ("initial", "lower", "upper"), strict=True):
                with col:
                    value = st.text_input(
                        suffix,
                        key=f"{prefix}_fit_{model}_{name}_{suffix}",
                        label_visibility="collapsed",
                        placeholder=suffix.capitalize(),
                    )
                fields.append(value)

            def parse_parameter(value: str) -> float | None:
                if not value.strip():
                    return None
                try:
                    return float(value)
                except ValueError:
                    return float("nan")

            values = [parse_parameter(value) for value in fields]
            if values[0] is not None:
                initial[name] = values[0]
            if values[1] is not None or values[2] is not None:
                bounds[name] = (values[1], values[2])
    return FitConfig(
        model,
        fit_range=fit_range,
        initial_values=initial,
        bounds=bounds,
        use_variances=use_variances,
        expression=expression,
    )


def distribution_calculator_controls(
    prefix: str, plot_type: str
) -> tuple[float | None, float | None]:
    """Return an optional threshold or quantile percentile for a 1D histogram."""

    if plot_type != "1D histogram":
        return None, None
    st.markdown("**Distribution calculator**")
    enabled = st.checkbox(
        "Calculate threshold / quantile",
        value=False,
        key=f"{prefix}_distribution_enabled",
        help="Uses every selected observation after cuts, including histogram underflow and overflow.",
    )
    if not enabled:
        return None, None
    st.caption(
        "Uses the full selected sample after cuts, including values outside the histogram "
        "range. Histogram flow settings do not change these calculations."
    )
    mode = st.selectbox(
        "Calculator mode",
        ["Threshold", "Quantile"],
        key=f"{prefix}_distribution_mode",
    )
    if mode == "Threshold":
        threshold = st.number_input(
            "Threshold", value=0.0, format="%.6g", key=f"{prefix}_threshold"
        )
        return float(threshold), None
    percentile = st.number_input(
        "Quantile percentile (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0,
        format="%.3g",
        key=f"{prefix}_quantile_percentile",
    )
    return None, float(percentile)


def range_cut_controls(
    prefix: str,
    fields: list[str],
    *,
    item_name: str,
    field_info: Sequence[FieldInfo] = (),
    show_field_details: bool = False,
) -> tuple[list[VariableSpec], list[tuple[float | None, float | None]]]:
    st.markdown("### Cuts", anchors=False)
    n_cuts = st.number_input(
        "Number of cuts",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        key=f"{prefix}_n_cuts",
        help="Leave the minimum empty for no lower bound, and leave the maximum empty for no upper bound.",
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
                    field_info=field_info,
                    show_field_details=show_field_details,
                    allow_event_reductions=False,
                    display_prefix=f"cut_{i + 1}",
                )
            )
            lo = st.text_input(
                f"Cut {i + 1} minimum",
                value="",
                key=f"{prefix}_cut_{i}_min",
                placeholder="No lower bound",
            )
            hi = st.text_input(
                f"Cut {i + 1} maximum",
                value="",
                key=f"{prefix}_cut_{i}_max",
                placeholder="No upper bound",
            )
            lo_value = parse_optional_float(lo)
            hi_value = parse_optional_float(hi)
            if str(lo).strip() and lo_value is None:
                st.warning(f"Cut {i + 1} min is not a valid number and will be ignored.")
            if str(hi).strip() and hi_value is None:
                st.warning(f"Cut {i + 1} max is not a valid number and will be ignored.")
            cut_bounds.append((lo_value, hi_value))
    return cut_specs, cut_bounds


def histogram_analysis_controls(
    prefix: str,
    plot_type: str,
    *,
    default_bins: int = 100,
) -> dict[str, Any]:
    """Configure the calculation of 1D and 2D histograms, not their presentation."""

    if not supports_histogram_weights(plot_type):
        return {}

    style: dict[str, Any] = {}
    if plot_type == "1D histogram":
        style["manual_histogram_limits"] = st.checkbox(
            "Custom histogram range and flow handling",
            value=False,
            key=f"{prefix}_manual_histogram_limits",
        )
        style["hist_x_min_text"] = ""
        style["hist_x_max_text"] = ""
        style["hist_flow_mode"] = "ignore"
        if style["manual_histogram_limits"]:
            st.caption(
                "Set histogram bin edges and choose how observations outside them are handled. "
                "Leave either bound blank for its automatic value; both blank retains "
                "automatic binning. Statistical calculators still use the full selected sample."
            )
            hmin_col, hmax_col = st.columns(2)
            with hmin_col:
                style["hist_x_min_text"] = st.text_input(
                    "Histogram minimum", key=f"{prefix}_hist_x_min", placeholder="Automatic"
                )
            with hmax_col:
                style["hist_x_max_text"] = st.text_input(
                    "Histogram maximum", key=f"{prefix}_hist_x_max", placeholder="Automatic"
                )
            style["hist_flow_mode"] = st.selectbox(
                "Underflow / overflow",
                ["ignore", "fold", "separate"],
                format_func=lambda value: {
                    "ignore": "Ignore (report separately)",
                    "fold": "Fold into edge bins",
                    "separate": "Show separate bins",
                }[value],
                key=f"{prefix}_hist_flow_mode",
            )
        style["bins_x"] = st.slider("Bins", 5, 500, default_bins, key=f"{prefix}_bins")
        separate_flow = (
            style["manual_histogram_limits"]
            and style["hist_flow_mode"] == "separate"
            and bool(style["hist_x_min_text"].strip() or style["hist_x_max_text"].strip())
        )
        # Clear the widget value before creating it; disabled Streamlit checkboxes
        # otherwise retain a stale, visually checked session-state value.
        if separate_flow:
            st.session_state[f"{prefix}_normalise_unit_area"] = False
        normalise_requested = st.checkbox(
            "Normalise to unit area",
            value=False,
            key=f"{prefix}_normalise_unit_area",
            disabled=separate_flow,
            help=(
                "Display a density: the sum of (bin height times bin width) is one. "
                "Only nominal bins contribute in Ignore mode; folded flow is included in Fold mode."
            ),
        )
        style["normalise_unit_area"] = bool(normalise_requested and not separate_flow)
        if separate_flow:
            st.caption(
                "Unit-area density cannot include separate underflow/overflow categories "
                "because they have no finite bin widths. Choose Ignore or Fold."
            )
    else:
        style["bins_x"] = st.slider("X bins", 5, 500, default_bins, key=f"{prefix}_bins_x")
        style["bins_y"] = st.slider("Y bins", 5, 500, default_bins, key=f"{prefix}_bins_y")

    style["show_kde"] = st.checkbox("Show KDE", value=False, key=f"{prefix}_show_kde")
    style["kde_bandwidth"] = "scott"
    style["kde_custom_factor"] = 1.0
    if style["show_kde"]:
        bandwidth_label = st.selectbox(
            "KDE bandwidth",
            ["Scott", "Silverman", "Custom factor"],
            key=f"{prefix}_kde_bandwidth",
            help=(
                "KDE construction uses at most 5,000 deterministically selected points "
                "to keep large plots responsive."
            ),
        )
        style["kde_bandwidth"] = bandwidth_label.casefold().split()[0]
        if style["kde_bandwidth"] == "custom":
            style["kde_custom_factor"] = st.number_input(
                "KDE bandwidth factor",
                min_value=0.001,
                value=1.0,
                step=0.1,
                format="%.3f",
                key=f"{prefix}_kde_custom_factor",
            )

    if plot_type == "2D histogram":
        style["show_contours"] = st.checkbox(
            "Show contours", value=False, key=f"{prefix}_show_contours"
        )
        style["contour_levels"] = []
        if style["show_contours"]:
            contour_text = st.text_input(
                "Contour levels (%)",
                value="68, 95, 99",
                key=f"{prefix}_contour_levels",
                help="Highest-density regions enclosing these percentages of the histogram content.",
            )
            try:
                style["contour_levels"] = parse_contour_levels(contour_text)
            except ContourConfigurationError as exc:
                style["show_contours"] = False
                st.warning(str(exc))

    return style


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

    st.markdown("### Style", anchors=False)
    style: dict[str, Any] = {}

    if plot_type in {"1D histogram", "2D histogram", "2D scatter"}:
        style["fig_width"] = st.slider(
            "Figure width",
            4.0,
            16.0,
            8.0,
            key=f"{prefix}_fig_width",
            help=f"Static previews are rendered at {DEFAULT_STATIC_DISPLAY_DPI} DPI.",
        )
        style["fig_height"] = st.slider("Figure height", 3.0, 12.0, 6.0, key=f"{prefix}_fig_height")
        style["display_dpi"] = DEFAULT_STATIC_DISPLAY_DPI
        style["log_x"] = st.checkbox("Log x-axis", value=False, key=f"{prefix}_log_x")
        style["log_y"] = st.checkbox("Log y-axis", value=False, key=f"{prefix}_log_y")
        style["manual_axis_limits"] = st.checkbox(
            "Manual plot limits", value=False, key=f"{prefix}_manual_axis_limits"
        )
        style["x_min_text"] = ""
        style["x_max_text"] = ""
        style["y_min_text"] = ""
        style["y_max_text"] = ""
        if style["manual_axis_limits"]:
            st.caption("Leave a bound blank to use the automatic limit for that side.")
            x_min_column, x_max_column = st.columns(2)
            with x_min_column:
                style["x_min_text"] = st.text_input(
                    "X minimum", value="", key=f"{prefix}_x_min", placeholder="Automatic"
                )
            with x_max_column:
                style["x_max_text"] = st.text_input(
                    "X maximum", value="", key=f"{prefix}_x_max", placeholder="Automatic"
                )
            y_min_column, y_max_column = st.columns(2)
            with y_min_column:
                style["y_min_text"] = st.text_input(
                    "Y minimum", value="", key=f"{prefix}_y_min", placeholder="Automatic"
                )
            with y_max_column:
                style["y_max_text"] = st.text_input(
                    "Y maximum", value="", key=f"{prefix}_y_max", placeholder="Automatic"
                )

    if plot_type == "1D histogram":
        style["grid"] = st.checkbox("Grid", value=True, key=f"{prefix}_grid")

    elif plot_type == "2D histogram":
        style.update(
            color_scale_controls(
                prefix, "_hist2d", log_style_key="log_z", log_widget_key=f"{prefix}_log_z"
            )
        )
        style["equal_aspect"] = st.checkbox(
            "Equal aspect", value=default_equal_aspect, key=f"{prefix}_equal_aspect"
        )
        style["grid"] = st.checkbox("Grid", value=True, key=f"{prefix}_grid")

    elif plot_type == "2D scatter":
        style["marker_size_2d"] = st.slider(
            "Marker size", 0.2, 50.0, default_marker_size_2d, key=f"{prefix}_marker_size_2d"
        )
        style["opacity_2d"] = st.slider(
            "Marker opacity", 0.05, 1.0, default_opacity_2d, key=f"{prefix}_opacity_2d"
        )
        style["max_points"] = st.number_input(
            "Maximum scatter points",
            min_value=1_000,
            max_value=10_000_000,
            value=DEFAULT_MAX_POINTS,
            step=10_000,
            key=f"{prefix}_max_points_2d",
        )
        if color_enabled:
            style.update(color_scale_controls(prefix, "_2d", log_style_key="log_color"))
        style["equal_aspect"] = st.checkbox(
            "Equal aspect", value=default_equal_aspect, key=f"{prefix}_equal_aspect"
        )
        style["grid"] = st.checkbox("Grid", value=True, key=f"{prefix}_grid")

    elif plot_type == "3D scatter interactive":
        style["marker_size_3d"] = st.slider(
            "Marker size", 1.0, 20.0, default_marker_size_3d, key=f"{prefix}_marker_size_3d"
        )
        style["opacity_3d"] = st.slider(
            "Marker opacity", 0.05, 1.0, default_opacity_3d, key=f"{prefix}_opacity_3d"
        )
        style["max_points"] = st.number_input(
            "Maximum scatter points",
            min_value=1_000,
            max_value=10_000_000,
            value=DEFAULT_MAX_POINTS,
            step=10_000,
            key=f"{prefix}_max_points_3d",
        )
        style["aspectmode"] = st.selectbox(
            "Aspect mode", ["data", "cube", "auto"], key=f"{prefix}_aspectmode"
        )
        if color_enabled:
            style.update(color_scale_controls(prefix, "_3d"))

    return style


def label_controls(
    prefix: str, plot_type: str, *, default_title: str = "", has_colorbar: bool = False
) -> dict[str, Any]:
    controls = {
        "title": st.text_input("Title", value=default_title, key=f"{prefix}_title"),
        "custom_x": st.text_input("X-axis label override", value="", key=f"{prefix}_custom_x"),
        "custom_y": st.text_input("Y-axis label override", value="", key=f"{prefix}_custom_y"),
        "custom_z": "",
        "custom_colorbar": "",
    }
    if needs_z_variable(plot_type):
        controls["custom_z"] = st.text_input(
            "Z-axis label override", value="", key=f"{prefix}_custom_z"
        )
    if has_colorbar and plot_type in {"2D histogram", "2D scatter", "3D scatter interactive"}:
        controls["custom_colorbar"] = st.text_input(
            "Colour-bar label override", value="", key=f"{prefix}_custom_colorbar"
        )
    return controls


def export_controls(prefix: str, plot_kind: str, *, default_save_name: str) -> dict[str, Any]:
    """Render all export settings beneath an already-generated plot."""

    st.markdown("### Export", anchors=False)
    controls: dict[str, Any] = {}
    controls["save_directory"] = st.text_input(
        "Save directory",
        value="root_explorer_plots",
        key=f"{prefix}_save_directory",
        help=(
            "Directory on the filesystem of the machine running ROOT Explorer. "
            "Use the Download buttons instead to save files through your browser."
        ),
    )
    controls["save_name"] = st.text_input(
        "Filename",
        value=default_save_name,
        key=f"{prefix}_save_name",
    )
    if plot_kind == "matplotlib":
        controls["save_png"] = st.checkbox("Save PNG", value=True, key=f"{prefix}_save_png")
        controls["save_pdf"] = st.checkbox("Save PDF", value=False, key=f"{prefix}_save_pdf")
        controls["export_dpi"] = st.selectbox(
            "Raster export resolution",
            STATIC_EXPORT_DPI_OPTIONS,
            index=STATIC_EXPORT_DPI_OPTIONS.index(DEFAULT_STATIC_EXPORT_DPI),
            format_func=lambda dpi: f"{dpi} DPI",
            key=f"{prefix}_export_dpi",
            help=(
                "Controls PNG pixel dimensions and the resolution of any rasterized content "
                "inside PDF files. PDF text and vector lines remain resolution-independent."
            ),
        )
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
