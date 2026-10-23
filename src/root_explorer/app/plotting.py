"""Streamlit plot tabs and their shared plot-generation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ..analysis.arrays import (
    ArrayShapeMismatchError,
    EmptySelectionError,
    array_length,
    cut_values_for_values,
    excluded_values_note,
    histogram_weights_for_values,
    reduction_can_lack_value,
    require_values,
    validate_variable_event_shapes,
    variable_values_for_values,
)
from ..analysis.fitting import FitConfig, fit_histogram_1d
from ..analysis.histograms import InvalidHistogramRangeError
from ..analysis.statistics import (
    calculate_quantile_result,
    calculate_threshold_result,
    describe_weighted_array,
)
from ..analysis.variables import evaluate_variable
from ..geometry.overlay import overlay_geometry_on_plotly
from ..io.root import FieldReadError
from ..io.source import SourceUnavailableError
from ..models import VariableSpec
from ..plotting.codegen import generate_static_plot_code
from ..plotting.pipeline import (
    STATIC_PLOT_TYPES,
    StaticPlotSpec,
    prepare_static_plot,
    render_prepared_static_plot,
)
from ..plotting.plots import (
    InvalidAxisLimitError,
    InvalidAxisScaleError,
    InvalidColorScaleError,
    InvalidLabelError,
    build_plotly_3d,
    calculate_color_clipping,
    calculate_correlations,
    describe_array,
    downsample_arrays,
    render_static_plot,
    scatter_color_scale_linear,
    select_plot_values,
    validate_plot_labels,
)
from .controls import (
    PLOT_TYPES,
    distribution_calculator_controls,
    empty_collection_caption,
    fit_controls,
    histogram_analysis_controls,
    histogram_weight_selector,
    label_controls,
    needs_y_variable,
    needs_z_variable,
    plot_style_controls,
    range_cut_controls,
    supports_point_color,
    variable_selector,
)
from .results import (
    discard_plot_result,
    show_cached_plot_result,
    show_technical_details,
    store_plot_result,
)
from .state import DataContext, GeometryState


@dataclass(frozen=True)
class PlotRequest:
    """All values required to create one global or single-event plot."""

    context: DataContext
    plot_type: str
    x_spec: VariableSpec
    y_spec: VariableSpec | None
    z_spec: VariableSpec | None
    color_spec: VariableSpec | None
    weight_branch: str | None
    cut_specs: list[VariableSpec]
    cut_bounds: list[tuple[float | None, float | None]]
    style: dict[str, Any]
    labels: dict[str, Any]
    geometry: GeometryState
    event_index: int | None
    preview: bool
    file_name: str
    threshold: float | None = None
    quantile_percentile: float | None = None
    fit_config: FitConfig | None = None

    def to_static_spec(self) -> StaticPlotSpec:
        """Return the shared Streamlit-independent configuration for static plots."""

        return StaticPlotSpec(
            root_path=self.context.root_path,
            tree_name=self.context.event_container_name,
            entry_start=self.context.entry_start,
            entry_stop=self.context.entry_stop,
            plot_type=self.plot_type,
            x_spec=self.x_spec,
            y_spec=self.y_spec,
            color_spec=self.color_spec,
            weight_branch=self.weight_branch,
            cut_specs=tuple(self.cut_specs),
            cut_bounds=tuple(self.cut_bounds),
            style=self.style,
            labels=self.labels,
            event_index=self.event_index,
            threshold=self.threshold,
            quantile_percentile=self.quantile_percentile,
            fit_config=self.fit_config,
        )


def render_plot_tab(
    context: DataContext, geometry: GeometryState, *, shortcuts_enabled: bool = True
) -> None:
    """Render controls and output for plotting the selected entry range."""

    controls, output = st.columns([0.36, 0.64], gap="large")
    with controls:
        st.subheader("Plot configuration", anchor=False)
        request, make_plot = _render_plot_controls(
            context,
            geometry,
            prefix="plot",
            event_index=None,
            default_bins=100,
            default_marker_size_2d=2.0,
            default_marker_size_3d=2.5,
            default_opacity_2d=0.45,
            default_opacity_3d=0.65,
            file_name="root_explorer_plot",
            shortcuts_enabled=shortcuts_enabled,
        )
    with output:
        st.subheader("Output", anchor=False)
        _generate_and_show(request, make_plot, "plot_result", "plot_export")


def render_event_plot_tab(context: DataContext, geometry: GeometryState, event_index: int) -> None:
    """Render controls and output for one selected event."""

    controls, output = st.columns([0.36, 0.64], gap="large")
    with controls:
        st.subheader("Plot configuration", anchor=False)
        request, make_plot = _render_plot_controls(
            context,
            geometry,
            prefix="event",
            event_index=event_index,
            default_bins=50,
            default_marker_size_2d=8.0,
            default_marker_size_3d=4.0,
            default_opacity_2d=0.75,
            default_opacity_3d=0.75,
            file_name="root_explorer_event_plot",
        )
    with output:
        st.subheader(f"Output: event {event_index}", anchor=False)
        _generate_and_show(request, make_plot, "event_plot_result", "event_export")


def _render_plot_controls(
    context: DataContext,
    geometry: GeometryState,
    *,
    prefix: str,
    event_index: int | None,
    default_bins: int,
    default_marker_size_2d: float,
    default_marker_size_3d: float,
    default_opacity_2d: float,
    default_opacity_3d: float,
    file_name: str,
    shortcuts_enabled: bool = True,
) -> tuple[PlotRequest, bool]:
    """Render the shared plot-control surface with stable per-tab widget keys."""

    is_event_plot = event_index is not None
    plot_type = st.selectbox(
        "Plot type", PLOT_TYPES, index=0, key="event_plot_type" if is_event_plot else "plot_type"
    )
    show_field_details = st.toggle(
        f"Show {context.data_item_name} details in selectors",
        value=False,
        key=f"{prefix}_show_field_details",
        help="Add structural-shape and logical-dtype badges after each name in selectors.",
    )
    st.markdown("### Variables", anchors=False)
    axis_prefix = (lambda axis: f"{prefix}_{axis}") if is_event_plot else (lambda axis: axis)
    x_spec = variable_selector(
        axis_prefix("x"),
        context.fields,
        ["x", "pt", "eta"] if event_index is not None else ["pt", "energy", "eta", "x"],
        item_name=context.data_item_name,
        field_info=context.field_info,
        show_field_details=show_field_details,
        display_prefix="x",
        allow_event_reductions=event_index is None,
    )
    y_spec = z_spec = color_spec = None
    if needs_y_variable(plot_type):
        y_spec = variable_selector(
            axis_prefix("y"),
            context.fields,
            ["y", "eta", "phi"] if event_index is not None else ["eta", "phi", "y"],
            item_name=context.data_item_name,
            field_info=context.field_info,
            show_field_details=show_field_details,
            display_prefix="y",
            allow_event_reductions=event_index is None,
        )
    if needs_z_variable(plot_type):
        z_spec = variable_selector(
            axis_prefix("z"),
            context.fields,
            ["z", "phi"] if event_index is not None else ["phi", "z"],
            item_name=context.data_item_name,
            field_info=context.field_info,
            show_field_details=show_field_details,
            display_prefix="z",
            allow_event_reductions=event_index is None,
        )
    color_enabled = False
    if supports_point_color(plot_type):
        color_enabled = st.checkbox(
            "Color points",
            value=False,
            key=f"{prefix}_color_points",
        )
        if color_enabled:
            color_spec = variable_selector(
                axis_prefix("color"),
                context.fields,
                ["energy", "pt", "charge", "pdgId"]
                if event_index is not None
                else ["pt", "energy", "mass", "pdgId", "charge"],
                item_name=context.data_item_name,
                field_info=context.field_info,
                show_field_details=show_field_details,
                display_prefix="color",
                allow_event_reductions=event_index is None,
            )
            empty_collection_caption(color_spec, context.field_info, role="colour")

    cut_specs, cut_bounds = range_cut_controls(
        prefix,
        context.fields,
        item_name=context.data_item_name,
        field_info=context.field_info,
        show_field_details=show_field_details,
        allow_event_reductions=event_index is None,
    )
    analysis_style: dict[str, Any] = {}
    weight_branch = None
    threshold = quantile_percentile = None
    fit_config = None
    if plot_type in {"1D histogram", "2D histogram"}:
        st.markdown("### Histogram analysis", anchors=False)
        weight_branch = histogram_weight_selector(
            prefix,
            plot_type,
            context.fields,
            item_name=context.data_item_name,
            field_info=context.field_info,
            show_field_details=show_field_details,
        )
        analysis_style = histogram_analysis_controls(prefix, plot_type, default_bins=default_bins)
        if plot_type == "1D histogram":
            threshold, quantile_percentile = distribution_calculator_controls(prefix, plot_type)
            fit_config = fit_controls(
                prefix, plot_type, default_use_variances=weight_branch is not None
            )

    # Presentation settings are collapsed by default; everything that changes the
    # plotted data stays visible above. Collapsed widgets still run and keep their
    # values, so they remain part of the plot configuration and generated code.
    with st.expander("Style"):
        presentation_style = plot_style_controls(
            prefix,
            plot_type,
            color_enabled=color_enabled,
            default_marker_size_2d=default_marker_size_2d,
            default_marker_size_3d=default_marker_size_3d,
            default_opacity_2d=default_opacity_2d,
            default_opacity_3d=default_opacity_3d,
            default_equal_aspect=False,
        )
    style = {**analysis_style, **presentation_style}
    with st.expander("Labels"):
        labels = label_controls(
            prefix,
            plot_type,
            has_colorbar=plot_type == "2D histogram"
            or (supports_point_color(plot_type) and color_enabled),
        )
    make_plot = st.button(
        "Make event plot" if is_event_plot else "Make plot",
        type="primary",
        key=f"{prefix}_make",
        shortcut=("Shift+P" if is_event_plot else "P") if shortcuts_enabled else None,
    )
    return (
        PlotRequest(
            context=context,
            plot_type=plot_type,
            x_spec=x_spec,
            y_spec=y_spec,
            z_spec=z_spec,
            color_spec=color_spec,
            weight_branch=weight_branch,
            cut_specs=cut_specs,
            cut_bounds=cut_bounds,
            style=style,
            labels=labels,
            geometry=geometry,
            event_index=event_index,
            preview=is_event_plot,
            file_name=file_name,
            threshold=threshold,
            quantile_percentile=quantile_percentile,
            fit_config=fit_config,
        ),
        make_plot,
    )


def _generate_and_show(
    request: PlotRequest, make_plot: bool, result_key: str, download_key: str
) -> None:
    if make_plot:
        try:
            if request.plot_type in STATIC_PLOT_TYPES:
                # Static plots report their individual steps themselves.
                result = _build_plot_result(request)
            else:
                with st.spinner("Reading data and building the 3D plot…", show_time=True):
                    result = _build_plot_result(request)
            store_plot_result(result_key, result)
        except ArrayShapeMismatchError as exc:
            discard_plot_result(result_key)
            st.warning(str(exc))
        except (
            InvalidAxisLimitError,
            InvalidAxisScaleError,
            InvalidColorScaleError,
            InvalidHistogramRangeError,
            InvalidLabelError,
            SourceUnavailableError,
            EmptySelectionError,
        ) as exc:
            discard_plot_result(result_key)
            st.warning(str(exc))
        except FieldReadError as exc:
            # Routine data problems: a concise message, with the reader's error on request.
            discard_plot_result(result_key)
            st.error(str(exc))
            show_technical_details(exc)
        except Exception as exc:
            # Unexpected failures are likely bugs, so the traceback stays one click away.
            discard_plot_result(result_key)
            prefix = "Event plot" if request.event_index is not None else "Plot"
            st.error(f"{prefix} failed: {exc}")
            show_technical_details(exc)
    show_cached_plot_result(
        result_key,
        export_prefix="event" if request.event_index is not None else "plot",
        download_key=download_key,
        make_plot_label="Make event plot" if request.event_index is not None else "Make plot",
        make_plot_shortcut="Shift + P" if request.event_index is not None else "P",
        empty_message=(
            "Choose an event and settings, then click **Make event plot** or **press Shift + P**."
            if request.event_index is not None
            else "Configure settings on the left, then click **Make plot** or **press P**."
        ),
    )


def _safe_generate_plot_code(request: PlotRequest, warnings: list[str]) -> str | None:
    """Keep the histogram available even when its standalone export cannot be built."""
    try:
        return generate_static_plot_code(request)
    except (ValueError, TypeError, RuntimeError) as exc:
        warnings.append(f"Plot-code export unavailable: {exc}")
        return None


def _exclusion_note(request: PlotRequest, excluded_count: int) -> str | None:
    """Describe excluded values, naming colour or cut reductions as a possible cause."""

    if not excluded_count:
        return None
    note = excluded_values_note(excluded_count)
    if any(
        reduction_can_lack_value(spec)
        for spec in [request.color_spec, *request.cut_specs]
        if spec is not None
    ):
        note += (
            " This count can include the objects of events whose collection is empty or "
            "too short for a colour or cut reduction (nth, mean, max or min)."
        )
    return note


def _sampling_note(shown: int, total: int) -> str | None:
    """Explain that a scatter plot draws only a sample of its points."""

    if shown >= total:
        return None
    return (
        f"Showing a fixed random sample of {shown:,} of {total:,} points. Summary statistics "
        "use all points. Increase **Maximum scatter points** under **Style** to draw more."
    )


def _weighted_summary_notes(weight_branch: str | None, weights: np.ndarray) -> list[str]:
    """Explain which summary columns are weighted and when they are unavailable."""

    notes = [
        f"Weighted statistics use the weights from `{weight_branch}`; "
        "N_eff = (Σw)² / Σw². The summary table above is unweighted."
    ]
    negative = int(np.count_nonzero(weights < 0))
    if negative:
        notes.append(f"{negative:,} of {len(weights):,} weights are negative.")
    if float(np.sum(weights)) <= 0:
        notes.append(
            "The weighted mean and standard deviation are unavailable because the sum of "
            "weights is not positive."
        )
    return notes


def _build_static_plot_result(request: PlotRequest) -> dict[str, Any]:
    """Create a cached result for a static plot through the shared plotting pipeline."""

    validate_plot_labels(request.labels)
    spec = request.to_static_spec()
    # Spinners appear only after a short delay, so fast steps do not flicker.
    with st.spinner("Reading and selecting data…", show_time=True):
        prepared = prepare_static_plot(spec)

    summary_rows = [describe_array(prepared.x_label, prepared.x)]
    if prepared.y is not None:
        summary_rows.append(describe_array(prepared.y_label, prepared.y))
    weighted_summary = None
    summary_notes: list[str] = []
    if prepared.weights is not None:
        # Weights are aligned with the plotted x/y values, so each row describes
        # the weighted distribution that is actually histogrammed.
        weighted_summary = pd.DataFrame(
            [
                {"variable": row["variable"], **describe_weighted_array(values, prepared.weights)}
                for row, values in zip(
                    summary_rows,
                    [prepared.x] + ([prepared.y] if prepared.y is not None else []),
                    strict=True,
                )
            ]
        )
        summary_notes = _weighted_summary_notes(request.weight_branch, prepared.weights)
    if prepared.color is not None:
        summary_rows.append(describe_array(prepared.color_label or "color", prepared.color))
    summary = pd.DataFrame(summary_rows)

    correlation_result = None
    if request.plot_type == "2D scatter" and prepared.y is not None:
        correlation_result = calculate_correlations(prepared.x, prepared.y)

    render_x, render_y, render_color = prepared.render_arrays(request.plot_type)
    sampling_note = _sampling_note(len(render_x), len(prepared.x))
    preview = (
        _preview_dataframe(render_x, render_y, None, prepared.weights, render_color)
        if request.preview
        else None
    )
    fit_result = None
    if request.fit_config is not None and prepared.display_histogram_1d is not None:
        with st.spinner("Fitting histogram…"):
            fit_result = fit_histogram_1d(prepared.display_histogram_1d, request.fit_config)
    figure = None
    try:
        with st.spinner("Rendering plot…"):
            figure, plot_warnings = render_prepared_static_plot(
                spec, prepared, fit_result=fit_result
            )
        if prepared.normalisation_result is not None and prepared.normalisation_result.error:
            plot_warnings.append(prepared.normalisation_result.error)
        elif prepared.normalisation_result is not None and prepared.normalisation_result.warning:
            plot_warnings.append(prepared.normalisation_result.warning)
        file_name = (
            f"{request.file_name}_{request.event_index}"
            if request.event_index is not None
            else request.file_name
        )
        return {
            "kind": "matplotlib",
            "plot_type": request.plot_type,
            "figure": figure,
            "code": _safe_generate_plot_code(request, plot_warnings),
            "summary": summary,
            "exclusion_note": _exclusion_note(request, prepared.excluded_count),
            "weighted_summary": weighted_summary,
            "summary_notes": summary_notes,
            "preview": preview,
            "sampling_note": sampling_note,
            "threshold_result": prepared.threshold_result,
            "quantile_result": prepared.quantile_result,
            "fit_result": fit_result,
            "histogram_flow_summary": prepared.histogram_flow_summary,
            "histogram_flow_mode": prepared.histogram_flow_mode,
            "normalisation_result": prepared.normalisation_result,
            "normalisation_range": (
                tuple(float(edge) for edge in prepared.histogram_1d.axes[0].edges[[0, -1]])
                if prepared.histogram_1d is not None and prepared.normalisation_result is not None
                else None
            ),
            "correlation_result": correlation_result,
            "color_clipping_result": prepared.color_clipping_result,
            "warnings": plot_warnings,
            "file_name": file_name,
        }
    except Exception:
        if figure is not None:
            plt.close(figure)
        raise


def _build_plot_result(request: PlotRequest) -> dict[str, Any]:
    """Create the cached plot payload shared by global and event plot tabs."""

    if request.plot_type in STATIC_PLOT_TYPES:
        return _build_static_plot_result(request)

    context = request.context
    event_index = request.event_index
    validate_variable_event_shapes(
        [spec for spec in [request.x_spec, request.y_spec, request.z_spec] if spec is not None],
        context.root_path,
        context.event_container_name,
        context.entry_start,
        context.entry_stop,
        event_index,
        context="event plot-axis variables" if event_index is not None else "plot-axis variables",
    )
    arrays: list[np.ndarray] = []
    axis_labels: list[str] = []
    x, x_label = evaluate_variable(
        request.x_spec,
        context.root_path,
        context.event_container_name,
        context.entry_start,
        context.entry_stop,
        event_index,
    )
    arrays.append(x)
    axis_labels.append(x_label)
    y, _y_label = _evaluate_optional(request.y_spec, context, event_index, arrays, axis_labels)
    z, _z_label = _evaluate_optional(request.z_spec, context, event_index, arrays, axis_labels)

    values_length = array_length(x)
    weights = None
    if request.weight_branch is not None:
        weights = histogram_weights_for_values(
            context.root_path,
            context.event_container_name,
            request.weight_branch,
            request.x_spec,
            values_length,
            context.entry_start,
            context.entry_stop,
            event_index,
        )
        arrays.append(weights)
    color = color_label = None
    if request.color_spec is not None:
        color, color_label = variable_values_for_values(
            request.color_spec,
            request.x_spec,
            values_length,
            context.root_path,
            context.event_container_name,
            context.entry_start,
            context.entry_stop,
            event_index,
        )
        arrays.append(color)

    cuts: list[tuple[np.ndarray, float | None, float | None]] = []
    for spec, bounds in zip(request.cut_specs, request.cut_bounds, strict=True):
        values, _ = cut_values_for_values(
            spec,
            request.x_spec,
            values_length,
            context.root_path,
            context.event_container_name,
            context.entry_start,
            context.entry_stop,
            event_index,
        )
        cuts.append((values, bounds[0], bounds[1]))
    arrays, excluded_count = select_plot_values(arrays, cuts)
    x, y, z, weights, color = _unpack_arrays(
        arrays,
        has_y=request.y_spec is not None,
        has_z=request.z_spec is not None,
        has_weights=request.weight_branch is not None,
        has_color=request.color_spec is not None,
    )
    require_values(x, excluded_count)
    threshold_result = None
    quantile_result = None
    reference_x = None
    reference_label = None
    if request.plot_type == "1D histogram" and request.threshold is not None:
        threshold_result = calculate_threshold_result(x, request.threshold, weights)
        reference_x = request.threshold
        reference_label = f"threshold = {request.threshold:g}"
    elif request.plot_type == "1D histogram" and request.quantile_percentile is not None:
        quantile_result = calculate_quantile_result(x, request.quantile_percentile, weights)
        if quantile_result.value is not None:
            reference_x = quantile_result.value
            reference_label = (
                f"{request.quantile_percentile:g}% quantile = {quantile_result.value:g}"
            )

    summary_rows = [describe_array(axis_labels[0], x)]
    if y is not None:
        summary_rows.append(describe_array(axis_labels[1], y))
    if z is not None:
        summary_rows.append(describe_array(axis_labels[2], z))
    if color is not None:
        summary_rows.append(describe_array(color_label or "color", color))
    summary = pd.DataFrame(summary_rows)
    correlation_result = None
    color_clipping_result = None
    if request.plot_type == "2D scatter":
        if y is None:
            raise ValueError("2D scatter requires a y variable.")
        correlation_result = calculate_correlations(x, y)
    if supports_point_color(request.plot_type) and color is not None:
        color_clipping_result = calculate_color_clipping(color, request.style, weights)

    value_count = len(x)
    x, y, z, color = _downsample_scatter(request.plot_type, request.style, x, y, z, color)
    sampling_note = _sampling_note(len(x), value_count)
    x_label = request.labels["custom_x"] or axis_labels[0]
    default_y_label = axis_labels[1] if request.y_spec is not None else "entries"
    if request.y_spec is None and request.weight_branch is not None:
        default_y_label = f"sum weights ({request.weight_branch})"
    y_axis_label = request.labels["custom_y"] or default_y_label
    z_axis_label = request.labels["custom_z"] or (
        axis_labels[2] if request.z_spec is not None else ""
    )
    title = request.labels["title"] or (
        f"Event {event_index}" if event_index is not None else request.plot_type
    )
    file_name = (
        f"{request.file_name}_{event_index}" if event_index is not None else request.file_name
    )
    preview = _preview_dataframe(x, y, z, weights, color) if request.preview else None

    if request.plot_type == "3D scatter interactive":
        if y is None or z is None:
            raise ValueError("3D scatter requires y and z variables.")
        color_min, color_max = (
            scatter_color_scale_linear(request.style) if color is not None else (None, None)
        )
        figure = build_plotly_3d(
            x,
            y,
            z,
            x_label,
            y_axis_label,
            z_axis_label,
            color=color,
            color_label=request.labels.get("custom_colorbar") or color_label,
            title=title,
            marker_size=request.style["marker_size_3d"],
            opacity=request.style["opacity_3d"],
            colorscale=request.style.get("colorscale", "Viridis"),
            reversescale=request.style.get("reverse_colorscale", False),
            aspectmode=request.style["aspectmode"],
            color_min=color_min,
            color_max=color_max,
        )
        return {
            "kind": "plotly",
            "plot_type": request.plot_type,
            "figure": overlay_geometry_on_plotly(
                figure, request.geometry.scene, request.geometry.options
            ),
            "geometry_source": request.geometry.source,
            "summary": summary,
            "exclusion_note": _exclusion_note(request, excluded_count),
            "preview": preview,
            "sampling_note": sampling_note,
            "threshold_result": threshold_result,
            "quantile_result": quantile_result,
            "correlation_result": correlation_result,
            "color_clipping_result": color_clipping_result,
            "file_name": file_name,
        }
    figure, axes = plt.subplots(figsize=(request.style["fig_width"], request.style["fig_height"]))
    try:
        plot_warnings = render_static_plot(
            figure,
            axes,
            request.plot_type,
            x,
            y,
            x_label,
            y_axis_label,
            request.style,
            title=title,
            weights=weights,
            weight_branch=request.weight_branch,
            color=color,
            color_label=color_label,
            reference_x=reference_x,
            reference_label=reference_label,
            colorbar_extend=(
                color_clipping_result.extend if color_clipping_result is not None else "neither"
            ),
        )
        return {
            "kind": "matplotlib",
            "figure": figure,
            "code": generate_static_plot_code(request),
            "summary": summary,
            "preview": preview,
            "sampling_note": sampling_note,
            "threshold_result": threshold_result,
            "quantile_result": quantile_result,
            "correlation_result": correlation_result,
            "color_clipping_result": color_clipping_result,
            "warnings": plot_warnings,
            "file_name": file_name,
        }
    except Exception:
        plt.close(figure)
        raise


def _evaluate_optional(
    spec: VariableSpec | None,
    context: DataContext,
    event_index: int | None,
    arrays: list[np.ndarray],
    labels: list[str],
) -> tuple[np.ndarray | None, str]:
    if spec is None:
        return None, ""
    values, label = evaluate_variable(
        spec,
        context.root_path,
        context.event_container_name,
        context.entry_start,
        context.entry_stop,
        event_index,
    )
    arrays.append(values)
    labels.append(label)
    return values, label


def _unpack_arrays(
    arrays: list[np.ndarray], *, has_y: bool, has_z: bool, has_weights: bool, has_color: bool
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    index = 0
    x = arrays[index]
    index += 1
    y = arrays[index] if has_y else None
    index += int(has_y)
    z = arrays[index] if has_z else None
    index += int(has_z)
    weights = arrays[index] if has_weights else None
    index += int(has_weights)
    color = arrays[index] if has_color else None
    return x, y, z, weights, color


def _downsample_scatter(
    plot_type: str,
    style: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray | None,
    z: np.ndarray | None,
    color: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if plot_type not in {"2D scatter", "3D scatter interactive"}:
        return x, y, z, color
    arrays = [x] + [value for value in [y, z, color] if value is not None]
    sampled = downsample_arrays(arrays, int(style["max_points"]))
    index = 1
    sampled_y = sampled[index] if y is not None else None
    index += int(y is not None)
    sampled_z = sampled[index] if z is not None else None
    index += int(z is not None)
    sampled_color = sampled[index] if color is not None else None
    return sampled[0], sampled_y, sampled_z, sampled_color


def _preview_dataframe(
    x: np.ndarray,
    y: np.ndarray | None,
    z: np.ndarray | None,
    weights: np.ndarray | None,
    color: np.ndarray | None,
) -> pd.DataFrame | None:
    values: dict[str, np.ndarray] = {
        "x": x,
        **{
            name: array
            for name, array in [("y", y), ("z", z), ("weights", weights), ("color", color)]
            if array is not None
        },
    }
    size = min(len(array) for array in values.values()) if values else 0
    return (
        pd.DataFrame({name: array[:size] for name, array in values.items()}).head(500)
        if size
        else None
    )
