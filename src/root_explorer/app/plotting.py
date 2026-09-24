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
    apply_finite_filter,
    array_length,
    cut_values_for_values,
    histogram_weights_for_values,
    validate_variable_event_shapes,
    variable_values_for_values,
)
from ..analysis.statistics import calculate_threshold_result
from ..analysis.variables import evaluate_variable
from ..geometry.overlay import overlay_geometry_on_plotly
from ..models import VariableSpec
from ..plotting.codegen import generate_static_plot_code
from ..plotting.plots import (
    InvalidAxisScaleError,
    apply_range_cuts,
    build_plotly_3d,
    describe_array,
    downsample_arrays,
    parse_optional_float,
    render_static_plot,
)
from .controls import (
    PLOT_TYPES,
    histogram_weight_selector,
    label_controls,
    needs_y_variable,
    needs_z_variable,
    plot_style_controls,
    range_cut_controls,
    supports_point_color,
    threshold_calculator_controls,
    variable_selector,
)
from .results import show_cached_plot_result
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
    save_name: str
    threshold: float | None = None


def render_plot_tab(context: DataContext, geometry: GeometryState) -> None:
    """Render controls and output for plotting the selected entry range."""

    controls, output = st.columns([0.36, 0.64], gap="large")
    with controls:
        st.subheader("Plot configuration")
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
            save_name="root_explorer_plot",
        )
    with output:
        st.subheader("Output")
        _generate_and_show(request, make_plot, "plot_result", "plot_save_cached")


def render_event_plot_tab(context: DataContext, geometry: GeometryState, event_index: int) -> None:
    """Render controls and output for one selected event."""

    controls, output = st.columns([0.36, 0.64], gap="large")
    with controls:
        st.subheader("Plot configuration")
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
            save_name="root_explorer_event_plot",
        )
    with output:
        st.subheader(f"Output: event {event_index}")
        _generate_and_show(request, make_plot, "event_plot_result", "event_save_cached")


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
    save_name: str,
) -> tuple[PlotRequest, bool]:
    """Render the shared plot-control surface with stable per-tab widget keys."""

    is_event_plot = event_index is not None
    plot_type = st.selectbox(
        "Plot type", PLOT_TYPES, index=0, key="event_plot_type" if is_event_plot else "plot_type"
    )
    st.markdown("### Variables")
    axis_prefix = (lambda axis: f"{prefix}_{axis}") if is_event_plot else (lambda axis: axis)
    x_spec = variable_selector(
        axis_prefix("x"),
        context.fields,
        ["x", "pt", "eta"] if event_index is not None else ["pt", "energy", "eta", "x"],
        item_name=context.data_item_name,
        allow_event_reductions=event_index is None,
    )
    y_spec = z_spec = color_spec = None
    if needs_y_variable(plot_type):
        y_spec = variable_selector(
            axis_prefix("y"),
            context.fields,
            ["y", "eta", "phi"] if event_index is not None else ["eta", "phi", "y"],
            item_name=context.data_item_name,
            allow_event_reductions=event_index is None,
        )
    if needs_z_variable(plot_type):
        z_spec = variable_selector(
            axis_prefix("z"),
            context.fields,
            ["z", "phi"] if event_index is not None else ["phi", "z"],
            item_name=context.data_item_name,
            allow_event_reductions=event_index is None,
        )
    color_enabled = False
    if supports_point_color(plot_type):
        color_enabled = st.checkbox(
            "Color points",
            value=plot_type == "3D scatter interactive",
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
                allow_event_reductions=False,
            )

    weight_branch = histogram_weight_selector(
        prefix, plot_type, context.fields, item_name=context.data_item_name
    )
    threshold = threshold_calculator_controls(prefix, plot_type)
    cut_specs, cut_bounds = range_cut_controls(
        prefix, context.fields, item_name=context.data_item_name
    )
    style = plot_style_controls(
        prefix,
        plot_type,
        color_enabled=color_enabled,
        default_bins=default_bins,
        default_marker_size_2d=default_marker_size_2d,
        default_marker_size_3d=default_marker_size_3d,
        default_opacity_2d=default_opacity_2d,
        default_opacity_3d=default_opacity_3d,
        default_equal_aspect=False,
    )
    labels = label_controls(prefix, plot_type)
    make_plot = st.button(
        "Make event plot" if is_event_plot else "Make plot",
        type="primary",
        key=f"{prefix}_make",
        shortcut="Shift+P" if is_event_plot else "P",
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
            save_name=save_name,
            threshold=threshold,
        ),
        make_plot,
    )


def _generate_and_show(
    request: PlotRequest, make_plot: bool, result_key: str, save_button_key: str
) -> None:
    if make_plot:
        try:
            st.session_state[result_key] = _build_plot_result(request)
        except ArrayShapeMismatchError as exc:
            st.session_state.pop(result_key, None)
            st.warning(str(exc))
        except InvalidAxisScaleError as exc:
            st.session_state.pop(result_key, None)
            st.warning(str(exc))
        except Exception as exc:
            st.session_state.pop(result_key, None)
            prefix = "Event plot" if request.event_index is not None else "Plot"
            st.error(f"{prefix} failed: {exc}")
            st.exception(exc)
    show_cached_plot_result(
        result_key,
        export_prefix="event" if request.event_index is not None else "plot",
        save_button_key=save_button_key,
        make_plot_label="Make event plot" if request.event_index is not None else "Make plot",
        make_plot_shortcut="Shift + P" if request.event_index is not None else "P",
        empty_message=(
            "Choose an event and settings, then click Make event plot or press Shift + P."
            if request.event_index is not None
            else "Configure settings on the left, then click Make plot or press P."
        ),
    )


def _build_plot_result(request: PlotRequest) -> dict[str, Any]:
    """Create the cached plot payload shared by global and event plot tabs."""

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
    arrays = apply_finite_filter(apply_range_cuts(arrays, cuts))
    x, y, z, weights, color = _unpack_arrays(
        arrays,
        has_y=request.y_spec is not None,
        has_z=request.z_spec is not None,
        has_weights=request.weight_branch is not None,
        has_color=request.color_spec is not None,
    )
    threshold_result = None
    if request.plot_type == "1D histogram" and request.threshold is not None:
        threshold_result = calculate_threshold_result(x, request.threshold, weights)
    x, y, z, color = _downsample_scatter(request.plot_type, request.style, x, y, z, color)

    summary_rows = [describe_array(axis_labels[0], x)]
    if request.y_spec is not None:
        summary_rows.append(describe_array(axis_labels[1], y))
    if request.z_spec is not None:
        summary_rows.append(describe_array(axis_labels[2], z))
    if color is not None:
        summary_rows.append(describe_array(color_label or "color", color))
    summary = pd.DataFrame(summary_rows)
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
    save_name = (
        f"{request.save_name}_{event_index}" if event_index is not None else request.save_name
    )
    preview = _preview_dataframe(x, y, z, weights, color) if request.preview else None

    if request.plot_type == "3D scatter interactive":
        figure = build_plotly_3d(
            x,
            y,
            z,
            x_label,
            y_axis_label,
            z_axis_label,
            color=color,
            color_label=color_label,
            title=title,
            marker_size=request.style["marker_size_3d"],
            opacity=request.style["opacity_3d"],
            colorscale=request.style.get("colorscale", "Viridis"),
            reversescale=request.style.get("reverse_colorscale", False),
            aspectmode=request.style["aspectmode"],
            color_min=parse_optional_float(request.style.get("color_min_text", "")),
            color_max=parse_optional_float(request.style.get("color_max_text", "")),
        )
        return {
            "kind": "plotly",
            "figure": overlay_geometry_on_plotly(
                figure, request.geometry.scene, request.geometry.options
            ),
            "summary": summary,
            "preview": preview,
            "threshold_result": threshold_result,
            "save_name": save_name,
            "save_button_label": "Save HTML",
        }
    figure, axes = plt.subplots(figsize=(request.style["fig_width"], request.style["fig_height"]))
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
    )
    return {
        "kind": "matplotlib",
        "figure": figure,
        "code": generate_static_plot_code(request),
        "summary": summary,
        "preview": preview,
        "threshold_result": threshold_result,
        "warnings": plot_warnings,
        "save_name": save_name,
        "save_button_label": "Save selected formats",
    }


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
