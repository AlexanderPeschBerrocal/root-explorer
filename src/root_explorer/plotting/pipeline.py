"""Streamlit-independent preparation and rendering for static plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import hist
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.arrays import (
    array_length,
    filter_finite_counted,
    histogram_weights_for_values,
    require_values,
    validate_variable_event_shapes,
    variable_values_for_values,
)
from ..analysis.fitting import FitConfig, FitResult
from ..analysis.histograms import (
    HistogramFlowSummary,
    HistogramNormalisationResult,
    build_histogram_1d,
    build_histogram_2d,
    histogram_for_flow_mode,
    normalise_histogram_1d,
    resolve_histogram_range,
    summarise_histogram_flow,
)
from ..analysis.statistics import (
    QuantileResult,
    ThresholdResult,
    calculate_quantile_result,
    calculate_threshold_result,
)
from ..analysis.variables import evaluate_variable
from ..models import VariableSpec
from .plots import (
    ColorClippingResult,
    apply_range_cuts,
    calculate_color_clipping,
    create_static_figure,
    downsample_arrays,
    render_static_plot,
)

STATIC_PLOT_TYPES = frozenset({"1D histogram", "2D histogram", "2D scatter"})


@dataclass(frozen=True)
class StaticPlotSpec:
    """Complete, Streamlit-independent configuration for one static plot."""

    root_path: str
    tree_name: str
    entry_start: int
    entry_stop: int | None
    plot_type: str
    x_spec: VariableSpec
    y_spec: VariableSpec | None
    color_spec: VariableSpec | None
    weight_branch: str | None
    cut_specs: tuple[VariableSpec, ...]
    cut_bounds: tuple[tuple[float | None, float | None], ...]
    style: dict[str, Any]
    labels: dict[str, Any]
    event_index: int | None = None
    threshold: float | None = None
    quantile_percentile: float | None = None
    fit_config: FitConfig | None = None


@dataclass(frozen=True)
class PreparedStaticPlot:
    """Finite, aligned plot values plus resolved labels and annotations."""

    x: np.ndarray
    y: np.ndarray | None
    weights: np.ndarray | None
    color: np.ndarray | None
    histogram_1d: hist.Hist | None
    histogram_2d: hist.Hist | None
    x_label: str
    y_label: str
    color_label: str | None
    title: str
    reference_x: float | None
    reference_label: str | None
    threshold_result: ThresholdResult | None
    quantile_result: QuantileResult | None
    color_clipping_result: ColorClippingResult | None
    max_points: int
    display_histogram_1d: hist.Hist | None = None
    histogram_flow_summary: HistogramFlowSummary | None = None
    histogram_flow_mode: str = "ignore"
    normalisation_result: HistogramNormalisationResult | None = None
    excluded_count: int = 0

    def render_arrays(
        self, plot_type: str
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return plot arrays after deterministic scatter downsampling."""

        if plot_type != "2D scatter" or self.y is None:
            return self.x, self.y, self.color
        arrays = [self.x, self.y, *([self.color] if self.color is not None else [])]
        sampled = downsample_arrays(arrays, self.max_points)
        color = sampled[2] if self.color is not None else None
        return sampled[0], sampled[1], color


def prepare_static_plot(spec: StaticPlotSpec) -> PreparedStaticPlot:
    """Read, align, filter, and annotate values for a static plot."""

    if spec.plot_type not in STATIC_PLOT_TYPES:
        raise ValueError(f"Unsupported static plot type: {spec.plot_type}")
    if spec.plot_type in {"2D histogram", "2D scatter"} and spec.y_spec is None:
        raise ValueError(f"{spec.plot_type} requires a y variable.")

    axis_specs = [candidate for candidate in (spec.x_spec, spec.y_spec) if candidate is not None]
    validate_variable_event_shapes(
        axis_specs,
        spec.root_path,
        spec.tree_name,
        spec.entry_start,
        spec.entry_stop,
        spec.event_index,
        context=(
            "event plot-axis variables" if spec.event_index is not None else "plot-axis variables"
        ),
    )

    x, default_x_label = evaluate_variable(
        spec.x_spec,
        spec.root_path,
        spec.tree_name,
        spec.entry_start,
        spec.entry_stop,
        spec.event_index,
    )
    y = None
    default_y_label = "entries"
    arrays: list[np.ndarray] = [x]
    if spec.y_spec is not None:
        y, default_y_label = evaluate_variable(
            spec.y_spec,
            spec.root_path,
            spec.tree_name,
            spec.entry_start,
            spec.entry_stop,
            spec.event_index,
        )
        arrays.append(y)

    value_count = array_length(x)
    weights = None
    if spec.weight_branch is not None:
        weights = histogram_weights_for_values(
            spec.root_path,
            spec.tree_name,
            spec.weight_branch,
            spec.x_spec,
            value_count,
            spec.entry_start,
            spec.entry_stop,
            spec.event_index,
        )
        arrays.append(weights)

    color = None
    color_label = None
    if spec.color_spec is not None:
        color, color_label = variable_values_for_values(
            spec.color_spec,
            spec.x_spec,
            value_count,
            spec.root_path,
            spec.tree_name,
            spec.entry_start,
            spec.entry_stop,
            spec.event_index,
        )
        arrays.append(color)

    cuts: list[tuple[np.ndarray, float | None, float | None]] = []
    for cut_spec, bounds in zip(spec.cut_specs, spec.cut_bounds, strict=True):
        cut_values, _ = variable_values_for_values(
            cut_spec,
            spec.x_spec,
            value_count,
            spec.root_path,
            spec.tree_name,
            spec.entry_start,
            spec.entry_stop,
            spec.event_index,
        )
        cuts.append((cut_values, bounds[0], bounds[1]))

    arrays, excluded_count = filter_finite_counted(apply_range_cuts(arrays, cuts))
    x, y, weights, color = _unpack_prepared_arrays(
        arrays,
        has_y=spec.y_spec is not None,
        has_weights=spec.weight_branch is not None,
        has_color=spec.color_spec is not None,
    )
    require_values(x, excluded_count)

    threshold_result = None
    quantile_result = None
    reference_x = None
    reference_label = None
    if spec.plot_type == "1D histogram" and spec.threshold is not None:
        threshold_result = calculate_threshold_result(x, spec.threshold, weights)
        reference_x = spec.threshold
        reference_label = f"threshold = {spec.threshold:g}"
    elif spec.plot_type == "1D histogram" and spec.quantile_percentile is not None:
        quantile_result = calculate_quantile_result(x, spec.quantile_percentile, weights)
        if quantile_result.value is not None:
            reference_x = quantile_result.value
            reference_label = f"{spec.quantile_percentile:g}% quantile = {quantile_result.value:g}"

    color_clipping_result = None
    if spec.plot_type == "2D scatter" and color is not None:
        color_clipping_result = calculate_color_clipping(color, spec.style, weights)

    x_label = spec.labels.get("custom_x") or default_x_label
    if spec.y_spec is None and spec.weight_branch is not None:
        default_y_label = f"sum weights ({spec.weight_branch})"
    y_label = spec.labels.get("custom_y") or default_y_label
    title = spec.labels.get("title") or (
        f"Event {spec.event_index}" if spec.event_index is not None else spec.plot_type
    )
    histogram_1d = (
        build_histogram_1d(
            x,
            int(spec.style["bins_x"]),
            weights=weights,
            value_range=resolve_histogram_range(x, int(spec.style["bins_x"]), spec.style),
            label=x_label,
        )
        if spec.plot_type == "1D histogram"
        else None
    )

    histogram_range = (
        tuple(histogram_1d.axes[0].edges[[0, -1]])
        if histogram_1d is not None
        and spec.style.get("manual_histogram_limits", False)
        and (
            str(spec.style.get("hist_x_min_text", "")).strip()
            or str(spec.style.get("hist_x_max_text", "")).strip()
        )
        else None
    )
    flow_mode = str(spec.style.get("hist_flow_mode", "ignore"))
    display_histogram_1d = (
        histogram_for_flow_mode(histogram_1d, flow_mode)
        if histogram_1d is not None and histogram_range is not None
        else histogram_1d
    )
    flow_summary = (
        summarise_histogram_flow(histogram_1d, x, weighted=weights is not None)
        if histogram_1d is not None and histogram_range is not None
        else None
    )
    if histogram_1d is not None and flow_mode not in {"ignore", "fold", "separate"}:
        raise ValueError(f"Unsupported histogram flow mode: {flow_mode}")
    if histogram_range is None:
        flow_mode = "ignore"

    normalisation_result = None
    if histogram_1d is not None and spec.style.get("normalise_unit_area", False):
        if flow_mode == "separate" and histogram_range is not None:
            normalisation_result = HistogramNormalisationResult(
                None,
                float(np.sum(display_histogram_1d.values())),
                bool(weights is not None and np.any(weights < 0)),
                error=(
                    "Unit-area density is incompatible with separate categorical flow bins. "
                    "Choose Ignore or Fold instead."
                ),
            )
        else:
            within = (
                (x >= histogram_range[0]) & (x <= histogram_range[1])
                if histogram_range is not None and flow_mode == "ignore"
                else np.ones(len(x), dtype=bool)
            )
            relevant_weights = weights[within] if weights is not None else None
            absolute_total = (
                float(np.sum(np.abs(relevant_weights)))
                if relevant_weights is not None
                else float(np.count_nonzero(within))
            )
            negative = bool(relevant_weights is not None and np.any(relevant_weights < 0))
            normalisation_result = normalise_histogram_1d(
                display_histogram_1d,
                absolute_weight_sum=absolute_total,
                has_negative_weights=negative,
            )

    if (
        normalisation_result is not None
        and normalisation_result.valid
        and not spec.labels.get("custom_y")
    ):
        y_label = "Normalised density"

    histogram_2d = (
        build_histogram_2d(
            x,
            y,
            int(spec.style["bins_x"]),
            int(spec.style["bins_y"]),
            weights=weights,
            x_label=x_label,
            y_label=y_label,
        )
        if spec.plot_type == "2D histogram" and y is not None
        else None
    )

    return PreparedStaticPlot(
        x=x,
        y=y,
        weights=weights,
        color=color,
        histogram_1d=histogram_1d,
        histogram_2d=histogram_2d,
        x_label=x_label,
        y_label=y_label,
        color_label=color_label,
        title=title,
        reference_x=reference_x,
        reference_label=reference_label,
        threshold_result=threshold_result,
        quantile_result=quantile_result,
        color_clipping_result=color_clipping_result,
        max_points=int(spec.style.get("max_points", 100_000)),
        display_histogram_1d=display_histogram_1d,
        histogram_flow_summary=flow_summary,
        histogram_flow_mode=flow_mode,
        normalisation_result=normalisation_result,
        excluded_count=excluded_count,
    )


def render_prepared_static_plot(
    spec: StaticPlotSpec,
    prepared: PreparedStaticPlot,
    *,
    fit_result: FitResult | None = None,
):
    """Render a prepared static plot with the canonical Matplotlib implementation."""

    x, y, color = prepared.render_arrays(spec.plot_type)
    figure, axes = create_static_figure(spec.style)
    try:
        warnings = render_static_plot(
            figure,
            axes,
            spec.plot_type,
            x,
            y,
            prepared.x_label,
            prepared.y_label,
            spec.style,
            title=prepared.title,
            histogram_1d=(
                prepared.normalisation_result.histogram
                if prepared.normalisation_result is not None and prepared.normalisation_result.valid
                else prepared.display_histogram_1d
            ),
            fit_histogram_1d=prepared.display_histogram_1d,
            normalisation_result=prepared.normalisation_result,
            histogram_flow_summary=prepared.histogram_flow_summary,
            histogram_flow_mode=prepared.histogram_flow_mode,
            histogram_2d=prepared.histogram_2d,
            weights=prepared.weights,
            weight_branch=spec.weight_branch,
            color=color,
            color_label=prepared.color_label,
            colorbar_label=spec.labels.get("custom_colorbar") or None,
            reference_x=prepared.reference_x,
            reference_label=prepared.reference_label,
            fit_result=fit_result,
            colorbar_extend=(
                prepared.color_clipping_result.extend
                if prepared.color_clipping_result is not None
                else "neither"
            ),
        )
        return figure, warnings
    except Exception:
        plt.close(figure)
        raise


def _unpack_prepared_arrays(
    arrays: list[np.ndarray],
    *,
    has_y: bool,
    has_weights: bool,
    has_color: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    index = 0
    x = arrays[index]
    index += 1
    y = arrays[index] if has_y else None
    index += int(has_y)
    weights = arrays[index] if has_weights else None
    index += int(has_weights)
    color = arrays[index] if has_color else None
    return x, y, weights, color
