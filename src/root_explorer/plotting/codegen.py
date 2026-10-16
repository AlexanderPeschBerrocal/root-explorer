"""Standalone Python source generation for static ROOT Explorer plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..analysis.arrays import (
    align_variable_values_to_reference,
    apply_finite_filter,
    array_length,
    histogram_weights_for_values,
    validate_variable_event_shapes,
)
from ..analysis.fitting import FitConfig, fit_histogram_1d
from ..analysis.histograms import (
    build_histogram_1d,
    build_histogram_2d,
    histogram_for_flow_mode,
    resolve_histogram_range,
    summarise_histogram_flow,
)
from ..analysis.statistics import calculate_quantile_result, calculate_threshold_result
from ..analysis.variables import VariableEvaluator, variable_evaluator
from .pipeline import STATIC_PLOT_TYPES, StaticPlotSpec
from .plots import (
    apply_log_histogram_y_axis,
    apply_log_x_axis,
    apply_log_y_axis,
    apply_range_cuts,
    calculate_color_clipping,
    create_static_figure,
    downsample_arrays,
    draw_colored_scatter_2d_linear,
    draw_colored_scatter_2d_log,
    draw_histogram_1d,
    draw_histogram_2d,
    draw_histogram_contours,
    draw_kde_1d_overlay,
    draw_kde_2d_overlay,
    draw_scatter_2d,
    finish_static_axes,
)
from .sourcegen import bundle_python_source, python_source_literal

if TYPE_CHECKING:
    from ..app.plotting import PlotRequest


@dataclass(frozen=True)
class StaticPlotExportPlan:
    """Resolved source roots and selected operations for one static plot export."""

    spec: StaticPlotSpec
    x_evaluator: VariableEvaluator
    y_evaluator: VariableEvaluator | None
    color_evaluator: VariableEvaluator | None
    cut_evaluators: tuple[VariableEvaluator, ...]
    helper_roots: tuple[Any, ...]
    has_weights: bool
    has_color: bool
    show_kde: bool
    show_contours: bool
    has_threshold: bool
    has_quantile: bool
    has_fit: bool
    log_x: bool
    log_y: bool
    log_z: bool
    log_color: bool


def build_static_plot_export_plan(spec: StaticPlotSpec) -> StaticPlotExportPlan:
    """Resolve the exact canonical helpers needed by one selected static plot."""

    if spec.plot_type not in STATIC_PLOT_TYPES:
        raise ValueError(f"Unsupported static plot type: {spec.plot_type}")
    if spec.plot_type in {"2D histogram", "2D scatter"} and spec.y_spec is None:
        raise ValueError(f"{spec.plot_type} requires a y variable.")

    has_weights = spec.weight_branch is not None
    has_color = spec.color_spec is not None
    show_kde = spec.plot_type in {"1D histogram", "2D histogram"} and bool(
        spec.style.get("show_kde", False)
    )
    show_contours = spec.plot_type == "2D histogram" and bool(
        spec.style.get("show_contours", False)
    )
    has_threshold = spec.plot_type == "1D histogram" and spec.threshold is not None
    has_quantile = spec.plot_type == "1D histogram" and spec.quantile_percentile is not None
    has_fit = spec.plot_type == "1D histogram" and spec.fit_config is not None
    log_x = bool(spec.style.get("log_x", False))
    log_y = bool(spec.style.get("log_y", False))
    log_z = spec.plot_type == "2D histogram" and bool(spec.style.get("log_z", False))
    log_color = (
        spec.plot_type == "2D scatter" and has_color and bool(spec.style.get("log_color", False))
    )

    x_evaluator = variable_evaluator(spec.x_spec)
    y_evaluator = variable_evaluator(spec.y_spec) if spec.y_spec is not None else None
    color_evaluator = variable_evaluator(spec.color_spec) if spec.color_spec is not None else None
    cut_evaluators = tuple(variable_evaluator(cut_spec) for cut_spec in spec.cut_specs)

    roots: list[Any] = [
        StaticPlotSpec,
        create_static_figure,
        x_evaluator,
        array_length,
        apply_finite_filter,
        finish_static_axes,
    ]
    if y_evaluator is not None:
        roots.extend([validate_variable_event_shapes, y_evaluator])
    if has_weights:
        roots.append(histogram_weights_for_values)
    if color_evaluator is not None:
        roots.extend([color_evaluator, align_variable_values_to_reference])
    if cut_evaluators:
        roots.append(apply_range_cuts)
        roots.extend(cut_evaluators)
        roots.append(align_variable_values_to_reference)
    if has_threshold:
        roots.append(calculate_threshold_result)
    if has_quantile:
        roots.append(calculate_quantile_result)
    if spec.plot_type == "1D histogram":
        roots.extend(
            [
                build_histogram_1d,
                draw_histogram_1d,
                resolve_histogram_range,
                histogram_for_flow_mode,
                summarise_histogram_flow,
            ]
        )
        if has_fit:
            from .plots import draw_fit_overlay

            roots.extend([FitConfig, fit_histogram_1d, draw_fit_overlay])
        if show_kde:
            roots.append(draw_kde_1d_overlay)
        if log_x:
            roots.append(apply_log_x_axis)
        if log_y:
            roots.append(apply_log_histogram_y_axis)
    elif spec.plot_type == "2D histogram":
        roots.extend([build_histogram_2d, draw_histogram_2d])
        if show_contours:
            roots.append(draw_histogram_contours)
        if show_kde:
            roots.append(draw_kde_2d_overlay)
        if log_x:
            roots.append(apply_log_x_axis)
        if log_y:
            roots.append(apply_log_y_axis)
    else:
        roots.append(downsample_arrays)
        if has_color:
            roots.append(calculate_color_clipping)
            roots.append(
                draw_colored_scatter_2d_log if log_color else draw_colored_scatter_2d_linear
            )
        else:
            roots.append(draw_scatter_2d)
        if log_x:
            roots.append(apply_log_x_axis)
        if log_y:
            roots.append(apply_log_y_axis)

    return StaticPlotExportPlan(
        spec=spec,
        x_evaluator=x_evaluator,
        y_evaluator=y_evaluator,
        color_evaluator=color_evaluator,
        cut_evaluators=cut_evaluators,
        helper_roots=tuple(roots),
        has_weights=has_weights,
        has_color=has_color,
        show_kde=show_kde,
        show_contours=show_contours,
        has_threshold=has_threshold,
        has_quantile=has_quantile,
        has_fit=has_fit,
        log_x=log_x,
        log_y=log_y,
        log_z=log_z,
        log_color=log_color,
    )


def generate_static_plot_code(request: PlotRequest) -> str:
    """Return a standalone script assembled from a plot-specific export plan."""

    if request.plot_type not in STATIC_PLOT_TYPES:
        raise ValueError(f"Plot-code download is not available for {request.plot_type!r}.")

    plan = build_static_plot_export_plan(request.to_static_spec())
    helpers = bundle_python_source(*plan.helper_roots).rstrip()
    configuration = python_source_literal(plan.spec)
    main_lines = _main_lines(plan)

    return (
        '"""Standalone plot exported by ROOT Explorer.\n\n'
        "This script is specialised to the operations selected when it was exported.\n"
        "You may edit input paths, branch names, labels, ranges, and non-structural styles.\n"
        "Regenerate it after changing plot type, variable modes, enabled features, or log scales.\n"
        '"""\n\n'
        f"{helpers}\n\n"
        "# Plot configuration\n"
        f"PLOT = {configuration}\n\n\n"
        "def _validate_export_configuration() -> None:\n"
        + "\n".join(f"    {line}" if line else "" for line in _compatibility_lines(plan))
        + "\n\n\n"
        "def main() -> None:\n"
        + "\n".join(f"    {line}" if line else "" for line in main_lines)
        + "\n\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )


def _compatibility_lines(plan: StaticPlotExportPlan) -> list[str]:
    """Emit validation that keeps a lightweight export aligned with its plan."""

    spec = plan.spec
    lines = [
        f"if PLOT.plot_type != {spec.plot_type!r}:",
        "    raise ValueError('This export is specialised to a different plot type; regenerate it.')",
        f"if PLOT.x_spec.mode != {spec.x_spec.mode!r}:",
        "    raise ValueError('This export is specialised to different variable modes; regenerate it.')",
        f"if (PLOT.y_spec is not None) != {spec.y_spec is not None!r}:",
        "    raise ValueError('This export is specialised to different plot inputs; regenerate it.')",
        f"if (PLOT.color_spec is not None) != {plan.has_color!r}:",
        "    raise ValueError('This export is specialised to different plot inputs; regenerate it.')",
        f"if (PLOT.weight_branch is not None) != {plan.has_weights!r}:",
        "    raise ValueError('This export is specialised to different plot inputs; regenerate it.')",
        f"if len(PLOT.cut_specs) != {len(spec.cut_specs)}:",
        "    raise ValueError('This export is specialised to different cuts; regenerate it.')",
        (
            "if (PLOT.plot_type in {'1D histogram', '2D histogram'} "
            f"and bool(PLOT.style.get('show_kde', False))) != {plan.show_kde!r}:"
        ),
        "    raise ValueError('This export is specialised to different optional features; regenerate it.')",
        (
            "if (PLOT.plot_type == '2D histogram' "
            f"and bool(PLOT.style.get('show_contours', False))) != {plan.show_contours!r}:"
        ),
        "    raise ValueError('This export is specialised to different optional features; regenerate it.')",
        (
            "if (PLOT.plot_type == '1D histogram' and PLOT.threshold is not None) "
            f"!= {plan.has_threshold!r}:"
        ),
        "    raise ValueError('This export is specialised to different statistics; regenerate it.')",
        (
            "if (PLOT.plot_type == '1D histogram' and PLOT.quantile_percentile is not None) "
            f"!= {plan.has_quantile!r}:"
        ),
        "    raise ValueError('This export is specialised to different statistics; regenerate it.')",
        f"if bool(PLOT.style.get('log_x', False)) != {plan.log_x!r}:",
        "    raise ValueError('This export is specialised to different log scales; regenerate it.')",
        f"if bool(PLOT.style.get('log_y', False)) != {plan.log_y!r}:",
        "    raise ValueError('This export is specialised to different log scales; regenerate it.')",
        (
            "if (PLOT.plot_type == '2D histogram' and bool(PLOT.style.get('log_z', False))) "
            f"!= {plan.log_z!r}:"
        ),
        "    raise ValueError('This export is specialised to different log scales; regenerate it.')",
        (
            "if (PLOT.plot_type == '2D scatter' and PLOT.color_spec is not None "
            f"and bool(PLOT.style.get('log_color', False))) != {plan.log_color!r}:"
        ),
        "    raise ValueError('This export is specialised to different log scales; regenerate it.')",
    ]
    lines.extend(
        [
            "if (PLOT.fit_config is not None) != " + repr(plan.has_fit) + ":",
            "    raise ValueError('This export is specialised to different fitting features; regenerate it.')",
        ]
    )
    if spec.y_spec is not None:
        lines.extend(
            [
                f"if PLOT.y_spec is None or PLOT.y_spec.mode != {spec.y_spec.mode!r}:",
                "    raise ValueError('This export is specialised to different variable modes; regenerate it.')",
            ]
        )
    if spec.color_spec is not None:
        lines.extend(
            [
                f"if PLOT.color_spec is None or PLOT.color_spec.mode != {spec.color_spec.mode!r}:",
                "    raise ValueError('This export is specialised to different variable modes; regenerate it.')",
            ]
        )
    for index, cut_spec in enumerate(spec.cut_specs):
        lines.extend(
            [
                f"if PLOT.cut_specs[{index}].mode != {cut_spec.mode!r}:",
                "    raise ValueError('This export is specialised to different variable modes; regenerate it.')",
            ]
        )
    return lines


def _main_lines(plan: StaticPlotExportPlan) -> list[str]:
    """Emit the small plot-specific recipe that invokes canonical helper functions."""

    spec = plan.spec
    x_evaluator = plan.x_evaluator.__name__
    lines = [
        "_validate_export_configuration()",
        (
            f"x, default_x_label = {x_evaluator}(PLOT.x_spec, PLOT.root_path, PLOT.tree_name, "
            "PLOT.entry_start, PLOT.entry_stop, PLOT.event_index)"
        ),
        "arrays = [x]",
        "value_count = array_length(x)",
    ]

    if plan.y_evaluator is not None:
        y_evaluator = plan.y_evaluator.__name__
        lines[1:1] = [
            "validate_variable_event_shapes(",
            "    [PLOT.x_spec, PLOT.y_spec], PLOT.root_path, PLOT.tree_name, PLOT.entry_start,",
            "    PLOT.entry_stop, PLOT.event_index, context=(",
            "        'event plot-axis variables' if PLOT.event_index is not None else 'plot-axis variables'",
            "    ),",
            ")",
        ]
        lines.extend(
            [
                (
                    f"y, default_y_label = {y_evaluator}(PLOT.y_spec, PLOT.root_path, "
                    "PLOT.tree_name, PLOT.entry_start, PLOT.entry_stop, PLOT.event_index)"
                ),
                "arrays.append(y)",
            ]
        )
    else:
        lines.extend(["y = None", "default_y_label = 'entries'"])

    if plan.has_weights:
        lines.extend(
            [
                "weights = histogram_weights_for_values(",
                "    PLOT.root_path, PLOT.tree_name, str(PLOT.weight_branch), PLOT.x_spec, value_count,",
                "    PLOT.entry_start, PLOT.entry_stop, PLOT.event_index,",
                ")",
                "arrays.append(weights)",
            ]
        )
    else:
        lines.append("weights = None")

    if plan.color_evaluator is not None:
        color_evaluator = plan.color_evaluator.__name__
        lines.extend(
            [
                (
                    f"color, color_label = {color_evaluator}(PLOT.color_spec, PLOT.root_path, "
                    "PLOT.tree_name, PLOT.entry_start, PLOT.entry_stop, PLOT.event_index)"
                ),
                "color = align_variable_values_to_reference(",
                "    color, color_label, PLOT.color_spec, PLOT.x_spec, value_count, PLOT.root_path,",
                "    PLOT.tree_name, PLOT.entry_start, PLOT.entry_stop, PLOT.event_index,",
                ")",
                "arrays.append(color)",
            ]
        )
    else:
        lines.extend(["color = None", "color_label = None"])

    if plan.cut_evaluators:
        lines.append("cuts = []")
        for index, evaluator in enumerate(plan.cut_evaluators):
            lines.extend(
                [
                    (
                        f"cut_{index}, cut_{index}_label = {evaluator.__name__}(PLOT.cut_specs[{index}], "
                        "PLOT.root_path, PLOT.tree_name, PLOT.entry_start, PLOT.entry_stop, "
                        "PLOT.event_index)"
                    ),
                    f"cut_{index} = align_variable_values_to_reference(",
                    f"    cut_{index}, cut_{index}_label, PLOT.cut_specs[{index}], PLOT.x_spec, value_count,",
                    "    PLOT.root_path, PLOT.tree_name, PLOT.entry_start, PLOT.entry_stop,",
                    "    PLOT.event_index,",
                    ")",
                    f"cuts.append((cut_{index}, PLOT.cut_bounds[{index}][0], PLOT.cut_bounds[{index}][1]))",
                ]
            )
        lines.append("arrays = apply_range_cuts(arrays, cuts)")

    lines.extend(["arrays = apply_finite_filter(arrays)"])
    index = 0
    lines.append(f"x = arrays[{index}]")
    index += 1
    if plan.y_evaluator is not None:
        lines.append(f"y = arrays[{index}]")
        index += 1
    if plan.has_weights:
        lines.append(f"weights = arrays[{index}]")
        index += 1
    if plan.has_color:
        lines.append(f"color = arrays[{index}]")

    lines.extend(
        [
            "x_label = PLOT.labels.get('custom_x') or default_x_label",
            "if PLOT.y_spec is None and PLOT.weight_branch is not None:",
            "    default_y_label = f'sum weights ({PLOT.weight_branch})'",
            "y_label = PLOT.labels.get('custom_y') or default_y_label",
            "title = PLOT.labels.get('title') or (",
            "    f'Event {PLOT.event_index}' if PLOT.event_index is not None else PLOT.plot_type",
            ")",
            "reference_x = None",
            "reference_label = None",
        ]
    )
    if plan.has_threshold:
        lines.extend(
            [
                "threshold_result = calculate_threshold_result(x, float(PLOT.threshold), weights)",
                "reference_x = float(PLOT.threshold)",
                "reference_label = f'threshold = {PLOT.threshold:g}'",
            ]
        )
    elif plan.has_quantile:
        lines.extend(
            [
                "quantile_result = calculate_quantile_result(x, float(PLOT.quantile_percentile), weights)",
                "if quantile_result.value is not None:",
                "    reference_x = quantile_result.value",
                "    reference_label = f'{PLOT.quantile_percentile:g}% quantile = {quantile_result.value:g}'",
            ]
        )

    if spec.plot_type == "2D scatter":
        if plan.has_color:
            lines.append(
                "colorbar_extend = calculate_color_clipping(color, PLOT.style, weights).extend"
            )
        else:
            lines.append("colorbar_extend = 'neither'")
        lines.extend(
            [
                "render_arrays = [x, y] + ([color] if color is not None else [])",
                "render_arrays = downsample_arrays(render_arrays, int(PLOT.style.get('max_points', 100_000)))",
                "x, y = render_arrays[:2]",
                "if color is not None:",
                "    color = render_arrays[2]",
            ]
        )

    if spec.plot_type == "1D histogram":
        lines.extend(
            [
                "histogram_1d = build_histogram_1d(",
                "    x, int(PLOT.style['bins_x']), weights=weights,",
                "    value_range=resolve_histogram_range(x, int(PLOT.style['bins_x']), PLOT.style),",
                "    label=x_label,",
                ")",
                "flow_range = resolve_histogram_range(x, int(PLOT.style['bins_x']), PLOT.style)",
                "flow_mode = PLOT.style.get('hist_flow_mode', 'ignore') if flow_range is not None else 'ignore'",
                "flow_summary = (summarise_histogram_flow(histogram_1d, x, weighted=weights is not None)",
                "                if flow_range is not None else None)",
                "display_histogram = histogram_for_flow_mode(histogram_1d, flow_mode)",
            ]
        )

    if plan.has_fit:
        lines.extend(
            [
                "fit_result = fit_histogram_1d(display_histogram, PLOT.fit_config)",
                "if not fit_result.success:",
                "    print(f'ROOT Explorer fit warning: {fit_result.message}')",
            ]
        )

    if spec.plot_type == "2D histogram":
        lines.extend(
            [
                "histogram_2d = build_histogram_2d(",
                "    x, y, int(PLOT.style['bins_x']), int(PLOT.style['bins_y']),",
                "    weights=weights, x_label=x_label, y_label=y_label,",
                ")",
            ]
        )

    lines.extend(["figure, axes = create_static_figure(PLOT.style)", "warnings = []"])
    if spec.plot_type == "1D histogram":
        lines.extend(
            [
                "draw_histogram_1d(",
                "    axes, display_histogram, y_label, reference_x=reference_x,",
                "    reference_label=reference_label, flow_summary=flow_summary, flow_mode=flow_mode,",
                "    log_x=bool(PLOT.style.get('log_x', False)),",
                ")",
                "histogram_counts = display_histogram.values()",
                "histogram_edges = histogram_1d.axes[0].edges",
            ]
        )
        if plan.has_fit:
            lines.append(
                "if fit_result.success: draw_fit_overlay(axes, display_histogram, fit_result)"
            )
        if plan.show_kde:
            lines.extend(
                [
                    "inside = (x >= histogram_edges[0]) & (x <= histogram_edges[-1]) if flow_range is not None else slice(None)",
                    "warning = draw_kde_1d_overlay(axes, x[inside], histogram_edges, PLOT.style,",
                    "                              weights=weights[inside] if weights is not None else None)",
                    "if warning is not None:",
                    "    warnings.append(warning)",
                ]
            )
        if plan.log_x:
            lines.append("apply_log_x_axis(axes, histogram_edges)")
        if plan.log_y:
            lines.append("apply_log_histogram_y_axis(axes, histogram_counts)")
    elif spec.plot_type == "2D histogram":
        lines.extend(
            [
                "draw_histogram_2d(",
                "    figure, axes, histogram_2d,",
                "    log_scale=PLOT.style['log_z'],",
                "    colorbar_label=('entries' if weights is None else f'sum weights ({PLOT.weight_branch})'),",
                ")",
                "xedges = histogram_2d.axes[0].edges",
                "yedges = histogram_2d.axes[1].edges",
            ]
        )
        if plan.show_contours:
            lines.extend(
                [
                    "warning = draw_histogram_contours(axes, histogram_2d, PLOT.style)",
                    "if warning is not None:",
                    "    warnings.append(warning)",
                ]
            )
        if plan.show_kde:
            lines.extend(
                [
                    "warning = draw_kde_2d_overlay(axes, x, y, xedges, yedges, PLOT.style, weights=weights)",
                    "if warning is not None:",
                    "    warnings.append(warning)",
                ]
            )
        if plan.log_x:
            lines.append("apply_log_x_axis(axes, x)")
        if plan.log_y:
            lines.append("apply_log_y_axis(axes, y)")
    elif plan.has_color:
        renderer = (
            "draw_colored_scatter_2d_log" if plan.log_color else "draw_colored_scatter_2d_linear"
        )
        lines.extend(
            [
                f"{renderer}(",
                "    figure, axes, x, y, color, x_label, y_label, color_label, PLOT.style,",
                "    colorbar_extend=colorbar_extend,",
                ")",
            ]
        )
        if plan.log_x:
            lines.append("apply_log_x_axis(axes, x)")
        if plan.log_y:
            lines.append("apply_log_y_axis(axes, y)")
    else:
        lines.append("draw_scatter_2d(axes, x, y, x_label, y_label, PLOT.style)")
        if plan.log_x:
            lines.append("apply_log_x_axis(axes, x)")
        if plan.log_y:
            lines.append("apply_log_y_axis(axes, y)")

    lines.extend(
        [
            "finish_static_axes(axes, PLOT.plot_type, title, PLOT.style)",
            "for warning in warnings:",
            "    print(f'ROOT Explorer warning: {warning}')",
            "plt.show()",
        ]
    )
    return lines
