"""Graph renderer tests independent of ROOT test fixtures."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from root_explorer.models import GraphData
from root_explorer.plotting.graphs import clipped_error_bar_warnings, render_graph_data
from root_explorer.plotting.plots import InvalidAxisLimitError, InvalidAxisScaleError

matplotlib.use("Agg")


def style(**overrides):
    data = {"fig_width": 8, "fig_height": 6, "graph_mode": "Line + markers"}
    data.update(overrides)
    return data


def test_graph_retains_original_point_order():
    graph = GraphData(x=np.array([2.0, 1.0, 3.0]), y=np.array([4.0, 5.0, 6.0]))
    fig = render_graph_data(graph, style(), {})
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), graph.x)


def test_graph_titles_and_overrides():
    graph = GraphData(
        x=np.array([1.0]), y=np.array([2.0]), title="Original", x_label="pT", y_label="Efficiency"
    )
    fig = render_graph_data(graph, style(), {"custom_x": "Momentum"})
    assert fig.axes[0].get_title() == "Original"
    assert fig.axes[0].get_xlabel() == "Momentum"
    assert fig.axes[0].get_ylabel() == "Efficiency"


def test_errors_are_used_when_visible():
    graph = GraphData(
        x=np.array([1.0, 2.0]),
        y=np.array([3.0, 4.0]),
        yerr_low=np.array([0.1, 0.2]),
        yerr_high=np.array([0.4, 0.5]),
    )
    fig = render_graph_data(graph, style(show_errors=True), {})
    assert fig.axes[0].collections


@pytest.mark.parametrize("axis", ["x", "y"])
def test_log_axis_clips_lower_error_bars_reaching_zero(axis):
    errors = {
        f"{axis}err_low": np.array([0.3, 0.1, 0.3]),
        f"{axis}err_high": np.array([0.1, 0.1, 0.1]),
    }
    graph = GraphData(x=np.array([0.3, 0.5, 0.2]), y=np.array([0.3, 0.5, 0.2]), **errors)
    expected = (
        f"2 lower {axis} error bar(s) reach zero or below and are cut off "
        f"at the bottom of the logarithmic {axis}-axis."
    )
    assert clipped_error_bar_warnings(graph, style(**{f"log_{axis}": True})) == [expected]
    assert (
        clipped_error_bar_warnings(graph, style(**{f"log_{axis}": True}, show_errors=False)) == []
    )
    assert clipped_error_bar_warnings(graph, style()) == []
    with pytest.warns(UserWarning, match="2 lower"):
        fig = render_graph_data(graph, style(**{f"log_{axis}": True}), {})
    try:
        ax = fig.axes[0]
        fig.canvas.draw()
        assert (ax.get_xscale() if axis == "x" else ax.get_yscale()) == "log"
        # The clipped bars do not pull the axis towards zero.
        lower = (ax.get_xlim() if axis == "x" else ax.get_ylim())[0]
        assert 0.05 < lower < 0.2
    finally:
        plt.close(fig)


def test_log_axis_still_rejects_nonpositive_points_with_errors():
    graph = GraphData(
        x=np.array([1.0, 2.0]),
        y=np.array([0.0, 0.5]),
        yerr_low=np.array([0.1, 0.1]),
        yerr_high=np.array([0.1, 0.1]),
    )
    assert clipped_error_bar_warnings(graph, style(log_y=True)) == []
    with pytest.raises(InvalidAxisScaleError):
        render_graph_data(graph, style(log_y=True), {})


@pytest.mark.parametrize(
    ("graph", "invalid_style", "expected_error"),
    [
        (
            GraphData(
                x=np.array([1.0]),
                y=np.array([-0.2]),
                yerr_low=np.array([0.3]),
                yerr_high=np.array([0.1]),
            ),
            {"log_y": True},
            InvalidAxisScaleError,
        ),
        (
            GraphData(x=np.array([-1.0, 2.0]), y=np.array([1.0, 2.0])),
            {"log_x": True},
            InvalidAxisScaleError,
        ),
        (
            GraphData(x=np.array([1.0, 2.0]), y=np.array([1.0, 2.0])),
            {"x_min_text": "3", "x_max_text": "1"},
            InvalidAxisLimitError,
        ),
    ],
)
def test_failed_graph_rendering_closes_matplotlib_figure(
    graph: GraphData, invalid_style: dict[str, object], expected_error: type[Exception]
) -> None:
    """Early and post-creation validation failures must leave no open figure."""
    existing_figures = set(plt.get_fignums())
    with pytest.raises(expected_error):
        render_graph_data(graph, style(**invalid_style), {})
    assert set(plt.get_fignums()) == existing_figures


def test_nonfinite_coordinates_are_omitted_with_a_warning() -> None:
    graph = GraphData(
        x=np.array([2.0, np.nan, 1.0, np.inf]),
        y=np.array([4.0, 5.0, 6.0, 7.0]),
    )

    with pytest.warns(UserWarning, match="Omitted 2 graph point"):
        fig = render_graph_data(graph, style(), {})

    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), [2.0, 1.0])
    np.testing.assert_array_equal(graph.x, [2.0, np.nan, 1.0, np.inf])
