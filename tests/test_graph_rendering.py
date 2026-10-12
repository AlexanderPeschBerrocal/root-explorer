"""Graph renderer tests independent of ROOT test fixtures."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from root_explorer.models import GraphData
from root_explorer.plotting.graphs import render_graph_data
from root_explorer.plotting.plots import InvalidAxisScaleError

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


def test_log_axis_rejects_crossing_uncertainty():
    graph = GraphData(
        x=np.array([1.0]),
        y=np.array([0.2]),
        yerr_low=np.array([0.3]),
        yerr_high=np.array([0.1]),
    )
    with pytest.raises(InvalidAxisScaleError, match="uncertainties cross zero"):
        render_graph_data(graph, style(log_y=True), {})


def test_nonfinite_coordinates_are_omitted_with_a_warning() -> None:
    graph = GraphData(
        x=np.array([2.0, np.nan, 1.0, np.inf]),
        y=np.array([4.0, 5.0, 6.0, 7.0]),
    )

    with pytest.warns(UserWarning, match="Omitted 2 graph point"):
        fig = render_graph_data(graph, style(), {})

    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), [2.0, 1.0])
    np.testing.assert_array_equal(graph.x, [2.0, np.nan, 1.0, np.inf])
