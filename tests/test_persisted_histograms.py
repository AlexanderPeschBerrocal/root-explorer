"""Persisted ROOT histogram integration and renderer tests."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import uproot

from root_explorer.io.histograms import (
    UnsupportedHistogramError,
    read_histogram,
)
from root_explorer.plotting.histogram_codegen import generate_histogram_plot_code
from root_explorer.plotting.histograms import (
    HistogramPlotSpec,
    render_histogram_data,
)

matplotlib.use("Agg")
FIXTURE = Path(__file__).parent / "data" / "test_objects.root"


def _style(**changes):
    return {"fig_width": 8.0, "fig_height": 5.0, **changes}


def test_reads_root_histogram_1d_with_weights():
    data = read_histogram(str(FIXTURE), "histogram_1d;1")
    assert data.kind == "COUNT"
    assert data.histogram.ndim == 1
    np.testing.assert_allclose(data.histogram.values(), [1.0, 2.0, 0.0])
    np.testing.assert_allclose(data.histogram.variances(), [1.0, 4.0, 0.0])


def test_reads_root_histogram_2d():
    data = read_histogram(str(FIXTURE), "histogram_2d;1")
    assert data.histogram.ndim == 2
    assert np.sum(data.histogram.values()) == 1


def test_nested_variable_width_histogram(tmp_path):
    path = tmp_path / "nested.root"
    with uproot.recreate(path) as file:
        file["nested/variable"] = (
            np.array([2.0, 3.0]),
            np.array([0.0, 1.0, 4.0]),
        )
    data = read_histogram(str(path), "nested/variable;1")
    np.testing.assert_array_equal(data.histogram.axes[0].edges, [0, 1, 4])
    np.testing.assert_array_equal(data.histogram.values(), [2, 3])


def test_unsupported_root_class():
    with pytest.raises(UnsupportedHistogramError):
        read_histogram(str(FIXTURE), "plain_graph;1")


@pytest.mark.parametrize("name", ["histogram_1d;1", "histogram_2d;1"])
def test_renderer_and_standalone_export(name):
    data = read_histogram(str(FIXTURE), name)
    style = _style()
    labels = {"title": "Persisted ROOT object"}
    figure = render_histogram_data(data, style, labels)
    try:
        assert figure.axes[0].get_title() == labels["title"]
        source = generate_histogram_plot_code(HistogramPlotSpec(str(FIXTURE), name, style, labels))
        ast.parse(source)
        assert "import streamlit" not in source
        assert "import root_explorer" not in source
        assert "histogram_weights_for_values" not in source
        assert "read_histogram" in source
    finally:
        plt.close(figure)


def test_log_scale_fails_without_positive_bins(tmp_path):
    path = tmp_path / "empty.root"
    with uproot.recreate(path) as file:
        file["empty"] = np.histogram(np.array([], dtype=float), bins=3, range=(1, 4))
    data = read_histogram(str(path), "empty")
    initial_figures = set(plt.get_fignums())
    with pytest.raises(ValueError, match="positive"):
        render_histogram_data(data, _style(log_y=True), {})
    assert set(plt.get_fignums()) == initial_figures
