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
    histogram_errors,
    render_histogram_data,
)

matplotlib.use("Agg")
FIXTURE = Path(__file__).parent / "data" / "test_objects.root"


def _style(**changes):
    return {"fig_width": 8.0, "fig_height": 5.0, **changes}


def test_reads_root_histogram_1d_with_sumw2():
    data = read_histogram(str(FIXTURE), "histogram_1d;1")
    assert data.kind == "COUNT"
    assert data.histogram.ndim == 1
    np.testing.assert_allclose(data.histogram.values(), [1.0, 2.0, 0.0])
    # The regenerated fixture explicitly enables Sumw2 before filling:
    # the variance of a bin filled once with weight 2 is 2**2, not 2.
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


@pytest.mark.parametrize(
    ("object_path", "classname", "expected_means", "expected_errors"),
    [
        (
            "profile_1d;1",
            "TProfile",
            [3.0, 3.0, 6.0],
            [np.sqrt(0.5), np.sqrt(8.0 / 9.0), np.sqrt(0.5)],
        ),
        (
            "nested/profile_2d;1",
            "TProfile2D",
            [[3.0, 2.0], [6.0, 10.0]],
            [[np.sqrt(0.5), np.sqrt(0.5)], [np.sqrt(0.5), np.sqrt(0.5)]],
        ),
    ],
)
def test_committed_root_profiles_preserve_means_and_errors(
    object_path, classname, expected_means, expected_errors
):
    """Validate the actual PyROOT-produced fixture, not only Uproot-created files."""
    with uproot.open(FIXTURE) as root_file:
        assert object_path in root_file, (
            "The committed test_objects.root has not been regenerated with profiles. "
            "Run python tests/data/generate_graph_fixture.py and commit the result."
        )
        source = root_file[object_path]
        assert source.classname == classname
        assert source.kind == "MEAN"
        np.testing.assert_allclose(source.values(), expected_means)
        np.testing.assert_allclose(source.errors(), expected_errors)

    data = read_histogram(str(FIXTURE), object_path)
    assert data.is_profile
    assert data.histogram.ndim == (2 if classname == "TProfile2D" else 1)
    np.testing.assert_allclose(data.histogram.values(), expected_means)
    np.testing.assert_allclose(histogram_errors(data), expected_errors)
    assert data.x_label == "#eta"

    if classname == "TProfile2D":
        assert data.y_label == "p_{T} [GeV]"
        assert data.colorbar_label == "Mean response"
        np.testing.assert_allclose(data.histogram.variances(), np.square(expected_errors))
    else:
        assert data.y_label == "Mean E [GeV]"
        np.testing.assert_allclose(data.histogram.values(flow=True), [11.0, 3.0, 3.0, 6.0, 12.0])


def test_log_scale_fails_without_positive_bins(tmp_path):
    path = tmp_path / "empty.root"
    with uproot.recreate(path) as file:
        file["empty"] = np.histogram(np.array([], dtype=float), bins=3, range=(1, 4))
    data = read_histogram(str(path), "empty")
    initial_figures = set(plt.get_fignums())
    with pytest.raises(ValueError, match="positive"):
        render_histogram_data(data, _style(log_y=True), {})
    assert set(plt.get_fignums()) == initial_figures


def test_persisted_root_histogram_fit_export_executes():
    import sys
    from types import ModuleType

    from root_explorer.analysis.fitting import FitConfig, fit_histogram_1d

    spec = HistogramPlotSpec(
        str(FIXTURE),
        "histogram_1d;1",
        _style(),
        {},
        fit_config=FitConfig("Exponential"),
    )
    source = generate_histogram_plot_code(spec)
    assert "import root_explorer" not in source
    assert "import streamlit" not in source
    assert "def fit_histogram_1d(" in source
    module = ModuleType("persisted_histogram_standalone")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, "persisted_plot.py", "exec"), module.__dict__)
        module.plt.show = lambda: None
        module.main()
        figure = module.plt.gcf()
        data = read_histogram(str(FIXTURE), "histogram_1d;1")
        result = fit_histogram_1d(data.histogram, spec.fit_config)
        assert (
            result.success
            == module.fit_histogram_1d(data.histogram, module.PLOT.fit_config).success
        )
        assert figure.axes[0].get_title()
        plt.close(figure)
    finally:
        sys.modules.pop(module.__name__, None)


def test_persisted_custom_fit_export_executes():
    import sys
    from types import ModuleType

    from root_explorer.analysis.fitting import FitConfig

    spec = HistogramPlotSpec(
        str(FIXTURE),
        "histogram_1d;1",
        _style(),
        {},
        fit_config=FitConfig(
            "Custom expression",
            expression="a*x + b",
            initial_values={"a": 1, "b": 1},
        ),
    )
    source = generate_histogram_plot_code(spec)
    assert "from root_explorer" not in source
    assert "import streamlit" not in source
    module = ModuleType("persisted_custom_fit_standalone")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, "persisted_custom.py", "exec"), module.__dict__)
        module.plt.show = lambda: None
        module.main()
        result = module.fit_histogram_1d(
            read_histogram(str(FIXTURE), "histogram_1d;1").histogram,
            module.PLOT.fit_config,
        )
        assert result.model == "Custom expression"
        assert result.expression == "a*x + b"
    finally:
        plt.close("all")
        sys.modules.pop(module.__name__, None)
