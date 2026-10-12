"""Standalone graph exports use the same renderer as ROOT Explorer."""

from __future__ import annotations

import ast
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pytest
import uproot

from root_explorer.plotting.graph_codegen import generate_graph_plot_code
from root_explorer.plotting.graphs import GraphPlotSpec, render_graph_plot

GRAPH_FIXTURE = Path(__file__).parent / "data" / "test_objects.root"


@contextmanager
def _executed_generated_module(source: str) -> Generator[ModuleType, None, None]:
    """Execute an export in a registered module without invoking ``main`` eagerly."""
    module_name = "standalone_root_explorer_graph_export"
    module = ModuleType(module_name)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(source, "root_explorer_graph_export.py", "exec"), module.__dict__)
        yield module
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


def _execute_export(spec: GraphPlotSpec):
    source = generate_graph_plot_code(spec)
    assert "from root_explorer" not in source
    assert "import root_explorer" not in source
    assert "import streamlit" not in source

    canonical_figure = render_graph_plot(spec)
    with _executed_generated_module(source) as module:
        module.plt.show = lambda: None
        module.main()
        generated_figure = module.plt.gcf()
        assert generated_figure.axes
    return canonical_figure, generated_figure


def _line_data(figure) -> tuple[np.ndarray, np.ndarray]:
    line = figure.axes[0].lines[0]
    return np.asarray(line.get_xdata()), np.asarray(line.get_ydata())


def test_graph_export_is_standalone():
    spec = GraphPlotSpec(
        "file.root",
        "Efficiency/graph;2",
        {"fig_width": 8, "fig_height": 6, "graph_mode": "Markers"},
        {"title": "Efficiency"},
    )
    code = generate_graph_plot_code(spec)
    ast.parse(code)
    assert "import uproot" in code
    assert "import streamlit" not in code
    assert "import root_explorer" not in code
    assert "histogram_weights_for_values" not in code
    assert "PLOT = GraphPlotSpec(" in code
    assert "Efficiency/graph;2" in code


@pytest.mark.parametrize(
    ("object_path", "style", "labels", "expected_scale"),
    [
        (
            "plain_graph;1",
            {
                "fig_width": 7.0,
                "fig_height": 4.0,
                "graph_mode": "Markers",
                "marker_size": 7.0,
                "line_width": 2.0,
                "grid": True,
            },
            {"title": "Custom plain", "custom_x": "Elapsed time", "custom_y": "Signal"},
            ("linear", "linear"),
        ),
        (
            "symmetric_graph;1",
            {
                "fig_width": 8.0,
                "fig_height": 5.0,
                "graph_mode": "Line",
                "show_errors": True,
                "marker_size": 5.0,
                "line_width": 3.0,
                "log_x": True,
                "grid": False,
            },
            {},
            ("log", "linear"),
        ),
        (
            "asymmetric_graph;1",
            {
                "fig_width": 9.0,
                "fig_height": 5.0,
                "graph_mode": "Line + markers",
                "show_errors": True,
                "marker_size": 6.0,
                "line_width": 2.5,
                "log_y": True,
                "grid": True,
            },
            {"custom_y": "Custom rate"},
            ("linear", "log"),
        ),
    ],
)
def test_executed_graph_export_matches_canonical_renderer(
    object_path: str,
    style: dict[str, object],
    labels: dict[str, str],
    expected_scale: tuple[str, str],
) -> None:
    spec = GraphPlotSpec(str(GRAPH_FIXTURE), object_path, style, labels)
    canonical_figure, generated_figure = _execute_export(spec)
    try:
        canonical_axis = canonical_figure.axes[0]
        generated_axis = generated_figure.axes[0]
        generated_x, generated_y = _line_data(generated_figure)
        canonical_x, canonical_y = _line_data(canonical_figure)
        np.testing.assert_allclose(generated_x, canonical_x)
        np.testing.assert_allclose(generated_y, canonical_y)
        assert generated_axis.get_title() == canonical_axis.get_title()
        assert generated_axis.get_xlabel() == canonical_axis.get_xlabel()
        assert generated_axis.get_ylabel() == canonical_axis.get_ylabel()
        assert (generated_axis.get_xscale(), generated_axis.get_yscale()) == expected_scale
        assert generated_axis.lines[0].get_marker() == canonical_axis.lines[0].get_marker()
        assert generated_axis.lines[0].get_linestyle() == canonical_axis.lines[0].get_linestyle()
        assert generated_axis.lines[0].get_markersize() == canonical_axis.lines[0].get_markersize()
        assert generated_axis.lines[0].get_linewidth() == canonical_axis.lines[0].get_linewidth()
        assert len(generated_axis.collections) == len(canonical_axis.collections)
    finally:
        plt.close(canonical_figure)
        plt.close(generated_figure)


def test_executed_graph_export_warns_and_omits_nonfinite_coordinates(tmp_path) -> None:
    path = tmp_path / "nonfinite.root"
    with uproot.recreate(path) as root_file:
        root_file["graph"] = uproot.as_TGraph({"x": [1.0, np.nan, 3.0], "y": [2.0, 4.0, 6.0]})
    spec = GraphPlotSpec(
        str(path),
        "graph;1",
        {"fig_width": 8.0, "fig_height": 6.0, "graph_mode": "Line"},
        {},
    )

    with pytest.warns(UserWarning, match="Omitted 1 graph point"):
        canonical_figure, generated_figure = _execute_export(spec)
    try:
        np.testing.assert_array_equal(_line_data(generated_figure)[0], [1.0, 3.0])
        np.testing.assert_array_equal(_line_data(canonical_figure)[0], [1.0, 3.0])
    finally:
        plt.close(canonical_figure)
        plt.close(generated_figure)
