"""Standalone graph exports use the same renderer as ROOT Explorer."""

from __future__ import annotations

import ast

from root_explorer.plotting.graph_codegen import generate_graph_plot_code
from root_explorer.plotting.graphs import GraphPlotSpec


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
