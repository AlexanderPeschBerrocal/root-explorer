"""Integration coverage for persisted ROOT graph objects read through Uproot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from root_explorer.io.graphs import UnsupportedGraphError, read_graph

GRAPH_FIXTURE = Path(__file__).parent / "data" / "test_objects.root"


def test_reads_plain_graph_coordinates_order_and_labels() -> None:
    graph = read_graph(str(GRAPH_FIXTURE), "plain_graph;1")

    np.testing.assert_array_equal(graph.x, [2.0, 1.0, 3.0])
    np.testing.assert_array_equal(graph.y, [4.0, 5.0, 6.0])
    assert (graph.title, graph.x_label, graph.y_label) == (
        "Plain graph",
        "Time [ns]",
        "Amplitude",
    )
    assert graph.xerr_low is graph.xerr_high is None
    assert graph.yerr_low is graph.yerr_high is None


def test_reads_symmetric_graph_uncertainties_and_labels() -> None:
    graph = read_graph(str(GRAPH_FIXTURE), "symmetric_graph;1")

    np.testing.assert_array_equal(graph.xerr_low, [0.05, 0.10, 0.15])
    np.testing.assert_array_equal(graph.xerr_high, [0.05, 0.10, 0.15])
    np.testing.assert_array_equal(graph.yerr_low, [0.20, 0.25, 0.30])
    np.testing.assert_array_equal(graph.yerr_high, [0.20, 0.25, 0.30])
    assert (graph.title, graph.x_label, graph.y_label) == (
        "Symmetric graph",
        "Momentum [GeV]",
        "Efficiency",
    )


def test_reads_asymmetric_graph_uncertainties_and_labels() -> None:
    graph = read_graph(str(GRAPH_FIXTURE), "asymmetric_graph;1")

    np.testing.assert_array_equal(graph.xerr_low, [0.1, 0.1, 0.1])
    np.testing.assert_array_equal(graph.xerr_high, [0.2, 0.2, 0.2])
    np.testing.assert_array_equal(graph.yerr_low, [0.15, 0.15, 0.15])
    np.testing.assert_array_equal(graph.yerr_high, [0.3, 0.3, 0.3])
    assert (graph.title, graph.x_label, graph.y_label) == (
        "Asymmetric graph",
        "Position [cm]",
        "Rate [Hz]",
    )


def test_reads_nested_graph_and_explicit_root_cycles() -> None:
    nested = read_graph(str(GRAPH_FIXTURE), "nested/nested_graph;1")
    first_cycle = read_graph(str(GRAPH_FIXTURE), "cycled_graph;1")
    second_cycle = read_graph(str(GRAPH_FIXTURE), "cycled_graph;2")

    assert nested.title == "Nested graph"
    assert first_cycle.title == "First cycle"
    assert second_cycle.title == "Second cycle"
    np.testing.assert_array_equal(first_cycle.x, [2.0, 1.0, 3.0])
    np.testing.assert_array_equal(second_cycle.x, [12.0, 11.0, 13.0])


def test_rejects_non_graph_objects_gracefully() -> None:
    with pytest.raises(UnsupportedGraphError, match="TH1D"):
        read_graph(str(GRAPH_FIXTURE), "histogram_1d;1")
