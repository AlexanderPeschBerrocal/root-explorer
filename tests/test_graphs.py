"""Tests for the ROOT-independent graph adapter and its supported ROOT members."""

from __future__ import annotations

import numpy as np
import pytest

from root_explorer.io.graphs import (
    InvalidGraphDataError,
    UnsupportedGraphError,
    graph_data_from_object,
)


class FakeGraph:
    def __init__(self, classname, **members):
        self.classname = classname
        self.members = members

    def member(self, key):
        return self.members[key]


def make_graph(kind, **changes):
    data = {
        "fNpoints": 3,
        "fX": [2.0, 1.0, 3.0, 999.0],
        "fY": [4.0, 5.0, 6.0, 999.0],
        "fTitle": "Efficiency;Photon pT;Efficiency",
        "fEX": [0.1, 0.2, 0.3],
        "fEY": [0.4, 0.5, 0.6],
        "fEXlow": [0.1, 0.2, 0.3],
        "fEXhigh": [0.4, 0.5, 0.6],
        "fEYlow": [0.7, 0.8, 0.9],
        "fEYhigh": [1.0, 1.1, 1.2],
    }
    data.update(changes)
    return FakeGraph(kind, **data)


@pytest.mark.parametrize("kind", ["TGraph", "TGraphErrors", "TGraphAsymmErrors"])
def test_coordinates_and_title_are_preserved(kind):
    graph = graph_data_from_object(make_graph(kind))
    np.testing.assert_array_equal(graph.x, [2, 1, 3])
    np.testing.assert_array_equal(graph.y, [4, 5, 6])
    assert (graph.title, graph.x_label, graph.y_label) == ("Efficiency", "Photon pT", "Efficiency")


def test_plain_graph_has_no_errors():
    graph = graph_data_from_object(make_graph("TGraph"))
    assert graph.xerr_low is graph.xerr_high is None
    assert graph.yerr_low is graph.yerr_high is None


def test_symmetric_errors_remain_symmetric():
    graph = graph_data_from_object(make_graph("TGraphErrors"))
    np.testing.assert_array_equal(graph.xerr_low, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(graph.xerr_high, graph.xerr_low)
    np.testing.assert_array_equal(graph.yerr_low, [0.4, 0.5, 0.6])
    np.testing.assert_array_equal(graph.yerr_high, graph.yerr_low)


def test_asymmetric_errors_keep_original_order():
    graph = graph_data_from_object(make_graph("TGraphAsymmErrors"))
    np.testing.assert_array_equal(graph.xerr_low, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(graph.xerr_high, [0.4, 0.5, 0.6])
    np.testing.assert_array_equal(graph.yerr_low, [0.7, 0.8, 0.9])
    np.testing.assert_array_equal(graph.yerr_high, [1.0, 1.1, 1.2])


def test_empty_graph_is_valid():
    graph = graph_data_from_object(make_graph("TGraph", fNpoints=0))
    assert graph.x.size == graph.y.size == 0


@pytest.mark.parametrize("cls", ["TGraph2D", "TGraphBentErrors", "TMultiGraph"])
def test_unsupported_classes_get_clear_message(cls):
    with pytest.raises(UnsupportedGraphError, match=cls):
        graph_data_from_object(make_graph(cls))


def test_inconsistent_point_count_is_rejected():
    with pytest.raises(InvalidGraphDataError, match="fX"):
        graph_data_from_object(make_graph("TGraph", fNpoints=5))


@pytest.mark.parametrize("error", [float("nan"), -1.0])
def test_invalid_uncertainties_are_rejected(error):
    with pytest.raises(InvalidGraphDataError, match="uncertainties"):
        graph_data_from_object(make_graph("TGraphErrors", fEX=[0.1, error, 0.3]))
