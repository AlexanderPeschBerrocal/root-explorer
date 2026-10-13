"""Read persisted ROOT graphs into a small, ROOT-independent numerical model."""

from __future__ import annotations

from typing import Any

import numpy as np
import uproot

from ..models import GraphData

SUPPORTED_GRAPH_CLASSES = frozenset({"TGraph", "TGraphErrors", "TGraphAsymmErrors"})


class UnsupportedGraphError(ValueError):
    """A graph subclass is visible in the catalogue but cannot be displayed."""


class InvalidGraphDataError(ValueError):
    """The selected graph contains invalid or inconsistent stored data."""


def _graph_member_array(graph: Any, member: str, n_points: int) -> np.ndarray:
    """Extract exactly the used entries; ROOT often allocates extra capacity."""
    values = np.asarray(graph.member(member), dtype=np.float64)
    if values.ndim != 1 or len(values) < n_points:
        raise InvalidGraphDataError(
            f"Graph member {member} must contain at least {n_points} one-dimensional values."
        )
    return values[:n_points].copy()


def _graph_title_fields(graph: Any) -> tuple[str, str, str]:
    """Decode ROOT's conventional 'title;x axis;y axis' convention."""
    try:
        raw = str(graph.member("fTitle") or "")
    except (KeyError, AttributeError):
        raw = ""
    parts = raw.split(";")
    title = parts[0].strip()
    x_label = parts[1].strip() if len(parts) > 1 else ""
    y_label = parts[2].strip() if len(parts) > 2 else ""

    # ROOT's TGraph::SetTitle("title;x;y") splits the string: TGraph's
    # fTitle retains just the main title while the axis titles are persisted
    # in the accompanying TH1F (fHistogram). Older files may instead retain
    # the full semicolon-delimited fTitle, so support both representations.
    if not x_label or not y_label:
        try:
            histogram = graph.member("fHistogram", none_if_missing=True)
        except (AttributeError, KeyError):
            histogram = None
        if histogram is not None:
            for axis_member, current in (("fXaxis", x_label), ("fYaxis", y_label)):
                if current:
                    continue
                try:
                    axis = histogram.member(axis_member)
                    axis_label = str(axis.member("fTitle") or "").strip()
                except (AttributeError, KeyError):
                    axis_label = ""
                if axis_member == "fXaxis":
                    x_label = axis_label
                else:
                    y_label = axis_label
    return title, x_label, y_label


def graph_data_from_object(graph: Any) -> GraphData:
    """Convert exactly the three supported persisted TGraph types to NumPy."""
    classname = str(graph.classname)
    if classname not in SUPPORTED_GRAPH_CLASSES:
        raise UnsupportedGraphError(
            f"ROOT class {classname} is not supported for graph plotting. "
            "Supported types: TGraph, TGraphErrors, TGraphAsymmErrors."
        )
    n_points = int(graph.member("fNpoints"))
    if n_points < 0:
        raise InvalidGraphDataError("Graph has a negative number of points.")

    x = _graph_member_array(graph, "fX", n_points)
    y = _graph_member_array(graph, "fY", n_points)
    errors: dict[str, np.ndarray] = {}
    if classname == "TGraphErrors":
        errors = {
            "xerr_low": _graph_member_array(graph, "fEX", n_points),
            "xerr_high": _graph_member_array(graph, "fEX", n_points),
            "yerr_low": _graph_member_array(graph, "fEY", n_points),
            "yerr_high": _graph_member_array(graph, "fEY", n_points),
        }
    elif classname == "TGraphAsymmErrors":
        errors = {
            "xerr_low": _graph_member_array(graph, "fEXlow", n_points),
            "xerr_high": _graph_member_array(graph, "fEXhigh", n_points),
            "yerr_low": _graph_member_array(graph, "fEYlow", n_points),
            "yerr_high": _graph_member_array(graph, "fEYhigh", n_points),
        }

    for name, values in errors.items():
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise InvalidGraphDataError(f"Graph member {name} contains invalid uncertainties.")

    title, x_label, y_label = _graph_title_fields(graph)
    return GraphData(x=x, y=y, title=title, x_label=x_label, y_label=y_label, **errors)


def read_graph(root_path: str, object_path: str) -> GraphData:
    """Load only the selected graph; object_path may include directories and cycle."""
    with uproot.open(root_path) as root_file:
        return graph_data_from_object(root_file[object_path])
