"""Streamlit-independent rendering of persisted ROOT graph data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..io.graphs import read_graph
from ..models import GraphData
from .plots import (
    InvalidAxisScaleError,
    apply_axis_limits,
    apply_log_axis_scales,
    create_static_figure,
)


@dataclass(frozen=True)
class GraphPlotSpec:
    root_path: str
    object_path: str
    style: dict[str, Any]
    labels: dict[str, str]


def _error_array(low: np.ndarray | None, high: np.ndarray | None) -> np.ndarray | None:
    if low is None or high is None:
        return None
    return np.stack((low, high))


def render_graph_data(
    graph: GraphData, style: dict[str, Any], labels: dict[str, str]
) -> Any:
    """Draw all stored points in order, without altering graph semantics."""
    if graph.x.shape != graph.y.shape or graph.x.ndim != 1:
        raise ValueError("Graph coordinates must be aligned one-dimensional arrays.")

    fig, ax = create_static_figure(style)
    xerr = _error_array(graph.xerr_low, graph.xerr_high)
    yerr = _error_array(graph.yerr_low, graph.yerr_high)
    show_errors = bool(style.get("show_errors", True))
    mode = str(style.get("graph_mode", "Markers"))
    if mode not in {"Markers", "Line", "Line + markers"}:
        raise ValueError(f"Unknown graph drawing mode: {mode}")
    marker = "o" if mode != "Line" else "None"
    linestyle = "-" if mode != "Markers" else "None"
    valid = np.isfinite(graph.x) & np.isfinite(graph.y)
    if not np.all(valid):
        # Do not silently drop values from GraphData: only exclude invalid points
        # from the Matplotlib call.
        graph_x = graph.x[valid]
        graph_y = graph.y[valid]
        xerr = xerr[:, valid] if xerr is not None else None
        yerr = yerr[:, valid] if yerr is not None else None
    else:
        graph_x, graph_y = graph.x, graph.y

    if bool(style.get("log_x", False)) and xerr is not None and show_errors:
        if np.any(graph_x - xerr[0] <= 0):
            raise InvalidAxisScaleError("X uncertainties cross zero on a logarithmic x-axis.")
    if bool(style.get("log_y", False)) and yerr is not None and show_errors:
        if np.any(graph_y - yerr[0] <= 0):
            raise InvalidAxisScaleError("Y uncertainties cross zero on a logarithmic y-axis.")

    if show_errors and (xerr is not None or yerr is not None):
        ax.errorbar(
            graph_x,
            graph_y,
            xerr=xerr,
            yerr=yerr,
            marker=marker,
            linestyle=linestyle,
            markersize=float(style.get("marker_size", 5)),
            linewidth=float(style.get("line_width", 1.5)),
            capsize=2,
        )
    else:
        ax.plot(
            graph_x,
            graph_y,
            marker=marker,
            linestyle=linestyle,
            markersize=float(style.get("marker_size", 5)),
            linewidth=float(style.get("line_width", 1.5)),
        )
    ax.set_title(labels.get("title") or graph.title or "ROOT graph")
    ax.set_xlabel(labels.get("custom_x") or graph.x_label or "x")
    ax.set_ylabel(labels.get("custom_y") or graph.y_label or "y")
    if len(graph_x) > 0:
        apply_log_axis_scales(ax, style, x=graph_x, y=graph_y)
    apply_axis_limits(ax, style)
    if style.get("grid", True):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def render_graph_plot(spec: GraphPlotSpec) -> Any:
    """Load and render the selected graph using the canonical graph renderer."""
    return render_graph_data(
        read_graph(spec.root_path, spec.object_path), spec.style, spec.labels
    )
