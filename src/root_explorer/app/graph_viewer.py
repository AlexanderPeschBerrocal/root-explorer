"""Graph inspection controls for the ROOT catalogue master-detail view."""

from __future__ import annotations

from typing import Any

from pathlib import Path

import streamlit as st

from ..io.graphs import UnsupportedGraphError, read_graph
from ..plotting.graphs import GraphPlotSpec, render_graph_data
from ..plotting.plots import InvalidAxisLimitError, InvalidAxisScaleError
from .results import show_cached_plot_result


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_graph(root_path: str, object_path: str, file_version: int | None):
    """Cache selected payloads while invalidating local files after changes."""
    return read_graph(root_path, object_path)


def _file_version(root_path: str) -> int | None:
    path = Path(root_path)
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _graph_controls(graph, object_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    st.subheader("Graph configuration", anchor=False)
    style: dict[str, Any] = {}
    style["graph_mode"] = st.selectbox(
        "Drawing mode", ("Markers", "Line", "Line + markers"), key="graph_drawing_mode"
    )
    if graph.xerr_low is not None or graph.yerr_low is not None:
        style["show_errors"] = st.checkbox(
            "Show error bars", value=True, key="graph_show_errors"
        )
    style["marker_size"] = st.slider(
        "Marker size", 0.5, 20.0, 5.0, key="graph_marker_size"
    )
    style["line_width"] = st.slider(
        "Line width", 0.5, 10.0, 1.5, key="graph_line_width"
    )
    style["fig_width"] = st.slider(
        "Figure width", 4.0, 16.0, 8.0, key="graph_fig_width"
    )
    style["fig_height"] = st.slider(
        "Figure height", 3.0, 12.0, 6.0, key="graph_fig_height"
    )
    style["display_dpi"] = 200
    style["log_x"] = st.checkbox("Log x-axis", key="graph_log_x")
    style["log_y"] = st.checkbox("Log y-axis", key="graph_log_y")
    style["grid"] = st.checkbox("Grid", value=True, key="graph_grid")
    manual = st.checkbox("Apply manual axis limits", key="graph_manual_limits")
    for key, title in (
        ("x_min_text", "x minimum"), ("x_max_text", "x maximum"),
        ("y_min_text", "y minimum"), ("y_max_text", "y maximum"),
    ):
        style[key] = st.text_input(title, key=f"graph_{key}") if manual else ""
    st.markdown("#### Labels", anchors=False)
    labels = {
        "title": st.text_input("Title", value=graph.title, key=f"graph_title_{object_id}"),
        "custom_x": st.text_input("X label", value=graph.x_label, key=f"graph_x_label_{object_id}"),
        "custom_y": st.text_input("Y label", value=graph.y_label, key=f"graph_y_label_{object_id}"),
    }
    return style, labels


def render_graph_controls(
    root_path: str,
    object_path: str,
    classname: str,
    *,
    shortcuts_enabled: bool = True,
) -> None:
    """Render a graph after deliberate opening, retaining its plot across reruns."""
    st.subheader(f"Graph: {object_path.rsplit('/', 1)[-1]}", anchor=False)
    st.caption(f"`{classname}` · `{object_path}`")
    try:
        graph = _cached_graph(root_path, object_path, _file_version(root_path))
    except (UnsupportedGraphError, ValueError, OSError) as exc:
        st.warning(str(exc))
        return
    st.caption(f"Points: {len(graph.x):,}")
    style, labels = _graph_controls(graph, str(abs(hash((root_path, object_path)))))
    # A deliberate Open action triggers the initial plot. A mere row selection does not.
    make_plot = st.button(
        "Make plot", key="graph_make_plot", type="primary",
        shortcut="P" if shortcuts_enabled else None,
    )
    if "root_graph_result" not in st.session_state or make_plot:
        try:
            spec = GraphPlotSpec(root_path, object_path, style, labels)
            figure = render_graph_data(graph, spec.style, spec.labels)
            from ..plotting.graph_codegen import generate_graph_plot_code
            st.session_state["root_graph_result"] = {
                "kind": "matplotlib",
                "figure": figure,
                "code": generate_graph_plot_code(spec),
                "save_name": "root_graph",
                "warnings": [],
            }
        except (InvalidAxisLimitError, InvalidAxisScaleError, ValueError) as exc:
            st.session_state.pop("root_graph_result", None)
            st.warning(str(exc))
    # The output is rendered separately in the right-hand master-detail pane.


def render_graph_output() -> None:
    show_cached_plot_result(
        "root_graph_result", export_prefix="graph", save_button_key="graph_save",
        make_plot_label="Make plot", make_plot_shortcut="P",
        empty_message="Configure the graph and click Make plot.",
    )
