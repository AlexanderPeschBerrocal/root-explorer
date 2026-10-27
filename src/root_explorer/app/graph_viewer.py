"""Graph inspection controls for the ROOT catalogue master-detail view."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ..io.graphs import UnsupportedGraphError, read_graph
from ..io.root import RemoteBackendUnavailableError, explain_missing_remote_backend
from ..io.source import SourceVersion, source_version
from ..models import GraphData
from ..plotting.graphs import GraphPlotSpec, nonfinite_coordinate_count, render_graph_data
from ..plotting.plots import InvalidAxisLimitError, InvalidAxisScaleError, validate_plot_labels
from .controls import slider_or_number_input
from .results import (
    close_plot_result,
    discard_plot_result,
    show_cached_plot_result,
    store_plot_result,
)


@st.cache_data(show_spinner="Reading graph…", max_entries=16)
def _cached_graph(root_path: str, object_path: str, version: SourceVersion) -> GraphData:
    """Load one graph; ``version`` invalidates the cache when the file changes."""
    with explain_missing_remote_backend(root_path):
        return read_graph(root_path, object_path)


def _graph_controls(graph: GraphData, object_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    st.subheader("Graph configuration", anchor=False)
    style: dict[str, Any] = {}
    style["graph_mode"] = st.selectbox(
        "Drawing mode", ("Markers", "Line", "Line + markers"), key="graph_drawing_mode"
    )
    if graph.xerr_low is not None or graph.yerr_low is not None:
        style["show_errors"] = st.checkbox("Show error bars", value=True, key="graph_show_errors")
    style["marker_size"] = slider_or_number_input(
        "Marker size", 0.5, 20.0, 5.0, key="graph_marker_size"
    )
    style["line_width"] = slider_or_number_input(
        "Line width", 0.5, 10.0, 1.5, key="graph_line_width"
    )
    style["fig_width"] = slider_or_number_input(
        "Figure width", 4.0, 16.0, 8.0, key="graph_fig_width"
    )
    style["fig_height"] = slider_or_number_input(
        "Figure height", 3.0, 12.0, 6.0, key="graph_fig_height"
    )
    style["display_dpi"] = 200
    style["log_x"] = st.checkbox("Log x-axis", key="graph_log_x")
    style["log_y"] = st.checkbox("Log y-axis", key="graph_log_y")
    style["grid"] = st.checkbox("Grid", value=True, key="graph_grid")
    manual = st.checkbox("Apply manual axis limits", key="graph_manual_limits")
    for key, title in (
        ("x_min_text", "X minimum"),
        ("x_max_text", "X maximum"),
        ("y_min_text", "Y minimum"),
        ("y_max_text", "Y maximum"),
    ):
        style[key] = (
            st.text_input(title, key=f"graph_{key}", placeholder="Automatic") if manual else ""
        )
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
) -> tuple[Any, ...] | None:
    """Render a graph after deliberate opening, retaining its plot across reruns.

    Returns the current settings, which the output compares with those of the
    displayed plot, or ``None`` if the graph cannot be shown.
    """
    st.subheader(f"Graph: {object_path.rsplit('/', 1)[-1]}", anchor=False)
    st.caption(f"`{classname}` · `{object_path}`")
    try:
        version = source_version(root_path)
        graph = _cached_graph(root_path, object_path, version)
    except (
        UnsupportedGraphError,
        RemoteBackendUnavailableError,
        ValueError,
        OSError,
        KeyError,
    ) as exc:
        discard_plot_result("root_graph_result")
        st.warning(str(exc))
        return None
    st.caption(f"Points: {len(graph.x):,}")
    omitted_points = nonfinite_coordinate_count(graph)
    if omitted_points:
        st.warning(f"Omitted {omitted_points} graph point(s) with non-finite coordinates.")
    style, labels = _graph_controls(graph, str(abs(hash((root_path, object_path, version)))))
    settings = (style, labels)
    # A deliberate Open action triggers the initial plot. A mere row selection does not.
    make_plot = st.button(
        "Make plot",
        key="graph_make_plot",
        type="primary",
        shortcut="P" if shortcuts_enabled else None,
    )
    if "root_graph_result" not in st.session_state or make_plot:
        figure = None
        try:
            validate_plot_labels(labels)
            spec = GraphPlotSpec(root_path, object_path, style, labels)
            with st.spinner("Rendering plot…"):
                figure = render_graph_data(graph, spec.style, spec.labels)
            from ..plotting.graph_codegen import generate_graph_plot_code

            warnings = []
            try:
                code = generate_graph_plot_code(spec)
            except (ValueError, TypeError, RuntimeError, OSError) as exc:
                code = None
                warnings.append(f"Plot-code export unavailable: {exc}")
            store_plot_result(
                "root_graph_result",
                {
                    "kind": "matplotlib",
                    "figure": figure,
                    "code": code,
                    "file_name": "root_graph",
                    "warnings": warnings,
                    "settings": settings,
                },
            )
        except (InvalidAxisLimitError, InvalidAxisScaleError, ValueError) as exc:
            if figure is not None:
                close_plot_result({"kind": "matplotlib", "figure": figure})
            discard_plot_result("root_graph_result")
            st.warning(str(exc))
    # The output is rendered separately in the right-hand master-detail pane.
    return settings


def render_graph_output(current_settings: tuple[Any, ...] | None = None) -> None:
    show_cached_plot_result(
        "root_graph_result",
        current_settings=current_settings,
        export_prefix="graph",
        download_key="graph_export",
        make_plot_label="Make plot",
        make_plot_shortcut="P",
        empty_message="Configure the graph and click Make plot.",
    )
