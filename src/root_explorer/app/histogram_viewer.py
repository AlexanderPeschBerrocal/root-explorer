"""Streamlit master-detail controls for persisted ROOT histogram objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from ..io.histograms import UnsupportedHistogramError, read_histogram
from ..plotting.histograms import HistogramPlotSpec, render_histogram_data
from ..plotting.plots import InvalidAxisLimitError, InvalidAxisScaleError, InvalidColorScaleError
from .results import show_cached_plot_result


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_histogram(root_path: str, object_path: str, version: int | None):
    return read_histogram(root_path, object_path)


def _file_version(root_path: str) -> int | None:
    try:
        return Path(root_path).stat().st_mtime_ns
    except OSError:
        return None


def render_histogram_controls(
    root_path: str, object_path: str, classname: str, *, shortcuts_enabled: bool = True
) -> None:
    """Load only the explicitly opened histogram and configure its presentation."""
    st.subheader(f"Histogram: {object_path.rsplit('/', 1)[-1]}", anchor=False)
    st.caption(f"`{classname}` · `{object_path}`")
    try:
        data = _cached_histogram(root_path, object_path, _file_version(root_path))
    except (UnsupportedHistogramError, OSError, ValueError) as exc:
        st.warning(str(exc))
        return
    h = data.histogram
    st.caption(" × ".join(f"{len(axis)} bins" for axis in h.axes))
    if data.is_profile:
        st.caption("Profile: bin values are means, not event counts.")
    else:
        st.caption(f"Visible sum of bin contents: {float(np.sum(h.values())):g}")
    st.markdown("#### Histogram configuration", anchors=False)
    style: dict[str, object] = {
        "fig_width": st.slider("Figure width", 4.0, 16.0, 8.0, key="hist_fig_width"),
        "fig_height": st.slider("Figure height", 3.0, 12.0, 6.0, key="hist_fig_height"),
        "display_dpi": 200,
    }
    if h.ndim == 1:
        modes = ("Step", "Error bars", "Step + error bars", "Filled")
        style["histogram_mode"] = st.selectbox(
            "Drawing mode", modes, index=1 if data.is_profile else 0, key=f"hist_mode_{object_path}"
        )
    else:
        style["cmap"] = st.selectbox(
            "Colour map", ("viridis", "plasma", "inferno", "magma", "cividis", "Greys", "coolwarm"),
            key="hist_cmap",
        )
        style["log_z"] = st.checkbox("Log colour scale", key="hist_log_z")
    style["log_x"] = st.checkbox("Log x-axis", key="hist_log_x")
    style["log_y"] = st.checkbox("Log y-axis", key="hist_log_y")
    style["grid"] = st.checkbox("Grid", key="hist_grid")
    manual = st.checkbox("Apply manual axis limits", key="hist_manual_limits")
    for key, title in (
        ("x_min_text", "x minimum"), ("x_max_text", "x maximum"),
        ("y_min_text", "y minimum"), ("y_max_text", "y maximum"),
    ):
        style[key] = st.text_input(title, key=f"hist_{key}") if manual else ""
    st.markdown("#### Labels", anchors=False)
    identity = str(abs(hash((root_path, object_path))))
    labels = {
        "title": st.text_input("Title", value=data.title, key=f"hist_title_{identity}"),
        "custom_x": st.text_input("X label", value=data.x_label, key=f"hist_x_{identity}"),
        "custom_y": st.text_input("Y label", value=data.y_label, key=f"hist_y_{identity}"),
    }
    if h.ndim == 2:
        labels["custom_z"] = st.text_input(
            "Colour-bar label", value=data.colorbar_label, key=f"hist_z_{identity}"
        )
    make_plot = st.button(
        "Make plot", key="hist_make_plot", type="primary",
        shortcut="P" if shortcuts_enabled else None,
    )
    if "root_histogram_result" not in st.session_state or make_plot:
        try:
            spec = HistogramPlotSpec(root_path, object_path, style, labels)
            figure = render_histogram_data(data, spec.style, spec.labels)
            from ..plotting.histogram_codegen import generate_histogram_plot_code

            st.session_state["root_histogram_result"] = {
                "kind": "matplotlib", "figure": figure,
                "code": generate_histogram_plot_code(spec),
                "save_name": "root_histogram", "warnings": [],
            }
        except (InvalidAxisLimitError, InvalidAxisScaleError, InvalidColorScaleError, ValueError) as exc:
            st.session_state.pop("root_histogram_result", None)
            st.warning(str(exc))


def render_histogram_output() -> None:
    show_cached_plot_result(
        "root_histogram_result", export_prefix="histogram",
        save_button_key="histogram_save", make_plot_label="Make plot",
        make_plot_shortcut="P", empty_message="Configure the histogram and click Make plot.",
    )
