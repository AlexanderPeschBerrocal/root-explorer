"""Streamlit master-detail controls for persisted ROOT histogram objects."""

from __future__ import annotations

import numpy as np
import streamlit as st

from ..analysis.fitting import fit_histogram_1d
from ..io.histograms import UnsupportedHistogramError, read_histogram
from ..io.source import SourceVersion, source_version
from ..plotting.histograms import HistogramPlotSpec, render_histogram_data
from ..plotting.plots import (
    InvalidAxisLimitError,
    InvalidAxisScaleError,
    InvalidColorScaleError,
    draw_fit_overlay,
    validate_plot_labels,
)
from .controls import color_scale_controls, fit_controls, slider_or_number_input
from .results import (
    close_plot_result,
    discard_plot_result,
    show_cached_plot_result,
    store_plot_result,
)


@st.cache_data(show_spinner="Reading histogram…", max_entries=16)
def _cached_histogram(root_path: str, object_path: str, version: SourceVersion):
    """Load one histogram; ``version`` invalidates the cache when the file changes."""
    return read_histogram(root_path, object_path)


def render_histogram_controls(
    root_path: str, object_path: str, classname: str, *, shortcuts_enabled: bool = True
) -> None:
    """Load only the explicitly opened histogram and configure its presentation."""
    st.subheader(f"Histogram: {object_path.rsplit('/', 1)[-1]}", anchor=False)
    st.caption(f"`{classname}` · `{object_path}`")
    try:
        version = source_version(root_path)
        data = _cached_histogram(root_path, object_path, version)
    except (UnsupportedHistogramError, OSError, ValueError, KeyError) as exc:
        discard_plot_result("root_histogram_result")
        st.warning(str(exc))
        return
    h = data.histogram
    st.caption(" x ".join(f"{len(axis)} bins" for axis in h.axes))
    if data.is_profile:
        st.caption("Profile: bin values are means, not event counts.")
    else:
        st.caption(f"Visible sum of bin contents: {float(np.sum(h.values())):g}")
    st.markdown("#### Histogram configuration", anchors=False)
    style: dict[str, object] = {
        "fig_width": slider_or_number_input("Figure width", 4.0, 16.0, 8.0, key="hist_fig_width"),
        "fig_height": slider_or_number_input(
            "Figure height", 3.0, 12.0, 6.0, key="hist_fig_height"
        ),
        "display_dpi": 200,
    }
    if h.ndim == 1:
        modes = ("Step", "Error bars", "Step + error bars", "Filled")
        style["histogram_mode"] = st.selectbox(
            "Drawing mode", modes, index=1 if data.is_profile else 0, key=f"hist_mode_{object_path}"
        )
    else:
        style.update(
            color_scale_controls("hist", log_style_key="log_z", log_widget_key="hist_log_z")
        )
    style["log_x"] = st.checkbox("Log x-axis", key="hist_log_x")
    style["log_y"] = st.checkbox("Log y-axis", key="hist_log_y")
    style["grid"] = st.checkbox("Grid", key="hist_grid")
    manual = st.checkbox("Apply manual axis limits", key="hist_manual_limits")
    for key, title in (
        ("x_min_text", "X minimum"),
        ("x_max_text", "X maximum"),
        ("y_min_text", "Y minimum"),
        ("y_max_text", "Y maximum"),
    ):
        style[key] = (
            st.text_input(title, key=f"hist_{key}", placeholder="Automatic") if manual else ""
        )
    st.markdown("#### Labels", anchors=False)
    identity = str(abs(hash((root_path, object_path, version))))
    labels = {
        "title": st.text_input("Title", value=data.title, key=f"hist_title_{identity}"),
        "custom_x": st.text_input("X label", value=data.x_label, key=f"hist_x_{identity}"),
        "custom_y": st.text_input("Y label", value=data.y_label, key=f"hist_y_{identity}"),
    }
    if h.ndim == 2:
        labels["custom_z"] = st.text_input(
            "Colour-bar label", value=data.colorbar_label, key=f"hist_z_{identity}"
        )
    fit_config = (
        fit_controls(f"persisted_{identity}", "1D histogram")
        if h.ndim == 1 and not data.is_profile
        else None
    )
    make_plot = st.button(
        "Make plot",
        key="hist_make_plot",
        type="primary",
        shortcut="P" if shortcuts_enabled else None,
    )
    if "root_histogram_result" not in st.session_state or make_plot:
        figure = None
        try:
            validate_plot_labels(labels)
            spec = HistogramPlotSpec(root_path, object_path, style, labels, fit_config=fit_config)
            fit_result = None
            if fit_config is not None:
                with st.spinner("Fitting histogram…"):
                    fit_result = fit_histogram_1d(h, fit_config)
            with st.spinner("Rendering plot…"):
                figure = render_histogram_data(data, spec.style, spec.labels)
                if fit_result is not None and fit_result.success:
                    draw_fit_overlay(figure.axes[0], h, fit_result)
                    figure.tight_layout()
            from ..plotting.histogram_codegen import generate_histogram_plot_code

            warnings = []
            try:
                code = generate_histogram_plot_code(spec)
            except (ValueError, TypeError, RuntimeError) as exc:
                code = None
                warnings.append(f"Plot-code export unavailable: {exc}")
            store_plot_result(
                "root_histogram_result",
                {
                    "kind": "matplotlib",
                    "figure": figure,
                    "code": code,
                    "save_name": "root_histogram",
                    "warnings": warnings,
                    "fit_result": fit_result,
                },
            )
        except (
            InvalidAxisLimitError,
            InvalidAxisScaleError,
            InvalidColorScaleError,
            ValueError,
        ) as exc:
            if figure is not None:
                close_plot_result({"kind": "matplotlib", "figure": figure})
            discard_plot_result("root_histogram_result")
            st.warning(str(exc))


def render_histogram_output() -> None:
    show_cached_plot_result(
        "root_histogram_result",
        export_prefix="histogram",
        save_button_key="histogram_save",
        make_plot_label="Make plot",
        make_plot_shortcut="P",
        empty_message="Configure the histogram and click Make plot.",
    )
