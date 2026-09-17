"""Thin Streamlit application orchestration entrypoint."""

from __future__ import annotations

import streamlit as st

from ..config import APP_TITLE
from .geometry import render_geometry_sidebar, render_geometry_tab
from .input import invalidate_data_results, render_data_input
from .pages import render_field_tab, render_help_tab
from .plotting import render_event_plot_tab, render_plot_tab
from .ranking import render_event_selection


def main() -> None:
    """Configure the page and compose the focused Streamlit UI sections."""

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "A general-purpose Streamlit app for interactive plotting and event inspection of HEP ROOT TTrees and RNTuples in the web browser."
    )
    with st.sidebar:
        context = render_data_input()
        geometry = render_geometry_sidebar()

    invalidate_data_results(context)
    plot_tab, event_tab, field_tab, geometry_tab, help_tab = st.tabs(
        ["Plotter", "Event viewer", "Fields", "Geometry", "Help"]
    )
    with field_tab:
        render_field_tab(context)
    with geometry_tab:
        render_geometry_tab(geometry)
    with help_tab:
        render_help_tab()
    with plot_tab:
        render_plot_tab(context, geometry)
    with event_tab:
        event_index = render_event_selection(context)
        render_event_plot_tab(context, geometry, event_index)
