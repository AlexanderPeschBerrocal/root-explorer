"""Thin Streamlit application orchestration entrypoint."""

from __future__ import annotations

from importlib.resources import files

import streamlit as st

from .. import __version__
from ..config import APP_TITLE
from .geometry import render_geometry_sidebar, render_geometry_tab
from .input import invalidate_data_results, render_data_input
from .pages import render_field_tab, render_help_tab
from .plotting import render_event_plot_tab, render_plot_tab
from .ranking import render_event_selection

APP_ICON = files("root_explorer").joinpath("assets/root-explorer-app-icon-transparent-edges.png")

GITHUB_URL = "https://github.com/AlexanderPeschBerrocal/root-explorer"

ABOUT_TEXT = f"""
### {APP_TITLE}

**Version:** `{__version__}`

[View the project on GitHub]({GITHUB_URL})
"""


def main() -> None:
    """Configure the page and compose the focused Streamlit UI sections."""

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON.read_bytes(),
        layout="wide",
        menu_items={"About": ABOUT_TEXT},
    )
    st.title(APP_TITLE, anchor=False)
    st.caption(
        "A general-purpose Streamlit app for interactive plotting and event inspection of HEP ROOT TTrees and RNTuples in the web browser."
    )
    with st.sidebar:
        file_context = render_data_input()
        geometry = render_geometry_sidebar()

    context = file_context.data
    if context is None:
        st.info("This ROOT file has no event dataset to plot.")
        geometry_tab = st.tabs(["Geometry"])[0]
        with geometry_tab:
            render_geometry_tab(geometry)
        return

    invalidate_data_results(context)
    plot_tab, event_tab, field_tab, geometry_tab, help_tab = st.tabs(
        ["Plotter", "Event inspection", context.data_item_plural.title(), "Geometry", "Help"]
    )
    with field_tab:
        render_field_tab(context)
    with geometry_tab:
        render_geometry_tab(geometry)
    with help_tab:
        render_help_tab(context)
    with plot_tab:
        render_plot_tab(context, geometry)
    with event_tab:
        event_index = render_event_selection(context)
        render_event_plot_tab(context, geometry, event_index)
