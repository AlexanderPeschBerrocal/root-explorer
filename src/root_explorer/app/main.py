"""Thin Streamlit application orchestration entrypoint."""

from __future__ import annotations

from importlib.resources import files

import streamlit as st

from .. import __version__
from ..config import APP_TITLE
from .geometry import render_geometry_sidebar, render_geometry_tab
from .input import clear_data_results, invalidate_data_results, render_data_input
from .pages import render_contents_tab, render_help_tab, render_root_objects_browser
from .plotting import render_event_plot_tab, render_plot_tab
from .ranking import render_event_selection

APP_ICON = files("root_explorer").joinpath("assets/root-explorer-app-icon-transparent-edges.png")

GITHUB_URL = "https://github.com/AlexanderPeschBerrocal/root-explorer"

ABOUT_TEXT = f"""
**{APP_TITLE}**

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
        # Results from a previously selected container must not reappear later.
        clear_data_results()
        root_objects_tab, geometry_tab = st.tabs(
            ["ROOT objects", "Geometry"], key="object_only_tabs", on_change="rerun"
        )
        with root_objects_tab:
            render_root_objects_browser(
                file_context.catalogue,
                file_name=file_context.root_path,
                shortcuts_enabled=bool(root_objects_tab.open),
            )
        with geometry_tab:
            render_geometry_tab(geometry)
        return

    invalidate_data_results(context)
    contents_label = f"{context.data_item_plural.title()} and objects"
    plot_tab, event_tab, contents_tab, geometry_tab, help_tab = st.tabs(
        [
            "Plotter",
            "Event inspection",
            contents_label,
            "Geometry",
            "Help",
        ],
        key="event_file_tabs",
        on_change="rerun",
    )
    with contents_tab:
        render_contents_tab(
            context,
            file_context.catalogue,
            file_name=file_context.root_path,
            shortcuts_enabled=bool(contents_tab.open),
        )
    with geometry_tab:
        render_geometry_tab(geometry)
    with help_tab:
        render_help_tab(context)
    with plot_tab:
        render_plot_tab(context, geometry, shortcuts_enabled=bool(plot_tab.open))
    with event_tab:
        event_index = render_event_selection(context)
        render_event_plot_tab(context, geometry, event_index)
