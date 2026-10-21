"""ROOT-file input controls for the Streamlit application."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..config import DEFAULT_MAX_ENTRIES
from ..io.root import (
    event_container_infos,
    event_container_kind_from_info,
    get_entry_count,
    get_field_info,
    get_root_catalogue,
    numeric_fields,
    parse_cli_rootfile,
)
from .results import discard_plot_result, store_uploaded_file
from .state import DataContext, RootFileContext


def render_data_input() -> RootFileContext:
    """Render sidebar ROOT input controls and return the selected data context.

    This function deliberately uses ``st.stop`` for invalid or incomplete input,
    matching the former single-file application flow.
    """

    cli_path = parse_cli_rootfile()
    st.header("Input", anchor=False)
    source_mode = st.radio("File source", ["Path", "Upload"], horizontal=True)
    root_path = ""
    if source_mode == "Path":
        root_path = st.text_input("ROOT file path", value=cli_path or "", key="root_path")
    else:
        uploaded = st.file_uploader("Upload ROOT file", type=["root"])
        if uploaded is not None:
            root_path = store_uploaded_file(uploaded)
            st.caption(f"Uploaded copy: `{root_path}`")

    if not root_path:
        st.info("Provide a ROOT file to begin.")
        st.stop()

    root_path_obj = Path(root_path)
    if (
        source_mode == "Path"
        and not root_path_obj.exists()
        and not root_path.startswith(("root://", "http://", "https://"))
    ):
        st.error(f"File does not exist: {root_path}")
        st.stop()

    try:
        catalogue = get_root_catalogue(root_path)
    except Exception as exc:
        st.error(f"Could not open ROOT file: {exc}")
        st.stop()

    available_container_infos = event_container_infos(catalogue)
    if not available_container_infos:
        st.info(
            "No TTrees or RNTuples containing event data were found. "
            "The ROOT object browser remains available."
        )
        return RootFileContext(root_path=root_path, catalogue=catalogue, data=None)

    available_containers = [info.path for info in available_container_infos]
    event_container_name = st.selectbox("Event container", available_containers, index=0)
    selected_container = next(
        info for info in available_container_infos if info.path == event_container_name
    )
    container_kind = event_container_kind_from_info(selected_container)
    n_entries = get_entry_count(root_path, event_container_name)
    st.write(f"Entries/events: **{n_entries:,}**")
    if n_entries <= 0:
        st.warning("This event container has no entries to inspect.")
        st.stop()

    st.subheader("Read range", anchor=False)
    entry_start = st.number_input(
        "entry start", min_value=0, max_value=max(n_entries - 1, 0), value=0, step=1
    )
    entry_stop = st.number_input(
        "entry stop",
        min_value=1,
        max_value=n_entries,
        value=min(n_entries, DEFAULT_MAX_ENTRIES),
        step=1,
    )
    if entry_stop <= entry_start:
        st.warning("entry stop must be larger than entry start.")
        st.stop()

    field_info = get_field_info(root_path, event_container_name)
    data_context = DataContext(
        root_path=root_path,
        event_container_name=event_container_name,
        n_entries=n_entries,
        entry_start=int(entry_start),
        entry_stop=int(entry_stop),
        field_info=field_info,
        fields=numeric_fields(field_info),
        container_kind=container_kind,
    )
    return RootFileContext(root_path=root_path, catalogue=catalogue, data=data_context)


def invalidate_data_results(context: DataContext) -> None:
    """Clear result state whenever the selected data range changes."""

    if st.session_state.get("data_context") == context.cache_key:
        return
    for key in ["plot_result", "event_plot_result"]:
        discard_plot_result(key)
    for key in ["rank_df", "rank_label"]:
        st.session_state.pop(key, None)
    st.session_state["event_index"] = context.entry_start
    st.session_state["data_context"] = context.cache_key
