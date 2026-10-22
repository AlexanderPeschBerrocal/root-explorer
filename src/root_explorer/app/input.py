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
from ..io.source import (
    SourceUnavailableError,
    is_remote_source,
    reload_source,
    source_version,
)
from ..models import RootObjectInfo
from .results import discard_plot_result, store_uploaded_file
from .state import DataContext, EventDataStatus, RootFileContext


def render_data_input() -> RootFileContext:
    """Render sidebar ROOT input controls and return the selected data context.

    A missing or unreadable file stops the run with ``st.stop``. Once the file's
    catalogue is available, the run continues even if the selected event container
    is empty or unreadable, so the ROOT object browser and geometry remain usable.
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
        version = source_version(root_path)
    except SourceUnavailableError as exc:
        st.error(str(exc))
        st.stop()
    if is_remote_source(root_path) and st.button(
        "Reload source",
        key="reload_remote_source",
        help=(
            "Remote files are cached until they are reloaded. "
            "Read this file again to pick up changes."
        ),
    ):
        reload_source(root_path)
        version = source_version(root_path)

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
        return RootFileContext(
            root_path=root_path,
            catalogue=catalogue,
            data=None,
            event_data_status="no_event_containers",
        )

    available_containers = [info.path for info in available_container_infos]
    event_container_name = st.selectbox("Event container", available_containers, index=0)
    selected_container = next(
        info for info in available_container_infos if info.path == event_container_name
    )
    container_kind = event_container_kind_from_info(selected_container)
    try:
        n_entries = get_entry_count(root_path, event_container_name)
    except Exception as exc:
        return _without_event_data(
            root_path, catalogue, "unreadable_container", container_kind, exc
        )
    st.write(f"Entries/events: **{n_entries:,}**")
    if n_entries <= 0:
        return _without_event_data(root_path, catalogue, "empty_container", container_kind)

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

    try:
        field_info = get_field_info(root_path, event_container_name)
    except Exception as exc:
        return _without_event_data(
            root_path, catalogue, "unreadable_container", container_kind, exc
        )
    data_context = DataContext(
        root_path=root_path,
        event_container_name=event_container_name,
        n_entries=n_entries,
        entry_start=int(entry_start),
        entry_stop=int(entry_stop),
        field_info=field_info,
        fields=numeric_fields(field_info),
        container_kind=container_kind,
        source_version=version,
    )
    return RootFileContext(root_path=root_path, catalogue=catalogue, data=data_context)


def _without_event_data(
    root_path: str,
    catalogue: tuple[RootObjectInfo, ...],
    status: EventDataStatus,
    container_kind: str,
    error: Exception | None = None,
) -> RootFileContext:
    """Report an unusable event container while keeping the rest of the file available."""

    if status == "empty_container":
        st.warning(
            f"The selected {container_kind} has no entries to plot or inspect. "
            "You can still browse the other ROOT objects in this file."
        )
    else:
        st.error(
            f"Could not read the selected {container_kind}: {error}. "
            "You can still browse the other ROOT objects in this file."
        )
    return RootFileContext(
        root_path=root_path, catalogue=catalogue, data=None, event_data_status=status
    )


def clear_data_results() -> None:
    """Discard plot, event-plot and ranking results that belong to event data."""

    for key in ["plot_result", "event_plot_result"]:
        discard_plot_result(key)
    for key in ["rank_df", "rank_label", "event_index", "data_context"]:
        st.session_state.pop(key, None)


def invalidate_data_results(context: DataContext) -> None:
    """Clear result state whenever the selected data range changes."""

    if st.session_state.get("data_context") == context.cache_key:
        return
    clear_data_results()
    st.session_state["event_index"] = context.entry_start
    st.session_state["data_context"] = context.cache_key
