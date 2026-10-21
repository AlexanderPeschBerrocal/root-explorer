"""Event selection and ranking controls."""

from __future__ import annotations

import streamlit as st

from ..analysis.ranking import compute_event_metric, metric_dataframe
from ..io.root import choose_default
from .controls import field_option_labels
from .state import DataContext
from .tables import render_dataframe


def render_event_selection(context: DataContext) -> int:
    """Render event index and ranking controls, returning the selected event."""

    st.subheader("Event inspection", anchor=False)
    if not context.entry_start <= st.session_state.get("event_index", -1) < context.entry_stop:
        st.session_state["event_index"] = context.entry_start
    event_index = st.number_input(
        "event index",
        min_value=context.entry_start,
        max_value=context.entry_stop - 1,
        value=int(st.session_state["event_index"]),
        step=1,
    )
    st.session_state["event_index"] = int(event_index)

    with st.expander("Rank events", expanded=True):
        option_labels = (
            field_option_labels(context.field_info)
            if st.session_state.get("event_show_field_details", False)
            else {}
        )
        metric_branch = st.selectbox(
            f"metric {context.data_item_name}",
            context.fields,
            index=choose_default(context.fields, ["pt", "energy", "mass", "x"]),
            format_func=lambda field_name: option_labels.get(field_name, field_name),
            key="metric_branch",
        )
        metric_mode = st.selectbox(
            "metric",
            ["multiplicity", "sum", "mean", "max", "min", "count_above", "count_below"],
            key="metric_mode",
        )
        metric_abs = st.checkbox("use absolute values", value=False)
        threshold = None
        if metric_mode in {"count_above", "count_below"}:
            threshold = st.number_input("threshold", value=0.0)
        st.caption(
            f"Ranking entries {context.entry_start}-{context.entry_stop - 1} "
            f"({context.entry_stop - context.entry_start} events)."
        )
        descending = st.checkbox("largest first", value=True)
        max_rank_rows = st.slider("rows", 5, 200, 10)
        if st.button("Compute ranking", key="rank_compute"):
            try:
                events, values = compute_event_metric(
                    context.root_path,
                    context.event_container_name,
                    metric_branch,
                    metric_mode,
                    context.entry_start,
                    context.entry_stop,
                    threshold,
                    metric_abs,
                )
                st.session_state["rank_df"] = metric_dataframe(
                    events, values, descending, int(max_rank_rows)
                )
                metric_input = (
                    f"|{metric_branch}|"
                    if metric_abs and metric_mode != "multiplicity"
                    else metric_branch
                )
                st.session_state["rank_label"] = f"{metric_mode}({metric_input})"
            except Exception as exc:
                st.error(f"Ranking failed: {exc}")
        if "rank_df" in st.session_state:
            st.write(f"Metric: **{st.session_state.get('rank_label', 'metric')}**")
            render_dataframe(st.session_state["rank_df"], height=220)
            ranked_events = st.session_state["rank_df"]["event"].astype(int).tolist()
            if ranked_events:
                selected = st.selectbox(
                    "jump to event", ranked_events, format_func=lambda event: f"event {event}"
                )
                if st.button("Use selected event"):
                    st.session_state["event_index"] = int(selected)
                    st.rerun()
    return int(st.session_state["event_index"])
