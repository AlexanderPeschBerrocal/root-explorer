"""Event selection and ranking controls."""

from __future__ import annotations

import streamlit as st

from ..analysis.ranking import compute_event_metric, metric_dataframe
from ..io.root import choose_default
from .state import DataContext


def render_event_selection(context: DataContext) -> int:
    """Render event index and ranking controls, returning the selected event."""

    st.subheader("Event inspection", anchor=False)
    if "event_index" not in st.session_state:
        st.session_state["event_index"] = 0
    event_index = st.number_input(
        "event index",
        min_value=0,
        max_value=max(context.n_entries - 1, 0),
        value=int(st.session_state["event_index"]),
        step=1,
    )
    st.session_state["event_index"] = int(event_index)

    with st.expander("Rank events", expanded=True):
        metric_branch = st.selectbox(
            f"metric {context.data_item_name}",
            context.fields,
            index=choose_default(context.fields, ["pt", "energy", "mass", "x"]),
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
                st.session_state["rank_label"] = f"{metric_mode}({metric_branch})"
            except Exception as exc:
                st.error(f"Ranking failed: {exc}")
        if "rank_df" in st.session_state:
            st.write(f"Metric: **{st.session_state.get('rank_label', 'metric')}**")
            st.dataframe(st.session_state["rank_df"], width="stretch", height=220)
            ranked_events = st.session_state["rank_df"]["event"].astype(int).tolist()
            if ranked_events:
                selected = st.selectbox(
                    "jump to event", ranked_events, format_func=lambda event: f"event {event}"
                )
                if st.button("Use selected event"):
                    st.session_state["event_index"] = int(selected)
                    st.rerun()
    return int(st.session_state["event_index"])
