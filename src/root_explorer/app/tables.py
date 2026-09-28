"""Shared Streamlit dataframe rendering helpers."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st

DataframeHeight = int | Literal["auto", "content", "stretch"]


def render_dataframe(dataframe: pd.DataFrame, *, height: DataframeHeight = "auto") -> None:
    """Render an interactive, full-width dataframe with centred cell contents."""

    st.dataframe(
        dataframe,
        width="stretch",
        height=height,
        hide_index=True,
        column_config={
            column: st.column_config.Column(alignment="center") for column in dataframe.columns
        },
    )
