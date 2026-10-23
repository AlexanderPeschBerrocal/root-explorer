"""Adaptive layout of the plot tabs: 36/64 columns, or equal thirds on wide screens."""

from __future__ import annotations

import math
from collections.abc import Iterable

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

# Below this width a third is too narrow for comfortable selectors and plots.
MIN_COLUMN_WIDTH_REM = 22.0
# Conservative average width of one selector character, and the selector's
# padding and drop-down arrow, in rem of the default 16 px base font.
SELECTOR_CHARACTER_REM = 0.45
SELECTOR_CHROME_REM = 4.0
# Gaps of Streamlit's "large" and "medium" container gap sizes.
COLUMN_GAP_REM = 4.0
GROUP_GAP_REM = 2.0
# Streamlit stacks st.columns below this viewport width; keep the same behaviour.
MOBILE_BREAKPOINT_PX = 640


def column_min_width_rem(option_labels: Iterable[str]) -> float:
    """Return the narrowest column width that shows every option label in full."""

    longest = max((len(label) for label in option_labels), default=0)
    required = math.ceil(longest * SELECTOR_CHARACTER_REM + SELECTOR_CHROME_REM)
    return max(MIN_COLUMN_WIDTH_REM, float(required))


def three_column_min_width_rem(option_labels: Iterable[str]) -> float:
    """Return the plot-area width from which equal thirds are used."""

    return 3 * column_min_width_rem(option_labels) + COLUMN_GAP_REM + GROUP_GAP_REM


def _layout_css(prefix: str, option_labels: Iterable[str]) -> str:
    layout = f".st-key-{prefix}_layout"
    groups = f".st-key-{prefix}_control_groups"
    threshold = three_column_min_width_rem(option_labels)
    gaps = COLUMN_GAP_REM + GROUP_GAP_REM
    # Streamlit adds a container's key as the CSS class ``st-key-<key>``; the
    # direct children of a horizontal container are its flex items. The mobile
    # rule uses :nth-child(n) to match the specificity of the column rules.
    return f"""<style>
{layout} {{ container-type: inline-size; }}
{layout} > * {{ min-width: 0 !important; }}
{layout} > :nth-child(1) {{ flex: 36 1 0 !important; }}
{layout} > :nth-child(2) {{ flex: 64 1 0 !important; }}
{groups} > * {{ flex: 1 1 100% !important; min-width: 0 !important; }}
@container (min-width: {threshold:g}rem) {{
  {layout} > :nth-child(1) {{ flex: 1 1 0 !important; }}
  {layout} > :nth-child(2) {{ flex: 0 0 calc((100% - {gaps:g}rem) / 3) !important; }}
  {groups} > * {{ flex: 1 1 0 !important; }}
}}
@media (max-width: {MOBILE_BREAKPOINT_PX}px) {{
  {layout} > :nth-child(n) {{ flex: 1 1 100% !important; }}
}}
</style>"""


def adaptive_plot_columns(
    prefix: str, option_labels: Iterable[str]
) -> tuple[DeltaGenerator, DeltaGenerator]:
    """Return the controls and output columns of a plot tab.

    The server does not know the browser width, so a CSS container query on the
    plot area chooses the layout. A narrow plot area keeps the 36/64 split with
    all controls in one column. Once each of three equal columns can show the
    longest option label in full, the two control groups from
    :func:`control_groups` and the output each take a third of the width.
    """

    st.html(_layout_css(prefix, option_labels))
    with st.container(horizontal=True, gap="large", key=f"{prefix}_layout"):
        controls = st.container(key=f"{prefix}_controls")
        output = st.container(key=f"{prefix}_output")
    return controls, output


def control_groups(prefix: str) -> tuple[DeltaGenerator, DeltaGenerator]:
    """Return the selection and appearance control groups of a plot tab.

    The groups are stacked unless :func:`adaptive_plot_columns` switched the
    plot area to equal thirds.
    """

    with st.container(horizontal=True, gap="medium", key=f"{prefix}_control_groups"):
        selection = st.container(key=f"{prefix}_control_groups_first")
        appearance = st.container(key=f"{prefix}_control_groups_second")
    return selection, appearance
