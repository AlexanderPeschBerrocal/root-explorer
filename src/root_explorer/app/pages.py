"""Informational and data-browser Streamlit tabs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .results import collection_summary
from .state import DataContext


def render_branch_tab(context: DataContext) -> None:
    """Render the branch and collection browser."""

    st.subheader("Branch and collection browser")
    collections, branches = st.columns([0.35, 0.65])
    with collections:
        st.markdown("#### Collections")
        st.dataframe(collection_summary(context.branch_info), width="stretch", height=540)
    with branches:
        st.markdown("#### Branches")
        query = st.text_input("Filter branches", value="", key="branch_table_filter")
        dataframe = pd.DataFrame([branch.__dict__ for branch in context.branch_info])
        if query:
            folded_query = query.casefold()
            dataframe = dataframe[
                dataframe.apply(
                    lambda row: folded_query in " ".join(map(str, row.values)).casefold(), axis=1
                )
            ]
        st.dataframe(dataframe, width="stretch", height=540)


def render_help_tab() -> None:
    """Render the unchanged user-facing quick guide."""

    st.subheader("How to use this app")
    st.markdown(
        """
**Basic workflow**
1. Choose a ROOT file, TTree, and read range in the sidebar. Start with a small range, then increase it once the plot setup looks right.
2. Use the **Plotter** tab for histogram or scatter plots over the selected entry range.
3. Use the **Event viewer** tab to inspect one event at a time. The ranking tool can help find events with large multiplicities, sums, extrema, or threshold counts.
4. Pick variables from the branch selectors. Besides direct branches, the app can derive `r`, `phi`, `eta`, ratios, sums, differences, products, `delta phi`, and `delta R`.
5. Click **Make plot** or **Make event plot** after changing settings. The output area keeps showing the most recently generated plot until you make a new one.

**Jagged collections and cuts**
- For object collections, direct branch values are flattened for plotting.
- In the Plotter, **Multiplicity per event**, **nth object per event**, and per-event reductions such as sum, mean, max, and min are useful for turning jagged collections into one value per event.
- Range cuts are combined with logical AND and applied simultaneously before plotting.
- Histogram weights support the common HEP case where one event-level branch such as `genWeight` should weight every plotted object from that event. Object-level weight branches are also supported when they already match the plotted values.

**Detector geometry overlay**
- Enable **3D detector overlay** in the sidebar and provide a `.gdml` file by path or upload.
- Geometry is drawn only on **3D scatter interactive** plots.
- Use the geometry scale and offset controls if the ROOT coordinates and GDML geometry use different units or origins.
- Keep the geometry vertex limits modest when exploring large detector descriptions.

**Saving plots**
- Static matplotlib plots can be saved as PNG, PDF, or both using the export checkboxes.
- Interactive 3D Plotly plots are saved as HTML.
- The save button exports the currently displayed generated plot, not merely the current control settings.

**Performance tips**
- Large entry ranges, many cached branches, high scatter point limits, and detailed GDML meshes can consume a lot of memory.
- Use smaller read ranges while configuring plots.
- For scatter plots, tune **max scatter points** in the Style section.
- For geometry, reduce the parse/draw vertex limits or disable container/world volumes unless you need them.

Good generic starting points:
- Object kinematics: `pt`, `eta`, `phi`, `mass`, `energy`.
- Event quantities: `MET`, `HT`, `nMuon`, `nElectron`, `run`, `event`, `genWeight`.
- Geometry studies: `x`, `y`, `z`, `r`, `phi`, `eta`.

This app is meant for quick inspection and plot prototyping. Keep final selections and publication plots in a reproducible analysis script.
"""
    )
