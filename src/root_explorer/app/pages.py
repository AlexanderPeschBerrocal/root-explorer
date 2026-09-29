"""Informational and data-browser Streamlit tabs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .results import collection_summary
from .state import DataContext


def render_field_tab(context: DataContext) -> None:
    """Render the branch or field and collection browser."""

    item_name = context.data_item_name
    item_plural = context.data_item_plural
    st.subheader(f"{item_name.title()} and collection browser", anchor=False)
    st.markdown(f"#### {item_plural.title()}", anchors=False)
    query = st.text_input(f"Filter {item_plural}", value="", key="field_table_filter")
    dataframe = pd.DataFrame([field.__dict__ for field in context.field_info]).rename(
        columns={"name": item_name, "field": f"{item_name} member"}
    )
    if "description" in dataframe:
        if dataframe["description"].notna().any():
            dataframe["description"] = dataframe["description"].fillna("")
        else:
            dataframe = dataframe.drop(columns="description")
    if query:
        folded_query = query.casefold()
        dataframe = dataframe[
            dataframe.apply(
                lambda row: folded_query in " ".join(map(str, row.values)).casefold(), axis=1
            )
        ]
    column_config = {}
    if "description" in dataframe:
        column_config["description"] = st.column_config.TextColumn(
            "description",
            help="Documentation supplied by the ROOT file, when available.",
            width="large",
        )
    st.dataframe(dataframe, width="stretch", height=540, column_config=column_config)

    st.markdown("#### Collections", anchors=False)
    st.dataframe(
        collection_summary(context.field_info, item_name=item_name), width="stretch", height=540
    )


def render_help_tab(context: DataContext) -> None:
    """Render the user-facing quick guide for the selected container type."""

    item_name = context.data_item_name
    st.subheader("How to use this app", anchor=False)
    st.markdown(
        f"""
**Basic workflow**
1. Choose a ROOT file, event dataset (TTree or RNTuple), and read range in the sidebar. Start with a small range, then increase it once the plot setup looks right.
2. Use the **Plotter** tab for histogram or scatter plots over the selected entry range.
3. Use the **Event inspection** tab to inspect one event at a time. The ranking tool can help find events with large multiplicities, sums, extrema, or threshold counts.
4. Pick variables from the {item_name} selectors. Besides direct {item_name}s, the app can derive `r`, `phi`, `eta`, ratios, sums, differences, products, `delta phi`, `delta eta`, and `delta R`.
5. Click **Make plot** or **Make event plot** after changing settings. The output area keeps showing the most recently generated plot until you make a new one.

**Jagged collections and cuts**
- For object collections, direct {item_name} values are flattened for plotting.
- In the Plotter, **Multiplicity per event**, **nth object per event**, and per-event reductions such as sum, mean, max, and min are useful for turning jagged collections into one value per event.
- Range cuts are combined with logical AND and applied simultaneously before plotting.
- Histogram weights support the common HEP case where one event-level {item_name} such as `genWeight` should weight every plotted object from that event. Object-level weight {item_name}s are also supported when they already match the plotted values.

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
- Large entry ranges, many cached {item_name}s, high scatter point limits, and detailed GDML meshes can consume a lot of memory.
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
