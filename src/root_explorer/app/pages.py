"""Informational and data-browser Streamlit tabs."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from ..io.root import filter_root_objects
from ..models import RootObjectInfo
from .controls import field_structure_label
from .results import collection_summary
from .state import DataContext


def render_field_browser(context: DataContext) -> None:
    """Render metadata for the selected event container's branches or fields."""

    item_name = context.data_item_name
    item_plural = context.data_item_plural
    st.subheader(
        f"Selected {context.container_kind}: {context.event_container_name}",
        anchor=False,
    )
    st.markdown(f"#### {item_plural.title()}", anchors=False)
    query = st.text_input(f"Filter {item_plural}", value="", key="field_table_filter")
    dataframe = pd.DataFrame(
        [
            {
                **field.__dict__,
                "structure": field_structure_label(field),
            }
            for field in context.field_info
        ]
    ).rename(columns={"name": item_name, "field": f"{item_name} member"})
    if "fixed_shape" in dataframe:
        dataframe = dataframe.drop(columns="fixed_shape")
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
    column_config = {
        "structure": st.column_config.TextColumn(
            "structure",
            help="Per-event shape: scalar, fixed-size, jagged, nested/record, or unsupported.",
        ),
        "dtype": st.column_config.TextColumn(
            "dtype",
            help="Logical primitive data type reported by the source schema.",
        ),
    }
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


def _root_object_display_name(info: RootObjectInfo) -> str:
    """Return an indented display label that preserves directory hierarchy."""

    depth = info.path.count("/")
    prefix = "" if depth == 0 else "    " * (depth - 1) + "↳ "
    return f"{prefix}{info.name}"


def render_root_objects_browser(
    catalogue: tuple[RootObjectInfo, ...],
    *,
    file_name: str | None = None,
    shortcuts_enabled: bool = True,
) -> None:
    """Browse metadata and open supported plot objects deliberately."""

    opened = st.session_state.get("root_inspector_open")
    if opened is not None and opened[0] != file_name:
        st.session_state.pop("root_inspector_open", None)
        st.session_state.pop("root_graph_result", None)
        st.session_state.pop("root_histogram_result", None)
        opened = None

    def render_catalogue(*, compact: bool) -> None:
        st.subheader("ROOT file contents", anchor=False)
        if file_name:
            st.caption(f"Objects in `{file_name}`. Selecting a row does not load its contents.")

        show_all_cycles = st.toggle(
            "Show all ROOT cycles",
            value=False,
            key="root_object_show_all_cycles",
            help="By default only the highest cycle for each object path is shown.",
        )
        query = st.text_input("Filter ROOT objects", value="", key="root_object_filter")
        objects = filter_root_objects(catalogue, latest_only=not show_all_cycles)
        if query:
            folded = query.casefold()
            objects = tuple(
                info
                for info in objects
                if folded
                in " ".join(
                    (
                        info.path,
                        info.name,
                        info.classname,
                        info.kind,
                        info.parent,
                        "" if info.cycle is None else str(info.cycle),
                    )
                ).casefold()
            )
        dataframe = pd.DataFrame(
            [
                {
                    "object": _root_object_display_name(info),
                    "ROOT class": info.classname,
                    "kind": info.kind,
                    "cycle": info.cycle,
                    "path": info.path,
                }
                for info in objects
            ],
            columns=["object", "ROOT class", "kind", "cycle", "path"],
        )
        shown = dataframe[["object", "ROOT class"]] if compact else dataframe
        # A changed filter, cycle view or layout must not reuse an old row index.
        catalogue_identity = (
            file_name,
            compact,
            tuple(info.qualified_path for info in objects),
        )
        selection_key = hashlib.sha256(repr(catalogue_identity).encode("utf-8")).hexdigest()[:16]
        selection = st.dataframe(
            shown,
            width="stretch",
            height="content",
            on_select="rerun",
            selection_mode="single-row",
            key=f"root_object_selection_{selection_key}",
        )
        if not objects:
            st.info("No ROOT objects match the current filter.")
            return
        rows = selection.selection.rows
        if rows:
            row = rows[0]
            if 0 <= row < len(objects):
                info = objects[row]
                st.caption(f"Selected: `{info.qualified_path}` ({info.classname})")
                if info.kind in {"graph", "histogram"} and st.button(
                    "Open selected object",
                    key="root_object_open",
                    shortcut="O" if shortcuts_enabled else None,
                ):
                    st.session_state["root_inspector_open"] = (
                        file_name,
                        info.qualified_path,
                        info.classname,
                    )
                    st.session_state.pop("root_graph_result", None)
                    st.session_state.pop("root_histogram_result", None)
                    st.rerun()

    if opened is None:
        render_catalogue(compact=False)
        return

    left, right = st.columns([0.36, 0.64], gap="large")
    with left:
        render_catalogue(compact=True)
        if st.button(
            "Close inspection",
            key="root_inspector_close",
            shortcut="Esc" if shortcuts_enabled else None,
        ):
            st.session_state.pop("root_inspector_open", None)
            st.session_state.pop("root_graph_result", None)
            st.session_state.pop("root_histogram_result", None)
            st.rerun()
    from .graph_viewer import render_graph_controls, render_graph_output

    if opened[2].startswith(("TH1", "TH2", "TH3", "TProfile")):
        from .histogram_viewer import render_histogram_controls, render_histogram_output

        with left:
            render_histogram_controls(
                opened[0], opened[1], opened[2], shortcuts_enabled=shortcuts_enabled
            )
        with right:
            render_histogram_output()
    else:
        with left:
            render_graph_controls(
                opened[0], opened[1], opened[2], shortcuts_enabled=shortcuts_enabled
            )
        with right:
            render_graph_output()


def render_contents_tab(
    context: DataContext,
    catalogue: tuple[RootObjectInfo, ...],
    *,
    file_name: str | None = None,
    shortcuts_enabled: bool = True,
) -> None:
    """Render file-level ROOT objects followed by the selected event-container schema."""

    render_root_objects_browser(catalogue, file_name=file_name, shortcuts_enabled=shortcuts_enabled)
    st.divider()
    render_field_browser(context)


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

**Field badges, jagged collections, and cuts**
- Enable **Show {context.data_item_name} details in selectors** below the plot-type selection in the Plotter or Event inspection tab to add name-first badges such as `[scalar] [float32]`, `[fixed 3]`, or `[jagged] [float32]`. These describe the per-event structure and logical data type without loading the full field.
- `[nested/record]` identifies structured or multiply nested data, `[unknown]` means the schema does not provide enough information for a safe classification, and `[unsupported]` identifies fields that cannot be plotted directly. All categories and source type details are available in the field browser.
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
