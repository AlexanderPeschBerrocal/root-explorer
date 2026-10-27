"""Informational and data-browser Streamlit tabs."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from ..config import DOCS_URL
from ..io.root import filter_root_objects
from ..io.source import source_version
from ..models import RootObjectInfo
from .controls import field_structure_label
from .results import collection_summary, discard_plot_result, local_absolute_path
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
            help="Per-event shape: scalar, fixed-size, jagged, nested/record, unknown, or unsupported.",
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
    st.dataframe(dataframe, width="stretch", height="auto", column_config=column_config)

    st.markdown("#### Collections", anchors=False)
    st.dataframe(
        collection_summary(context.field_info, item_name=item_name), width="stretch", height="auto"
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
    identity = (file_name, source_version(file_name)) if file_name else None
    if st.session_state.get("root_object_source") != identity:
        discard_plot_result("root_graph_result")
        discard_plot_result("root_histogram_result")
        st.session_state["root_object_source"] = identity
        # A replacement can remove an object or change its class. Keep an
        # inspection open only if the new catalogue still contains that object.
        if opened is not None and not any(
            (info.qualified_path, info.classname) == opened[1:] for info in catalogue
        ):
            st.session_state.pop("root_inspector_open", None)
            opened = None
    if opened is not None and opened[0] != file_name:
        st.session_state.pop("root_inspector_open", None)
        discard_plot_result("root_graph_result")
        discard_plot_result("root_histogram_result")
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
        # The dataframe selection is controlled by the browser frontend. When
        # selection events are unavailable (or no row is selected), provide an
        # explicit object selector so the Open action is never hidden.
        candidates = tuple(info for info in objects if info.kind in {"graph", "histogram"})
        selected_info = None
        rows = selection.selection.rows
        if rows:
            row = rows[0]
            if 0 <= row < len(objects) and objects[row].kind in {"graph", "histogram"}:
                selected_info = objects[row]

        if selected_info is None and candidates:
            candidate_paths = [info.qualified_path for info in candidates]
            # Opening an object switches to the compact layout. The selector must not
            # depend on the layout, and it starts at an opened object, so it keeps
            # naming the object on display.
            target_identity = (file_name, tuple(candidate_paths))
            target_key = hashlib.sha256(repr(target_identity).encode("utf-8")).hexdigest()[:16]
            opened_path = opened[1] if opened is not None else None
            fallback_path = st.selectbox(
                "Object to inspect",
                candidate_paths,
                index=candidate_paths.index(opened_path) if opened_path in candidate_paths else 0,
                format_func=lambda path: next(
                    f"{info.path} ({info.classname})"
                    for info in candidates
                    if info.qualified_path == path
                ),
                key=f"root_object_target_{target_key}",
                help="Use this selector if clicking a row does not select it.",
            )
            selected_info = next(
                info for info in candidates if info.qualified_path == fallback_path
            )

        if selected_info is not None:
            st.caption(
                f"Ready to open: `{selected_info.qualified_path}` ({selected_info.classname})"
            )
        if (
            st.button(
                "Open selected object",
                key="root_object_open",
                shortcut="O" if shortcuts_enabled else None,
                disabled=selected_info is None,
            )
            and selected_info is not None
        ):
            st.session_state["root_inspector_open"] = (
                file_name,
                selected_info.qualified_path,
                selected_info.classname,
            )
            discard_plot_result("root_graph_result")
            discard_plot_result("root_histogram_result")
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
            discard_plot_result("root_graph_result")
            discard_plot_result("root_histogram_result")
            st.rerun()
    from .graph_viewer import render_graph_controls, render_graph_output

    # Exported plot scripts embed the ROOT path, so it must not be relative.
    if opened[2].startswith(("TH1", "TH2", "TH3", "TProfile")):
        from .histogram_viewer import render_histogram_controls, render_histogram_output

        with left:
            settings = render_histogram_controls(
                local_absolute_path(opened[0]),
                opened[1],
                opened[2],
                shortcuts_enabled=shortcuts_enabled,
            )
        with right:
            render_histogram_output(settings)
    else:
        with left:
            settings = render_graph_controls(
                local_absolute_path(opened[0]),
                opened[1],
                opened[2],
                shortcuts_enabled=shortcuts_enabled,
            )
        with right:
            render_graph_output(settings)


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
    st.markdown(f"For detailed guides and explanations, see the [full documentation]({DOCS_URL}).")
    st.subheader("How to use this app", anchor=False)
    st.markdown(
        f"""
**Basic workflow**
1. Choose a ROOT file, event container (TTree or RNTuple), and read range in the sidebar. Start with a small range, then increase it once the plot setup looks right.
2. Use the **Plotter** tab for histogram or scatter plots over the selected entry range.
3. Use the **Event inspection** tab to inspect one event at a time. The ranking tool can help find events with large multiplicities, sums, extrema, or threshold counts.
4. Pick variables from the {item_name} selectors. Besides direct {item_name}s, the app can derive absolute values, 3D magnitudes, `r`, `phi`, `eta`, massless and massive energies, single-particle invariant masses, ratios, sums, differences, products, `delta phi`, `delta eta`, and `delta R`.
5. Click **Make plot** (shortcut **P**) or **Make event plot** (shortcut **Shift+P**) after changing settings. The output area keeps showing the most recently generated plot until you make a new one; a notice says when the settings no longer match the displayed plot.
6. Titles and axis labels of static plots support math between dollar signs, for example `$p_T$ [GeV]`. Interactive 3D plots show labels literally.

**Field badges, jagged collections, and cuts**
- Enable **Show {context.data_item_name} details in selectors** below the plot-type selection in the Plotter or Event inspection tab to add name-first badges such as `[scalar] [float32]`, `[fixed 3]`, or `[jagged] [float32]`. These describe the per-event structure and logical data type without loading the full field.
- `[nested/record]` identifies structured or multiply nested data, `[unknown]` means the schema does not provide enough information for a safe classification, and `[unsupported]` identifies fields that cannot be plotted directly. All categories and source type details are available in the **{context.data_item_plural.title()} and objects** tab.
- For object collections, direct {item_name} values are flattened for plotting.
- In the Plotter, **Multiplicity per event**, **nth object per event**, and per-event reductions such as sum, mean, max, and min are useful for turning jagged collections into one value per event. They are also available for point colours and cuts, where the event's value applies to all of its plotted objects.
- Range cuts are combined with logical AND and applied simultaneously before plotting.
- Histogram weights support the common HEP case where one event-level {item_name} such as `genWeight` should weight every plotted object from that event. Object-level weight {item_name}s are also supported when they already match the plotted values.

**Detector geometry overlay**
- Turn on **Enable 3D detector overlay** in the sidebar and provide a `.gdml` file by path or upload.
- Geometry is drawn only on **3D scatter interactive** plots.
- Use the geometry scale and offset controls if the ROOT coordinates and GDML geometry use different units or origins.
- Keep the geometry vertex limits modest when exploring large detector descriptions.

**Downloading plots**
- Static matplotlib plots can be downloaded as PNG or PDF with **Download PNG** and **Download PDF**; the raster export resolution sets the PNG size.
- Interactive 3D Plotly plots can be downloaded as HTML with **Download HTML**.
- **Download plot code** downloads a Python script that can be used as a starting point for reproducing a static plot.
- Downloads are sent through your browser to your own computer, also when ROOT Explorer runs in Docker or on a remote server.
- Downloads use the currently displayed generated plot, not merely the current control settings.

**Performance tips**
- Large entry ranges, many cached {item_name}s, high scatter point limits, and detailed GDML meshes can consume a lot of memory.
- Use smaller read ranges while configuring plots.
- For scatter plots, tune **Maximum scatter points** in the collapsed **Style** section. A caption below the plot states when only a sample of the points is drawn.
- For geometry, reduce the parse/draw vertex limits or disable container/world volumes unless you need them.

Good generic starting points:
- Object kinematics: `pt`, `eta`, `phi`, `mass`, `energy`.
- Event quantities: `MET`, `HT`, `nMuon`, `nElectron`, `run`, `event`, `genWeight`.
- Geometry studies: `x`, `y`, `z`, `r`, `phi`, `eta`.

This app is meant for quick inspection and plot prototyping. Keep final selections and publication plots in a reproducible analysis script.
"""
    )
