"""Streamlit application entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ..analysis.arrays import (
    ArrayShapeMismatchError,
    apply_finite_filter,
    array_length,
    cut_values_for_values,
    histogram_weights_for_values,
    validate_variable_event_shapes,
    variable_values_for_values,
)
from ..analysis.ranking import compute_event_metric, metric_dataframe
from ..analysis.variables import evaluate_variable
from ..config import APP_TITLE, DEFAULT_MAX_ENTRIES
from ..geometry.overlay import (
    DEFAULT_DISPLAY_GEOMETRY_VERTICES,
    DEFAULT_MAX_GEOMETRY_OBJECTS,
    DEFAULT_MAX_GEOMETRY_VERTICES,
    GEOMETRY_QUALITY_LEVELS,
    geometry_materials,
    geometry_summary_dataframe,
    load_geometry_scene,
    overlay_geometry_on_plotly,
)
from ..io.root import (
    choose_default,
    get_branch_info,
    numeric_branches,
    open_root_file,
    parse_cli_rootfile,
    walk_ttrees,
)
from ..plotting.plots import (
    apply_range_cuts,
    build_plotly_3d,
    describe_array,
    downsample_arrays,
    parse_optional_float,
    render_static_plot,
)
from .controls import (
    PLOT_TYPES,
    histogram_weight_selector,
    label_export_controls,
    needs_y_variable,
    needs_z_variable,
    plot_style_controls,
    range_cut_controls,
    selected_static_formats,
    supports_point_color,
    variable_selector,
)
from .results import collection_summary, show_cached_plot_result, store_uploaded_file


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # App body
    # -----------------------------------------------------------------------------

    cli_path = parse_cli_rootfile()
    st.title(APP_TITLE)
    st.caption(
        "A general-purpose Streamlit app for interactive plotting and event inspection of HEP ROOT TTrees in the web browser."
    )

    with st.sidebar:
        st.header("Input")
        source_mode = st.radio("File source", ["Path", "Upload"], horizontal=True)
        root_path = ""
        if source_mode == "Path":
            root_path = st.text_input("ROOT file path", value=cli_path or "")
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
            and not str(root_path).startswith(("root://", "http://", "https://"))
        ):
            st.error(f"File does not exist: {root_path}")
            st.stop()

        try:
            root_file = open_root_file(root_path)
        except Exception as exc:
            st.error(f"Could not open ROOT file: {exc}")
            st.stop()

        available_trees = walk_ttrees(root_file)
        if not available_trees:
            st.error("No TTrees were found. This app currently focuses on TTree-like event data.")
            st.stop()

        tree_name = st.selectbox("TTree", available_trees, index=0)
        tree = root_file[tree_name]
        n_entries = int(tree.num_entries)
        st.write(f"Entries/events: **{n_entries:,}**")
        if n_entries <= 0:
            st.warning("This TTree has no entries to inspect.")
            st.stop()

        st.subheader("Read range")
        entry_start = st.number_input(
            "entry start", min_value=0, max_value=max(n_entries - 1, 0), value=0, step=1
        )
        default_stop = min(n_entries, DEFAULT_MAX_ENTRIES)
        entry_stop = st.number_input(
            "entry stop", min_value=1, max_value=n_entries, value=default_stop, step=1
        )
        if entry_stop <= entry_start:
            st.warning("entry stop must be larger than entry start.")
            st.stop()

        st.subheader("Detector geometry overlay")
        geometry_enabled = st.checkbox("Enable 3D detector overlay", value=False)
        geometry_scene = None
        geometry_options: dict[str, Any] = {"enabled": False}
        if geometry_enabled:
            geometry_source_mode = st.radio(
                "Geometry source", ["Path", "Upload"], horizontal=True, key="geometry_source_mode"
            )
            geometry_path = ""
            if geometry_source_mode == "Path":
                geometry_path = st.text_input("GDML geometry path", value="", key="geometry_path")
            else:
                geometry_uploaded = st.file_uploader(
                    "Upload GDML geometry", type=["gdml"], key="geometry_upload"
                )
                if geometry_uploaded is not None:
                    geometry_path = store_uploaded_file(geometry_uploaded)
                    st.caption(f"Uploaded copy: `{geometry_path}`")

            geometry_quality = st.selectbox("mesh quality", GEOMETRY_QUALITY_LEVELS, index=1)
            include_containers = st.checkbox(
                "include container volumes", value=False, help="Usually leave off."
            )
            include_world = st.checkbox(
                "include world volume", value=False, help="Usually leave off."
            )
            geometry_build_max_objects = st.number_input(
                "max geometry objects to parse",
                min_value=100,
                max_value=500_000,
                value=DEFAULT_MAX_GEOMETRY_OBJECTS,
                step=1_000,
            )
            geometry_build_max_vertices = st.number_input(
                "max geometry vertices to parse",
                min_value=10_000,
                max_value=20_000_000,
                value=DEFAULT_MAX_GEOMETRY_VERTICES,
                step=100_000,
            )
            geometry_display_max_vertices = st.number_input(
                "max geometry vertices to draw",
                min_value=10_000,
                max_value=20_000_000,
                value=DEFAULT_DISPLAY_GEOMETRY_VERTICES,
                step=50_000,
            )
            geometry_opacity = st.slider("geometry opacity scale", 0.02, 1.0, 0.30, step=0.02)
            geometry_scale = st.number_input(
                "geometry coordinate scale",
                value=1.0,
                format="%.8g",
                help="Use this if ROOT vertices and GDML use different length units.",
            )
            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                geometry_offset_x = st.number_input("geom dx", value=0.0, format="%.8g")
            with gc2:
                geometry_offset_y = st.number_input("geom dy", value=0.0, format="%.8g")
            with gc3:
                geometry_offset_z = st.number_input("geom dz", value=0.0, format="%.8g")
            geometry_wireframe = st.checkbox("add wireframe edges", value=False)

            if geometry_path:
                geometry_path_obj = Path(geometry_path)
                if geometry_path_obj.suffix.casefold() != ".gdml":
                    st.error("Only `.gdml` geometry files are supported.")
                    geometry_path = ""
                elif geometry_source_mode == "Path" and not geometry_path_obj.exists():
                    st.error(f"Geometry file does not exist: {geometry_path}")
                    geometry_path = ""

            if geometry_path:
                try:
                    geometry_scene = load_geometry_scene(
                        geometry_path,
                        geometry_quality,
                        bool(include_containers),
                        bool(include_world),
                        int(geometry_build_max_objects),
                        int(geometry_build_max_vertices),
                    )
                    meta = geometry_scene.get("meta", {})
                    st.success(
                        f"Loaded geometry: {meta.get('objects', 0):,} objects, {meta.get('triangles', 0):,} triangles"
                    )
                    materials = geometry_materials(geometry_scene)
                    default_materials = materials[: min(len(materials), 30)]
                    selected_materials = st.multiselect(
                        "materials to draw", materials, default=default_materials
                    )
                    geometry_options = {
                        "enabled": True,
                        "materials": selected_materials,
                        "opacity": float(geometry_opacity),
                        "scale": float(geometry_scale),
                        "offset_x": float(geometry_offset_x),
                        "offset_y": float(geometry_offset_y),
                        "offset_z": float(geometry_offset_z),
                        "wireframe": bool(geometry_wireframe),
                        "max_vertices": int(geometry_display_max_vertices),
                    }
                    warnings = meta.get("warnings") or []
                    if warnings:
                        st.warning("Geometry warnings: " + "; ".join(str(w) for w in warnings[:3]))
                except Exception as exc:
                    geometry_scene = None
                    geometry_options = {"enabled": False}
                    st.error(f"Could not load geometry: {exc}")
            else:
                st.info(
                    "Provide a `.gdml` detector description by path or upload to overlay it on 3D plots."
                )

    branch_info = get_branch_info(root_path, tree_name)
    all_branches = [b.name for b in branch_info]
    branches = numeric_branches(branch_info)

    current_data_context = (root_path, tree_name, int(entry_start), int(entry_stop))
    if st.session_state.get("data_context") != current_data_context:
        for key in ["plot_result", "event_plot_result", "rank_df", "rank_label"]:
            st.session_state.pop(key, None)
        st.session_state["event_index"] = 0
        st.session_state["data_context"] = current_data_context

    plot_tab, event_tab, branch_tab, geometry_tab, help_tab = st.tabs(
        ["Plotter", "Event viewer", "Branches", "Geometry", "Help"]
    )

    with branch_tab:
        st.subheader("Branch and collection browser")
        c1, c2 = st.columns([0.35, 0.65])
        with c1:
            st.markdown("#### Collections")
            st.dataframe(collection_summary(branch_info), width="stretch", height=540)
        with c2:
            st.markdown("#### Branches")
            query = st.text_input("Filter branches", value="", key="branch_table_filter")
            df = pd.DataFrame([b.__dict__ for b in branch_info])
            if query:
                q = query.casefold()
                df = df[
                    df.apply(lambda row: q in " ".join(map(str, row.values)).casefold(), axis=1)
                ]
            st.dataframe(df, width="stretch", height=540)

    with geometry_tab:
        st.subheader("Detector geometry overlay")
        if geometry_scene is None:
            st.info(
                "Enable the 3D detector overlay in the sidebar and provide a GDML geometry file by path or upload."
            )
        else:
            st.dataframe(geometry_summary_dataframe(geometry_scene), width="stretch")
            meta = geometry_scene.get("meta", {})
            materials = meta.get("materials", {})
            if isinstance(materials, dict) and materials:
                mat_df = pd.DataFrame(
                    [
                        {"material": k, "objects": v}
                        for k, v in sorted(materials.items(), key=lambda kv: (-kv[1], kv[0]))
                    ]
                )
                st.markdown("#### Materials")
                st.dataframe(mat_df, width="stretch")
            solid_types = meta.get("solidTypes", {})
            if isinstance(solid_types, dict) and solid_types:
                st.markdown("#### Solid types")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"solid type": k, "objects": v}
                            for k, v in sorted(solid_types.items(), key=lambda kv: (-kv[1], kv[0]))
                        ]
                    ),
                    width="stretch",
                )
            warnings = meta.get("warnings") or []
            if warnings:
                st.markdown("#### Geometry parser warnings")
                st.write("\n".join(f"- {w}" for w in warnings[:50]))
            st.markdown(
                """
    The overlay is only drawn for **3D scatter interactive** plots. The GDML geometry is converted to triangle meshes and added as transparent Plotly `Mesh3d` traces underneath your event or global vertex scatter.

    Geometry files can be loaded either from a local `.gdml` path or via upload in the sidebar. Use the scale and offset controls when your ROOT coordinates and GDML coordinates use different units or origins.
    """
            )

    with help_tab:
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
    - Event quantities: `MET`, `HT`, `nMuon`, `nElectron`, `nJet`, `run`, `event`, `genWeight`.
    - Geometry studies: `x`, `y`, `z`, `r`, `phi`, `eta`.

    This app is meant for quick inspection and plot prototyping. Keep final selections and publication plots in a reproducible analysis script.
    """
        )

    with plot_tab:
        controls, output = st.columns([0.36, 0.64], gap="large")
        with controls:
            st.subheader("Global plot configuration")
            plot_type = st.selectbox("Plot type", PLOT_TYPES, index=0, key="plot_type")

            st.markdown("### Variables")
            x_spec = variable_selector("x", branches, ["pt", "energy", "eta", "x"])
            y_spec = z_spec = color_spec = None
            if needs_y_variable(plot_type):
                y_spec = variable_selector("y", branches, ["eta", "phi", "y"])
            if needs_z_variable(plot_type):
                z_spec = variable_selector("z", branches, ["phi", "z"])

            use_color = False
            if supports_point_color(plot_type):
                use_color = st.checkbox(
                    "Color points",
                    value=plot_type == "3D scatter interactive",
                    key="plot_color_points",
                )
                if use_color:
                    color_spec = variable_selector(
                        "color",
                        branches,
                        ["pt", "energy", "mass", "pdgId", "charge"],
                        allow_event_reductions=False,
                    )

            weight_branch = histogram_weight_selector("plot", plot_type, branches)
            cut_specs, cut_bounds = range_cut_controls("plot", branches)
            style = plot_style_controls(
                "plot",
                plot_type,
                color_enabled=use_color,
                default_bins=100,
                default_marker_size_2d=2.0,
                default_marker_size_3d=2.5,
                default_opacity_2d=0.45,
                default_opacity_3d=0.65,
                default_equal_aspect=False,
            )
            label_controls = label_export_controls(
                "plot", plot_type, default_save_name="root_explorer_plot"
            )
            make_plot = st.button(
                "Make plot",
                type="primary",
                key="plot_make",
                shortcut="P",
            )

        with output:
            st.subheader("Output")
            if make_plot:
                try:
                    validate_variable_event_shapes(
                        [spec for spec in [x_spec, y_spec, z_spec] if spec is not None],
                        root_path,
                        tree_name,
                        int(entry_start),
                        int(entry_stop),
                        None,
                        context="plot-axis variables",
                    )
                    arrays: list[np.ndarray] = []
                    labels: list[str] = []
                    x, x_label = evaluate_variable(
                        x_spec, root_path, tree_name, int(entry_start), int(entry_stop)
                    )
                    arrays.append(x)
                    labels.append(x_label)
                    if y_spec is not None:
                        y, y_label = evaluate_variable(
                            y_spec, root_path, tree_name, int(entry_start), int(entry_stop)
                        )
                        arrays.append(y)
                        labels.append(y_label)
                    else:
                        y = None
                        y_label = ""
                    if z_spec is not None:
                        z, z_label = evaluate_variable(
                            z_spec, root_path, tree_name, int(entry_start), int(entry_stop)
                        )
                        arrays.append(z)
                        labels.append(z_label)
                    else:
                        z = None
                        z_label = ""
                    weights = None
                    if weight_branch is not None:
                        weights = histogram_weights_for_values(
                            root_path,
                            tree_name,
                            weight_branch,
                            x_spec,
                            len(x),
                            int(entry_start),
                            int(entry_stop),
                            None,
                        )
                        arrays.append(weights)
                    color = None
                    color_label = None
                    if color_spec is not None:
                        color, color_label = variable_values_for_values(
                            color_spec,
                            x_spec,
                            len(x),
                            root_path,
                            tree_name,
                            int(entry_start),
                            int(entry_stop),
                            None,
                        )
                        arrays.append(color)

                    cut_definitions: list[tuple[np.ndarray, Optional[float], Optional[float]]] = []
                    for spec, bounds in zip(cut_specs, cut_bounds):
                        cut_values, _ = cut_values_for_values(
                            spec,
                            x_spec,
                            len(x),
                            root_path,
                            tree_name,
                            int(entry_start),
                            int(entry_stop),
                            None,
                        )
                        cut_definitions.append((cut_values, bounds[0], bounds[1]))
                    arrays = apply_range_cuts(arrays, cut_definitions)

                    arrays = apply_finite_filter(arrays)
                    idx = 0
                    x = arrays[idx]
                    idx += 1
                    if y_spec is not None:
                        y = arrays[idx]
                        idx += 1
                    if z_spec is not None:
                        z = arrays[idx]
                        idx += 1
                    if weight_branch is not None:
                        weights = arrays[idx]
                        idx += 1
                    if color_spec is not None:
                        color = arrays[idx]
                        idx += 1

                    if plot_type in {"2D scatter", "3D scatter interactive"}:
                        ds = [x]
                        if y_spec is not None:
                            ds.append(y)
                        if z_spec is not None:
                            ds.append(z)
                        if color is not None:
                            ds.append(color)
                        ds = downsample_arrays(ds, int(style["max_points"]))
                        x = ds[0]
                        if y_spec is not None:
                            y = ds[1]
                        if z_spec is not None:
                            z = ds[2]
                        if color is not None:
                            color = ds[-1]

                    rows = [describe_array(labels[0], x)]
                    if y_spec is not None:
                        rows.append(describe_array(labels[1], y))
                    if z_spec is not None:
                        rows.append(describe_array(labels[2], z))
                    if color is not None:
                        rows.append(describe_array(color_label or "color", color))
                    summary_df = pd.DataFrame(rows)

                    xlab = label_controls["custom_x"] or labels[0]
                    ylab = label_controls["custom_y"] or (
                        labels[1] if y_spec is not None else "entries"
                    )
                    zlab = label_controls["custom_z"] or (labels[2] if z_spec is not None else "")
                    title = label_controls["title"] or plot_type
                    save_name = label_controls["save_name"]

                    if plot_type == "3D scatter interactive":
                        cmin = parse_optional_float(style.get("color_min_text", ""))
                        cmax = parse_optional_float(style.get("color_max_text", ""))
                        fig3d = build_plotly_3d(
                            x,
                            y,
                            z,
                            xlab,
                            ylab,
                            zlab,
                            color=color,
                            color_label=color_label,
                            title=title,
                            marker_size=style["marker_size_3d"],
                            opacity=style["opacity_3d"],
                            colorscale=style.get("colorscale", "Viridis"),
                            reversescale=style.get("reverse_colorscale", False),
                            aspectmode=style["aspectmode"],
                            color_min=cmin,
                            color_max=cmax,
                        )
                        fig3d = overlay_geometry_on_plotly(fig3d, geometry_scene, geometry_options)
                        st.session_state["plot_result"] = {
                            "kind": "plotly",
                            "figure": fig3d,
                            "summary": summary_df,
                            "preview": None,
                            "save_name": save_name,
                            "save_button_label": "Save 3D HTML",
                        }
                    else:
                        fig, ax = plt.subplots(figsize=(style["fig_width"], style["fig_height"]))
                        render_static_plot(
                            fig,
                            ax,
                            plot_type,
                            x,
                            y,
                            xlab,
                            ylab,
                            style,
                            title=title,
                            weights=weights,
                            weight_branch=weight_branch,
                            color=color,
                            color_label=color_label,
                        )
                        st.session_state["plot_result"] = {
                            "kind": "matplotlib",
                            "figure": fig,
                            "summary": summary_df,
                            "preview": None,
                            "save_name": save_name,
                            "save_formats": selected_static_formats(label_controls),
                            "save_button_label": "Save selected formats",
                        }
                except ArrayShapeMismatchError as exc:
                    st.session_state.pop("plot_result", None)
                    st.warning(str(exc))
                except Exception as exc:
                    st.session_state.pop("plot_result", None)
                    st.error(f"Plot failed: {exc}")
                    st.exception(exc)

            show_cached_plot_result(
                "plot_result",
                Path(label_controls["save_directory"]),
                save_button_key="plot_save_cached",
                empty_message="Configure variables on the left, then click **Make plot**.",
            )

    with event_tab:
        controls, output = st.columns([0.36, 0.64], gap="large")
        with controls:
            st.subheader("Event inspection")
            if "event_index" not in st.session_state:
                st.session_state["event_index"] = 0
            event_index = st.number_input(
                "event index",
                min_value=0,
                max_value=max(n_entries - 1, 0),
                value=int(st.session_state["event_index"]),
                step=1,
            )
            st.session_state["event_index"] = int(event_index)

            with st.expander("Rank events", expanded=True):
                metric_branch = st.selectbox(
                    "metric branch",
                    branches,
                    index=choose_default(branches, ["pt", "energy", "mass", "x"]),
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
                r1, r2 = st.columns(2)
                with r1:
                    metric_start = st.number_input(
                        "ranking start", min_value=0, max_value=max(n_entries - 1, 0), value=0
                    )
                with r2:
                    metric_stop = st.number_input(
                        "ranking stop",
                        min_value=1,
                        max_value=n_entries,
                        value=min(n_entries, DEFAULT_MAX_ENTRIES),
                    )
                descending = st.checkbox("largest first", value=True)
                max_rank_rows = st.slider("rows", 5, 200, 10)
                if st.button("Compute ranking"):
                    if int(metric_stop) <= int(metric_start):
                        st.error("ranking stop must be larger than ranking start.")
                    else:
                        try:
                            ev, vals = compute_event_metric(
                                root_path,
                                tree_name,
                                metric_branch,
                                metric_mode,
                                int(metric_start),
                                int(metric_stop),
                                threshold,
                                metric_abs,
                            )
                            st.session_state["rank_df"] = metric_dataframe(
                                ev, vals, descending, int(max_rank_rows)
                            )
                            st.session_state["rank_label"] = f"{metric_mode}({metric_branch})"
                        except Exception as exc:
                            st.error(f"Ranking failed: {exc}")
                if "rank_df" in st.session_state:
                    st.write(f"Metric: **{st.session_state.get('rank_label', 'metric')}**")
                    st.dataframe(st.session_state["rank_df"], width="stretch", height=220)
                    events = st.session_state["rank_df"]["event"].astype(int).tolist()
                    if events:
                        selected = st.selectbox(
                            "jump to event", events, format_func=lambda e: f"event {e}"
                        )
                        if st.button("Use selected event"):
                            st.session_state["event_index"] = int(selected)
                            st.rerun()

            st.subheader("Event plot configuration")
            event_plot_type = st.selectbox("Plot type", PLOT_TYPES, index=0, key="event_plot_type")

            st.markdown("### Variables")
            ex_spec = variable_selector(
                "event_x", branches, ["x", "pt", "eta"], allow_event_reductions=False
            )
            ey_spec = ez_spec = event_color_spec = None
            if needs_y_variable(event_plot_type):
                ey_spec = variable_selector(
                    "event_y", branches, ["y", "eta", "phi"], allow_event_reductions=False
                )
            if needs_z_variable(event_plot_type):
                ez_spec = variable_selector(
                    "event_z", branches, ["z", "phi"], allow_event_reductions=False
                )

            event_use_color = False
            if supports_point_color(event_plot_type):
                event_use_color = st.checkbox(
                    "Color points",
                    value=event_plot_type == "3D scatter interactive",
                    key="event_color_points",
                )
                if event_use_color:
                    event_color_spec = variable_selector(
                        "event_color",
                        branches,
                        ["energy", "pt", "charge", "pdgId"],
                        allow_event_reductions=False,
                    )

            event_weight_branch = histogram_weight_selector("event", event_plot_type, branches)
            event_cut_specs, event_cut_bounds = range_cut_controls("event", branches)
            event_style = plot_style_controls(
                "event",
                event_plot_type,
                color_enabled=event_use_color,
                default_bins=50,
                default_marker_size_2d=8.0,
                default_marker_size_3d=4.0,
                default_opacity_2d=0.75,
                default_opacity_3d=0.75,
                default_equal_aspect=False,
            )
            event_label_controls = label_export_controls(
                "event",
                event_plot_type,
                default_title="",
                default_save_name="root_explorer_event_plot",
            )
            make_event_plot = st.button(
                "Make event plot", type="primary", key="event_make", shortcut="Shift+P"
            )

        with output:
            st.subheader(f"Output: event {int(event_index)}")
            if make_event_plot:
                try:
                    validate_variable_event_shapes(
                        [spec for spec in [ex_spec, ey_spec, ez_spec] if spec is not None],
                        root_path,
                        tree_name,
                        int(entry_start),
                        int(entry_stop),
                        int(event_index),
                        context="event plot-axis variables",
                    )
                    arrays: list[np.ndarray] = []
                    labels: list[str] = []
                    ex, ex_label = evaluate_variable(
                        ex_spec,
                        root_path,
                        tree_name,
                        int(entry_start),
                        int(entry_stop),
                        int(event_index),
                    )
                    arrays.append(ex)
                    labels.append(ex_label)
                    if ey_spec is not None:
                        ey, ey_label = evaluate_variable(
                            ey_spec,
                            root_path,
                            tree_name,
                            int(entry_start),
                            int(entry_stop),
                            int(event_index),
                        )
                        arrays.append(ey)
                        labels.append(ey_label)
                    else:
                        ey = None
                        ey_label = ""
                    if ez_spec is not None:
                        ez, ez_label = evaluate_variable(
                            ez_spec,
                            root_path,
                            tree_name,
                            int(entry_start),
                            int(entry_stop),
                            int(event_index),
                        )
                        arrays.append(ez)
                        labels.append(ez_label)
                    else:
                        ez = None
                        ez_label = ""
                    event_weights = None
                    if event_weight_branch is not None:
                        event_weights = histogram_weights_for_values(
                            root_path,
                            tree_name,
                            event_weight_branch,
                            ex_spec,
                            array_length(ex),
                            int(entry_start),
                            int(entry_stop),
                            int(event_index),
                        )
                        arrays.append(event_weights)
                    event_color = None
                    event_color_label = None
                    if event_color_spec is not None:
                        event_color, event_color_label = variable_values_for_values(
                            event_color_spec,
                            ex_spec,
                            array_length(ex),
                            root_path,
                            tree_name,
                            int(entry_start),
                            int(entry_stop),
                            int(event_index),
                        )
                        arrays.append(event_color)
                    event_cut_definitions: list[
                        tuple[np.ndarray, Optional[float], Optional[float]]
                    ] = []
                    for spec, bounds in zip(event_cut_specs, event_cut_bounds):
                        cv, _ = cut_values_for_values(
                            spec,
                            ex_spec,
                            array_length(ex),
                            root_path,
                            tree_name,
                            int(entry_start),
                            int(entry_stop),
                            int(event_index),
                        )
                        event_cut_definitions.append((cv, bounds[0], bounds[1]))
                    arrays = apply_range_cuts(arrays, event_cut_definitions)
                    arrays = apply_finite_filter(arrays)
                    idx = 0
                    ex = arrays[idx]
                    idx += 1
                    if ey_spec is not None:
                        ey = arrays[idx]
                        idx += 1
                    if ez_spec is not None:
                        ez = arrays[idx]
                        idx += 1
                    if event_weight_branch is not None:
                        event_weights = arrays[idx]
                        idx += 1
                    if event_color_spec is not None:
                        event_color = arrays[idx]
                        idx += 1

                    if event_plot_type in {"2D scatter", "3D scatter interactive"}:
                        ds = [ex]
                        if ey_spec is not None:
                            ds.append(ey)
                        if ez_spec is not None:
                            ds.append(ez)
                        if event_color is not None:
                            ds.append(event_color)
                        ds = downsample_arrays(ds, int(event_style["max_points"]))
                        ex = ds[0]
                        if ey_spec is not None:
                            ey = ds[1]
                        if ez_spec is not None:
                            ez = ds[2]
                        if event_color is not None:
                            event_color = ds[-1]

                    rows = [describe_array(labels[0], ex)]
                    if ey_spec is not None:
                        rows.append(describe_array(labels[1], ey))
                    if ez_spec is not None:
                        rows.append(describe_array(labels[2], ez))
                    if event_color is not None:
                        rows.append(describe_array(event_color_label or "color", event_color))
                    summary_df = pd.DataFrame(rows)

                    preview = {"x": ex}
                    if ey_spec is not None:
                        preview["y"] = ey
                    if ez_spec is not None:
                        preview["z"] = ez
                    if event_weights is not None:
                        preview["weights"] = event_weights
                    if event_color is not None:
                        preview["color"] = event_color
                    nprev = min(len(v) for v in preview.values()) if preview else 0
                    preview_df = None
                    if nprev:
                        preview_df = pd.DataFrame({k: v[:nprev] for k, v in preview.items()}).head(
                            500
                        )

                    exlab = event_label_controls["custom_x"] or labels[0]
                    eylab = event_label_controls["custom_y"] or (
                        labels[1] if ey_spec is not None else "entries"
                    )
                    ezlab = event_label_controls["custom_z"] or (
                        labels[2] if ez_spec is not None else ""
                    )
                    event_title = event_label_controls["title"] or f"Event {int(event_index)}"
                    event_save_name = f"{event_label_controls['save_name']}_{int(event_index)}"

                    if event_plot_type == "3D scatter interactive":
                        cmin = parse_optional_float(event_style.get("color_min_text", ""))
                        cmax = parse_optional_float(event_style.get("color_max_text", ""))
                        fig3d = build_plotly_3d(
                            ex,
                            ey,
                            ez,
                            exlab,
                            eylab,
                            ezlab,
                            color=event_color,
                            color_label=event_color_label,
                            title=event_title,
                            marker_size=event_style["marker_size_3d"],
                            opacity=event_style["opacity_3d"],
                            colorscale=event_style.get("colorscale", "Viridis"),
                            reversescale=event_style.get("reverse_colorscale", False),
                            aspectmode=event_style["aspectmode"],
                            color_min=cmin,
                            color_max=cmax,
                        )
                        fig3d = overlay_geometry_on_plotly(fig3d, geometry_scene, geometry_options)
                        st.session_state["event_plot_result"] = {
                            "kind": "plotly",
                            "figure": fig3d,
                            "summary": summary_df,
                            "preview": preview_df,
                            "save_name": event_save_name,
                            "save_button_label": "Save 3D HTML",
                        }
                    else:
                        fig, ax = plt.subplots(
                            figsize=(event_style["fig_width"], event_style["fig_height"])
                        )
                        render_static_plot(
                            fig,
                            ax,
                            event_plot_type,
                            ex,
                            ey,
                            exlab,
                            eylab,
                            event_style,
                            title=event_title,
                            weights=event_weights,
                            weight_branch=event_weight_branch,
                            color=event_color,
                            color_label=event_color_label,
                        )
                        st.session_state["event_plot_result"] = {
                            "kind": "matplotlib",
                            "figure": fig,
                            "summary": summary_df,
                            "preview": preview_df,
                            "save_name": event_save_name,
                            "save_formats": selected_static_formats(event_label_controls),
                            "save_button_label": "Save selected formats",
                        }
                except ArrayShapeMismatchError as exc:
                    st.session_state.pop("event_plot_result", None)
                    st.warning(str(exc))
                except Exception as exc:
                    st.session_state.pop("event_plot_result", None)
                    st.error(f"Event plot failed: {exc}")
                    st.exception(exc)

            show_cached_plot_result(
                "event_plot_result",
                Path(event_label_controls["save_directory"]),
                save_button_key="event_save_cached",
                empty_message="Choose an event and variables, then click **Make event plot**.",
            )
