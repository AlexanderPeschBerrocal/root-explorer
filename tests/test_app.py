from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import uproot
from streamlit.testing.v1 import AppTest

from root_explorer.app.main import APP_ICON

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_app_test() -> AppTest:
    return AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=10)


def test_app_starts_without_input() -> None:
    at = make_app_test().run()

    assert not at.exception
    assert APP_ICON.is_file()
    assert at.title[0].value == "ROOT Explorer"
    assert at.text_input(key="root_path").label == "ROOT file path"


def test_app_shows_invalid_path_error() -> None:
    at = make_app_test().run()
    at.text_input(key="root_path").input("/not/a/real/file.root").run()

    assert not at.exception
    assert at.error[0].value == "File does not exist: /not/a/real/file.root"


def _load_root_file(at: AppTest, root_path) -> AppTest:
    at.run()
    return at.text_input(key="root_path").input(str(root_path)).run()


def test_app_loads_root_file_and_makes_static_plot(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)

    assert not at.exception
    assert [tab.label for tab in at.tabs] == [
        "Plotter",
        "Event inspection",
        "Branches and objects",
        "Geometry",
        "Help",
    ]
    assert len(at.dataframe) >= 2
    assert at.selectbox(key="x_mode").options == [
        "Direct branch values: A",
        "|A|",
        "Multiplicity per event",
        "nth object per event",
        "Sum per event",
        "Mean per event",
        "Max per event",
        "Min per event",
        "A + B",
        "A - B",
        "A / B",
        "A * B",
        "sqrt(A² + B² + C²)",
        "r = sqrt(x² + y²)",
        "phi = atan2(y, x)",
        "eta = asinh(z / sqrt(x²+y²))",
        "Energy (massless)",
        "Energy (with mass)",
        "Invariant mass (single particle)",
        "delta phi",
        "delta eta",
        "delta R",
    ]
    assert at.selectbox(key="x_branch").label == "x: branch"
    assert "Muon_pt" in at.selectbox(key="x_branch").options
    assert not any("[jagged]" in option for option in at.selectbox(key="x_branch").options)
    assert at.toggle(key="plot_show_field_details").label == "Show branch details in selectors"
    assert not at.toggle(key="plot_show_field_details").value
    assert at.toggle(key="event_show_field_details").label == "Show branch details in selectors"
    assert not at.toggle(key="event_show_field_details").value
    branch_table = next(
        dataframe.value
        for dataframe in at.dataframe
        if {"structure", "dtype"}.issubset(dataframe.value.columns)
    )
    assert {"structure", "dtype"}.issubset(branch_table.columns)
    initial_dataframes = len(at.dataframe)
    at.selectbox(key="x_branch").select("Muon_pt").run()
    assert at.selectbox(key="x_branch").value == "Muon_pt"

    at.button(key="plot_make").click().run()
    assert not at.exception
    at.text_input(key="plot_save_name").input("muon_pt").run()
    assert not at.exception
    result = at.session_state["plot_result"]
    assert len(at.dataframe) > initial_dataframes
    assert result["kind"] == "matplotlib"
    assert result["summary"].iloc[0]["n"] > 0
    assert result["figure"].dpi == 200
    assert at.text_input(key="plot_save_name").value == "muon_pt"
    assert at.selectbox(key="plot_export_dpi").value == 300
    assert at.selectbox(key="plot_export_dpi").options == ["150 DPI", "300 DPI", "600 DPI"]
    assert any("Make plot button again or press P" in caption.value for caption in at.caption)
    assert result["code"].startswith('"""Standalone plot exported by ROOT Explorer.')
    code_download = next(
        button
        for button in at.get("download_button")
        if button.key == "plot_save_cached_download_code"
    )
    assert code_download.label == "Download plot code"


def test_root_object_browser_shows_nontrivial_catalogue(catalogue_root_path) -> None:
    at = _load_root_file(make_app_test(), catalogue_root_path)

    assert not at.exception
    assert "Branches and objects" in [tab.label for tab in at.tabs]
    table = next(
        dataframe.value
        for dataframe in at.dataframe
        if {"ROOT class", "kind", "cycle", "path"}.issubset(dataframe.value.columns)
    )
    assert {
        "Events",
        "Analysis",
        "Analysis/nested",
        "Analysis/nested/Events",
        "Analysis/h_mass",
        "Control/h_mass",
        "Analysis/repeated",
        "metadata",
    }.issubset(set(table["path"]))
    assert table.loc[table["path"] == "Analysis/repeated", "cycle"].item() == 2

    object_table_index = next(
        i
        for i, dataframe in enumerate(at.dataframe)
        if {"ROOT class", "kind", "cycle", "path"}.issubset(dataframe.value.columns)
    )
    field_table_index = next(
        i
        for i, dataframe in enumerate(at.dataframe)
        if {"structure", "dtype"}.issubset(dataframe.value.columns)
    )
    assert object_table_index < field_table_index

    at.toggle(key="root_object_show_all_cycles").set_value(True).run()
    table = next(
        dataframe.value
        for dataframe in at.dataframe
        if {"ROOT class", "kind", "cycle", "path"}.issubset(dataframe.value.columns)
    )
    repeated = table[table["path"] == "Analysis/repeated"]
    assert repeated["cycle"].tolist() == [1, 2]


def test_object_only_root_file_remains_browseable(object_only_root_path) -> None:
    at = _load_root_file(make_app_test(), object_only_root_path)

    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["ROOT objects", "Geometry"]
    assert any(
        "No TTrees or RNTuples containing event data were found" in info.value for info in at.info
    )
    table = next(
        dataframe.value
        for dataframe in at.dataframe
        if {"ROOT class", "kind", "cycle", "path"}.issubset(dataframe.value.columns)
    )
    assert set(table["path"]) >= {"Analysis", "Analysis/h_values", "Analysis/note"}
    assert "histogram" in set(table["kind"])
    assert "other" in set(table["kind"])


def test_new_single_particle_variable_controls_are_available(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)
    mode = at.selectbox(key="x_mode")

    assert mode.options[:8] == [
        "Direct branch values: A",
        "|A|",
        "Multiplicity per event",
        "nth object per event",
        "Sum per event",
        "Mean per event",
        "Max per event",
        "Min per event",
    ]
    assert {
        "|A|",
        "sqrt(A² + B² + C²)",
        "Energy (massless)",
        "Energy (with mass)",
        "Invariant mass (single particle)",
    }.issubset(mode.options)

    mode.select("Energy (with mass)").run()
    coordinate = at.selectbox(key="x_longitudinal_kind")
    assert coordinate.options == ["eta (pseudorapidity)", "y (rapidity)"]
    coordinate.select("y (rapidity)").run()
    assert at.selectbox(key="x_longitudinal").label == "x: y"


def test_field_detail_toggle_applies_badges_to_all_field_selectors(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)
    at.toggle(key="plot_show_field_details").set_value(True).run()

    badge = "Muon_pt  [jagged] [float32]"
    assert badge in at.selectbox(key="x_branch").options

    at.checkbox(key="plot_use_weights").check().run()
    assert "genWeight  [scalar] [float32]" in at.selectbox(key="plot_weight_branch").options

    at.number_input(key="plot_n_cuts").set_value(1).run()
    assert badge in at.selectbox(key="plot_cut_0_branch").options

    at.selectbox(key="plot_type").select("2D scatter").run()
    at.checkbox(key="plot_color_points").check().run()
    assert badge in at.selectbox(key="color_branch").options

    assert not any("[jagged]" in option for option in at.selectbox(key="event_x_branch").options)
    at.toggle(key="event_show_field_details").set_value(True).run()
    assert badge in at.selectbox(key="event_x_branch").options
    assert badge in at.selectbox(key="metric_branch").options


def test_event_reduction_explains_the_resulting_shape(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)
    at.toggle(key="plot_show_field_details").set_value(True).run()
    at.selectbox(key="x_mode").select("Sum per event").run()
    at.selectbox(key="x_branch").select("Muon_pt").run()

    assert any(
        "one event-level scalar per event from [jagged] [float32] input" in caption.value
        for caption in at.caption
    )


def test_app_labels_weighted_histogram_with_weight_sum(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)
    at.toggle(key="plot_show_field_details").set_value(True).run()
    at.selectbox(key="x_branch").select("Muon_pt").run()
    at.checkbox(key="plot_use_weights").check().run()
    assert "genWeight  [scalar] [float32]" in at.selectbox(key="plot_weight_branch").options
    at.button(key="plot_make").click().run()

    assert not at.exception
    axes = at.session_state["plot_result"]["figure"].axes[0]
    assert axes.get_ylabel() == "sum weights (genWeight)"


def test_histogram_kde_controls_offer_all_bandwidth_modes(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)

    at.checkbox(key="plot_show_kde").check().run()
    bandwidth = at.selectbox(key="plot_kde_bandwidth")
    assert bandwidth.options == ["Scott", "Silverman", "Custom factor"]
    bandwidth.select("Custom factor").run()
    assert at.number_input(key="plot_kde_custom_factor").value == 1.0

    at.selectbox(key="plot_type").select("2D histogram").run()
    assert at.checkbox(key="plot_show_kde").label == "Show KDE"


def test_scatter_correlations_use_full_filtered_sample_before_downsampling(tmp_path) -> None:
    root_path = tmp_path / "correlations.root"
    x = np.linspace(-5.0, 5.0, 3_000)
    y = x**3
    y[-1] = np.nan
    with uproot.recreate(root_path) as root_file:
        root_file.mktree("Events", {"x": x, "y": y})

    at = _load_root_file(make_app_test(), root_path)
    initial_dataframes = len(at.dataframe)
    at.selectbox(key="plot_type").select("2D scatter").run()
    at.selectbox(key="x_branch").select("x")
    at.selectbox(key="y_branch").select("y")
    at.number_input(key="plot_n_cuts").set_value(1).run()
    at.selectbox(key="plot_cut_0_branch").select("x")
    at.text_input(key="plot_cut_0_min").input("0").run()
    at.number_input(key="plot_max_points_2d").set_value(1_000).run()
    at.button(key="plot_make").click().run()

    limited = at.session_state["plot_result"]
    limited_correlation = limited["correlation_result"]
    assert limited_correlation.n_pairs == 1_499
    assert limited["summary"].iloc[0]["n"] == 1_499
    assert len(limited["figure"].axes[0].collections[0].get_offsets()) == 1_000
    assert len(at.dataframe) >= initial_dataframes + 2
    assert any(markdown.value == "#### Correlations" for markdown in at.markdown)

    at.number_input(key="plot_max_points_2d").set_value(2_000).run()
    at.button(key="plot_make").click().run()
    full = at.session_state["plot_result"]
    full_correlation = full["correlation_result"]

    assert len(full["figure"].axes[0].collections[0].get_offsets()) == 1_499
    assert full_correlation.pearson_r == pytest.approx(limited_correlation.pearson_r)
    assert full_correlation.spearman_rho == pytest.approx(limited_correlation.spearman_rho)


def test_app_warns_for_invalid_logarithmic_axis(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)
    at.selectbox(key="x_branch").select("Muon_eta").run()
    at.checkbox(key="plot_log_x").check().run()
    at.button(key="plot_make").click().run()

    assert not at.exception
    assert "plot_result" not in at.session_state
    assert any("strictly positive" in warning.value for warning in at.warning)


def test_app_applies_and_validates_explicit_axis_limits(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)
    at.selectbox(key="x_branch").select("Muon_pt").run()
    assert not any(widget.key == "plot_x_min" for widget in at.text_input)
    at.checkbox(key="plot_manual_axis_limits").check().run()
    at.text_input(key="plot_x_min").input("10").run()
    at.text_input(key="plot_x_max").input("80").run()
    at.button(key="plot_make").click().run()

    assert not at.exception
    assert at.session_state["plot_result"]["figure"].axes[0].get_xlim() == pytest.approx(
        (10.0, 80.0)
    )

    at.text_input(key="plot_x_min").input("100").run()
    at.text_input(key="plot_x_max").input("10").run()
    at.button(key="plot_make").click().run()

    assert not at.exception
    assert "plot_result" not in at.session_state
    assert any("minimum must be smaller" in warning.value for warning in at.warning)


def test_app_calculates_threshold_fractions_with_optional_weights(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)
    initial_dataframes = len(at.dataframe)
    at.selectbox(key="x_branch").select("Muon_pt").run()
    at.checkbox(key="plot_distribution_enabled").check().run()
    at.selectbox(key="plot_distribution_mode").select("Threshold fractions").run()
    at.number_input(key="plot_threshold").set_value(50.0).run()
    at.button(key="plot_make").click().run()

    result = at.session_state["plot_result"]
    threshold = result["threshold_result"]
    assert threshold is not None
    assert not threshold.weighted
    assert threshold.below_count + threshold.above_count == threshold.total_count
    assert threshold.total_count == result["summary"].iloc[0]["n"]
    assert threshold.below_fraction + threshold.above_fraction == pytest.approx(1.0)
    assert len(at.dataframe) >= initial_dataframes + 2

    at.number_input(key="plot_n_cuts").set_value(1).run()
    at.selectbox(key="plot_cut_0_branch").select("Muon_pt").run()
    at.text_input(key="plot_cut_0_min").input("40").run()
    at.button(key="plot_make").click().run()
    cut_result = at.session_state["plot_result"]
    cut_threshold = cut_result["threshold_result"]
    assert cut_threshold.total_count == cut_result["summary"].iloc[0]["n"]
    assert cut_threshold.total_count < threshold.total_count

    at.checkbox(key="plot_use_weights").check().run()
    at.button(key="plot_make").click().run()
    weighted = at.session_state["plot_result"]["threshold_result"]
    assert weighted.weighted
    assert weighted.total_weight == pytest.approx(weighted.below_weight + weighted.above_weight)


def test_app_loads_rntuple_and_makes_static_plot(rntuple_path) -> None:
    at = _load_root_file(make_app_test(), rntuple_path)

    assert not at.exception
    assert (
        next(widget for widget in at.selectbox if widget.label == "Event dataset").value == "Events"
    )
    assert "Fields and objects" in [tab.label for tab in at.tabs]
    assert at.selectbox(key="x_mode").options[0] == "Direct field values: A"
    assert at.selectbox(key="x_branch").label == "x: field"
    assert at.toggle(key="plot_show_field_details").label == "Show field details in selectors"
    assert at.selectbox(key="x_branch").options.count("Muon_pt") == 1
    at.toggle(key="plot_show_field_details").set_value(True).run()
    assert "Muon_pt  [jagged] [float32]" in at.selectbox(key="x_branch").options
    assert "event  [scalar] [uint64]" in at.selectbox(key="x_branch").options
    at.selectbox(key="x_branch").select("Muon_pt").run()
    at.button(key="plot_make").click().run()

    assert not at.exception
    result = at.session_state["plot_result"]
    assert result["kind"] == "matplotlib"
    assert result["summary"].iloc[0]["n"] > 0


def test_app_updates_data_item_terminology_for_the_selected_container(
    mixed_event_containers_path,
) -> None:
    at = _load_root_file(make_app_test(), mixed_event_containers_path)
    container = next(widget for widget in at.selectbox if widget.label == "Event dataset")

    container.select("ntuples/Events").run()
    assert at.selectbox(key="x_mode").options[0] == "Direct field values: A"
    assert at.selectbox(key="x_branch").label == "x: field"

    container = next(widget for widget in at.selectbox if widget.label == "Event dataset")
    container.select("trees/Events").run()
    assert at.selectbox(key="x_mode").options[0] == "Direct branch values: A"
    assert at.selectbox(key="x_branch").label == "x: branch"


def test_app_ranks_events_and_makes_3d_event_plot(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)
    initial_dataframes = len(at.dataframe)
    at.button(key="rank_compute").click().run()
    assert not at.exception
    assert not at.session_state["rank_df"].empty
    assert len(at.dataframe) > initial_dataframes

    at.selectbox(key="event_plot_type").select("3D scatter interactive").run()
    at.selectbox(key="event_x_branch").select("Muon_px")
    at.selectbox(key="event_y_branch").select("Muon_py")
    at.selectbox(key="event_z_branch").select("Muon_pz").run()
    at.button(key="event_make").click().run()

    assert not at.exception
    result = at.session_state["event_plot_result"]
    assert result["kind"] == "plotly"
    assert result["figure"].data[0].type == "scatter3d"
    assert any(
        "Make event plot button again or press Shift + P" in caption.value for caption in at.caption
    )


def test_ranking_selection_updates_event_index(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)
    at.button(key="rank_compute").click().run()

    ranked_events = at.session_state["rank_df"]["event"].astype(int).tolist()
    selected = ranked_events[-1]
    next(widget for widget in at.selectbox if widget.label == "jump to event").select(selected)
    next(widget for widget in at.button if widget.label == "Use selected event").click().run()

    assert not at.exception
    assert at.session_state["event_index"] == selected


def test_app_loads_geometry_path(tiny_gdml_path, root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)
    at.checkbox(key="geometry_enabled").check().run()
    at.text_input(key="geometry_path").input(str(tiny_gdml_path)).run()

    assert not at.exception
    assert any("Loaded geometry" in success.value for success in at.success)


def test_app_calculates_quantile_and_marks_histogram(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)
    initial_dataframes = len(at.dataframe)
    at.selectbox(key="x_branch").select("Muon_pt").run()
    at.checkbox(key="plot_distribution_enabled").check().run()
    at.selectbox(key="plot_distribution_mode").select("Quantile").run()
    at.number_input(key="plot_quantile_percentile").set_value(50.0).run()
    at.button(key="plot_make").click().run()

    result = at.session_state["plot_result"]
    quantile = result["quantile_result"]
    assert quantile is not None
    assert quantile.value is not None
    assert len(at.dataframe) >= initial_dataframes + 2
    axes = result["figure"].axes[0]
    assert axes.lines[-1].get_xdata()[0] == pytest.approx(quantile.value)


def test_2d_scatter_offers_log_color_scale(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)
    at.selectbox(key="plot_type").select("2D scatter").run()
    at.checkbox(key="plot_color_points").check().run()
    assert at.checkbox(key="plot_log_color_2d").label == "log color scale"
