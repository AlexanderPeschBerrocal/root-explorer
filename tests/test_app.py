from __future__ import annotations

from pathlib import Path

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
        "Branches",
        "Geometry",
        "Help",
    ]
    assert len(at.dataframe) >= 2
    assert at.selectbox(key="x_mode").options[0] == "Direct branch values"
    assert at.selectbox(key="x_mode").options[-3:] == ["delta phi", "delta eta", "delta R"]
    assert at.selectbox(key="x_branch").label == "x: branch"
    at.selectbox(key="x_branch").select("Muon_pt").run()
    assert at.selectbox(key="x_branch").value == "Muon_pt"

    at.button(key="plot_make").click().run()
    assert not at.exception
    at.text_input(key="plot_save_name").input("muon_pt").run()
    assert not at.exception
    result = at.session_state["plot_result"]
    assert result["kind"] == "matplotlib"
    assert result["summary"].iloc[0]["n"] > 0
    assert at.text_input(key="plot_save_name").value == "muon_pt"
    assert any("Make plot button again or press P" in caption.value for caption in at.caption)
    assert result["code"].startswith('"""Standalone plot exported by ROOT Explorer.')
    code_download = next(
        button
        for button in at.get("download_button")
        if button.key == "plot_save_cached_download_code"
    )
    assert code_download.label == "Download plot code"


def test_app_labels_weighted_histogram_with_weight_sum(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)
    at.selectbox(key="x_branch").select("Muon_pt").run()
    at.checkbox(key="plot_use_weights").check().run()
    at.button(key="plot_make").click().run()

    assert not at.exception
    axes = at.session_state["plot_result"]["figure"].axes[0]
    assert axes.get_ylabel() == "sum weights (genWeight)"


def test_app_loads_rntuple_and_makes_static_plot(rntuple_path) -> None:
    at = _load_root_file(make_app_test(), rntuple_path)

    assert not at.exception
    assert (
        next(widget for widget in at.selectbox if widget.label == "Event dataset").value
        == "Events"
    )
    assert at.selectbox(key="x_mode").options[0] == "Direct field values"
    assert at.selectbox(key="x_branch").label == "x: field"
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
    assert at.selectbox(key="x_mode").options[0] == "Direct field values"
    assert at.selectbox(key="x_branch").label == "x: field"

    container = next(widget for widget in at.selectbox if widget.label == "Event dataset")
    container.select("trees/Events").run()
    assert at.selectbox(key="x_mode").options[0] == "Direct branch values"
    assert at.selectbox(key="x_branch").label == "x: branch"


def test_app_ranks_events_and_makes_3d_event_plot(root_path) -> None:
    at = _load_root_file(make_app_test(), root_path)
    at.button(key="rank_compute").click().run()
    assert not at.exception
    assert not at.session_state["rank_df"].empty

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
        "Make event plot button again or press Shift + P" in caption.value
        for caption in at.caption
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
