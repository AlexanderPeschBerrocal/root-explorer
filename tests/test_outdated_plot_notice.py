"""A notice appears only when the displayed plot no longer matches the settings (#90)."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECTS_PATH = PROJECT_ROOT / "tests" / "data" / "test_objects.root"


def _app(path: Path) -> AppTest:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20).run()
    return at.text_input(key="root_path").input(str(path)).run()


def _notices(at: AppTest) -> list[str]:
    return [info.value for info in at.info if "settings have changed" in info.value]


def test_plotter_notice_follows_the_settings(root_path) -> None:
    at = _app(root_path)
    at.button(key="plot_make").click().run()
    assert not _notices(at)

    # Settings in a collapsed section count as well.
    at.checkbox(key="plot_log_y").check().run()
    assert _notices(at) == [
        "The settings have changed since this plot was made. "
        "Click **Make plot** or **press P** to update it."
    ]

    at.button(key="plot_make").click().run()
    assert not _notices(at)

    # Returning to the settings of the displayed plot removes the notice.
    at.checkbox(key="plot_log_y").uncheck().run()
    assert _notices(at)
    at.checkbox(key="plot_log_y").check().run()
    assert not _notices(at)


def test_event_plot_is_outdated_for_another_event(root_path) -> None:
    at = _app(root_path)
    at.button(key="event_make").click().run()
    assert not _notices(at)

    next(item for item in at.number_input if item.label == "event index").set_value(3).run()
    assert _notices(at) == [
        "The settings have changed since this plot was made. "
        "Click **Make event plot** or **press Shift + P** to update it."
    ]


def test_geometry_settings_do_not_outdate_static_plots(root_path, tiny_gdml_path) -> None:
    at = _app(root_path)
    at.button(key="plot_make").click().run()
    at.checkbox(key="geometry_enabled").check().run()
    at.text_input(key="geometry_path").input(str(tiny_gdml_path)).run()
    assert not at.exception
    assert not _notices(at)


def test_persisted_viewers_report_outdated_plots() -> None:
    at = _app(OBJECTS_PATH)
    selector = next(item for item in at.selectbox if item.label == "Object to inspect")
    selector.select("histogram_1d (TH1D)").run()
    at.button(key="root_object_open").click().run()
    assert "root_histogram_result" in at.session_state
    assert not _notices(at)

    label = next(widget for widget in at.text_input if widget.label == "X label")
    label.input("Changed").run()
    assert _notices(at)
    at.button(key="hist_make_plot").click().run()
    assert not _notices(at)
