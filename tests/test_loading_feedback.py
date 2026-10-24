"""Tests for progress feedback while plots and data are loading (#133)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from root_explorer.analysis import ranking
from root_explorer.app import graph_viewer, histogram_viewer
from root_explorer.geometry import overlay
from root_explorer.io import root as root_io

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECTS = PROJECT_ROOT / "tests" / "data" / "test_objects.root"
CACHED_FUNCTIONS = [
    root_io._cached_get_root_catalogue,
    root_io._cached_get_entry_count,
    root_io._cached_get_field_info,
    root_io._cached_read_field,
    root_io._cached_read_event_range,
    ranking._cached_event_metric,
    overlay._cached_geometry_scene,
    histogram_viewer._cached_histogram,
    graph_viewer._cached_graph,
]


@pytest.fixture
def spinner_messages(monkeypatch) -> list[str]:
    """Record the text of every ``st.spinner`` used while the app runs."""

    messages: list[str] = []

    @contextmanager
    def recording_spinner(text: str = "", *args, **kwargs):
        messages.append(text)
        yield

    monkeypatch.setattr("streamlit.spinner", recording_spinner)
    return messages


def _plot_steps(messages: list[str]) -> list[str]:
    """Drop the sidebar input spinners, which every rerun passes through.

    Those reads are cached and finish instantly, so Streamlit's spinner delay keeps
    them invisible; only the plot-generation steps matter here.
    """

    return [
        message
        for message in messages
        if message != "Reading the ROOT file catalogue…" and "of the selected" not in message
    ]


def _app(path: Path) -> AppTest:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20).run()
    return at.text_input(key="root_path").input(str(path)).run()


@pytest.mark.parametrize("function", CACHED_FUNCTIONS, ids=lambda function: function.__name__)
def test_cache_spinners_never_expose_internal_function_names(function) -> None:
    text = function._info.show_spinner
    assert text is False or (isinstance(text, str) and "(" not in text and "_" not in text)


def test_opening_a_file_reports_catalogue_and_field_reading(root_path, spinner_messages) -> None:
    _app(root_path)
    assert "Reading the ROOT file catalogue…" in spinner_messages
    assert "Reading branches of the selected TTree…" in spinner_messages


def test_static_plot_reports_each_step_in_order(root_path, spinner_messages) -> None:
    at = _app(root_path)
    at.checkbox(key="plot_fit_enabled").check().run()
    spinner_messages.clear()
    at.button(key="plot_make").click().run()
    assert not at.exception
    assert _plot_steps(spinner_messages) == [
        "Reading and selecting data…",
        "Fitting histogram…",
        "Rendering plot…",
    ]


def test_fit_step_is_only_reported_when_a_fit_is_enabled(root_path, spinner_messages) -> None:
    at = _app(root_path)
    spinner_messages.clear()
    at.button(key="plot_make").click().run()
    assert _plot_steps(spinner_messages) == ["Reading and selecting data…", "Rendering plot…"]


def test_event_inspection_plot_reports_progress(root_path, spinner_messages) -> None:
    at = _app(root_path)
    spinner_messages.clear()
    at.button(key="event_make").click().run()
    assert not at.exception
    assert "Reading and selecting data…" in spinner_messages
    assert "Rendering plot…" in spinner_messages


def test_3d_scatter_reports_progress(root_path, spinner_messages) -> None:
    at = _app(root_path)
    at.selectbox(key="plot_type").select("3D scatter interactive").run()
    spinner_messages.clear()
    at.button(key="plot_make").click().run()
    assert not at.exception
    assert _plot_steps(spinner_messages) == ["Reading data and building the 3D plot…"]


def test_failed_rendering_shows_warning_without_leftover_indicator(root_path) -> None:
    at = _app(root_path)
    at.text_input(key="plot_custom_x").input(r"$\pt$").run()
    at.button(key="plot_make").click().run()
    assert not at.exception
    assert any("invalid math" in warning.value for warning in at.warning)
    assert "plot_result" not in at.session_state
    # A working plot afterwards succeeds normally.
    at.text_input(key="plot_custom_x").input("pT").run()
    at.button(key="plot_make").click().run()
    assert at.session_state["plot_result"]["kind"] == "matplotlib"


@pytest.mark.parametrize(
    ("option", "make_key"),
    [("histogram_1d (TH1D)", "hist_make_plot"), ("plain_graph (TGraph)", "graph_make_plot")],
)
def test_persisted_viewers_report_rendering(option, make_key, spinner_messages) -> None:
    at = _app(OBJECTS)
    next(item for item in at.selectbox if item.label == "Object to inspect").select(option).run()
    at.button(key="root_object_open").click().run()
    spinner_messages.clear()
    at.button(key=make_key).click().run()
    assert not at.exception
    assert _plot_steps(spinner_messages) == ["Rendering plot…"]


def test_failure_inside_a_step_leaves_no_running_indicator(root_path, spinner_messages) -> None:
    at = _app(root_path)
    # Reversed axis limits are rejected while rendering, inside the "Rendering plot…" step.
    at.checkbox(key="plot_manual_axis_limits").check().run()
    at.text_input(key="plot_x_min").input("50").run()
    at.text_input(key="plot_x_max").input("10").run()
    spinner_messages.clear()
    at.button(key="plot_make").click().run()
    assert "Rendering plot…" in _plot_steps(spinner_messages)
    assert not at.exception
    assert at.warning
    assert "plot_result" not in at.session_state
    assert not at.get("spinner")


def test_ranking_and_geometry_describe_their_work() -> None:
    assert ranking._cached_event_metric._info.show_spinner == "Computing event ranking…"
    assert overlay._cached_geometry_scene._info.show_spinner == "Loading GDML geometry…"
