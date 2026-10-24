"""Regression tests for widget state that must follow the displayed results."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from root_explorer.app.plotting import PlotRequest
from root_explorer.app.results import local_absolute_path, settings_equal
from root_explorer.app.state import DataContext, GeometryState
from root_explorer.models import VariableSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECTS_PATH = PROJECT_ROOT / "tests" / "data" / "test_objects.root"


def _app(path: Path | str) -> AppTest:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20).run()
    return at.text_input(key="root_path").input(str(path)).run()


def _notices(at: AppTest, text: str = "settings have changed") -> list[str]:
    return [info.value for info in at.info if text in info.value]


@dataclass(frozen=True)
class _Settings:
    bounds: tuple[float | None, float | None]
    options: dict[str, list[float]]


def test_settings_equal_treats_nested_nan_as_equal() -> None:
    nan = float("nan")
    assert settings_equal(
        _Settings((float("nan"), 1.0), {"values": [float("nan"), 2.0]}),
        _Settings((float("nan"), 1.0), {"values": [float("nan"), 2.0]}),
    )
    assert not settings_equal(_Settings((nan, 1.0), {}), _Settings((nan, 2.0), {}))
    assert not settings_equal(_Settings((nan, 1.0), {}), _Settings((0.0, 1.0), {}))
    assert not settings_equal({"a": [nan]}, {"a": [nan, nan]})
    assert not settings_equal({"a": nan}, {"b": nan})
    assert not settings_equal((nan,), [nan])
    assert not settings_equal(None, _Settings((None, None), {}))


def _enable_fit(at: AppTest) -> AppTest:
    at.checkbox(key="plot_fit_enabled").check().run()
    return at


@pytest.mark.parametrize("text", ["abc", "nan", "inf"])
def test_invalid_fit_minimum_warns_fails_the_fit_and_clears_the_notice(root_path, text) -> None:
    at = _enable_fit(_app(root_path))
    at.checkbox(key="plot_fit_range_enabled").check().run()
    at.text_input(key="plot_fit_min").input(text).run()
    assert [warning.value for warning in at.warning] == [
        "Fit minimum must be a finite number. Correct it before plotting."
    ]

    at.button(key="plot_make").click().run()
    assert not at.exception
    fit_result = at.session_state["plot_result"]["fit_result"]
    assert not fit_result.success
    assert "Fit range must have finite bounds" in fit_result.message
    # The invalid value is stored as NaN, which must still match the current settings.
    assert not _notices(at)
    at.run()
    assert not _notices(at)


def test_invalid_fit_parameters_warn_and_fail_the_fit(root_path) -> None:
    at = _enable_fit(_app(root_path))
    at.checkbox(key="plot_fit_override").check().run()
    initial = next(widget for widget in at.text_input if widget.key.endswith("_initial"))
    lower = next(widget for widget in at.text_input if widget.key.endswith("_lower"))
    name = initial.key.removeprefix("plot_fit_").removesuffix("_initial").split("_", 1)[1]
    initial.input("1e").run()
    lower = at.text_input(key=lower.key).input("x").run()
    assert {warning.value for warning in at.warning} == {
        f"{name} initial value must be a finite number. Correct it before plotting.",
        f"{name} lower bound must be a number. Correct it before plotting.",
    }

    at.button(key="plot_make").click().run()
    fit_result = at.session_state["plot_result"]["fit_result"]
    assert not fit_result.success
    assert fit_result.message.startswith("Invalid initial value or bounds")
    assert not _notices(at)


def test_infinite_fit_bounds_remain_valid(root_path) -> None:
    at = _enable_fit(_app(root_path))
    at.checkbox(key="plot_fit_override").check().run()
    upper = next(widget for widget in at.text_input if widget.key.endswith("_upper"))
    upper.input("inf").run()
    assert not at.warning


def test_quantile_percentile_displays_the_value_used(root_path) -> None:
    at = _app(root_path)
    at.checkbox(key="plot_distribution_enabled").check().run()
    at.selectbox(key="plot_distribution_mode").select("Quantile").run()
    percentile = at.number_input(key="plot_quantile_percentile")
    assert percentile.format == "%.6g"
    assert percentile.format % 99.95 == "99.95"


def _event_index(at: AppTest, value: int) -> AppTest:
    return next(item for item in at.number_input if item.label == "event index").set_value(value)


def test_event_plot_file_name_follows_the_plotted_event(root_path) -> None:
    at = _app(root_path)
    at.button(key="event_make").click().run()
    assert at.text_input(key="event_file_name").value == "root_explorer_event_plot_0"

    _event_index(at, 3).run()
    at.button(key="event_make").click().run()
    assert at.text_input(key="event_file_name").value == "root_explorer_event_plot_3"

    # A name entered by the user is kept for later events.
    at.text_input(key="event_file_name").input("my_event").run()
    _event_index(at, 5).run()
    at.button(key="event_make").click().run()
    assert at.text_input(key="event_file_name").value == "my_event"


def test_ranking_order_and_rows_update_the_cached_ranking(root_path) -> None:
    at = _app(root_path)
    at.selectbox(key="metric_branch").select("Muon_pt")
    at.selectbox(key="metric_mode").select("sum").run()
    at.button(key="rank_compute").click().run()
    descending = at.session_state["rank_df"]["metric"].tolist()
    assert descending == sorted(descending, reverse=True)

    next(widget for widget in at.checkbox if widget.label == "largest first").uncheck().run()
    ascending = at.session_state["rank_df"]["metric"].tolist()
    assert ascending == sorted(ascending)
    assert ascending != descending
    at.slider(key="rank_rows").set_value(5).run()
    assert len(at.session_state["rank_df"]) == 5
    assert not _notices(at, "metric settings have changed")

    next(widget for widget in at.checkbox if widget.label == "use absolute values").check().run()
    assert _notices(at, "metric settings have changed") == [
        "The metric settings have changed since this ranking was computed. "
        "Click **Compute ranking** to update it."
    ]
    at.button(key="rank_compute").click().run()
    assert at.session_state["rank_label"] == "sum(|Muon_pt|)"
    assert not _notices(at, "metric settings have changed")


def test_absolute_values_are_disabled_for_multiplicity(root_path) -> None:
    at = _app(root_path)
    at.selectbox(key="metric_mode").select("sum").run()
    absolute = next(widget for widget in at.checkbox if widget.label == "use absolute values")
    absolute.check().run()
    at.selectbox(key="metric_mode").select("multiplicity").run()
    absolute = next(widget for widget in at.checkbox if widget.label == "use absolute values")
    assert absolute.disabled
    at.button(key="rank_compute").click().run()
    assert at.session_state["rank_label"] == f"multiplicity({at.session_state['metric_branch']})"


def test_object_selector_keeps_the_opened_object_in_the_compact_layout() -> None:
    at = _app(OBJECTS_PATH)
    selector = next(item for item in at.selectbox if item.label == "Object to inspect")
    selector.select("plain_graph (TGraph)").run()
    at.button(key="root_object_open").click().run()
    assert at.session_state["root_inspector_open"][1] == "plain_graph;1"

    selector = next(item for item in at.selectbox if item.label == "Object to inspect")
    assert selector.value == "plain_graph;1"
    assert any("Ready to open: `plain_graph;1`" in caption.value for caption in at.caption)


def test_invalid_entry_range_keeps_root_objects_available(root_path) -> None:
    at = _app(root_path)
    next(item for item in at.number_input if item.label == "entry start").set_value(5).run()
    next(item for item in at.number_input if item.label == "entry stop").set_value(5).run()

    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["ROOT objects", "Geometry"]
    assert [warning.value for warning in at.sidebar.warning] == [
        "entry stop must be larger than entry start. "
        "You can still browse the other ROOT objects in this file."
    ]


def test_event_plot_shortcut_is_only_active_in_its_tab(root_path, monkeypatch) -> None:
    shortcuts: dict[str, object] = {}
    button = st.button

    def recording_button(*args, **kwargs):
        shortcuts[kwargs.get("key")] = kwargs.get("shortcut")
        return button(*args, **kwargs)

    monkeypatch.setattr(st, "button", recording_button)
    _app(root_path)
    # The Plotter tab is open initially.
    assert shortcuts["plot_make"] == "P"
    assert shortcuts["event_make"] is None


def _request(root_path: str) -> PlotRequest:
    context = DataContext(root_path, "Events", 1, 0, 1, [], ["x"])
    return PlotRequest(
        context=context,
        plot_type="1D histogram",
        x_spec=VariableSpec("direct", "x"),
        y_spec=None,
        z_spec=None,
        color_spec=None,
        weight_branch=None,
        cut_specs=[],
        cut_bounds=[],
        style={},
        labels={},
        geometry=GeometryState(scene=None, options={}),
        event_index=None,
        preview=False,
        file_name="plot",
    )


def test_exported_plot_spec_uses_absolute_local_paths() -> None:
    assert _request("data.root").to_static_spec().root_path == os.path.abspath("data.root")
    remote = "root://eos.example.org//store/data.root"
    assert _request(remote).to_static_spec().root_path == remote
    assert local_absolute_path("https://example.org/data.root") == "https://example.org/data.root"


def test_persisted_graph_code_uses_an_absolute_path(monkeypatch) -> None:
    monkeypatch.chdir(OBJECTS_PATH.parent)
    at = _app(OBJECTS_PATH.name)
    selector = next(item for item in at.selectbox if item.label == "Object to inspect")
    selector.select("plain_graph (TGraph)").run()
    at.button(key="root_object_open").click().run()
    assert not at.exception
    code = at.session_state["root_graph_result"]["code"]
    assert repr(str(OBJECTS_PATH)) in code


def test_stored_variance_default_follows_later_weight_changes(root_path) -> None:
    at = _app(root_path)
    at.checkbox(key="plot_fit_enabled").check().run()
    assert at.checkbox(key="plot_fit_variances").value is False

    at.checkbox(key="plot_use_weights").check().run()
    assert at.checkbox(key="plot_fit_variances").value is True
    at.checkbox(key="plot_use_weights").uncheck().run()
    assert at.checkbox(key="plot_fit_variances").value is False

    # An explicit choice is kept when the weight selection changes.
    at.checkbox(key="plot_fit_variances").check().run()
    at.checkbox(key="plot_use_weights").check().run()
    at.checkbox(key="plot_fit_variances").uncheck().run()
    at.checkbox(key="plot_use_weights").uncheck().run()
    assert at.checkbox(key="plot_fit_variances").value is False
    assert not at.exception
    assert not any("Session State API" in warning.value for warning in at.warning)
