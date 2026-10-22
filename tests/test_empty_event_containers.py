"""Integration tests for files whose selected event container has no entries (#138)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import uproot
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_FIXTURE = PROJECT_ROOT / "tests" / "data" / "test_objects.root"
EMPTY_WARNING = "has no entries to plot or inspect"
OBJECT_TABS = ["ROOT objects", "Geometry"]


def _app(root_path: Path) -> AppTest:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=10).run()
    return at.text_input(key="root_path").input(str(root_path)).run()


def _histogram() -> tuple[np.ndarray, np.ndarray]:
    return np.histogram([1.0, 2.0, 2.5, 3.0], bins=3, range=(0.0, 4.0))


def _write(path: Path, **objects) -> Path:
    """Write ``objects``; the string values ``"empty tree"``, ``"empty ntuple"`` and
    ``"filled tree"`` create the corresponding event containers."""

    with uproot.recreate(path) as root_file:
        for name, value in objects.items():
            if value == "empty tree":
                root_file.mktree(name, {"x": "f8"})
            elif value == "empty ntuple":
                root_file.mkrntuple(name, {"x": "float64"})
            elif value == "filled tree":
                # Assigning a dict would create an RNTuple; build a real TTree explicitly.
                root_file.mktree(name, {"x": "f8"}).extend({"x": np.arange(5, dtype=float)})
            else:
                root_file[name] = value
    return path


def _assert_object_mode(at: AppTest) -> None:
    assert not at.exception
    assert any(EMPTY_WARNING in warning.value for warning in at.sidebar.warning)
    assert [tab.label for tab in at.tabs] == OBJECT_TABS
    # No event range, event index, ranking or event-plot controls for an empty container.
    assert not any(
        widget.label in {"entry start", "entry stop", "event index"} for widget in at.number_input
    )
    assert not any(
        widget.key in {"plot_make", "event_make", "rank_compute"} for widget in at.button
    )


def _open_object(at: AppTest, option: str) -> AppTest:
    selector = next(item for item in at.selectbox if item.label == "Object to inspect")
    selector.select(option).run()
    return at.button(key="root_object_open").click().run()


@pytest.mark.parametrize(
    ("container", "kind"), [("empty tree", "TTree"), ("empty ntuple", "RNTuple")]
)
def test_empty_container_keeps_persisted_histogram_available(tmp_path, container, kind) -> None:
    path = _write(tmp_path / "empty.root", Events=container, histogram=_histogram())
    at = _app(path)
    _assert_object_mode(at)
    assert any(f"selected {kind}" in warning.value for warning in at.sidebar.warning)

    _open_object(at, "histogram (TH1D)")
    assert not at.exception
    result = at.session_state["root_histogram_result"]
    assert result["kind"] == "matplotlib"
    assert any(button.label == "Download PNG" for button in at.get("download_button"))


def test_empty_container_keeps_persisted_graph_available(tmp_path) -> None:
    graph = uproot.open(GRAPH_FIXTURE)["plain_graph"]
    path = _write(tmp_path / "graph.root", Events="empty tree", graph=graph)
    at = _app(path)
    _assert_object_mode(at)
    _open_object(at, "graph (TGraph)")
    assert not at.exception
    assert at.session_state["root_graph_result"]["kind"] == "matplotlib"
    assert any(button.label == "Download PNG" for button in at.get("download_button"))


def test_nested_objects_next_to_empty_container_can_be_opened(tmp_path) -> None:
    path = tmp_path / "nested.root"
    with uproot.recreate(path) as root_file:
        root_file.mktree("Events", {"x": "f8"})
        root_file.mkdir("analysis").mkdir("plots")["h_nested"] = _histogram()
    at = _app(path)
    _assert_object_mode(at)
    _open_object(at, "analysis/plots/h_nested (TH1D)")
    assert not at.exception
    assert at.session_state["root_histogram_result"]["kind"] == "matplotlib"


@pytest.mark.parametrize("container", ["empty tree", "empty ntuple"])
def test_only_an_empty_container_without_plot_objects(tmp_path, container) -> None:
    path = _write(tmp_path / "only_empty.root", Events=container)
    at = _app(path)
    _assert_object_mode(at)
    assert at.sidebar.checkbox(key="geometry_enabled")


def test_switching_between_empty_and_filled_containers(tmp_path) -> None:
    path = _write(
        tmp_path / "mixed.root",
        Empty="empty tree",
        Events="filled tree",
        histogram=_histogram(),
    )
    at = _app(path)
    container = next(item for item in at.sidebar.selectbox if item.label == "Event container")
    assert set(container.options) == {"Empty", "Events"}

    container.select("Events").run()
    assert not at.exception
    assert "Plotter" in [tab.label for tab in at.tabs]
    at.button(key="plot_make").click().run()
    at.button(key="event_make").click().run()
    at.button(key="rank_compute").click().run()
    for key in ("plot_result", "event_plot_result", "rank_df"):
        assert key in at.session_state

    # Switching to the empty container keeps the file browsable and drops event results.
    next(item for item in at.sidebar.selectbox if item.label == "Event container").select(
        "Empty"
    ).run()
    _assert_object_mode(at)
    for key in ("plot_result", "event_plot_result", "rank_df", "rank_label", "data_context"):
        assert key not in at.session_state

    # Switching back restores the normal workflow without stale results.
    next(item for item in at.sidebar.selectbox if item.label == "Event container").select(
        "Events"
    ).run()
    assert not at.exception
    assert "Plotter" in [tab.label for tab in at.tabs]
    assert "plot_result" not in at.session_state
    at.button(key="plot_make").click().run()
    assert at.session_state["plot_result"]["kind"] == "matplotlib"


def test_unreadable_container_keeps_object_browser_available(tmp_path, monkeypatch) -> None:
    path = _write(tmp_path / "broken.root", Events="filled tree", histogram=_histogram())

    def fail(*_args, **_kwargs):
        raise OSError("simulated read failure")

    monkeypatch.setattr("root_explorer.app.input.get_entry_count", fail)
    at = _app(path)
    assert not at.exception
    assert any("simulated read failure" in error.value for error in at.sidebar.error)
    assert [tab.label for tab in at.tabs] == OBJECT_TABS
    _open_object(at, "histogram (TH1D)")
    assert at.session_state["root_histogram_result"]["kind"] == "matplotlib"


def test_files_without_event_containers_behave_as_before(tmp_path) -> None:
    path = _write(tmp_path / "objects.root", histogram=_histogram())
    at = _app(path)
    assert not at.exception
    assert not at.sidebar.warning
    assert any("No TTrees or RNTuples" in info.value for info in at.sidebar.info)
    assert [tab.label for tab in at.tabs] == OBJECT_TABS
