"""Regression tests for cache invalidation when sources change at the same path (#117)."""

from __future__ import annotations

import os
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot
from streamlit.testing.v1 import AppTest

from root_explorer.analysis.ranking import compute_event_metric
from root_explorer.geometry.overlay import decode_geometry_scene, load_geometry_scene
from root_explorer.io import root as root_io
from root_explorer.io.root import (
    get_entry_count,
    get_field_info,
    get_root_catalogue,
    open_root_file,
    read_event_field,
    read_field,
)
from root_explorer.io.source import (
    SourceUnavailableError,
    is_remote_source,
    reload_source,
    source_version,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bump_mtime(path: Path) -> None:
    """Give a rewritten file a strictly newer modification time.

    Rewrites within one coarse filesystem timestamp tick could otherwise keep the
    same modification time; tests must not rely on sleeping.
    """

    status = path.stat()
    newer = status.st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(newer, newer))


def _write_events(path: Path, values: np.ndarray, *, extra_field: bool = False) -> None:
    with uproot.recreate(path) as root_file:
        branches = {"x": "f8", "jet": "var * float64"}
        if extra_field:
            branches["y"] = "f8"
        tree = root_file.mktree("Events", branches)
        data = {"x": values, "jet": ak.Array([[value, value + 1.0] for value in values])}
        if extra_field:
            data["y"] = values * 2
        tree.extend(data)
    _bump_mtime(path)


def test_local_source_version_changes_with_content_and_detects_missing_files(tmp_path) -> None:
    path = tmp_path / "events.root"
    _write_events(path, np.arange(3.0))
    first = source_version(str(path))
    assert source_version(str(path)) == first

    _write_events(path, np.arange(5.0))
    assert source_version(str(path)) != first

    path.unlink()
    with pytest.raises(SourceUnavailableError, match="Source is not available"):
        source_version(str(path))


def test_remote_source_version_changes_only_on_reload() -> None:
    url = "root://eos.example.org//store/data.root"
    assert is_remote_source(url)
    first = source_version(url)
    assert source_version(url) == first
    reload_source(url)
    assert source_version(url) != first


def test_catalogue_entry_count_and_field_info_follow_a_replaced_file(tmp_path) -> None:
    path = tmp_path / "events.root"
    _write_events(path, np.arange(100.0))
    assert get_entry_count(str(path), "Events") == 100
    assert "y" not in {field.name for field in get_field_info(str(path), "Events")}
    assert {info.path for info in get_root_catalogue(str(path))} == {"Events"}

    with uproot.recreate(path) as root_file:
        tree = root_file.mktree("Events", {"x": "f8", "jet": "var * float64", "y": "f8"})
        tree.extend({"x": np.arange(200.0), "jet": ak.Array([[1.0]] * 200), "y": np.zeros(200)})
        root_file["histogram"] = np.histogram([1.0, 2.0], bins=2)
    _bump_mtime(path)

    assert get_entry_count(str(path), "Events") == 200
    assert "y" in {field.name for field in get_field_info(str(path), "Events")}
    assert {info.path for info in get_root_catalogue(str(path))} == {"Events", "histogram"}


def test_range_and_event_reads_follow_a_replaced_file(tmp_path) -> None:
    path = tmp_path / "events.root"
    _write_events(path, np.array([1.0, 2.0, 3.0]))
    assert ak.to_list(read_field(str(path), "Events", "x", 0, 3)) == [1.0, 2.0, 3.0]
    assert ak.to_list(read_event_field(str(path), "Events", "jet", 1)) == [2.0, 3.0]
    first_handle = open_root_file(str(path))

    _write_events(path, np.array([10.0, 20.0, 30.0]))
    assert ak.to_list(read_field(str(path), "Events", "x", 0, 3)) == [10.0, 20.0, 30.0]
    assert ak.to_list(read_event_field(str(path), "Events", "jet", 1)) == [20.0, 21.0]
    assert open_root_file(str(path)) is not first_handle


def test_ranking_cache_does_not_mask_a_replaced_file(tmp_path) -> None:
    path = tmp_path / "events.root"
    _write_events(path, np.array([1.0, 5.0, 3.0]))
    _, before = compute_event_metric(str(path), "Events", "jet", "max", 0, 3, None, False)
    np.testing.assert_allclose(before, [2.0, 6.0, 4.0])

    _write_events(path, np.array([7.0, 0.0, 9.0]))
    _, after = compute_event_metric(str(path), "Events", "jet", "max", 0, 3, None, False)
    np.testing.assert_allclose(after, [8.0, 1.0, 10.0])


def test_unchanged_files_keep_using_the_cache(tmp_path, monkeypatch) -> None:
    path = tmp_path / "events.root"
    _write_events(path, np.arange(4.0))
    calls = []
    original = root_io.read_field_uncached

    def counting(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(root_io, "read_field_uncached", counting)
    for _ in range(3):
        read_field(str(path), "Events", "x", 0, 4)
    assert len(calls) == 1


def test_missing_file_raises_a_clear_error_instead_of_stale_data(tmp_path) -> None:
    path = tmp_path / "events.root"
    _write_events(path, np.arange(3.0))
    read_field(str(path), "Events", "x", 0, 3)
    path.unlink()
    with pytest.raises(SourceUnavailableError):
        read_field(str(path), "Events", "x", 0, 3)


def test_geometry_scene_is_rebuilt_after_gdml_replacement(tiny_gdml_path) -> None:
    def min_x() -> float:
        scene = load_geometry_scene(str(tiny_gdml_path), "fast", False, False, 10, 1_000)
        positions, _, _ = decode_geometry_scene(scene)
        return float(positions[:, 0].min())

    assert min_x() == pytest.approx(4.0)
    tiny_gdml_path.write_text(
        tiny_gdml_path.read_text(encoding="utf-8").replace('x="5"', 'x="8"'), encoding="utf-8"
    )
    _bump_mtime(tiny_gdml_path)
    assert min_x() == pytest.approx(7.0)


def test_retained_plot_is_discarded_when_the_file_is_replaced(tmp_path) -> None:
    path = tmp_path / "events.root"
    _write_events(path, np.arange(100.0))
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=10).run()
    at.text_input(key="root_path").input(str(path)).run()
    at.button(key="plot_make").click().run()
    at.button(key="rank_compute").click().run()
    assert "plot_result" in at.session_state
    assert "rank_df" in at.session_state

    _write_events(path, np.arange(200.0))
    at.run()
    assert not at.exception
    assert any("200" in markdown.value for markdown in at.sidebar.markdown)
    assert "plot_result" not in at.session_state
    assert "rank_df" not in at.session_state


def test_disappearing_file_fails_clearly_and_recovers(tmp_path) -> None:
    path = tmp_path / "events.root"
    _write_events(path, np.arange(10.0))
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=10).run()
    at.text_input(key="root_path").input(str(path)).run()
    at.button(key="plot_make").click().run()

    path.unlink()
    at.run()
    assert not at.exception
    assert any("does not exist" in error.value for error in at.sidebar.error)
    assert not at.tabs  # The previous plot is not displayed while the file is missing.

    _write_events(path, np.arange(10.0))
    at.run()
    assert not at.exception
    # The restored file is a new version, so the old plot is not reused.
    assert "plot_result" not in at.session_state
    at.button(key="plot_make").click().run()
    assert at.session_state["plot_result"]["kind"] == "matplotlib"


def test_persisted_graph_viewer_shows_replaced_contents(tmp_path) -> None:
    fixture = uproot.open(PROJECT_ROOT / "tests" / "data" / "test_objects.root")
    path = tmp_path / "graphs.root"

    def write(graph_name: str) -> list[float]:
        with uproot.recreate(path) as root_file:
            root_file["graph"] = fixture[graph_name]
        _bump_mtime(path)
        return [float(value) for value in fixture[graph_name].values()[1]]

    def plotted_y(at: AppTest) -> list[float]:
        axes = at.session_state["root_graph_result"]["figure"].axes[0]
        return [float(value) for line in axes.lines for value in line.get_ydata()]

    first = write("plain_graph")
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=10).run()
    at.text_input(key="root_path").input(str(path)).run()
    next(item for item in at.selectbox if item.label == "Object to inspect").select(
        "graph (TGraph)"
    ).run()
    at.button(key="root_object_open").click().run()
    assert plotted_y(at) == pytest.approx(first)

    second = write("cycled_graph")
    assert second != first
    at.button(key="graph_make_plot").click().run()
    assert not at.exception
    assert plotted_y(at) == pytest.approx(second)


def test_persisted_histogram_viewer_shows_replaced_contents(tmp_path) -> None:
    path = tmp_path / "objects.root"

    def write(counts: list[float]) -> None:
        with uproot.recreate(path) as root_file:
            root_file["histogram"] = (np.array(counts), np.array([0.0, 1.0, 2.0]))
        _bump_mtime(path)

    write([1.0, 2.0])
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=10).run()
    at.text_input(key="root_path").input(str(path)).run()
    next(item for item in at.selectbox if item.label == "Object to inspect").select(
        "histogram (TH1D)"
    ).run()
    at.button(key="root_object_open").click().run()
    assert any("sum of bin contents: 3" in caption.value for caption in at.caption)

    write([10.0, 20.0])
    at.button(key="hist_make_plot").click().run()
    assert not at.exception
    assert any("sum of bin contents: 30" in caption.value for caption in at.caption)


def test_reload_source_button_is_offered_only_for_remote_paths(tmp_path, monkeypatch) -> None:
    url = "root://eos.example.org//store/remote.root"
    catalogue_versions = []

    def fake_catalogue(path: str):
        catalogue_versions.append(source_version(path))
        raise OSError("remote access is not available in tests")

    monkeypatch.setattr("root_explorer.app.input.get_root_catalogue", fake_catalogue)
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=10).run()
    at.text_input(key="root_path").input(url).run()
    assert at.sidebar.button(key="reload_remote_source").label == "Reload source"

    at.sidebar.button(key="reload_remote_source").click().run()
    assert catalogue_versions[-1] != catalogue_versions[0]

    local = tmp_path / "local.root"
    _write_events(local, np.arange(3.0))
    at.text_input(key="root_path").input(str(local)).run()
    assert not any(button.key == "reload_remote_source" for button in at.sidebar.button)
