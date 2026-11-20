"""Remote sources and persisted-object reads fail clearly, not silently or fatally (#168).

Remote behaviour is exercised against local servers from ``http_servers``: one that
honours byte ranges, one that ignores them, one that never answers and one that drops
the connection while an object is read.
"""

from __future__ import annotations

import time
from pathlib import Path

import fsspec.exceptions
import numpy as np
import pytest
import uproot
from conftest import _generate_catalogue_root
from http_servers import serve_file
from streamlit.testing.v1 import AppTest

from root_explorer.app import graph_viewer, histogram_viewer
from root_explorer.io import root as root_io
from root_explorer.io.graphs import read_graph
from root_explorer.io.histograms import UnsupportedHistogramError, read_histogram
from root_explorer.io.source import source_version
from root_explorer.plotting.histogram_codegen import generate_histogram_plot_code
from root_explorer.plotting.histograms import HistogramPlotSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_FIXTURE = PROJECT_ROOT / "tests" / "data" / "test_objects.root"
RANGE_MESSAGE = "does not support HTTP range requests"
NO_EVENT_DATA = "No TTrees or RNTuples containing event data were found"


@pytest.fixture
def catalogue_file(tmp_path: Path) -> Path:
    return _generate_catalogue_root(tmp_path / "catalogue.root", n_events=50, seed=168)


@pytest.fixture
def short_timeout(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(root_io, "REMOTE_TIMEOUT_SECONDS", 1)
    return 1


def _app(source: str) -> AppTest:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=30).run()
    return at.text_input(key="root_path").input(source).run()


def _open_object(at: AppTest, name: str) -> AppTest:
    selector = next(item for item in at.selectbox if item.label == "Object to inspect")
    selector.select(next(option for option in selector.options if name in option)).run()
    return at.button(key="root_object_open").click().run()


def _payload_span(path: Path, object_path: str) -> tuple[int, int]:
    """Byte span of an object's stored (compressed) payload in a ROOT file."""

    with uproot.open(path) as root_file:
        directory, _, name = object_path.rpartition("/")
        key = (root_file[directory] if directory else root_file).key(name)
        start = key.data_cursor.index
        return start, start + key.data_compressed_bytes


def _corrupt_histogram_file(path: Path) -> Path:
    """Write a compressed TH1 next to a small TTree, then garble its zlib payload."""

    with uproot.recreate(path) as root_file:
        root_file.mktree("Events", {"x": "f8"}).extend({"x": np.arange(5, dtype=float)})
        root_file["h_corrupt"] = np.histogram(np.zeros(10), bins=2000, range=(0.0, 1.0))
    start, stop = _payload_span(path, "h_corrupt")
    data = bytearray(path.read_bytes())
    for index in range(start + 20, stop - 4, 7):  # keep ROOT's 9-byte block header
        data[index] ^= 0xFF
    path.write_bytes(bytes(data))
    return path


def _assert_other_tabs_render(at: AppTest) -> None:
    tabs = {tab.label: tab for tab in at.tabs}
    # Before #168 an escaping reader error left these tabs as a single empty element.
    assert len(list(tabs["Plotter"])) > 5
    assert len(list(tabs["Event inspection"])) > 5
    assert len(list(tabs["Help"])) > 1


# --- Open options -----------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_http_sources_require_byte_ranges_and_time_out(scheme) -> None:
    options = root_io.uproot_open_options(f"{scheme}://example.org/data.root")
    assert options["handler"] is uproot.source.http.HTTPSource
    assert options["timeout"] == float(root_io.REMOTE_TIMEOUT_SECONDS)


def test_xrootd_sources_time_out_in_whole_seconds() -> None:
    assert root_io.uproot_open_options("root://eos.example.org//data.root") == {
        "timeout": int(root_io.REMOTE_TIMEOUT_SECONDS)
    }


def test_local_sources_keep_uproot_defaults(tmp_path) -> None:
    assert root_io.uproot_open_options(str(tmp_path / "data.root")) == {}


def test_exported_scripts_use_the_same_remote_options() -> None:
    source = generate_histogram_plot_code(
        HistogramPlotSpec("https://example.org/f.root", "h;1", {"fig_width": 8.0}, {})
    )
    assert "def uproot_open_options(" in source
    assert f"REMOTE_TIMEOUT_SECONDS = {root_io.REMOTE_TIMEOUT_SECONDS}" in source
    assert "uproot.open(path, **uproot_open_options(path))" in source


# --- Error reasons ----------------------------------------------------------------------


def test_reasons_are_never_empty() -> None:
    assert root_io.read_error_reason(RuntimeError()) == "RuntimeError"
    assert root_io.read_error_reason(ValueError("first line\nsecond line")) == "first line"


@pytest.mark.parametrize(
    "error",
    [TimeoutError("timed out"), fsspec.exceptions.FSTimeoutError(), OSError("wrapper")],
    ids=["socket", "fsspec-empty-message", "chained"],
)
def test_timeouts_are_explained_wherever_they_occur(error) -> None:
    if str(error) == "wrapper":
        try:
            try:
                raise TimeoutError("timed out")
            except TimeoutError as inner:
                raise OSError("wrapper") from inner
        except OSError as outer:
            error = outer
    reason = root_io.read_error_reason(error)
    assert f"did not respond within {root_io.REMOTE_TIMEOUT_SECONDS} s" in reason
    assert "Reload source" in reason


# --- Real remote reads against local servers ------------------------------------------


def test_range_server_reads_catalogue_fields_and_objects(catalogue_file) -> None:
    with serve_file(catalogue_file, "range") as url:
        remote = root_io.get_root_catalogue(url)
        assert remote == root_io.get_root_catalogue(str(catalogue_file))
        values = root_io.read_field_uncached(url, "Events", "Muon_pt", 0, None)
        local = root_io.read_field_uncached(str(catalogue_file), "Events", "Muon_pt", 0, None)
        assert values.tolist() == local.tolist()
        histogram = read_histogram(url, "Analysis/h_mass").histogram
        expected = read_histogram(str(catalogue_file), "Analysis/h_mass").histogram
        assert histogram.values().tolist() == expected.values().tolist() == [1.0, 3.0]


def test_server_without_range_support_is_named_not_misread(catalogue_file) -> None:
    with serve_file(catalogue_file, "norange") as url:
        with pytest.raises(Exception) as error:
            root_io.get_root_catalogue(url)
        assert RANGE_MESSAGE in root_io.read_error_reason(error.value)

        at = _app(url)
        errors = [item.value for item in at.sidebar.error]
        assert any(RANGE_MESSAGE in message for message in errors), errors
        assert not any(NO_EVENT_DATA in item.value for item in at.sidebar.info)


def test_empty_remote_catalogue_is_not_reported_as_missing_event_data(monkeypatch) -> None:
    monkeypatch.setattr(root_io, "build_root_catalogue", lambda root_file: ())
    monkeypatch.setattr(root_io, "open_uproot_source", lambda path: _NullContext())
    with pytest.raises(OSError, match="lists no ROOT objects"):
        root_io.get_root_catalogue("https://example.org/empty-looking.root")


class _NullContext:
    def __enter__(self) -> _NullContext:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def test_hanging_server_times_out_with_a_message(catalogue_file, short_timeout) -> None:
    with serve_file(catalogue_file, "hang") as url:
        started = time.monotonic()
        at = _app(url)
        elapsed = time.monotonic() - started
    errors = [item.value for item in at.sidebar.error]
    assert any(
        message.startswith("Could not open ROOT file: The server did not respond within 1 s")
        for message in errors
    ), errors
    # One timeout per request that uproot makes before it gives up, not 300 s.
    assert elapsed < 20


def test_dropped_connection_while_reading_a_histogram(catalogue_file) -> None:
    span = _payload_span(catalogue_file, "Analysis/h_mass")
    with serve_file(catalogue_file, "drop", drop_span=span) as url:
        with pytest.raises(root_io.ObjectReadError, match="Could not read histogram") as error:
            histogram_viewer._cached_histogram(url, "Analysis/h_mass", source_version(url))
        assert url in str(error.value)
        assert error.value.__cause__ is not None

        at = _open_object(_app(url), "Analysis/h_mass")
        assert any("Could not read histogram" in item.value for item in at.warning)
        assert any(expander.label == "Technical details" for expander in at.expander)
        _assert_other_tabs_render(at)


def test_dropped_connection_while_reading_a_graph(tmp_path) -> None:
    path = tmp_path / "graph.root"
    with uproot.recreate(path) as root_file:
        root_file["graph"] = uproot.open(GRAPH_FIXTURE)["plain_graph"]
    with (
        serve_file(path, "drop", drop_span=_payload_span(path, "graph")) as url,
        pytest.raises(root_io.ObjectReadError, match="Could not read graph `graph`"),
    ):
        graph_viewer._cached_graph(url, "graph", source_version(url))


# --- Corrupt local payloads -----------------------------------------------------------


def test_corrupt_histogram_payload_keeps_the_app_usable(tmp_path) -> None:
    path = _corrupt_histogram_file(tmp_path / "corrupt.root")
    at = _open_object(_app(str(path)), "h_corrupt")

    warnings = [item.value for item in at.warning]
    assert any(
        message.startswith("Could not read histogram `h_corrupt;1` in `corrupt.root`: Error -3")
        for message in warnings
    ), warnings
    details = next(expander for expander in at.expander if expander.label == "Technical details")
    assert "decompressing" in str(details.exception[0].value)
    assert "root_histogram_result" not in at.session_state
    _assert_other_tabs_render(at)


def test_unsupported_histograms_keep_their_own_message(tmp_path, monkeypatch) -> None:
    def unsupported(root_path, object_path):
        raise UnsupportedHistogramError("TH3 histograms are not supported.")

    monkeypatch.setattr(histogram_viewer, "read_histogram", unsupported)
    with pytest.raises(UnsupportedHistogramError, match=r"^TH3 histograms are not supported\.$"):
        histogram_viewer._cached_histogram("file.root", "h", ("local", "file.root", 0, 0))


def test_local_graph_reader_still_reads(tmp_path) -> None:
    graph = read_graph(str(GRAPH_FIXTURE), "plain_graph")
    assert len(graph.x) > 0
