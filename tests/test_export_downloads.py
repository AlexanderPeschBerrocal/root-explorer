"""Browser downloads as the only export mechanism (#122).

Exports must work wherever ROOT Explorer runs, including the published Docker image,
whose working directory is not writable. No export writes to the server filesystem.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
from streamlit.testing.v1 import AppTest

from root_explorer.app import results
from root_explorer.plotting.plots import build_plotly_3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECTS_PATH = PROJECT_ROOT / "tests" / "data" / "test_objects.root"


def _app(path: Path) -> AppTest:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20).run()
    return at.text_input(key="root_path").input(str(path)).run()


def _plotter_static(at: AppTest) -> None:
    at.button(key="plot_make").click().run()


def _plotter_3d(at: AppTest) -> None:
    at.selectbox(key="plot_type").select("3D scatter interactive").run()
    at.button(key="plot_make").click().run()


def _event_plot(at: AppTest) -> None:
    at.selectbox(key="event_plot_type").select("2D scatter").run()
    at.button(key="event_make").click().run()


def _open_object(option: str):
    def open_object(at: AppTest) -> None:
        selector = next(item for item in at.selectbox if item.label == "Object to inspect")
        selector.select(option).run()
        at.button(key="root_object_open").click().run()

    return open_object


EXPORTS = {
    "plotter static": (
        _plotter_static,
        "plot",
        "root_explorer_plot",
        {"Download PNG", "Download PDF", "Download plot code"},
    ),
    "plotter 3D": (_plotter_3d, "plot", "root_explorer_plot", {"Download HTML"}),
    "event plot": (
        _event_plot,
        "event",
        "root_explorer_event_plot_0",
        {"Download PNG", "Download PDF", "Download plot code"},
    ),
    "persisted histogram": (
        _open_object("histogram_1d (TH1D)"),
        "histogram",
        "root_histogram",
        {"Download PNG", "Download PDF", "Download plot code"},
    ),
    "persisted graph": (
        _open_object("plain_graph (TGraph)"),
        "graph",
        "root_graph",
        {"Download PNG", "Download PDF", "Download plot code"},
    ),
}


@pytest.mark.parametrize("case", EXPORTS)
def test_every_plot_type_offers_only_browser_downloads(case, root_path) -> None:
    make_plot, prefix, default_name, expected_labels = EXPORTS[case]
    at = _app(OBJECTS_PATH if case.startswith("persisted") else root_path)
    make_plot(at)
    assert not at.exception

    labels = {
        button.proto.label
        for button in at.get("download_button")
        if button.key.startswith(f"{prefix}_export_download_")
    }
    assert labels == expected_labels
    assert at.text_input(key=f"{prefix}_file_name").value == default_name

    # Server-side Save and its settings are gone.
    widgets = [*at.button, *at.text_input, *at.checkbox]
    assert not any("save" in str(widget.label).casefold() for widget in widgets)
    assert not any("save" in str(widget.key).casefold() for widget in widgets)
    assert not any("filesystem" in caption.value for caption in at.caption)


class _Recorder:
    """Stand-ins for the Streamlit calls made while showing a cached plot."""

    def __init__(self) -> None:
        self.downloads: dict[str, dict[str, Any]] = {}
        self.warnings: list[str] = []

    def download_button(self, label: str, **kwargs) -> None:
        self.downloads[label] = kwargs

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _show(monkeypatch, result: dict[str, Any], export: dict[str, Any]) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(results.st, "session_state", {"test_result": result})
    for name in ["caption", "markdown", "pyplot", "plotly_chart"]:
        monkeypatch.setattr(results.st, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(results.st, "download_button", recorder.download_button)
    monkeypatch.setattr(results.st, "warning", recorder.warning)
    monkeypatch.setattr(results, "export_controls", lambda *args, **kwargs: export)
    results.show_cached_plot_result(
        "test_result",
        export_prefix="plot",
        download_key="plot_export",
        make_plot_label="Make plot",
        make_plot_shortcut="P",
        empty_message="No plot",
    )
    return recorder


def _png_size(data: bytes) -> tuple[int, int]:
    assert data.startswith(b"\x89PNG")
    return struct.unpack(">II", data[16:24])


def test_static_downloads_use_filename_and_resolution(monkeypatch) -> None:
    figure, axes = plt.subplots(figsize=(4.0, 3.0))
    axes.plot([0, 1], [0, 1])
    result = {
        "kind": "matplotlib",
        "figure": figure,
        "code": "print('plot')\n",
        "file_name": "root_explorer_plot",
    }
    try:
        downloads = {}
        for dpi in (150, 300):
            recorder = _show(monkeypatch, result, {"file_name": "muon pt / v2", "export_dpi": dpi})
            downloads[dpi] = recorder.downloads
            assert not recorder.warnings
    finally:
        plt.close(figure)

    files = {label: options["file_name"] for label, options in downloads[150].items()}
    assert files == {
        "Download PNG": "muon_pt___v2.png",
        "Download PDF": "muon_pt___v2.pdf",
        "Download plot code": "muon_pt___v2.py",
    }
    assert not downloads[150]["Download PNG"]["disabled"]
    width_150, height_150 = _png_size(downloads[150]["Download PNG"]["data"]())
    width_300, height_300 = _png_size(downloads[300]["Download PNG"]["data"]())
    assert width_300 == pytest.approx(2 * width_150, abs=2)
    assert height_300 == pytest.approx(2 * height_150, abs=2)
    assert downloads[150]["Download PDF"]["data"]().startswith(b"%PDF")
    assert downloads[150]["Download plot code"]["data"] == "print('plot')\n"


def test_oversized_png_is_disabled_while_pdf_stays_available(monkeypatch) -> None:
    figure, _ = plt.subplots(figsize=(16.0, 12.0))
    result = {"kind": "matplotlib", "figure": figure, "code": None, "file_name": "large"}
    try:
        recorder = _show(monkeypatch, result, {"file_name": "large", "export_dpi": 600})
    finally:
        plt.close(figure)

    assert set(recorder.downloads) == {"Download PNG", "Download PDF"}
    assert recorder.downloads["Download PNG"]["disabled"]
    assert not recorder.downloads["Download PDF"].get("disabled", False)
    assert any("megapixel safety limit" in warning for warning in recorder.warnings)


def test_interactive_download_is_html(monkeypatch) -> None:
    figure = build_plotly_3d([1], [2], [3], "x", "y", "z", title="event")
    result = {"kind": "plotly", "figure": figure, "file_name": "root_explorer_plot"}
    recorder = _show(monkeypatch, result, {"file_name": "event 7"})

    assert set(recorder.downloads) == {"Download HTML"}
    download = recorder.downloads["Download HTML"]
    assert download["file_name"] == "event_7.html"
    assert download["mime"] == "text/html"
    assert b"plotly" in download["data"]().lower()
