"""Uploads must retain plots and rankings across ordinary Streamlit reruns."""

import os
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from root_explorer.app.results import store_uploaded_file
from root_explorer.io.source import source_version


class Upload:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def getbuffer(self):
        return memoryview(self.data)


@pytest.mark.parametrize("suffix", ["root", "gdml"])
def test_upload_is_written_once_and_recreated_if_removed(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr("root_explorer.app.results.tempfile.gettempdir", lambda: str(tmp_path))
    upload = Upload(f"data.{suffix}", b"original data")
    path = Path(store_uploaded_file(upload))
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    version = source_version(str(path))
    assert store_uploaded_file(upload) == str(path)
    assert source_version(str(path)) == version
    other = Path(store_uploaded_file(Upload(upload.name, b"changed data")))
    assert other != path
    assert path.read_bytes() == b"original data"
    path.unlink()
    assert store_uploaded_file(upload) == str(path)
    assert path.read_bytes() == b"original data"


def test_uploaded_root_keeps_plot_and_ranking_on_widget_reruns(root_path, monkeypatch):
    upload = Upload("events.root", root_path.read_bytes())
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    at = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "streamlit_app.py", default_timeout=20
    ).run()
    at.radio[0].set_value("Upload").run()
    at.button(key="plot_make").click().run()
    assert not at.exception
    figure = at.session_state["plot_result"]["figure"]
    at.button(key="rank_compute").click().run()
    at.text_input(key="plot_title").set_value("New title").run()
    assert not at.exception
    assert at.session_state["plot_result"]["figure"] is figure
    assert "rank_df" in at.session_state
    assert any("settings have changed" in info.value for info in at.info)
