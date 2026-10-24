"""Uploads must retain plots and rankings across ordinary Streamlit reruns."""

import os
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from root_explorer.app import results
from root_explorer.app.results import UPLOAD_PATHS_KEY, store_uploaded_file
from root_explorer.io.source import source_version


class Upload:
    def __init__(self, name, data, file_id=None):
        self.name = name
        self.data = data
        if file_id is not None:
            self.file_id = file_id

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


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("root_explorer.app.results.tempfile.gettempdir", lambda: str(tmp_path))
    st.session_state.pop(UPLOAD_PATHS_KEY, None)
    yield tmp_path / "root_explorer_uploads"
    st.session_state.pop(UPLOAD_PATHS_KEY, None)


@pytest.mark.parametrize("stem", ["a" * 250, "é" * 200])
def test_long_upload_names_fit_the_filename_limit(upload_dir, stem):
    path = Path(store_uploaded_file(Upload(f"{stem}.root", b"data")))
    assert len(path.name.encode()) <= 160
    assert path.suffix == ".root"
    assert path.read_bytes() == b"data"


def test_upload_is_published_without_hard_link_support(upload_dir, monkeypatch):
    def no_links(source, destination):
        raise PermissionError("hard links are not supported")

    monkeypatch.setattr(results.os, "link", no_links)
    path = Path(store_uploaded_file(Upload("data.root", b"data")))
    assert path.read_bytes() == b"data"
    assert list(upload_dir.iterdir()) == [path]


def test_upload_is_hashed_once_per_file_id_and_recreated_if_removed(upload_dir, monkeypatch):
    hashes = []
    sha256 = results.hashlib.sha256

    def counting_sha256(data):
        hashes.append(len(data))
        return sha256(data)

    monkeypatch.setattr(results.hashlib, "sha256", counting_sha256)
    upload = Upload("data.root", b"data", file_id="upload-1")
    path = Path(store_uploaded_file(upload))
    assert store_uploaded_file(upload) == str(path)
    assert len(hashes) == 1

    path.unlink()
    assert store_uploaded_file(upload) == str(path)
    assert path.read_bytes() == b"data"
    assert len(hashes) == 2

    # A different upload of the same name is hashed separately.
    other = Path(store_uploaded_file(Upload("data.root", b"other", file_id="upload-2")))
    assert other != path
    assert len(hashes) == 3


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
