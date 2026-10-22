"""Tests for consistent version information (#118)."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import root_explorer
from root_explorer.app import main as app_main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _declared_version() -> str:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]["version"]


def test_declared_installed_and_runtime_versions_agree() -> None:
    declared = _declared_version()
    installed = version("root-explorer")
    assert installed == declared, (
        f"The installed package reports {installed}, but pyproject.toml declares {declared}. "
        "Reinstall the project, for example with `python -m pip install -e .`, after "
        "changing the version."
    )
    assert root_explorer.__version__ == installed
    assert root_explorer.installed_version() == installed


def test_declared_version_follows_calver() -> None:
    assert re.fullmatch(r"20\d{2}\.(?:[1-9]|1[0-2])\.\d+", _declared_version())


def test_about_text_shows_the_installed_version() -> None:
    text = app_main.about_text()
    assert f"**Version:** `{version('root-explorer')}`" in text
    assert app_main.GITHUB_URL in text


def test_about_text_reads_the_version_on_every_call(monkeypatch) -> None:
    # A reinstalled package must appear without restarting the server.
    monkeypatch.setattr(app_main, "installed_version", lambda: "2099.1.0")
    assert "`2099.1.0`" in app_main.about_text()


def test_launcher_reports_the_installed_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "root_explorer", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert result.stdout.strip() == f"root-explorer {version('root-explorer')}"


def test_app_passes_the_installed_version_to_the_about_menu(monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    menus = []
    original = app_main.st.set_page_config

    def recording_set_page_config(*args, **kwargs):
        menus.append(kwargs.get("menu_items", {}))
        return original(*args, **kwargs)

    monkeypatch.setattr("streamlit.set_page_config", recording_set_page_config)
    at = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=10).run()
    assert not at.exception
    assert f"`{version('root-explorer')}`" in menus[-1]["About"]
