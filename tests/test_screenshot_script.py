"""Command-line handling of the documentation screenshot generator."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "docs" / "scripts"


@pytest.fixture
def screenshot_script(monkeypatch) -> ModuleType:
    # Playwright is an optional extra; argument parsing does not need it.
    sync_api = ModuleType("playwright.sync_api")
    for name in ("Locator", "Page", "expect", "sync_playwright"):
        setattr(sync_api, name, object)
    monkeypatch.setitem(sys.modules, "playwright", ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.delitem(sys.modules, "generate_screenshots", raising=False)
    return importlib.import_module("generate_screenshots")


@pytest.mark.parametrize(
    ("argv", "base_url"),
    [
        ([], "http://127.0.0.1:8751"),
        (["--port", "9000"], "http://127.0.0.1:9000"),
        (["--port", "9000", "--base-url", "http://localhost:9000"], "http://localhost:9000"),
    ],
)
def test_base_url_follows_the_port_unless_given(screenshot_script, argv, base_url) -> None:
    assert screenshot_script.parse_arguments(argv).base_url == base_url
