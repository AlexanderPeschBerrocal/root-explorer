"""Tests for the ``root-explorer`` launcher (#132)."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from root_explorer import cli
from root_explorer.io.root import parse_cli_rootfile
from root_explorer.io.source import REMOTE_PREFIXES

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_console_script_is_declared() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["root-explorer"] == "root_explorer.cli:main"


def test_packaged_app_script_runs_the_application() -> None:
    script = Path(str(cli.APP_SCRIPT))
    assert script.is_file()
    assert script.parent == Path(cli.__file__).parent
    assert "from root_explorer.app.main import main" in script.read_text(encoding="utf-8")


def test_without_file_only_defaults_are_passed() -> None:
    arguments = cli.streamlit_arguments([])
    assert arguments == ["run", str(cli.APP_SCRIPT), *cli.DEFAULT_STREAMLIT_OPTIONS]
    assert "--" not in arguments


def test_relative_file_is_resolved_against_the_working_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    arguments = cli.streamlit_arguments(["data/events.root"])
    assert arguments[-2:] == ["--", str((tmp_path / "data" / "events.root").resolve())]


def test_absolute_and_home_paths_are_absolute() -> None:
    absolute = str(Path("/data/events.root").resolve())
    assert cli.streamlit_arguments([absolute])[-1] == absolute
    assert cli.streamlit_arguments(["~/events.root"])[-1] == str(
        Path("~/events.root").expanduser().resolve()
    )


@pytest.mark.parametrize(
    "url", ["root://eos.example.org//store/data.root", "https://example.org/data.root"]
)
def test_remote_sources_are_passed_unchanged(url) -> None:
    assert cli.streamlit_arguments([url])[-2:] == ["--", url]


def test_port_no_browser_and_forwarded_options() -> None:
    arguments = cli.streamlit_arguments(
        ["events.root", "--port", "8600", "--no-browser", "--server.maxUploadSize=200"]
    )
    assert "--server.port=8600" in arguments
    assert "--server.headless=true" in arguments
    # Forwarded options follow the defaults, so they take precedence.
    default = arguments.index("--server.maxUploadSize=4096")
    assert arguments.index("--server.maxUploadSize=200") > default
    # The ROOT file is passed to the app script after "--", not to Streamlit.
    assert arguments[-2] == "--"


def test_cli_and_source_detection_use_the_same_remote_prefixes() -> None:
    assert cli.REMOTE_PREFIXES == REMOTE_PREFIXES


def test_main_starts_streamlit_with_the_translated_arguments(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "streamlit.web.cli.main", lambda args, prog_name: calls.append((args, prog_name))
    )
    cli.main(["events.root", "--no-browser"])
    assert calls == [(cli.streamlit_arguments(["events.root", "--no-browser"]), "streamlit")]


def test_app_receives_the_file_passed_after_the_separator(monkeypatch) -> None:
    # Streamlit passes everything after "--" to the app script as its sys.argv.
    monkeypatch.setattr(sys, "argv", [str(cli.APP_SCRIPT), "/data/events.root"])
    assert parse_cli_rootfile() == "/data/events.root"


@pytest.mark.parametrize("option", ["--help", "--version"])
def test_python_module_entry_point_works_without_streamlit_warnings(option) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "root_explorer", option],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0
    assert "root-explorer" in result.stdout
    assert "WARNING" not in result.stderr


def test_packaged_app_prefills_and_opens_the_file_given_on_the_command_line(
    root_path, monkeypatch
) -> None:
    # Streamlit passes the arguments after "--" to the app script as sys.argv.
    monkeypatch.setattr(sys, "argv", [str(cli.APP_SCRIPT), str(root_path)])
    at = AppTest.from_file(str(cli.APP_SCRIPT), default_timeout=20).run()
    assert not at.exception
    assert at.text_input(key="root_path").value == str(root_path)
    assert "Plotter" in [tab.label for tab in at.tabs]


def test_python_module_launch_serves_the_application(root_path, tmp_path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "root_explorer",
            str(root_path),
            "--no-browser",
            "--port",
            str(port),
        ],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health") as response:
                    assert response.status == 200
                    break
            except OSError:
                assert process.poll() is None, "root-explorer exited before serving the app"
                assert time.monotonic() < deadline, "root-explorer did not start in time"
                time.sleep(0.5)
    finally:
        process.terminate()
        process.wait(timeout=20)
