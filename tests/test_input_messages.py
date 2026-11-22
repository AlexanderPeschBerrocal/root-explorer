"""Messages that name the actual cause of common input problems (#194)."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot
from conftest import _generate_test_root
from streamlit.testing.v1 import AppTest

from root_explorer.analysis.arrays import EmptySelectionError, require_values
from root_explorer.io.root import (
    FieldReadError,
    get_root_catalogue,
    read_error_reason,
    read_field_uncached,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRUNCATED = "The file is truncated or incomplete"


def _app() -> AppTest:
    return AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20).run()


# --- Truncated files ----------------------------------------------------------------------


def _truncated_files(tmp_path: Path) -> list[Path]:
    empty = tmp_path / "empty.root"
    empty.write_bytes(b"")
    full = _generate_test_root(tmp_path / "full.root", n_events=200, seed=194)
    half = tmp_path / "half.root"
    half.write_bytes(full.read_bytes()[: full.stat().st_size // 2])
    return [empty, half]


def test_truncated_files_are_named_as_such(tmp_path) -> None:
    empty, half = _truncated_files(tmp_path)
    # An empty file fails at the catalogue.
    with pytest.raises(Exception) as error:
        get_root_catalogue(str(empty))
    assert read_error_reason(error.value).startswith(f"{TRUNCATED}: a read expected 403 bytes")
    # A half-copied TTree file still lists its objects, but its branch data is cut off.
    assert get_root_catalogue(str(half))
    with pytest.raises(FieldReadError) as field_error:
        read_field_uncached(str(half), "Events", "Muon_pt", 0, None)
    assert f"`half.root` (entries from 0): {TRUNCATED}: a read expected " in str(field_error.value)
    assert "but received 0." in str(field_error.value)


def test_truncated_file_message_in_the_app(tmp_path) -> None:
    empty = _truncated_files(tmp_path)[0]
    at = _app()
    at.text_input(key="root_path").input(str(empty)).run()
    errors = [error.value for error in at.sidebar.error]
    assert errors == [
        f"Could not open ROOT file: {TRUNCATED}: a read expected 403 bytes but received 0. "
        "Check that the file was written and copied completely."
    ]


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("first part,\nsecond part\nthird part", "first part, second part"),
        ("one line", "one line"),
        ("first line\nsecond line", "first line"),
        ("", "ValueError"),
    ],
)
def test_reasons_keep_a_comma_continued_line(message, reason) -> None:
    assert read_error_reason(ValueError(message)) == reason


# --- Empty selections ---------------------------------------------------------------------


def test_empty_selection_mentions_cuts_only_when_there_are_cuts() -> None:
    with pytest.raises(EmptySelectionError, match=r"^No values remain to plot after applying"):
        require_values(np.array([]), 0, has_cuts=True)
    with pytest.raises(EmptySelectionError) as error:
        require_values(np.array([]), 0, has_cuts=False)
    assert str(error.value) == (
        "No values to plot: the selected variables have no values in these events."
    )
    require_values(np.array([1.0]), 0, has_cuts=False)  # values remain: no error


def test_empty_collection_without_cuts_in_the_app(tmp_path) -> None:
    path = tmp_path / "no_muons.root"
    with uproot.recreate(path) as root_file:
        root_file.mktree(
            "Events",
            {"Muon_pt": ak.Array([[], [], []]), "MET_pt": np.array([10.0, 20.0, 30.0])},
        )
    at = _app()
    at.text_input(key="root_path").input(str(path)).run()
    at.selectbox(key="x_branch").select("Muon_pt").run()
    at.button(key="plot_make").click().run()
    assert not at.exception
    warnings = [warning.value for warning in at.warning]
    assert any("have no values in these events" in warning for warning in warnings), warnings
    assert not any("cuts" in warning for warning in warnings)

    # With a cut, the message still names the cuts.
    at.number_input(key="plot_n_cuts").set_value(1).run()
    at.selectbox(key="plot_cut_0_branch").select("MET_pt").run()
    at.text_input(key="plot_cut_0_min").input("0").run()
    at.button(key="plot_make").click().run()
    assert any("after applying the cuts" in warning.value for warning in at.warning)


# --- Path input ---------------------------------------------------------------------------


def test_home_shorthand_and_surrounding_whitespace_are_accepted(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _generate_test_root(tmp_path / "events.root", n_events=5, seed=194)
    for typed in ("~/events.root", f"  {tmp_path / 'events.root'} \t"):
        at = _app()
        at.text_input(key="root_path").input(typed).run()
        assert not at.exception
        assert not [error.value for error in at.sidebar.error], typed
        assert at.selectbox(key="x_branch").options, typed


def test_missing_file_is_still_reported(tmp_path) -> None:
    at = _app()
    at.text_input(key="root_path").input(f" {tmp_path / 'missing.root'} ").run()
    assert [error.value for error in at.sidebar.error] == [
        f"File does not exist: {tmp_path / 'missing.root'}"
    ]
