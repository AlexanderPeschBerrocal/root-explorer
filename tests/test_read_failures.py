"""Concise, contextual messages for failed reads and no stale rankings (#44).

The fixture is a real TTree whose ``bad`` branch has a corrupted basket: the file
and its metadata open normally, but reading the branch fails while decompressing.
"""

from __future__ import annotations

import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import uproot
from streamlit.testing.v1 import AppTest
from test_codegen import _executed_generated_module, _request

from root_explorer.io.root import FieldReadError, read_event_field, read_field
from root_explorer.models import VariableSpec
from root_explorer.plotting.codegen import generate_static_plot_code

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_EVENTS = 1_000
BAD_BRANCH_MESSAGE = (
    "Could not read branch `bad` of TTree `Events` in `corrupt.root` (entries 0-999): "
    "Error -3 while decompressing data: incorrect data check"
)


@pytest.fixture
def corrupt_path(tmp_path) -> Path:
    path = tmp_path / "corrupt.root"
    with uproot.recreate(path) as root_file:
        tree = root_file.mktree("Events", {"good": "f8", "bad": "f8"})
        tree.extend({"good": np.arange(float(N_EVENTS)), "bad": np.linspace(0, 1, N_EVENTS)})
    with uproot.open(path) as root_file:
        branch = root_file["Events"]["bad"]
        seek = int(branch.member("fBasketSeek")[0])
        size = int(branch.member("fBasketBytes")[0])
    with open(path, "r+b") as raw:
        raw.seek(seek + size // 2)
        raw.write(b"\xff" * 64)
    return path


def test_failed_reads_name_the_branch_container_file_and_entries(corrupt_path) -> None:
    with pytest.raises(FieldReadError) as error:
        read_field(str(corrupt_path), "Events", "bad", 0, N_EVENTS)
    assert str(error.value) == BAD_BRANCH_MESSAGE
    # The server directory is not exposed; the original error stays available.
    assert str(corrupt_path.parent) not in str(error.value)
    assert isinstance(error.value.__cause__, zlib.error)

    with pytest.raises(FieldReadError, match=r"\(event 7\): Error -3 while decompressing"):
        read_event_field(str(corrupt_path), "Events", "bad", 7)

    # Readable branches of the same file are unaffected.
    assert read_field(str(corrupt_path), "Events", "good", 0, 3).tolist() == [0.0, 1.0, 2.0]


def test_rntuple_reads_say_field(tmp_path) -> None:
    path = tmp_path / "ntuple.root"
    with uproot.recreate(path) as root_file:
        root_file.mkrntuple("Events", {"x": np.arange(3.0)})

    with pytest.raises(FieldReadError, match=r"^Could not read field `y` of RNTuple `Events`"):
        read_field(str(path), "Events", "y", 0, 3)


def _app(path: Path) -> AppTest:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20).run()
    return at.text_input(key="root_path").input(str(path)).run()


def _assert_concise_error(at: AppTest, message: str) -> None:
    assert [error.value for error in at.error] == [message]
    # The only traceback is the reader's original error inside the collapsed
    # "Technical details" expander; nothing escaped as an uncaught exception.
    details = next(expander for expander in at.expander if expander.label == "Technical details")
    assert [exception.message for exception in details.exception] == [
        "Error -3 while decompressing data: incorrect data check"
    ]
    assert len(at.exception) == 1


def test_failed_plot_reports_the_read_and_discards_the_previous_plot(corrupt_path) -> None:
    at = _app(corrupt_path)
    at.selectbox(key="x_branch").select("good").run()
    at.button(key="plot_make").click().run()
    assert "plot_result" in at.session_state

    at.selectbox(key="x_branch").select("bad").run()
    at.button(key="plot_make").click().run()
    _assert_concise_error(at, BAD_BRANCH_MESSAGE)
    assert "plot_result" not in at.session_state


def test_failed_event_plot_names_the_event(corrupt_path) -> None:
    at = _app(corrupt_path)
    at.selectbox(key="event_x_branch").select("bad").run()
    at.button(key="event_make").click().run()
    _assert_concise_error(at, BAD_BRANCH_MESSAGE.replace("entries 0-999", "event 0"))
    assert "event_plot_result" not in at.session_state


def test_failed_ranking_discards_the_previous_ranking(corrupt_path) -> None:
    at = _app(corrupt_path)
    at.selectbox(key="metric_branch").select("good").run()
    at.button(key="rank_compute").click().run()
    assert "rank_df" in at.session_state

    at.selectbox(key="metric_branch").select("bad").run()
    at.button(key="rank_compute").click().run()
    _assert_concise_error(at, BAD_BRANCH_MESSAGE)
    assert "rank_df" not in at.session_state
    assert "rank_label" not in at.session_state
    assert not any(markdown.value.startswith("Metric:") for markdown in at.markdown)


def test_exported_code_reports_failed_reads_the_same_way(corrupt_path) -> None:
    request = _request(
        corrupt_path,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="good"),
        style={"fig_width": 6.0, "fig_height": 4.0, "bins_x": 10},
    )
    request = replace(
        request, context=replace(request.context, n_entries=N_EVENTS, entry_stop=N_EVENTS)
    )
    source = generate_static_plot_code(request)

    with _executed_generated_module(source) as module:
        module.PLOT = replace(module.PLOT, x_spec=VariableSpec(mode="branch", branch="bad"))
        with pytest.raises(module.FieldReadError) as error:
            module.main()
    assert str(error.value) == BAD_BRANCH_MESSAGE
