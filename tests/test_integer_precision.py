"""Warnings for integer branches that float64 cannot represent exactly (#166)."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot
from streamlit.testing.v1 import AppTest

from root_explorer.analysis.precision import (
    FLOAT64_EXACT_INTEGER_LIMIT,
    integer_precision_warnings,
    is_inexact_integer_bound,
    largest_inexact_integer,
    variable_value_branches,
)
from root_explorer.models import VariableSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIMIT = FLOAT64_EXACT_INTEGER_LIMIT
B = 1_700_000_000_000_000_000  # nanoseconds since the epoch, as in the issue
WARNING = "beyond 2⁵³"


def _event_fields() -> dict[str, object]:
    return {
        "a": np.array([B + 1, B + 256, B + 3], dtype=np.int64),
        "b": np.array([B, B, B], dtype=np.int64),
        "small": np.array([1, 2, 3], dtype=np.int64),
        "packed": np.array([2**64 - 1, 2**64 - 2, 2**63 + 1], dtype=np.uint64),
        "large_float": np.array([1.7e18, 1.7e18, 1.7e18]),
        "hit_id": ak.Array([[B, B + 1], [], [5]]),
        "x": np.array([1.0, 2.0, 3.0]),
    }


@pytest.fixture(params=["TTree", "RNTuple"])
def precision_file(request, tmp_path) -> Path:
    path = tmp_path / "timestamps.root"
    with uproot.recreate(path) as root_file:
        if request.param == "TTree":
            root_file.mktree("Events", _event_fields())
        else:
            root_file.mkrntuple("Events", _event_fields())
    return path


def _warnings(path: Path, *specs: VariableSpec, extra=(), event_index=None) -> list[str]:
    return integer_precision_warnings(list(specs), extra, str(path), "Events", 0, 3, event_index)


# --- Detection --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([LIMIT], dtype=np.int64), None),  # 2**53 itself is exact
        (np.array([LIMIT + 1], dtype=np.int64), LIMIT + 1),
        (np.array([-LIMIT - 1], dtype=np.int64), LIMIT + 1),
        (np.array([np.iinfo(np.int64).min], dtype=np.int64), 2**63),  # no abs() overflow
        (np.array([2**64 - 1], dtype=np.uint64), 2**64 - 1),
        (np.array([1, 2, 3], dtype=np.int32), None),
        (np.array([1.7e18]), None),  # floats were rounded before ROOT Explorer read them
        (np.array([], dtype=np.int64), None),
        (ak.Array([[B, None], [], None]), B),  # missing values and collections are ignored
        (ak.Array([{"x": B}]), B),  # integer fields of records are inspected too
        (ak.Array(["text"]), None),
        (B, B),  # a single event's scalar
    ],
)
def test_largest_inexact_integer(values, expected) -> None:
    assert largest_inexact_integer(values) == expected


@pytest.mark.parametrize(
    ("spec", "branches"),
    [
        (VariableSpec(mode="branch", branch="a"), ["a"]),
        (VariableSpec(mode="max", branch="hit_id"), ["hit_id"]),
        (VariableSpec(mode="multiplicity", branch="hit_id"), []),  # counts, never values
        (VariableSpec(mode="difference", branch_a="a", branch_b="b"), ["a", "b"]),
        (
            VariableSpec(mode="delta_r", branch_a="p", branch_b="q", branch_c="r", branch_d="s"),
            ["p", "q", "r", "s"],
        ),
    ],
)
def test_variable_value_branches(spec, branches) -> None:
    assert variable_value_branches(spec) == branches


@pytest.mark.parametrize(
    ("text", "inexact"),
    [
        ("9007199254740992", False),  # 2**53
        ("9007199254740993", True),  # 2**53 + 1
        (" -9007199254740993 ", True),
        (str(B + 1), True),
        ("1e18", False),  # not an integer literal; 1e18 is exact anyway
        ("1.5", False),
        ("40", False),
        ("inf", False),
        ("not a number", False),
        ("1" + "0" * 400, True),  # beyond float64 altogether
    ],
)
def test_inexact_integer_bounds(text, inexact) -> None:
    assert is_inexact_integer_bound(text) is inexact


# --- Real files -------------------------------------------------------------------------


def test_difference_of_timestamps_warns_for_both_inputs(precision_file) -> None:
    warnings = _warnings(
        precision_file, VariableSpec(mode="difference", branch_a="a", branch_b="b")
    )
    assert len(warnings) == 2
    assert warnings[0].startswith("`a` holds integers up to 1.7e+18 in magnitude, beyond 2⁵³")
    assert warnings[1].startswith("`b` holds integers up to 1.7e+18")
    # The spacing of float64 near 1.7e18, as in the issue's example.
    assert "values less than 256 apart become equal" in warnings[0]


def test_unsigned_branch_warns(precision_file) -> None:
    (warning,) = _warnings(precision_file, VariableSpec(mode="branch", branch="packed"))
    assert warning.startswith("`packed` holds integers up to 1.84e+19")


def test_ordinary_and_float_branches_do_not_warn(precision_file) -> None:
    assert not _warnings(
        precision_file,
        VariableSpec(mode="branch", branch="small"),
        VariableSpec(mode="branch", branch="large_float"),
        VariableSpec(mode="branch", branch="x"),
    )


def test_collections_warn_for_values_but_not_for_counts(precision_file) -> None:
    assert _warnings(precision_file, VariableSpec(mode="nth", branch="hit_id"))
    assert not _warnings(precision_file, VariableSpec(mode="multiplicity", branch="hit_id"))


def test_each_branch_warns_once_including_weights(precision_file) -> None:
    warnings = _warnings(
        precision_file,
        VariableSpec(mode="branch", branch="a"),
        VariableSpec(mode="branch", branch="a"),
        None,
        extra=["a", None],
    )
    assert len(warnings) == 1


def test_single_event_scope_checks_only_that_event(precision_file) -> None:
    spec = VariableSpec(mode="nth", branch="hit_id")
    assert _warnings(precision_file, spec, event_index=0)
    assert not _warnings(precision_file, spec, event_index=2)  # holds only [5]


def test_unreadable_branches_are_skipped(precision_file) -> None:
    assert not _warnings(precision_file, VariableSpec(mode="branch", branch="missing"))


# --- Application --------------------------------------------------------------------------


def _app(path: Path) -> AppTest:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20).run()
    return at.text_input(key="root_path").input(str(path)).run()


def _precision_warnings(elements) -> list[str]:
    return [element.value for element in elements if WARNING in element.value]


def test_plotting_a_large_integer_branch_warns_and_ordinary_does_not(tmp_path) -> None:
    path = tmp_path / "timestamps.root"
    with uproot.recreate(path) as root_file:
        root_file.mktree("Events", _event_fields())
    at = _app(path)

    at.selectbox(key="x_branch").select("small").run()
    at.button(key="plot_make").click().run()
    assert not at.exception
    assert not _precision_warnings(at.warning)

    # `a` spans only 255 near 1.7e18: 100 bins would be narrower than the float64
    # spacing of 256, which used to crash the app.
    at.selectbox(key="x_branch").select("a").run()
    at.button(key="plot_make").click().run()
    assert not at.exception
    assert any("Cannot create 100 distinct histogram bins" in w.value for w in at.warning)
    (warning,) = _precision_warnings(at.warning)
    assert warning.startswith("`a` holds integers")
    assert "plot_result" not in at.session_state

    # A widely spread large branch plots normally and keeps the warning with its result.
    at.selectbox(key="x_branch").select("packed").run()
    at.button(key="plot_make").click().run()
    assert not at.exception
    (warning,) = _precision_warnings(at.warning)
    assert warning.startswith("`packed` holds integers")
    assert warning in at.session_state["plot_result"]["warnings"]


def test_cut_on_a_large_integer_branch_and_bound_warn(tmp_path) -> None:
    path = tmp_path / "timestamps.root"
    with uproot.recreate(path) as root_file:
        root_file.mktree("Events", _event_fields())
    at = _app(path)
    at.selectbox(key="x_branch").select("x").run()
    at.number_input(key="plot_n_cuts").set_value(1).run()
    at.selectbox(key="plot_cut_0_branch").select("a")
    # B itself is a multiple of 256 and exact; B + 1 is not.
    at.text_input(key="plot_cut_0_max").input(str(B)).run()
    assert not [w.value for w in at.warning if w.value.startswith("Cut 1 maximum")]
    at.text_input(key="plot_cut_0_max").input(str(B + 1)).run()

    bound_warnings = [w.value for w in at.warning if w.value.startswith("Cut 1 maximum")]
    assert bound_warnings == [
        f"Cut 1 maximum {B + 1} is beyond 2⁵³ and is used as {float(B + 1):.17g}, the "
        "nearest floating-point value. See Numeric precision in the user guide."
    ]
    at.button(key="plot_make").click().run()
    assert not at.exception
    assert any(w.startswith("`a` holds integers") for w in _precision_warnings(at.warning))


def test_ranking_by_a_large_integer_branch_warns_but_multiplicity_does_not(tmp_path) -> None:
    path = tmp_path / "timestamps.root"
    with uproot.recreate(path) as root_file:
        root_file.mktree("Events", _event_fields())
    at = _app(path)
    at.selectbox(key="metric_branch").select("hit_id")
    at.selectbox(key="metric_mode").select("max").run()
    at.button(key="rank_compute").click().run()
    assert not at.exception
    assert any(w.startswith("`hit_id` holds integers") for w in _precision_warnings(at.warning))

    at.selectbox(key="metric_mode").select("multiplicity").run()
    at.button(key="rank_compute").click().run()
    assert not at.exception
    assert not _precision_warnings(at.warning)


def test_histogram_ranges_too_narrow_for_float64_raise_a_range_error() -> None:
    from root_explorer.analysis.histograms import (
        InvalidHistogramRangeError,
        build_histogram_1d,
        build_histogram_2d,
    )

    timestamps = np.array([B + 1, B + 256, B + 3], dtype=np.int64).astype(float)
    with pytest.raises(InvalidHistogramRangeError, match="Cannot create 100 distinct histogram"):
        build_histogram_1d(timestamps, 100)
    with pytest.raises(InvalidHistogramRangeError, match="distinct y histogram bins"):
        build_histogram_2d(np.array([1.0, 2.0, 3.0]), timestamps, 10, 10)
    # Few enough bins for the float64 spacing still work.
    assert build_histogram_1d(timestamps, 1).values().sum() == 3
