"""Regression tests for missing-value alignment (#137).

TTrees cannot store missing values, so the fixtures use RNTuples with option types.
Tests assert which event or object each value comes from, not only array lengths.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot
from streamlit.testing.v1 import AppTest
from test_codegen import _execute_export, _executed_generated_module, _request

from root_explorer.analysis.arrays import ArrayShapeMismatchError, EmptySelectionError
from root_explorer.analysis.ranking import compute_event_metric
from root_explorer.analysis.variables import evaluate_variable
from root_explorer.models import VariableSpec
from root_explorer.plotting.codegen import generate_static_plot_code
from root_explorer.plotting.pipeline import StaticPlotSpec, prepare_static_plot
from root_explorer.plotting.plots import select_plot_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_EVENTS = 5


@pytest.fixture
def optional_path(tmp_path) -> Path:
    """Five events with missing values at different positions in each field."""

    path = tmp_path / "optional.root"
    with uproot.recreate(path) as root_file:
        root_file.mkrntuple(
            "Events",
            {
                "evt": ak.Array([0.0, 1.0, 2.0, 3.0, 4.0]),
                "a": ak.Array([1.0, None, 3.0, 4.0, 5.0]),
                "b": ak.Array([None, 20.0, 30.0, 40.0, 50.0]),
                "w": ak.Array([1.0, 1.0, None, 2.0, 1.0]),
                "c": ak.Array([7.0, 7.0, 7.0, None, 7.0]),
                "jet": ak.Array([[1.0, None], [2.0], [None, 4.0], [], [6.0]]),
                "jet2": ak.Array([[10.0, 20.0], [None], [30.0, None], [], [60.0]]),
                "jet_other": ak.Array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
                "opt": ak.Array([[1.0, 2.0], None, [], [5.0], [6.0]]),
            },
        )
    return path


def _evaluate(path: Path, event_index: int | None = None, **spec) -> list[float]:
    values, _ = evaluate_variable(
        VariableSpec(**spec), str(path), "Events", 0, N_EVENTS, event_index
    )
    return np.asarray(values, dtype=float).tolist()


def _prepare(path: Path, x: VariableSpec, y: VariableSpec | None = None, **options):
    spec = StaticPlotSpec(
        root_path=str(path),
        tree_name="Events",
        entry_start=options.pop("entry_start", 0),
        entry_stop=N_EVENTS,
        plot_type="2D scatter" if y is not None else "1D histogram",
        x_spec=x,
        y_spec=y,
        color_spec=options.pop("color_spec", None),
        weight_branch=options.pop("weight_branch", None),
        cut_specs=tuple(options.pop("cut_specs", ())),
        cut_bounds=tuple(options.pop("cut_bounds", ())),
        style={"bins_x": 5, "max_points": 1_000},
        labels={},
    )
    return prepare_static_plot(spec)


def _branch(name: str) -> VariableSpec:
    return VariableSpec(mode="branch", branch=name)


def _pairs(prepared) -> list[tuple[float, float]]:
    return list(zip(prepared.x.tolist(), prepared.y.tolist(), strict=True))


def test_derived_variable_only_uses_events_where_both_inputs_exist(optional_path) -> None:
    values = _evaluate(optional_path, mode="difference", branch_a="a", branch_b="b")
    assert np.isnan(values[0]) and np.isnan(values[1])
    assert values[2:] == [-27.0, -36.0, -45.0]

    prepared = _prepare(
        optional_path,
        VariableSpec(mode="difference", branch_a="a", branch_b="b"),
        _branch("evt"),
    )
    # Each retained difference stays paired with the event it was computed from.
    assert _pairs(prepared) == [(-27.0, 2.0), (-36.0, 3.0), (-45.0, 4.0)]
    assert prepared.excluded_count == 2


def test_direct_variable_does_not_relabel_later_values_as_earlier_events(optional_path) -> None:
    prepared = _prepare(optional_path, _branch("a"), _branch("evt"))
    assert _pairs(prepared) == [(1.0, 0.0), (3.0, 2.0), (4.0, 3.0), (5.0, 4.0)]


def test_optional_values_inside_matching_jagged_collections(optional_path) -> None:
    values = _evaluate(optional_path, mode="sum_ab", branch_a="jet", branch_b="jet2")
    # Objects: event 0 -> [1+10, None+20], event 1 -> [2+None],
    # event 2 -> [None+30, 4+None], event 3 -> [], event 4 -> [6+60].
    assert [value for value in values if np.isfinite(value)] == [11.0, 66.0]
    assert len(values) == 6

    prepared = _prepare(optional_path, _branch("jet"), _branch("jet2"))
    assert _pairs(prepared) == [(1.0, 10.0), (6.0, 60.0)]


def test_incompatible_jagged_structures_still_fail(optional_path) -> None:
    with pytest.raises(ArrayShapeMismatchError):
        _evaluate(optional_path, mode="sum_ab", branch_a="jet", branch_b="jet_other")


def test_missing_and_empty_collections_remain_distinguishable(optional_path) -> None:
    multiplicity = _evaluate(optional_path, mode="multiplicity", branch="opt")
    total = _evaluate(optional_path, mode="sum", branch="opt")
    # Event 1 has a missing collection, event 2 an empty one.
    assert np.isnan(multiplicity[1]) and multiplicity[2] == 0.0
    assert np.isnan(total[1]) and total[2] == 0.0
    assert multiplicity[0] == 2.0 and total[0] == 3.0


def test_reductions_skip_missing_object_values(optional_path) -> None:
    means = _evaluate(optional_path, mode="mean", branch="jet")
    assert means[0] == 1.0 and means[2] == 4.0
    assert np.isnan(means[3])  # empty collection


@pytest.mark.parametrize(
    ("mode", "threshold", "expected"),
    [
        ("max", None, {0: 1.0, 2: 3.0, 3: 4.0, 4: 5.0}),
        ("count_above", 3.5, {0: 0.0, 2: 0.0, 3: 1.0, 4: 1.0}),
    ],
)
def test_ranking_reports_absolute_events(optional_path, mode, threshold, expected) -> None:
    events, values = compute_event_metric(
        str(optional_path), "Events", "a", mode, 0, N_EVENTS, threshold, False
    )
    finite = {
        int(event): float(value)
        for event, value in zip(events, values, strict=True)
        if np.isfinite(value)
    }
    assert finite == expected  # Event 1 is missing and has no metric.

    # With an entry offset the event numbers stay absolute.
    events, values = compute_event_metric(
        str(optional_path), "Events", "a", "max", 2, N_EVENTS, None, False
    )
    assert events.tolist() == [2, 3, 4]
    assert values.tolist() == [3.0, 4.0, 5.0]


def test_cuts_weights_and_colours_share_the_missing_value_mask(optional_path) -> None:
    prepared = _prepare(
        optional_path,
        _branch("evt"),
        _branch("a"),
        weight_branch="w",
        color_spec=_branch("c"),
        cut_specs=[_branch("b")],
        cut_bounds=[(0.0, None)],
    )
    # Event 0: b missing (cut); event 1: a missing; event 2: weight missing;
    # event 3: colour missing. Only event 4 remains, with its own values.
    assert prepared.x.tolist() == [4.0]
    assert prepared.y.tolist() == [5.0]
    assert prepared.weights.tolist() == [1.0]
    assert prepared.color.tolist() == [7.0]


def test_all_missing_selection_raises_a_concise_error(optional_path) -> None:
    with pytest.raises(EmptySelectionError, match="No values remain to plot"):
        _prepare(
            optional_path,
            VariableSpec(mode="difference", branch_a="a", branch_b="b"),
            cut_specs=[_branch("evt")],
            cut_bounds=[(0.0, 1.5)],
        )


def test_single_event_missing_values(optional_path) -> None:
    assert np.isnan(_evaluate(optional_path, 1, mode="sum", branch="a")[0])
    assert _evaluate(optional_path, 2, mode="difference", branch_a="a", branch_b="b") == [-27.0]
    with pytest.raises(EmptySelectionError):
        spec = StaticPlotSpec(
            root_path=str(optional_path),
            tree_name="Events",
            entry_start=0,
            entry_stop=N_EVENTS,
            plot_type="1D histogram",
            x_spec=VariableSpec(mode="difference", branch_a="a", branch_b="b"),
            y_spec=None,
            color_spec=None,
            weight_branch=None,
            cut_specs=(),
            cut_bounds=(),
            style={"bins_x": 5},
            labels={},
            event_index=1,
        )
        prepare_static_plot(spec)


def test_standalone_plot_code_reproduces_the_alignment(optional_path) -> None:
    request = _request(
        optional_path,
        plot_type="2D scatter",
        x_spec=VariableSpec(mode="difference", branch_a="a", branch_b="b"),
        y_spec=_branch("evt"),
        style={
            "fig_width": 6.0,
            "fig_height": 4.0,
            "marker_size_2d": 3.0,
            "opacity_2d": 0.5,
            "max_points": 1_000,
            "log_x": False,
            "log_y": False,
            "equal_aspect": False,
            "grid": False,
        },
    )
    _, canonical, generated = _execute_export(request)
    try:
        canonical_points = canonical.axes[0].collections[0].get_offsets().tolist()
        generated_points = generated.axes[0].collections[0].get_offsets().tolist()
        assert generated_points == canonical_points == [[-27.0, 2.0], [-36.0, 3.0], [-45.0, 4.0]]
    finally:
        import matplotlib.pyplot as plt

        plt.close(canonical)
        plt.close(generated)


def test_app_reports_excluded_values_and_empty_selections(optional_path) -> None:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20).run()
    at.text_input(key="root_path").input(str(optional_path)).run()
    at.selectbox(key="x_branch").select("a").run()
    at.button(key="plot_make").click().run()
    assert not at.exception
    assert any(
        "1 value was excluded because a plotted variable, weight, colour or cut value was missing"
        in caption.value
        for caption in at.caption
    )

    # A cut that excludes every event leaves nothing to plot.
    at.number_input(key="plot_n_cuts").set_value(1).run()
    at.selectbox(key="plot_cut_0_branch").select("evt").run()
    at.text_input(key="plot_cut_0_max").input("-1").run()
    at.button(key="plot_make").click().run()
    assert not at.exception
    assert any("No values remain to plot" in warning.value for warning in at.warning)
    assert "plot_result" not in at.session_state


def test_selection_counts_invalid_inputs_once_and_keeps_range_failures_separate() -> None:
    x = np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, np.inf, 1.0, 1.0])
    first_cut = np.array([np.nan, 2.0, np.nan, 2.0, 2.0, -1.0, 2.0])
    second_cut = np.array([2.0, np.inf, 2.0, 2.0, 2.0, 2.0, 2.0])

    selected, excluded = select_plot_values(
        [x, weights], [(first_cut, 0.0, None), (second_cut, None, None)]
    )

    # Positions 0, 1, 2 and 4 are invalid; position 2 is counted only once.
    # Position 5 fails an ordinary bound and must not increase the invalid count.
    assert excluded == 4
    assert selected[0].tolist() == [3.0, 6.0]
    assert selected[1].tolist() == [1.0, 1.0]


@pytest.mark.parametrize("plot_type", ["1D histogram", "3D scatter interactive"])
def test_app_counts_missing_cut_values(optional_path, plot_type) -> None:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20).run()
    at.text_input(key="root_path").input(str(optional_path)).run()
    at.selectbox(key="plot_type").select(plot_type).run()
    at.selectbox(key="x_branch").select("evt")
    if plot_type == "3D scatter interactive":
        at.selectbox(key="y_branch").select("evt")
        at.selectbox(key="z_branch").select("evt")
    at.run()
    at.number_input(key="plot_n_cuts").set_value(1).run()
    at.selectbox(key="plot_cut_0_branch").select("b").run()
    at.text_input(key="plot_cut_0_min").input("0").run()
    at.button(key="plot_make").click().run()

    assert not at.exception
    result = at.session_state["plot_result"]
    assert result["summary"].iloc[0]["n"] == 4
    assert result["exclusion_note"].startswith("1 value was excluded")
    assert any(result["exclusion_note"] == caption.value for caption in at.caption)


@pytest.mark.parametrize("plot_type", ["1D histogram", "2D histogram", "2D scatter"])
@pytest.mark.parametrize("empty_reason", ["missing_axis", "missing_cut", "range_cut"])
def test_export_rejects_empty_selections_like_the_app(
    optional_path, plot_type, empty_reason
) -> None:
    request = _request(
        optional_path,
        plot_type=plot_type,
        x_spec=_branch("b") if empty_reason == "missing_axis" else _branch("evt"),
        y_spec=None if plot_type == "1D histogram" else _branch("evt"),
        cut_specs=[_branch("b")] if empty_reason != "missing_axis" else [],
        cut_bounds=[(0.0, None)] if empty_reason != "missing_axis" else [],
        style={"fig_width": 6.0, "fig_height": 4.0, "bins_x": 5, "bins_y": 5},
    )
    # Export a usable selection first, then edit the script's range or cut bounds.
    source = generate_static_plot_code(request)
    spec = request.to_static_spec()
    if empty_reason == "range_cut":
        # Only finite cut values, all outside the requested range.
        spec = replace(spec, entry_start=1, entry_stop=N_EVENTS, cut_bounds=((100.0, None),))
    else:
        # Event 0 has b missing, either as an axis or as a cut value.
        spec = replace(spec, entry_start=0, entry_stop=1)
    with pytest.raises(EmptySelectionError) as app_error:
        prepare_static_plot(spec)

    with _executed_generated_module(source) as module:
        module.PLOT = replace(
            module.PLOT,
            entry_start=spec.entry_start,
            entry_stop=spec.entry_stop,
            cut_bounds=spec.cut_bounds,
        )
        with pytest.raises(module.EmptySelectionError) as export_error:
            module.main()
    assert str(export_error.value) == str(app_error.value)
    if empty_reason != "range_cut":
        assert "1 value was excluded" in str(export_error.value)


def test_3d_scatter_excludes_events_with_missing_x_y_or_z(optional_path) -> None:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20).run()
    at.text_input(key="root_path").input(str(optional_path)).run()
    at.selectbox(key="plot_type").select("3D scatter interactive").run()
    at.selectbox(key="x_branch").select("evt")
    at.selectbox(key="y_branch").select("a")  # missing in event 1
    at.selectbox(key="z_branch").select("b").run()  # missing in event 0
    at.button(key="plot_make").click().run()
    assert not at.exception
    trace = at.session_state["plot_result"]["figure"].data[0]
    # Every point keeps its own event's values: (evt, a, b).
    points = list(zip(trace.x, trace.y, trace.z, strict=True))
    assert points == [(2.0, 3.0, 30.0), (3.0, 4.0, 40.0), (4.0, 5.0, 50.0)]
    assert at.session_state["plot_result"]["exclusion_note"].startswith("2 values were excluded")

    at.number_input(key="plot_n_cuts").set_value(1).run()
    at.selectbox(key="plot_cut_0_branch").select("evt").run()
    at.text_input(key="plot_cut_0_max").input("-1").run()
    at.button(key="plot_make").click().run()
    assert not at.exception
    assert any("No values remain to plot" in warning.value for warning in at.warning)
