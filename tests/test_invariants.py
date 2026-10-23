"""Generated-input checks of alignment and histogram invariants (#153).

Each test writes a small generated RNTuple or TTree, runs the production code and
compares the result with an explicit reference written directly from the
documented rules. Examples are derandomised and bounded, so runs are
reproducible, and no example database is written.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from root_explorer.analysis.arrays import ArrayShapeMismatchError, EmptySelectionError
from root_explorer.analysis.histograms import build_histogram_1d, histogram_for_flow_mode
from root_explorer.analysis.variables import evaluate_variable
from root_explorer.models import VariableSpec
from root_explorer.plotting.pipeline import StaticPlotSpec, prepare_static_plot

BOUNDED = settings(
    max_examples=40,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)

values = st.floats(-10, 10, allow_nan=False).map(lambda value: round(value, 3))
maybe = st.one_of(st.none(), values)


def _write_rntuple(fields: dict[str, tuple[list, str]]) -> tuple[Path, tempfile.TemporaryDirectory]:
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / "generated.root"
    arrays = {
        name: ak.enforce_type(ak.from_iter(data), kind) for name, (data, kind) in fields.items()
    }
    with uproot.recreate(path) as root_file:
        root_file.mkrntuple("Events", arrays)
    return path, directory


def _spec(path: Path, n_events: int, plot_type: str, **options) -> StaticPlotSpec:
    return StaticPlotSpec(
        root_path=str(path),
        tree_name="Events",
        entry_start=0,
        entry_stop=n_events,
        plot_type=plot_type,
        x_spec=options["x_spec"],
        y_spec=options.get("y_spec"),
        color_spec=options.get("color_spec"),
        weight_branch=options.get("weight_branch"),
        cut_specs=tuple(options.get("cut_specs", ())),
        cut_bounds=tuple(options.get("cut_bounds", ())),
        style={"bins_x": 5, "bins_y": 5, "max_points": 100_000},
        labels={},
    )


events_with_objects = st.lists(
    st.tuples(
        st.lists(st.tuples(maybe, maybe, maybe), max_size=3),  # objects: a, b, c
        maybe,  # event weight w
        maybe,  # event cut value s
    ),
    min_size=1,
    max_size=8,
)


@BOUNDED
@given(events_with_objects)
def test_object_values_weights_and_cuts_share_one_selection(events) -> None:
    lower, upper = -2.0, 2.0
    path, directory = _write_rntuple(
        {
            "a": ([[obj[0] for obj in objects] for objects, _, _ in events], "var * ?float64"),
            "b": ([[obj[1] for obj in objects] for objects, _, _ in events], "var * ?float64"),
            "w": ([w for _, w, _ in events], "?float64"),
            "s": ([s for _, _, s in events], "?float64"),
        }
    )
    with directory:
        # Reference from the documented rules: an object is excluded, and counted
        # once, if any participating value is missing; range-cut failures are not
        # counted. The event weight and cut value apply to every object of an event.
        expected_x, expected_weights, expected_excluded = [], [], 0
        for objects, weight, cut_value in events:
            for a, b, _ in objects:
                if a is None or b is None or weight is None or cut_value is None:
                    expected_excluded += 1
                elif lower <= cut_value <= upper:
                    expected_x.append(a - b)
                    expected_weights.append(weight)

        spec = _spec(
            path,
            len(events),
            "1D histogram",
            x_spec=VariableSpec(mode="difference", branch_a="a", branch_b="b"),
            weight_branch="w",
            cut_specs=[VariableSpec(mode="branch", branch="s")],
            cut_bounds=[(lower, upper)],
        )
        if not expected_x:
            with pytest.raises(EmptySelectionError):
                prepare_static_plot(spec)
            return
        prepared = prepare_static_plot(spec)
        assert prepared.x.tolist() == expected_x
        assert prepared.weights.tolist() == expected_weights
        assert prepared.excluded_count == expected_excluded


@BOUNDED
@given(events_with_objects)
def test_scatter_keeps_each_object_with_its_own_coordinates_and_colour(events) -> None:
    objects = [list(event_objects) for event_objects, _, _ in events]
    path, directory = _write_rntuple(
        {
            name: ([[obj[index] for obj in event] for event in objects], "var * ?float64")
            for index, name in enumerate(["a", "b", "c"])
        }
    )
    with directory:
        expected = [
            (a, b, c)
            for event in objects
            for a, b, c in event
            if a is not None and b is not None and c is not None
        ]
        spec = _spec(
            path,
            len(events),
            "2D scatter",
            x_spec=VariableSpec(mode="branch", branch="a"),
            y_spec=VariableSpec(mode="branch", branch="b"),
            color_spec=VariableSpec(mode="branch", branch="c"),
        )
        if not expected:
            with pytest.raises(EmptySelectionError):
                prepare_static_plot(spec)
            return
        prepared = prepare_static_plot(spec)
        actual = list(
            zip(prepared.x.tolist(), prepared.y.tolist(), prepared.color.tolist(), strict=True)
        )
        assert actual == expected


collections = st.lists(
    st.one_of(st.none(), st.lists(maybe, max_size=4)),  # missing, empty or filled
    min_size=1,
    max_size=8,
)


def _reference_reduction(collection: list | None, mode: str, index: int) -> float:
    """Documented per-event reductions, computed without the production helpers."""

    if collection is None:
        return math.nan
    if mode == "nth":
        if index >= len(collection) or collection[index] is None:
            return math.nan
        return float(collection[index])
    present = [value for value in collection if value is not None]
    if mode == "sum":
        return float(sum(present))
    if mode == "mean":
        return sum(present) / len(present) if present else math.nan
    if mode == "max":
        return max(present) if present else -math.inf
    return min(present) if present else math.inf


@BOUNDED
@given(collections, st.sampled_from(["sum", "mean", "max", "min", "nth"]), st.integers(0, 2))
def test_reductions_follow_the_documented_rules(data, mode, index) -> None:
    path, directory = _write_rntuple({"o": (data, "?var * ?float64")})
    with directory:
        spec = VariableSpec(mode=mode, branch="o", index=index)
        actual, _ = evaluate_variable(spec, str(path), "Events", 0, len(data))
        expected = [_reference_reduction(collection, mode, index) for collection in data]
        np.testing.assert_allclose(np.asarray(actual, dtype=float), expected, rtol=1e-12)


@BOUNDED
@given(st.lists(st.one_of(st.none(), st.lists(values, max_size=4)), min_size=1, max_size=8))
def test_multiplicity_distinguishes_missing_and_empty_collections(data) -> None:
    path, directory = _write_rntuple({"o": (data, "?var * float64")})
    with directory:
        actual, _ = evaluate_variable(
            VariableSpec(mode="multiplicity", branch="o"), str(path), "Events", 0, len(data)
        )
        expected = [math.nan if collection is None else len(collection) for collection in data]
        np.testing.assert_array_equal(np.asarray(actual, dtype=float), expected)


@BOUNDED
@given(st.lists(st.integers(0, 3), min_size=2, max_size=6), st.randoms(use_true_random=False))
def test_equal_total_lengths_with_different_event_boundaries_are_rejected(counts, random) -> None:
    other = counts.copy()
    random.shuffle(other)
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / "boundaries.root"
    total = sum(counts)
    with uproot.recreate(path) as root_file:
        root_file.mktree(
            "Events",
            {
                "a": ak.unflatten(np.arange(total, dtype=float), counts),
                "b": ak.unflatten(np.arange(total, dtype=float) * 2, other),
            },
        )
    with directory:
        spec = VariableSpec(mode="difference", branch_a="a", branch_b="b")
        if other == counts:
            actual, _ = evaluate_variable(spec, str(path), "Events", 0, len(counts))
            np.testing.assert_array_equal(actual, -np.arange(total, dtype=float))
        else:
            # The flattened lengths match, but objects of different events must not pair.
            with pytest.raises(ArrayShapeMismatchError):
                evaluate_variable(spec, str(path), "Events", 0, len(counts))


@BOUNDED
@given(
    st.lists(
        st.tuples(st.floats(-20, 20, allow_nan=False), st.floats(-3, 3, allow_nan=False)),
        min_size=1,
        max_size=60,
    ),
    st.integers(1, 12),
)
def test_folding_flow_bins_conserves_weighted_yields(entries, bins) -> None:
    x = np.array([value for value, _ in entries])
    w = np.array([weight for _, weight in entries])
    lower, upper = -5.0, 5.0
    histogram = build_histogram_1d(x, bins, weights=w, value_range=(lower, upper))
    folded = histogram_for_flow_mode(histogram, "fold")

    # Folding moves underflow into the first and overflow into the last visible bin,
    # which is what histogramming the values clipped to the range does.
    edges = folded.axes[0].edges
    clipped = np.clip(x, lower, upper)
    expected_sum, _ = np.histogram(clipped, bins=edges, weights=w)
    expected_sum2, _ = np.histogram(clipped, bins=edges, weights=w * w)
    np.testing.assert_allclose(folded.values(), expected_sum, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(folded.variances(), expected_sum2, rtol=1e-12, atol=1e-12)

    # Nothing remains in the flow bins, and the totals are conserved.
    flow = folded.view(flow=True)
    assert flow.value[0] == flow.value[-1] == 0.0
    assert flow.variance[0] == flow.variance[-1] == 0.0
    np.testing.assert_allclose(folded.values().sum(), w.sum(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(folded.variances().sum(), (w * w).sum(), rtol=1e-12, atol=1e-12)
