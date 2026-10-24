from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from root_explorer.analysis import ranking


@pytest.mark.parametrize(
    ("mode", "threshold", "abs_value", "expected"),
    [
        ("multiplicity", None, False, [2.0, 0.0, 1.0]),
        ("sum", None, True, [3.0, 0.0, 3.0]),
        ("max", None, False, [1.0, np.nan, 3.0]),
        ("count_above", 1.5, False, [0.0, 0.0, 1.0]),
    ],
)
def test_compute_event_metric_for_jagged_branches(
    monkeypatch, mode: str, threshold: float | None, abs_value: bool, expected: list[float]
) -> None:
    monkeypatch.setattr(ranking, "read_field", lambda *args: ak.Array([[1.0, -2.0], [], [3.0]]))
    monkeypatch.setattr(ranking, "source_version", lambda path: ("local", path, 0, 0))

    events, values = ranking.compute_event_metric(
        "fixture.root", "Events", "Muon_pt", mode, 7, 10, threshold, abs_value
    )

    np.testing.assert_array_equal(events, [7, 8, 9])
    np.testing.assert_allclose(values, expected, equal_nan=True)


def test_metric_dataframe_drops_nonfinite_values_and_limits_rows() -> None:
    frame = ranking.metric_dataframe(
        np.array([10, 11, 12]), np.array([2.0, np.nan, 3.0]), descending=True, max_rows=1
    )

    assert frame.to_dict("records") == [{"event": 12, "metric": 3.0}]


@pytest.mark.parametrize("descending", [True, False])
def test_metric_dataframe_orders_ties_by_event_number(descending: bool) -> None:
    # Enough ties that an unstable sort would scramble the event order.
    rng = np.random.default_rng(1)
    events = rng.permutation(np.arange(100, 400))
    values = (events % 3).astype(float)
    frame = ranking.metric_dataframe(events, values, descending=descending, max_rows=300)

    metric_order = sorted({0.0, 1.0, 2.0}, reverse=descending)
    expected = [
        (int(event), metric)
        for metric in metric_order
        for event in range(100, 400)
        if event % 3 == metric
    ]
    assert list(zip(frame["event"], frame["metric"], strict=True)) == expected
