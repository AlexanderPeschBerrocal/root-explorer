from __future__ import annotations

import numpy as np
import pytest

from root_explorer.analysis.arrays import ArrayShapeMismatchError
from root_explorer.analysis.histograms import build_histogram_1d


def test_unweighted_histogram_matches_numpy_visible_bins() -> None:
    values = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

    histogram = build_histogram_1d(values, 4, label="observable")
    expected_counts, expected_edges = np.histogram(values, bins=4)

    np.testing.assert_allclose(histogram.values(), expected_counts)
    np.testing.assert_allclose(histogram.axes[0].edges, expected_edges)
    assert histogram.axes[0].label == "observable"
    assert histogram.values(flow=True).sum() == pytest.approx(len(values))


def test_weighted_histogram_preserves_sumw_sumw2_and_flow_bins() -> None:
    values = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    weights = np.array([2.0, 1.0, 3.0, 4.0, 5.0])

    histogram = build_histogram_1d(
        values,
        2,
        weights=weights,
        value_range=(0.0, 1.0),
        label="x",
    )

    np.testing.assert_allclose(histogram.values(flow=True), [2.0, 1.0, 7.0, 5.0])
    np.testing.assert_allclose(histogram.variances(flow=True), [4.0, 1.0, 25.0, 25.0])
    np.testing.assert_allclose(histogram.values(), [1.0, 7.0])
    np.testing.assert_allclose(histogram.variances(), [1.0, 25.0])


def test_histogram_rejects_misaligned_weights() -> None:
    with pytest.raises(ArrayShapeMismatchError, match="1D histogram values"):
        build_histogram_1d(
            np.array([1.0, 2.0]),
            2,
            weights=np.array([1.0]),
        )


@pytest.mark.parametrize(
    "value_range",
    [
        (1.0, 1.0),
        (2.0, 1.0),
        (np.nan, 1.0),
        (0.0, np.inf),
    ],
)
def test_histogram_rejects_invalid_explicit_range(
    value_range: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="range bounds"):
        build_histogram_1d(np.array([0.5]), 2, value_range=value_range)
