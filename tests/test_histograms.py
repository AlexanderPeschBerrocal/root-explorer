from __future__ import annotations

import numpy as np
import pytest

from root_explorer.analysis.arrays import ArrayShapeMismatchError
from root_explorer.analysis.histograms import build_histogram_1d, build_histogram_2d


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


@pytest.mark.parametrize(
    "x, y",
    [
        (np.array([0.0, 0.2, 0.7, 1.0]), np.array([-1.0, 0.0, 1.0, 2.0])),
        (np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])),
        (np.array([], dtype=float), np.array([], dtype=float)),
    ],
)
def test_unweighted_2d_histogram_matches_numpy_visible_bins(x: np.ndarray, y: np.ndarray) -> None:
    histogram = build_histogram_2d(x, y, 4, 3, x_label="x observable", y_label="y observable")
    values, x_edges, y_edges = np.histogram2d(x, y, bins=(4, 3))

    np.testing.assert_allclose(histogram.values(), values)
    np.testing.assert_allclose(histogram.axes[0].edges, x_edges)
    np.testing.assert_allclose(histogram.axes[1].edges, y_edges)
    assert histogram.axes[0].label == "x observable"
    assert histogram.axes[1].label == "y observable"
    assert histogram.values(flow=True).shape == (6, 5)
    assert histogram.values(flow=True).sum() == pytest.approx(len(x))


def test_weighted_2d_histogram_preserves_variances_and_all_flow_regions() -> None:
    x = np.array([-1.0, 0.0, 0.25, 1.0, 2.0, 1.0])
    y = np.array([-1.0, 0.25, 1.0, 2.0, 0.75, 1.0])
    weights = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    histogram = build_histogram_2d(
        x, y, 2, 2, weights=weights, x_range=(0.0, 1.0), y_range=(0.0, 1.0)
    )
    expected = np.zeros((4, 4))
    expected_variances = np.zeros((4, 4))
    for i, j, weight in (
        (0, 0, 2.0),
        (1, 1, 3.0),
        (1, 2, 5.0),
        (2, 3, 7.0),
        (3, 2, 11.0),
        (2, 2, 13.0),
    ):
        expected[i, j] = weight
        expected_variances[i, j] = weight**2
    np.testing.assert_allclose(histogram.values(flow=True), expected)
    np.testing.assert_allclose(histogram.variances(flow=True), expected_variances)
    np.testing.assert_allclose(histogram.values(), expected[1:-1, 1:-1])
    assert histogram.values(flow=True).sum() == pytest.approx(weights.sum())


@pytest.mark.parametrize(
    "x, y, weights",
    [
        (np.array([1.0, 2.0]), np.array([1.0]), None),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0])),
    ],
)
def test_2d_histogram_rejects_misaligned_inputs(x, y, weights) -> None:
    with pytest.raises(ArrayShapeMismatchError, match="2D histogram values"):
        build_histogram_2d(x, y, 2, 2, weights=weights)


@pytest.mark.parametrize(
    "x, y, weights",
    [
        (np.array([np.nan]), np.array([1.0]), None),
        (np.array([1.0]), np.array([np.inf]), None),
        (np.array([1.0]), np.array([2.0]), np.array([np.nan])),
    ],
)
def test_2d_histogram_rejects_nonfinite_inputs(x, y, weights) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        build_histogram_2d(x, y, 2, 2, weights=weights)


@pytest.mark.parametrize("bins_x, bins_y", [(0, 2), (2, -1), (2.5, 3)])
def test_2d_histogram_rejects_invalid_bins(bins_x, bins_y) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_histogram_2d(np.array([0.5]), np.array([0.5]), bins_x, bins_y)


@pytest.mark.parametrize(
    "x_range, y_range",
    [
        ((0.0, 0.0), None),
        (None, (1.0, 0.0)),
        ((-np.inf, 1.0), None),
        (None, (0.0, np.nan)),
    ],
)
def test_2d_histogram_rejects_invalid_ranges(x_range, y_range) -> None:
    with pytest.raises(ValueError, match="range bounds"):
        build_histogram_2d(np.array([0.5]), np.array([0.5]), 2, 2, x_range=x_range, y_range=y_range)
