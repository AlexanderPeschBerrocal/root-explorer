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


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("ignore", [1.0, 7.0]),
        ("fold", [3.0, 12.0]),
        ("separate", [1.0, 7.0]),
    ],
)
def test_weighted_flow_modes_preserve_original_and_expected_variances(mode, expected):
    from root_explorer.analysis.histograms import (
        histogram_for_flow_mode,
        summarise_histogram_flow,
    )

    values = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    weights = np.array([2.0, 1.0, 3.0, 4.0, 5.0])
    original = build_histogram_1d(values, 2, weights=weights, value_range=(0.0, 1.0))
    display = histogram_for_flow_mode(original, mode)
    np.testing.assert_allclose(display.values(), expected)
    np.testing.assert_allclose(original.values(flow=True), [2.0, 1.0, 7.0, 5.0])
    if mode == "fold":
        np.testing.assert_allclose(display.variances(), [5.0, 50.0])
        assert display.values().sum() == pytest.approx(weights.sum())
        assert display.variances().sum() == pytest.approx(np.square(weights).sum())
    summary = summarise_histogram_flow(original, values, weighted=True)
    assert (summary.underflow_entries, summary.in_range_entries, summary.overflow_entries) == (
        1,
        3,
        1,
    )
    assert (summary.underflow_sumw, summary.overflow_sumw) == pytest.approx((2.0, 5.0))
    assert (summary.underflow_sumw2, summary.overflow_sumw2) == pytest.approx((4.0, 25.0))


@pytest.mark.parametrize("mode", ["ignore", "fold", "separate"])
def test_unweighted_flow_modes_keep_entry_counts(mode):
    from root_explorer.analysis.histograms import histogram_for_flow_mode

    values = np.array([-1.0, 0.1, 0.8, 2.0])
    original = build_histogram_1d(values, 2, value_range=(0.0, 1.0))
    display = histogram_for_flow_mode(original, mode)
    assert original.values(flow=True).sum() == 4
    assert display.values().sum() == (4 if mode == "fold" else 2)


@pytest.mark.parametrize(
    "bounds",
    [
        {"hist_x_min_text": "nan", "hist_x_max_text": "2"},
        {"hist_x_min_text": "2", "hist_x_max_text": "1"},
        {"hist_x_min_text": "oops", "hist_x_max_text": "2"},
    ],
)
def test_invalid_manual_histogram_ranges_are_reported(bounds):
    from root_explorer.analysis.histograms import (
        InvalidHistogramRangeError,
        resolve_histogram_range,
    )

    with pytest.raises(InvalidHistogramRangeError, match="Histogram"):
        resolve_histogram_range(
            np.array([0.0, 1.0]), 10, {"manual_histogram_limits": True, **bounds}
        )


def test_histogram_limits_are_independent_of_plot_limits():
    from root_explorer.analysis.histograms import resolve_histogram_range

    values = np.array([0.0, 1.0, 5.0])
    style = {
        "manual_histogram_limits": True,
        "hist_x_min_text": "1",
        "hist_x_max_text": "4",
        "x_min_text": "-100",
        "x_max_text": "100",
    }
    assert resolve_histogram_range(values, 4, style) == (1.0, 4.0)
    style["x_min_text"] = "2"
    style["x_max_text"] = "3"
    assert resolve_histogram_range(values, 4, style) == (1.0, 4.0)


def test_unit_area_density_respects_variable_bin_widths() -> None:
    import hist

    from root_explorer.analysis.histograms import normalise_histogram_1d

    source = hist.Hist(hist.axis.Variable([0.0, 1.0, 3.0]), storage=hist.storage.Weight())
    source.fill([0.5, 2.0], weight=[2.0, 2.0])
    normalised = normalise_histogram_1d(source)
    assert normalised.valid
    np.testing.assert_allclose(normalised.histogram.values(), [0.5, 0.25])
    assert np.dot(normalised.histogram.values(), [1.0, 2.0]) == pytest.approx(1.0)
    np.testing.assert_allclose(normalised.histogram.variances(), [0.25, 0.0625])
    np.testing.assert_allclose(source.values(), [2.0, 2.0])


def test_unit_area_density_preserves_weighted_variance_scale_and_flow() -> None:
    from root_explorer.analysis.histograms import normalise_histogram_1d

    source = build_histogram_1d(
        [-1.0, 0.2, 0.7, 2.0],
        2,
        weights=np.array([4.0, 2.0, 3.0, 8.0]),
        value_range=(0.0, 1.0),
    )
    normalised = normalise_histogram_1d(source)
    assert normalised.valid
    assert normalised.denominator == pytest.approx(5.0)
    np.testing.assert_allclose(normalised.histogram.values(), [0.8, 1.2])
    np.testing.assert_allclose(normalised.histogram.variances(), [0.64, 1.44])
    np.testing.assert_allclose(normalised.histogram.values(flow=True)[[0, -1]], [0, 0])
    np.testing.assert_allclose(source.values(flow=True), [4, 2, 3, 8])


def test_unit_area_signed_weights_are_not_called_probabilities() -> None:
    from root_explorer.analysis.histograms import normalise_histogram_1d

    source = build_histogram_1d(
        [0.1, 0.8],
        2,
        weights=np.array([-1.0, 3.0]),
        value_range=(0.0, 1.0),
    )
    result = normalise_histogram_1d(
        source,
        absolute_weight_sum=4.0,
        has_negative_weights=True,
    )
    assert result.valid and result.signed_weights
    assert result.warning is not None
    np.testing.assert_allclose(result.histogram.values(), [-1.0, 3.0])
    assert np.dot(result.histogram.values(), [0.5, 0.5]) == pytest.approx(1.0)


@pytest.mark.parametrize("weights", [[], [1.0, -1.0], [-1.0, -2.0]])
def test_invalid_unit_area_returns_diagnostic_without_nan(weights) -> None:
    from root_explorer.analysis.histograms import normalise_histogram_1d

    values = np.arange(len(weights), dtype=float) + 0.5
    source = build_histogram_1d(
        values,
        2,
        weights=np.asarray(weights),
        value_range=(0.0, 3.0),
    )
    result = normalise_histogram_1d(source)
    assert not result.valid
    assert result.histogram is None
    assert result.error


def test_unit_area_rejects_nearly_cancelled_weights_but_not_small_positive_yields() -> None:
    from root_explorer.analysis.histograms import normalise_histogram_1d

    source = build_histogram_1d(
        [0.1],
        2,
        weights=np.array([1e-20]),
        value_range=(0.0, 1.0),
    )
    assert normalise_histogram_1d(source).valid
    assert not normalise_histogram_1d(
        source,
        absolute_weight_sum=1.0,
        has_negative_weights=True,
    ).valid


def test_unit_area_tolerates_roundoff_in_absolute_event_weight_sum() -> None:
    from root_explorer.analysis.histograms import normalise_histogram_1d

    source = build_histogram_1d([0.25], 2, weights=np.array([1.0]), value_range=(0.0, 1.0))
    # Event-wise and histogram-wise summation can differ by one ULP.
    rounded_event_sum = np.nextafter(1.0, 0.0)
    result = normalise_histogram_1d(source, absolute_weight_sum=rounded_event_sum)
    assert result.valid
    np.testing.assert_allclose(
        np.sum(result.histogram.values() * np.diff(result.histogram.axes[0].edges)),
        1.0,
    )
    assert not normalise_histogram_1d(source, absolute_weight_sum=0.9).valid
