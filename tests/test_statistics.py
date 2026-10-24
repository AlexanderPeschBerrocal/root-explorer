from __future__ import annotations

import numpy as np
import pytest

from root_explorer.analysis.arrays import ArrayShapeMismatchError
from root_explorer.analysis.statistics import (
    calculate_quantile_result,
    calculate_threshold_result,
    describe_weighted_array,
)


def test_unweighted_threshold_uses_complementary_boundary_convention() -> None:
    result = calculate_threshold_result(np.array([1.0, 2.0, 3.0]), 2.0)

    assert result.total_count == 3
    assert result.below_count == 2
    assert result.above_count == 1
    assert result.below_fraction == pytest.approx(2.0 / 3.0)
    assert result.above_fraction == pytest.approx(1.0 / 3.0)
    assert not result.weighted


def test_weighted_threshold_uses_weight_sums_for_both_fractions() -> None:
    result = calculate_threshold_result(
        np.array([1.0, 2.0, 3.0]),
        2.0,
        np.array([1.0, 2.0, 3.0]),
    )

    assert result.total_weight == pytest.approx(6.0)
    assert result.below_weight == pytest.approx(3.0)
    assert result.above_weight == pytest.approx(3.0)
    assert result.below_fraction == pytest.approx(0.5)
    assert result.above_fraction == pytest.approx(0.5)


def test_signed_weighted_fractions_are_not_clipped() -> None:
    result = calculate_threshold_result(
        np.array([1.0, 2.0, 3.0]),
        2.0,
        np.array([2.0, -3.0, 2.0]),
    )

    assert result.has_negative_weights
    assert result.total_weight == pytest.approx(1.0)
    assert result.below_fraction == pytest.approx(-1.0)
    assert result.above_fraction == pytest.approx(2.0)


def test_empty_sample_and_zero_total_weight_have_no_fraction() -> None:
    empty = calculate_threshold_result(np.array([]), 0.0)
    assert empty.total_count == 0
    assert empty.below_fraction is None
    assert empty.above_fraction is None

    zero_weight = calculate_threshold_result(
        np.array([1.0, 2.0]),
        1.0,
        np.array([1.0, -1.0]),
    )
    assert zero_weight.total_weight == pytest.approx(0.0)
    assert zero_weight.below_fraction is None
    assert zero_weight.above_fraction is None


def test_threshold_calculator_rejects_misaligned_or_nonfinite_inputs() -> None:
    with pytest.raises(ArrayShapeMismatchError, match="No values were truncated"):
        calculate_threshold_result(np.array([1.0, 2.0]), 1.0, np.array([1.0]))

    with pytest.raises(ValueError, match="finite"):
        calculate_threshold_result(np.array([1.0, np.nan]), 1.0)

    with pytest.raises(ValueError, match="Threshold must be finite"):
        calculate_threshold_result(np.array([1.0]), np.nan)


def test_quantiles_support_unweighted_and_nonnegative_weighted_samples() -> None:
    unweighted = calculate_quantile_result(np.array([1.0, 2.0, 3.0, 4.0]), 50.0)
    assert unweighted.value == pytest.approx(2.5)
    assert not unweighted.weighted

    weighted = calculate_quantile_result(np.array([1.0, 2.0, 3.0]), 50.0, np.array([1.0, 1.0, 8.0]))
    assert weighted.value == pytest.approx(3.0)
    assert weighted.weighted


def test_signed_weighted_quantile_is_explicitly_unavailable() -> None:
    result = calculate_quantile_result(np.array([1.0, 2.0, 3.0]), 50.0, np.array([1.0, -0.5, 1.0]))
    assert result.value is None
    assert result.has_negative_weights


def test_weighted_summary_matches_direct_calculation() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([1.0, 1.0, 2.0, 0.5])
    summary = describe_weighted_array(values, weights)
    mean = np.average(values, weights=weights)
    assert summary["sum of weights"] == pytest.approx(4.5)
    assert summary["N_eff"] == pytest.approx(4.5**2 / np.sum(weights**2))
    assert summary["weighted mean"] == pytest.approx(mean)
    assert summary["weighted std"] == pytest.approx(
        np.sqrt(np.average((values - mean) ** 2, weights=weights))
    )
    assert "weighted median" not in summary


def test_uniform_weights_reproduce_unweighted_statistics() -> None:
    values = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    summary = describe_weighted_array(values, np.full(len(values), 2.5))
    assert summary["weighted mean"] == pytest.approx(np.mean(values))
    assert summary["weighted std"] == pytest.approx(np.std(values))
    assert summary["N_eff"] == pytest.approx(len(values))


def test_negative_weights_with_positive_sum_keep_weighted_mean() -> None:
    summary = describe_weighted_array(np.array([1.0, 2.0, 3.0]), np.array([1.0, -0.5, 1.0]))
    assert summary["sum of weights"] == pytest.approx(1.5)
    assert summary["weighted mean"] == pytest.approx((1.0 - 1.0 + 3.0) / 1.5)


@pytest.mark.parametrize(
    ("values", "weights", "undefined"),
    [
        ([1.0, 2.0], [-1.0, 0.5], ("weighted mean", "weighted std")),  # sum of weights < 0
        ([1.0, 2.0], [1.0, -1.0], ("weighted mean", "weighted std")),  # sum of weights = 0
        ([0.0, 10.0], [2.0, -1.0], ("weighted std",)),  # negative weighted variance
    ],
)
def test_undefined_weighted_quantities_are_nan(values, weights, undefined) -> None:
    summary = describe_weighted_array(np.array(values), np.array(weights))
    for key in undefined:
        assert np.isnan(summary[key])
    assert np.isfinite(summary["sum of weights"])


def test_empty_weighted_summary() -> None:
    summary = describe_weighted_array(np.array([]), np.array([]))
    assert summary["sum of weights"] == 0.0
    assert all(np.isnan(summary[key]) for key in ("N_eff", "weighted mean", "weighted std"))


def test_weighted_summary_rejects_misaligned_inputs() -> None:
    with pytest.raises(ArrayShapeMismatchError):
        describe_weighted_array(np.array([1.0, 2.0]), np.array([1.0]))


@pytest.mark.parametrize("scale", [1e154, 1e200, 1e-200, 1e-300])
@pytest.mark.parametrize("weights", [[1.0, 1.0], [2.0, 1.0], [2.0, -1.0], [1.0, -1.0]])
def test_weighted_summary_is_invariant_under_extreme_weight_scaling(scale, weights) -> None:
    values = np.array([1.0, 2.0])
    weights = np.array(weights)
    expected = describe_weighted_array(values, weights)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        actual = describe_weighted_array(values, weights * scale)
    for key in ("N_eff", "weighted mean", "weighted std"):
        assert actual[key] == pytest.approx(expected[key], nan_ok=True)
    assert actual["sum of weights"] / scale == pytest.approx(expected["sum of weights"])


@pytest.mark.parametrize("percentile", [0, 50, 100])
def test_zero_weight_observations_do_not_set_quantile_endpoints(percentile):
    result = calculate_quantile_result(
        np.array([-10.0, 5.0, 100.0]), percentile, np.array([0.0, 1.0, 0.0])
    )
    assert result.value == 5.0


@pytest.mark.parametrize("scale", [1e-308, 1.0, 1e308])
@pytest.mark.parametrize(("percentile", "expected"), [(0, 0.0), (50, 10.0), (100, 20.0)])
def test_weighted_quantiles_are_invariant_to_weight_scale(scale, percentile, expected):
    result = calculate_quantile_result(np.array([0.0, 10.0, 20.0]), percentile, np.full(3, scale))
    assert result.value == expected


@pytest.mark.parametrize("scale", [1e-308, 1.0, 1e308])
def test_threshold_fractions_are_invariant_to_weight_scale(scale):
    result = calculate_threshold_result(np.array([0.0, 1.0]), 0.5, np.full(2, scale))
    assert result.below_fraction == pytest.approx(0.5)
    assert result.above_fraction == pytest.approx(0.5)
