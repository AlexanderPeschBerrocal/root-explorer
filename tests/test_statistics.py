from __future__ import annotations

import numpy as np
import pytest

from root_explorer.analysis.arrays import ArrayShapeMismatchError
from root_explorer.analysis.statistics import calculate_threshold_result


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
