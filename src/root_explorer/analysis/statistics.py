"""Small statistical summaries calculated from prepared plot values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arrays import align_arrays


@dataclass(frozen=True)
class ThresholdResult:
    """Counts and optional weighted sums on both sides of one threshold."""

    threshold: float
    total_count: int
    below_count: int
    above_count: int
    below_fraction: float | None
    above_fraction: float | None
    weighted: bool
    total_weight: float | None = None
    below_weight: float | None = None
    above_weight: float | None = None
    has_negative_weights: bool = False


def calculate_threshold_result(
    values: np.ndarray,
    threshold: float,
    weights: np.ndarray | None = None,
) -> ThresholdResult:
    """Summarize values at ``<= threshold`` and ``> threshold``.

    Weighted fractions use signed sums. A zero total weight leaves both
    fractions undefined instead of silently falling back to event counts.
    """

    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")

    arrays = [np.asarray(values, dtype=float)]
    names = ["values"]
    if weights is not None:
        arrays.append(np.asarray(weights, dtype=float))
        names.append("weights")
    aligned = align_arrays(arrays, names=names, context="threshold calculator")
    selected_values = aligned[0]
    selected_weights = aligned[1] if weights is not None else None

    if not np.all(np.isfinite(selected_values)) or (
        selected_weights is not None and not np.all(np.isfinite(selected_weights))
    ):
        raise ValueError("Threshold calculation requires finite values and weights.")

    below = selected_values <= float(threshold)
    above = ~below
    total_count = len(selected_values)
    below_count = int(np.count_nonzero(below))
    above_count = int(np.count_nonzero(above))

    if selected_weights is None:
        denominator = float(total_count)
        below_fraction = below_count / denominator if total_count else None
        above_fraction = above_count / denominator if total_count else None
        return ThresholdResult(
            threshold=float(threshold),
            total_count=total_count,
            below_count=below_count,
            above_count=above_count,
            below_fraction=below_fraction,
            above_fraction=above_fraction,
            weighted=False,
        )

    total_weight = float(np.sum(selected_weights))
    below_weight = float(np.sum(selected_weights[below]))
    above_weight = float(np.sum(selected_weights[above]))
    rounding_tolerance = 8.0 * np.finfo(float).eps * float(np.sum(np.abs(selected_weights)))
    zero_total_weight = abs(total_weight) <= rounding_tolerance
    below_fraction = None if zero_total_weight else below_weight / total_weight
    above_fraction = None if zero_total_weight else above_weight / total_weight
    return ThresholdResult(
        threshold=float(threshold),
        total_count=total_count,
        below_count=below_count,
        above_count=above_count,
        below_fraction=below_fraction,
        above_fraction=above_fraction,
        weighted=True,
        total_weight=total_weight,
        below_weight=below_weight,
        above_weight=above_weight,
        has_negative_weights=bool(np.any(selected_weights < 0)),
    )


@dataclass(frozen=True)
class QuantileResult:
    """One unweighted or non-negatively weighted empirical quantile."""

    percentile: float
    value: float | None
    total_count: int
    weighted: bool
    total_weight: float | None = None
    has_negative_weights: bool = False


def calculate_quantile_result(
    values: np.ndarray,
    percentile: float,
    weights: np.ndarray | None = None,
) -> QuantileResult:
    """Calculate a quantile, using the weighted empirical CDF when possible."""

    if not np.isfinite(percentile) or percentile < 0 or percentile > 100:
        raise ValueError("Quantile percentile must be between 0 and 100.")

    arrays = [np.asarray(values, dtype=float)]
    names = ["values"]
    if weights is not None:
        arrays.append(np.asarray(weights, dtype=float))
        names.append("weights")
    aligned = align_arrays(arrays, names=names, context="quantile calculator")
    selected_values = aligned[0]
    selected_weights = aligned[1] if weights is not None else None

    if not np.all(np.isfinite(selected_values)) or (
        selected_weights is not None and not np.all(np.isfinite(selected_weights))
    ):
        raise ValueError("Quantile calculation requires finite values and weights.")

    if len(selected_values) == 0:
        return QuantileResult(percentile, None, 0, selected_weights is not None)

    if selected_weights is None:
        return QuantileResult(
            percentile,
            float(np.quantile(selected_values, percentile / 100.0)),
            len(selected_values),
            False,
        )

    total_weight = float(np.sum(selected_weights))
    has_negative_weights = bool(np.any(selected_weights < 0))
    if has_negative_weights or total_weight <= 0:
        return QuantileResult(
            percentile,
            None,
            len(selected_values),
            True,
            total_weight=total_weight,
            has_negative_weights=has_negative_weights,
        )

    order = np.argsort(selected_values)
    sorted_values = selected_values[order]
    sorted_weights = selected_weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = percentile / 100.0 * total_weight
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, len(sorted_values) - 1)
    return QuantileResult(
        percentile,
        float(sorted_values[index]),
        len(selected_values),
        True,
        total_weight=total_weight,
    )
