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

    scale = float(np.max(np.abs(selected_weights))) if total_count else 0.0
    relative = selected_weights / scale if scale else selected_weights
    total = float(np.sum(relative))
    below_total = float(np.sum(relative[below]))
    above_total = float(np.sum(relative[above]))
    with np.errstate(over="ignore"):
        total_weight = float(np.float64(scale) * total)
        below_weight = float(np.float64(scale) * below_total)
        above_weight = float(np.float64(scale) * above_total)
    rounding_tolerance = 8.0 * np.finfo(float).eps * float(np.sum(np.abs(relative)))
    zero_total_weight = abs(total) <= rounding_tolerance
    below_fraction = None if zero_total_weight else below_total / total
    above_fraction = None if zero_total_weight else above_total / total
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

    with np.errstate(over="ignore"):
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

    # Zero-weight observations carry no probability, including at the endpoints.
    # Scale before summing so quantiles are independent of the weight units.
    positive = selected_weights > 0
    positive_values = selected_values[positive]
    positive_weights = selected_weights[positive]
    order = np.argsort(positive_values)
    sorted_values = positive_values[order]
    sorted_weights = positive_weights[order] / np.max(positive_weights)
    cumulative = np.cumsum(sorted_weights)
    target = percentile / 100.0 * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    if percentile == 100:
        index = len(sorted_values) - 1
    index = min(index, len(sorted_values) - 1)
    return QuantileResult(
        percentile,
        float(sorted_values[index]),
        len(selected_values),
        True,
        total_weight=total_weight,
    )


def describe_weighted_array(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Return weighted summary statistics for aligned values and weights.

    The weighted standard deviation uses the population convention
    ``sqrt(sum(w * (x - mean)**2) / sum(w))``, as ROOT does for weighted
    histograms, so uniform weights reproduce the unweighted mean and standard
    deviation. Undefined quantities are returned as NaN: the mean and standard
    deviation require a positive sum of weights, and a negative weighted variance
    (possible with negative weights) has no standard deviation. A weighted median
    is deliberately omitted: the step-wise weighted quantile used elsewhere does
    not interpolate and would contradict the unweighted median for uniform weights.
    """

    aligned = align_arrays(
        [np.asarray(values, dtype=float), np.asarray(weights, dtype=float)],
        names=["values", "weights"],
        context="weighted summary statistics",
    )
    selected_values, selected_weights = aligned
    # Relative statistics must not depend on the units or overall scale of the
    # weights. Squaring the original weights can overflow or underflow even
    # when all inputs and the final statistics are representable.
    scale = float(np.max(np.abs(selected_weights))) if len(selected_weights) else 0.0
    relative_weights = selected_weights / scale if scale > 0 else selected_weights
    sum_relative = float(np.sum(relative_weights))
    sum_relative2 = float(np.sum(relative_weights**2))
    with np.errstate(over="ignore"):
        sum_w = float(np.float64(scale) * sum_relative)
    result = {
        "sum of weights": sum_w,
        "N_eff": sum_relative**2 / sum_relative2 if sum_relative2 > 0 else np.nan,
        "weighted mean": np.nan,
        "weighted std": np.nan,
    }
    if len(selected_values) == 0 or sum_relative <= 0:
        return result

    mean = float(np.sum(relative_weights * selected_values) / sum_relative)
    variance = float(np.sum(relative_weights * (selected_values - mean) ** 2) / sum_relative)
    result["weighted mean"] = mean
    result["weighted std"] = float(np.sqrt(variance)) if variance >= 0 else np.nan
    return result
