"""Pure construction helpers for canonical ROOT Explorer histograms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import hist
import numpy as np

from .arrays import align_arrays


def build_histogram_1d(
    values: np.ndarray,
    bins: int,
    *,
    weights: np.ndarray | None = None,
    value_range: tuple[float, float] | None = None,
    label: str = "",
) -> hist.Hist:
    """Build a one-dimensional histogram with ROOT Explorer's canonical semantics.

    Bin edges follow NumPy's histogram convention so the migration from Matplotlib
    preserves the existing visible binning. Flow bins are retained internally, and
    weighted histograms use weight storage to preserve both sum(w) and sum(w^2).
    """

    arrays = [np.asarray(values, dtype=float)]
    names = ["values"]
    if weights is not None:
        arrays.append(np.asarray(weights, dtype=float))
        names.append("weights")
    aligned = align_arrays(arrays, names=names, context="1D histogram values")
    selected_values = aligned[0].astype(float, copy=False)
    selected_weights = aligned[1].astype(float, copy=False) if weights is not None else None

    if not np.all(np.isfinite(selected_values)):
        raise ValueError("1D histogram values must be finite.")
    if selected_weights is not None and not np.all(np.isfinite(selected_weights)):
        raise ValueError("1D histogram weights must be finite.")

    if int(bins) != bins or int(bins) <= 0:
        raise ValueError("The number of histogram bins must be a positive integer.")
    bin_count = int(bins)

    if value_range is not None:
        minimum, maximum = (float(value_range[0]), float(value_range[1]))
        if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
            raise ValueError("Histogram range bounds must be finite with minimum < maximum.")
        resolved_range = (minimum, maximum)
    else:
        resolved_range = None

    edges = np.histogram_bin_edges(selected_values, bins=bin_count, range=resolved_range)
    axis = hist.axis.Variable(
        edges,
        name="x",
        label=label,
        underflow=True,
        overflow=True,
    )
    storage = hist.storage.Weight() if selected_weights is not None else hist.storage.Double()
    histogram = hist.Hist(axis, storage=storage)

    fill_values = _numpy_compatible_fill_values(selected_values, edges)
    if selected_weights is None:
        histogram.fill(x=fill_values)
    else:
        histogram.fill(x=fill_values, weight=selected_weights)
    return histogram


def _numpy_compatible_fill_values(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Keep NumPy's inclusive final edge while retaining a true overflow bin."""

    if values.size == 0:
        return values
    upper_edge = float(edges[-1])
    on_upper_edge = values == upper_edge
    if not np.any(on_upper_edge):
        return values

    adjusted = values.copy()
    adjusted[on_upper_edge] = np.nextafter(upper_edge, float(edges[-2]))
    return adjusted


def build_histogram_2d(
    x: np.ndarray,
    y: np.ndarray,
    bins_x: int,
    bins_y: int,
    *,
    weights: np.ndarray | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    x_label: str = "",
    y_label: str = "",
) -> hist.Hist:
    """Build a canonical 2D histogram, including weighted variances and flow bins.

    Visible edges and upper-edge inclusion follow NumPy's histogram2d convention.
    Inputs must be finite, flat, and aligned; filtering belongs to the caller.
    """
    arrays = [np.asarray(x, dtype=float), np.asarray(y, dtype=float)]
    names = ["x", "y"]
    if weights is not None:
        arrays.append(np.asarray(weights, dtype=float))
        names.append("weights")
    aligned = align_arrays(arrays, names=names, context="2D histogram values")
    selected_x, selected_y = aligned[:2]
    selected_weights = aligned[2] if weights is not None else None

    if not np.all(np.isfinite(selected_x)) or not np.all(np.isfinite(selected_y)):
        raise ValueError("2D histogram coordinates must be finite.")
    if selected_weights is not None and not np.all(np.isfinite(selected_weights)):
        raise ValueError("2D histogram weights must be finite.")

    for bins, axis_name in ((bins_x, "x"), (bins_y, "y")):
        if int(bins) != bins or int(bins) <= 0:
            raise ValueError(
                f"The number of {axis_name} histogram bins must be a positive integer."
            )

    for bounds, axis_name in ((x_range, "x"), (y_range, "y")):
        if bounds is not None:
            minimum, maximum = (float(bounds[0]), float(bounds[1]))
            if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
                raise ValueError(
                    f"{axis_name} histogram range bounds must be finite with minimum < maximum."
                )

    x_edges = np.histogram_bin_edges(selected_x, bins=int(bins_x), range=x_range)
    y_edges = np.histogram_bin_edges(selected_y, bins=int(bins_y), range=y_range)
    histogram = hist.Hist(
        hist.axis.Variable(x_edges, name="x", label=x_label, underflow=True, overflow=True),
        hist.axis.Variable(y_edges, name="y", label=y_label, underflow=True, overflow=True),
        storage=hist.storage.Weight() if selected_weights is not None else hist.storage.Double(),
    )
    fill_x = _numpy_compatible_fill_values(selected_x, x_edges)
    fill_y = _numpy_compatible_fill_values(selected_y, y_edges)
    if selected_weights is None:
        histogram.fill(x=fill_x, y=fill_y)
    else:
        histogram.fill(x=fill_x, y=fill_y, weight=selected_weights)
    return histogram


class InvalidHistogramRangeError(ValueError):
    """An explicit histogram range cannot be resolved."""


@dataclass(frozen=True)
class HistogramFlowSummary:
    """Entries and weighted moments in and outside the nominal histogram range."""

    lower: float
    upper: float
    underflow_entries: int
    in_range_entries: int
    overflow_entries: int
    underflow_sumw: float
    in_range_sumw: float
    overflow_sumw: float
    underflow_sumw2: float
    in_range_sumw2: float
    overflow_sumw2: float
    weighted: bool


def resolve_histogram_range(
    values: np.ndarray, bins: int, style: dict[str, Any]
) -> tuple[float, float] | None:
    """Resolve independent histogram bounds; missing endpoints are automatic.

    Axis *display* limits have no effect on the returned range.
    """
    if not style.get("manual_histogram_limits", False):
        return None

    def bound(key: str) -> float | None:
        raw = style.get(key, "")
        if raw is None or not str(raw).strip():
            return None
        try:
            result = float(raw)
        except (ValueError, TypeError) as exc:
            raise InvalidHistogramRangeError(
                "Histogram minimum and maximum must be finite numbers."
            ) from exc
        if not np.isfinite(result):
            raise InvalidHistogramRangeError(
                "Histogram minimum and maximum must be finite numbers."
            )
        return result

    lower = bound("hist_x_min_text")
    upper = bound("hist_x_max_text")
    if lower is None and upper is None:
        return None
    automatic_edges = np.histogram_bin_edges(values, bins=int(bins))
    if lower is None:
        lower = float(automatic_edges[0])
    if upper is None:
        upper = float(automatic_edges[-1])
    if lower >= upper:
        raise InvalidHistogramRangeError(
            "Histogram minimum must be smaller than histogram maximum."
        )
    return lower, upper


def summarise_histogram_flow(
    histogram: hist.Hist, values: np.ndarray, *, weighted: bool
) -> HistogramFlowSummary:
    """Extract original flow yields and count entries without re-histogramming."""
    edges = np.asarray(histogram.axes[0].edges, dtype=float)
    raw = np.asarray(histogram.values(flow=True), dtype=float)
    variances = histogram.variances(flow=True)
    variance = np.asarray(variances, dtype=float) if variances is not None else raw
    x = np.asarray(values, dtype=float)
    n_under = int(np.count_nonzero(x < edges[0]))
    n_over = int(np.count_nonzero(x > edges[-1]))
    return HistogramFlowSummary(
        lower=float(edges[0]),
        upper=float(edges[-1]),
        underflow_entries=n_under,
        in_range_entries=int(len(x) - n_under - n_over),
        overflow_entries=n_over,
        underflow_sumw=float(raw[0]),
        in_range_sumw=float(raw[1:-1].sum()),
        overflow_sumw=float(raw[-1]),
        underflow_sumw2=float(variance[0]),
        in_range_sumw2=float(variance[1:-1].sum()),
        overflow_sumw2=float(variance[-1]),
        weighted=weighted,
    )


def histogram_for_flow_mode(histogram: hist.Hist, mode: str) -> hist.Hist:
    """Make a folded histogram without changing the canonical original."""
    if mode not in {"ignore", "fold", "separate"}:
        raise ValueError(f"Unsupported histogram flow mode: {mode}")
    if mode != "fold":
        return histogram
    result = histogram.copy()
    view = result.view(flow=True)
    if histogram.storage_type is hist.storage.Weight:
        for source, target in ((0, 1), (-1, -2)):
            view.value[target] += view.value[source]
            view.variance[target] += view.variance[source]
            view.value[source] = 0.0
            view.variance[source] = 0.0
    else:
        for source, target in ((0, 1), (-1, -2)):
            view[target] += view[source]
            view[source] = 0.0
    return result


class InvalidHistogramNormalisationError(ValueError):
    """The supplied 1D histogram cannot be normalised to unit area."""


@dataclass(frozen=True)
class HistogramNormalisationResult:
    """A density histogram and its denominator, or an explicit failure.

    Stored variances are rescaled for a *fixed* denominator. They do not
    represent the full covariance of a self-normalised random histogram.
    """

    histogram: hist.Hist | None
    denominator: float
    signed_weights: bool
    warning: str | None = None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.histogram is not None


def normalise_histogram_1d(
    histogram: hist.Hist,
    *,
    absolute_weight_sum: float | None = None,
    has_negative_weights: bool = False,
) -> HistogramNormalisationResult:
    """Convert nominal bin yields to a unit-area *density*, without mutation.

    For each regular bin, rho_i = sumw_i / (sum_j sumw_j * bin_width_i).
    Flow bins have no finite widths and are not represented in the resulting
    density. The caller must apply its flow policy (ignore/fold) first, and
    must reject separate categorical flow bins before requesting a density.

    Signed yields can have a unit *signed* integral, but are not necessarily
    probability densities. The caller can supply the sum of absolute event
    weights to detect cancellations hidden inside individual histogram bins.
    """
    if histogram.ndim != 1:
        raise ValueError("Unit-area normalisation requires a 1D histogram.")

    values = np.asarray(histogram.values(), dtype=float)
    edges = np.asarray(histogram.axes[0].edges, dtype=float)
    widths = np.diff(edges)
    denominator = float(np.sum(values))
    signed = bool(has_negative_weights or np.any(values < 0))

    def invalid(reason: str) -> HistogramNormalisationResult:
        return HistogramNormalisationResult(None, denominator, signed, error=reason)

    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(edges))
        or not np.all(np.isfinite(widths))
        or np.any(widths <= 0)
    ):
        return invalid("Unit-area normalisation requires finite bin contents and positive widths.")
    if values.size == 0 or not np.isfinite(denominator) or denominator <= 0:
        return invalid("Unit-area normalisation requires a finite, positive in-range yield.")

    abs_sum = (
        float(np.sum(np.abs(values))) if absolute_weight_sum is None else float(absolute_weight_sum)
    )
    if not np.isfinite(abs_sum) or abs_sum < 0 or abs_sum < abs(denominator):
        return invalid("Unit-area normalisation has invalid absolute-weight bookkeeping.")
    ratio = denominator / abs_sum if abs_sum else 0.0
    if ratio <= 64 * np.finfo(float).eps:
        return invalid(
            "Unit-area normalisation is numerically unstable due to weight cancellation."
        )

    variances = histogram.variances()
    if variances is None:
        return invalid("Unit-area normalisation requires histogram bin variances.")
    variance = np.asarray(variances, dtype=float)
    if not np.all(np.isfinite(variance)) or np.any(variance < 0):
        return invalid("Unit-area normalisation requires finite, non-negative bin variances.")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scale = denominator * widths
        density = values / scale
        density_variance = variance / np.square(scale)
    if not np.all(np.isfinite(density)) or not np.all(np.isfinite(density_variance)):
        return invalid("Unit-area normalisation produced non-finite densities or variances.")

    result = hist.Hist(*histogram.axes, storage=hist.storage.Weight())
    result_view = result.view()
    result_view.value[:] = density
    result_view.variance[:] = density_variance
    # No arbitrary widths can be assigned to underflow/overflow categories.
    warning = None
    if ratio < 1e-6:
        warning = "Strong cancellation of signed weights makes this normalisation sensitive."
    elif signed:
        warning = "Signed-weight unit integral is not necessarily a probability density."
    return HistogramNormalisationResult(result, denominator, signed, warning=warning)
