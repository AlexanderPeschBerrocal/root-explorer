"""Pure construction helpers for canonical ROOT Explorer histograms."""

from __future__ import annotations

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
            raise ValueError(f"The number of {axis_name} histogram bins must be a positive integer.")

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
