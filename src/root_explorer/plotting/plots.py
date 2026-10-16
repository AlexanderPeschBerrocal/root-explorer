"""Plot construction, filtering, and export helpers."""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import colormaps
from matplotlib.colors import LogNorm, to_hex
from scipy.stats import gaussian_kde

from ..analysis.arrays import align_arrays
from ..analysis.fitting import FitResult, model_bin_integrals


class InvalidAxisScaleError(ValueError):
    """Raised when selected values cannot be displayed on a requested axis scale."""


class InvalidAxisLimitError(ValueError):
    """Raised when explicit axis display limits are invalid."""


class InvalidColorScaleError(ValueError):
    """Raised when scatter colour normalization is invalid."""


class ContourConfigurationError(ValueError):
    """Raised when highest-density contours cannot be configured for a histogram."""


class KDEError(ValueError):
    """Raised when a KDE overlay cannot be constructed without misleading output."""


class InvalidPlotResolutionError(ValueError):
    """Raised when a raster export would be invalid or unreasonably large."""


KDE_MAX_POINTS = 5_000
KDE_1D_GRID_SIZE = 512
KDE_2D_MAX_GRID_SIZE = 100
DEFAULT_EXPORT_DPI = 300
MAX_EXPORT_DPI = 600
MAX_RASTER_PIXELS = 40_000_000


@dataclass(frozen=True)
class ColorClippingResult:
    """Counts and optional signed weight sums outside explicit scatter colour limits."""

    color_min: float | None
    color_max: float | None
    below_count: int
    above_count: int
    below_weight: float | None = None
    above_weight: float | None = None

    @property
    def extend(self) -> str:
        below = self.color_min is not None and self.below_count > 0
        above = self.color_max is not None and self.above_count > 0
        if below and above:
            return "both"
        if below:
            return "min"
        if above:
            return "max"
        return "neither"


@dataclass(frozen=True)
class CorrelationResult:
    """Pairwise correlation statistics for a finite, aligned x/y sample."""

    n_pairs: int
    pearson_r: float
    spearman_rho: float


def apply_log_axis_scales(
    ax: Any,
    style: dict[str, Any],
    *,
    x: np.ndarray,
    y: np.ndarray | None = None,
    histogram_counts: np.ndarray | None = None,
) -> None:
    """Validate and apply logarithmic axis scales without modifying plot data."""

    if style.get("log_x", False):
        apply_log_x_axis(ax, x)

    if style.get("log_y", False):
        if histogram_counts is not None:
            apply_log_histogram_y_axis(ax, histogram_counts)
        elif y is None:
            raise InvalidAxisScaleError(
                "Cannot use a logarithmic y-axis: no y coordinates are available."
            )
        else:
            apply_log_y_axis(ax, y)


def apply_log_x_axis(ax: Any, x: np.ndarray) -> None:
    """Validate and apply a logarithmic x-axis."""

    _require_positive_axis_values(x, "x")
    ax.set_xscale("log")


def apply_log_y_axis(ax: Any, y: np.ndarray) -> None:
    """Validate and apply a logarithmic y-axis for coordinate values."""

    _require_positive_axis_values(y, "y")
    ax.set_yscale("log")


def apply_log_histogram_y_axis(ax: Any, histogram_counts: np.ndarray) -> None:
    """Validate and apply a logarithmic y-axis for histogram bin contents."""

    counts = np.asarray(histogram_counts, dtype=float)
    if not np.any(np.isfinite(counts) & (counts > 0)):
        raise InvalidAxisScaleError(
            "Cannot use a logarithmic y-axis: the histogram has no positive bin contents."
        )
    ax.set_yscale("log")


def apply_axis_limits(ax: Any, style: dict[str, Any]) -> None:
    """Validate and apply optional viewport limits without filtering plot data."""

    x_min = _axis_limit_value(style, "x_min_text", "x-axis minimum")
    x_max = _axis_limit_value(style, "x_max_text", "x-axis maximum")
    y_min = _axis_limit_value(style, "y_min_text", "y-axis minimum")
    y_max = _axis_limit_value(style, "y_max_text", "y-axis maximum")

    _validate_axis_limit_pair(x_min, x_max, "x")
    _validate_axis_limit_pair(y_min, y_max, "y")
    _validate_log_axis_limits(x_min, x_max, "x", bool(style.get("log_x", False)))
    _validate_log_axis_limits(y_min, y_max, "y", bool(style.get("log_y", False)))

    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)


def _axis_limit_value(style: dict[str, Any], key: str, label: str) -> float | None:
    raw = style.get(key, "")
    if raw is None or not str(raw).strip():
        return None
    value = parse_optional_float(raw)
    if value is None or not np.isfinite(value):
        raise InvalidAxisLimitError(f"{label} must be a finite number.")
    return value


def _validate_axis_limit_pair(minimum: float | None, maximum: float | None, axis_name: str) -> None:
    if minimum is not None and maximum is not None and minimum >= maximum:
        raise InvalidAxisLimitError(
            f"{axis_name}-axis minimum must be smaller than {axis_name}-axis maximum."
        )


def _validate_log_axis_limits(
    minimum: float | None,
    maximum: float | None,
    axis_name: str,
    logarithmic: bool,
) -> None:
    if not logarithmic:
        return
    if minimum is not None and minimum <= 0:
        raise InvalidAxisLimitError(
            f"{axis_name}-axis minimum must be strictly positive for a logarithmic axis."
        )
    if maximum is not None and maximum <= 0:
        raise InvalidAxisLimitError(
            f"{axis_name}-axis maximum must be strictly positive for a logarithmic axis."
        )


def _require_positive_axis_values(values: np.ndarray, axis_name: str) -> None:
    coordinates = np.asarray(values, dtype=float)
    if coordinates.size == 0 or np.any(~np.isfinite(coordinates)) or np.any(coordinates <= 0):
        raise InvalidAxisScaleError(
            f"Cannot use a logarithmic {axis_name}-axis: all plotted {axis_name} coordinates "
            "must be finite and strictly positive."
        )


def downsample_arrays(
    arrays: Sequence[np.ndarray], max_points: int, seed: int = 12345
) -> list[np.ndarray]:
    arrays = align_arrays(arrays)
    if not arrays:
        return []
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(max_points), replace=False)
    return [a[idx] for a in arrays]


def resolve_kde_bandwidth(method: str, custom_factor: object = 1.0) -> str | float:
    """Return a SciPy KDE bandwidth method after validating UI configuration."""

    normalized = str(method).casefold()
    if normalized in {"scott", "silverman"}:
        return normalized
    if normalized != "custom":
        raise KDEError(f"Unknown KDE bandwidth method: {method!r}.")
    try:
        factor = float(custom_factor)
    except (TypeError, ValueError) as exc:
        raise KDEError(
            "The custom KDE bandwidth factor must be finite and greater than zero."
        ) from exc
    if not np.isfinite(factor) or factor <= 0:
        raise KDEError("The custom KDE bandwidth factor must be finite and greater than zero.")
    return factor


def _prepare_kde_sample(
    coordinates: Sequence[np.ndarray],
    weights: np.ndarray | None,
    *,
    max_points: int = KDE_MAX_POINTS,
) -> tuple[list[np.ndarray], np.ndarray | None, float]:
    """Validate KDE inputs and deterministically cap construction cost."""

    names = [f"coordinate {index + 1}" for index in range(len(coordinates))]
    arrays = align_arrays(
        [*coordinates, *([weights] if weights is not None else [])],
        names=[*names, *(["weights"] if weights is not None else [])],
        context="KDE values",
    )
    coordinate_values = [values.astype(float, copy=False) for values in arrays[: len(coordinates)]]
    weight_values = arrays[-1].astype(float, copy=False) if weights is not None else None
    finite = np.ones(len(coordinate_values[0]), dtype=bool)
    for values in coordinate_values:
        finite &= np.isfinite(values)
    if weight_values is not None:
        finite &= np.isfinite(weight_values)
    coordinate_values = [values[finite] for values in coordinate_values]
    if weight_values is not None:
        weight_values = weight_values[finite]
        if np.any(weight_values < 0):
            raise KDEError(
                "KDE is unavailable because the selected weights contain negative values."
            )
        total_weight = float(np.sum(weight_values))
        if total_weight <= 0:
            raise KDEError("KDE is unavailable because the total non-negative weight is zero.")
        positive_weight = weight_values > 0
        coordinate_values = [values[positive_weight] for values in coordinate_values]
        weight_values = weight_values[positive_weight]
    else:
        total_weight = float(len(coordinate_values[0]))

    minimum_points = len(coordinate_values) + 1
    if len(coordinate_values[0]) < minimum_points:
        raise KDEError(f"KDE requires at least {minimum_points} valid points for this plot.")
    if any(np.ptp(values) == 0 for values in coordinate_values):
        raise KDEError("KDE is unavailable for a constant variable.")

    sample_arrays = [*coordinate_values, *([weight_values] if weight_values is not None else [])]
    sample_arrays = downsample_arrays(sample_arrays, max_points)
    sampled_coordinates = sample_arrays[: len(coordinate_values)]
    sampled_weights = sample_arrays[-1] if weight_values is not None else None
    return sampled_coordinates, sampled_weights, total_weight


def _build_kde(
    coordinates: Sequence[np.ndarray],
    weights: np.ndarray | None,
    bandwidth: str | float,
):
    try:
        dataset = coordinates[0] if len(coordinates) == 1 else np.vstack(coordinates)
        return gaussian_kde(dataset, bw_method=bandwidth, weights=weights)
    except Exception as exc:
        raise KDEError(f"KDE could not be calculated: {exc}") from exc


def draw_kde_1d(
    ax,
    x: np.ndarray,
    histogram_edges: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    bandwidth: str | float = "scott",
):
    """Overlay a KDE scaled to the histogram's count or summed-weight units."""

    coordinates, sampled_weights, total_weight = _prepare_kde_sample([x], weights)
    kde = _build_kde(coordinates, sampled_weights, bandwidth)
    grid = np.linspace(float(histogram_edges[0]), float(histogram_edges[-1]), KDE_1D_GRID_SIZE)
    bin_widths = np.diff(histogram_edges)
    if not np.allclose(bin_widths, bin_widths[0]):
        raise KDEError("KDE scaling requires equal-width histogram bins.")
    curve = kde(grid) * total_weight * float(bin_widths[0])
    return ax.plot(grid, curve, color="tab:orange", linewidth=2.0, label="KDE")[0]


def draw_kde_2d(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    bandwidth: str | float = "scott",
):
    """Overlay KDE contour lines on the visible 2D histogram range."""

    coordinates, sampled_weights, _ = _prepare_kde_sample([x, y], weights)
    kde = _build_kde(coordinates, sampled_weights, bandwidth)
    x_grid = np.linspace(
        float(xedges[0]), float(xedges[-1]), min(len(xedges) - 1, KDE_2D_MAX_GRID_SIZE)
    )
    y_grid = np.linspace(
        float(yedges[0]), float(yedges[-1]), min(len(yedges) - 1, KDE_2D_MAX_GRID_SIZE)
    )
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    if not np.any(np.isfinite(density) & (density > 0)):
        raise KDEError("KDE produced no positive finite density values.")
    return ax.contour(
        grid_x,
        grid_y,
        density,
        levels=6,
        colors="tab:orange",
        linestyles="dashed",
        linewidths=1.2,
    )


def calculate_correlations(x: np.ndarray, y: np.ndarray) -> CorrelationResult:
    """Calculate Pearson and Spearman correlations for finite aligned pairs."""

    x_values, y_values = align_arrays([x, y], names=["x", "y"], context="correlation values")
    x_values = x_values.astype(float, copy=False)
    y_values = y_values.astype(float, copy=False)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    n_pairs = len(x_values)
    if n_pairs < 2:
        return CorrelationResult(n_pairs, np.nan, np.nan)

    pearson_r = _pearson_correlation(x_values, y_values)
    x_ranks = pd.Series(x_values).rank(method="average").to_numpy(dtype=float)
    y_ranks = pd.Series(y_values).rank(method="average").to_numpy(dtype=float)
    spearman_rho = _pearson_correlation(x_ranks, y_ranks)
    return CorrelationResult(n_pairs, pearson_r, spearman_rho)


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return a stable Pearson coefficient, or NaN when it is undefined."""

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    coefficient = float(np.dot(x_centered, y_centered) / denominator)
    return float(np.clip(coefficient, -1.0, 1.0))


def apply_range_cuts(
    arrays: Sequence[np.ndarray],
    cuts: Sequence[tuple[np.ndarray, float | None, float | None]],
) -> list[np.ndarray]:
    """Apply all range cuts with one combined mask.

    Every cut is evaluated on the same original read range/event. Building the
    mask once avoids the subtle bug where later cuts could be aligned against
    already-filtered plot arrays instead of the original entries/objects.
    """

    if not cuts:
        return align_arrays(arrays)

    cut_arrays = [cut_var for cut_var, _, _ in cuts]
    aligned = align_arrays([*arrays, *cut_arrays])
    if not aligned:
        return []

    values = aligned[: len(arrays)]
    aligned_cuts = aligned[len(arrays) :]
    n = len(values[0]) if values else len(aligned_cuts[0])
    mask = np.ones(n, dtype=bool)

    for cut, (_, lo, hi) in zip(aligned_cuts, cuts, strict=True):
        cut_values = cut.astype(float, copy=False)
        mask &= np.isfinite(cut_values)
        if lo is not None:
            mask &= cut_values >= lo
        if hi is not None:
            mask &= cut_values <= hi

    return [v[mask] for v in values]


def apply_range_cut(
    arrays: Sequence[np.ndarray],
    cut_var: np.ndarray | None,
    lo: float | None,
    hi: float | None,
) -> list[np.ndarray]:
    # Kept as a small compatibility wrapper for any local code that imports this helper.
    if cut_var is None:
        return align_arrays(arrays)
    return apply_range_cuts(arrays, [(cut_var, lo, hi)])


def _scatter_color_limits(style: dict[str, Any]) -> tuple[float | None, float | None]:
    """Parse and validate explicit 2D-scatter colour limits."""

    cmin_text = style.get("color_min_text", "")
    cmax_text = style.get("color_max_text", "")
    cmin = parse_optional_float(cmin_text)
    cmax = parse_optional_float(cmax_text)
    for label, raw, value in [
        ("color min", cmin_text, cmin),
        ("color max", cmax_text, cmax),
    ]:
        if str(raw).strip() and (value is None or not np.isfinite(value)):
            raise InvalidColorScaleError(f"{label} must be a finite number.")
    if cmin is not None and cmax is not None and cmin >= cmax:
        raise InvalidColorScaleError("color min must be smaller than color max.")
    return cmin, cmax


def calculate_color_clipping(
    color: np.ndarray,
    style: dict[str, Any],
    weights: np.ndarray | None = None,
) -> ColorClippingResult:
    """Summarize values clipped by explicit 2D-scatter colour limits."""

    cmin, cmax = _scatter_color_limits(style)
    arrays = [np.asarray(color, dtype=float)]
    names = ["color"]
    if weights is not None:
        arrays.append(np.asarray(weights, dtype=float))
        names.append("weights")
    aligned = align_arrays(arrays, names=names, context="scatter color clipping")
    values = aligned[0]
    selected_weights = aligned[1] if weights is not None else None

    below = values < cmin if cmin is not None else np.zeros(len(values), dtype=bool)
    above = values > cmax if cmax is not None else np.zeros(len(values), dtype=bool)
    below_weight = float(np.sum(selected_weights[below])) if selected_weights is not None else None
    above_weight = float(np.sum(selected_weights[above])) if selected_weights is not None else None
    return ColorClippingResult(
        color_min=cmin,
        color_max=cmax,
        below_count=int(np.count_nonzero(below)),
        above_count=int(np.count_nonzero(above)),
        below_weight=below_weight,
        above_weight=above_weight,
    )


def scatter_color_scale_linear(style: dict[str, Any]) -> tuple[float | None, float | None]:
    """Resolve explicit limits for a linear 2D-scatter colour scale."""

    return _scatter_color_limits(style)


def scatter_color_scale_log(color: np.ndarray, style: dict[str, Any]) -> LogNorm:
    """Validate and resolve logarithmic 2D-scatter colour normalization."""

    cmin, cmax = _scatter_color_limits(style)
    values = np.asarray(color, dtype=float)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise InvalidColorScaleError(
            "Logarithmic color scale requires all color values to be finite and strictly positive."
        )
    vmin = cmin if cmin is not None else float(np.min(values))
    vmax = cmax if cmax is not None else float(np.max(values))
    if vmin <= 0 or vmax <= 0:
        raise InvalidColorScaleError(
            "Logarithmic color scale requires positive color min and color max."
        )
    if vmin >= vmax:
        raise InvalidColorScaleError(
            "Logarithmic color scale requires color min to be smaller than color max."
        )
    return LogNorm(vmin=vmin, vmax=vmax)


def scatter_color_scale(
    color: np.ndarray, style: dict[str, Any]
) -> tuple[LogNorm | None, float | None, float | None]:
    """Validate and resolve 2D-scatter colour normalization."""

    if style.get("log_color", False):
        return scatter_color_scale_log(color, style), None, None
    cmin, cmax = scatter_color_scale_linear(style)
    return None, cmin, cmax


def parse_optional_float(text: str) -> float | None:
    stripped = str(text).strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def parse_contour_levels(value: str | Sequence[float]) -> list[float]:
    """Return sorted, unique contour percentages after strict validation."""

    if isinstance(value, str):
        tokens = value.replace(",", " ").split()
        if not tokens:
            raise ContourConfigurationError("Enter at least one contour level.")
        try:
            levels = [float(token) for token in tokens]
        except ValueError as exc:
            raise ContourConfigurationError(
                "Contour levels must be numbers separated by commas or spaces."
            ) from exc
    else:
        levels = [float(level) for level in value]

    if not levels:
        raise ContourConfigurationError("Enter at least one contour level.")
    if any(not np.isfinite(level) or level <= 0 or level >= 100 for level in levels):
        raise ContourConfigurationError("Contour levels must be between 0 and 100 percent.")
    return sorted(set(levels))


def highest_density_contour_thresholds(
    histogram: np.ndarray, levels: str | Sequence[float]
) -> dict[float, tuple[float, ...]]:
    """Map bin-content thresholds to the percentages enclosed above them.

    Bins are ordered from highest to lowest content. Each requested threshold is
    the lowest included bin content once the cumulative content reaches the
    requested fraction of the total. Equal thresholds are grouped because
    Matplotlib requires strictly increasing contour values.
    """

    percentages = parse_contour_levels(levels)
    contents = np.asarray(histogram, dtype=float)
    if np.any(~np.isfinite(contents)):
        raise ContourConfigurationError("Contour bin contents must be finite.")
    if np.any(contents < 0):
        raise ContourConfigurationError(
            "Contours are unavailable because signed weights produced negative bin contents."
        )

    positive = np.sort(contents[contents > 0])[::-1]
    if positive.size == 0:
        raise ContourConfigurationError(
            "Contours are unavailable because the histogram has no positive bin contents."
        )
    cumulative = np.cumsum(positive)
    total = float(cumulative[-1])
    grouped: dict[float, list[float]] = {}
    for percentage in percentages:
        index = int(np.searchsorted(cumulative, percentage / 100.0 * total, side="left"))
        threshold = float(positive[min(index, positive.size - 1)])
        grouped.setdefault(threshold, []).append(percentage)
    return {threshold: tuple(values) for threshold, values in grouped.items()}


def describe_array(name: str, arr: np.ndarray) -> dict[str, Any]:
    if len(arr) == 0:
        return {
            "variable": name,
            "n": 0,
            "min": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "max": np.nan,
        }
    values = arr.astype(float, copy=False)
    return {
        "variable": name,
        "n": len(values),
        "min": float(np.nanmin(values)),
        "median": float(np.nanmedian(values)),
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "max": float(np.nanmax(values)),
    }


def draw_hist2d_linear(ax: Any, histogram: hist.Hist):
    """Draw visible contents of a prepared 2D histogram with a linear colour scale."""
    values = np.asarray(histogram.values(), dtype=float)
    return ax.pcolormesh(histogram.axes[0].edges, histogram.axes[1].edges, values.T, shading="auto")


def draw_hist2d_log(ax: Any, histogram: hist.Hist):
    """Draw visible contents with the existing logarithmic-colour fallback."""
    values = np.asarray(histogram.values(), dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    norm = None
    if len(positive) > 1 and np.nanmax(positive) > np.nanmin(positive):
        norm = LogNorm(vmin=float(np.nanmin(positive)), vmax=float(np.nanmax(positive)))
    return ax.pcolormesh(
        histogram.axes[0].edges,
        histogram.axes[1].edges,
        values.T,
        shading="auto",
        norm=norm,
    )


def draw_highest_density_contours(
    ax,
    histogram: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    levels: str | Sequence[float],
):
    """Draw and label meaningful highest-density contours on a 2D histogram."""

    grouped = highest_density_contour_thresholds(histogram, levels)
    finite = histogram[np.isfinite(histogram)]
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    thresholds = sorted(threshold for threshold in grouped if minimum < threshold < maximum)
    if not thresholds:
        raise ContourConfigurationError(
            "No meaningful contour can be drawn for these levels and histogram bins."
        )

    xcenters = (xedges[:-1] + xedges[1:]) / 2.0
    ycenters = (yedges[:-1] + yedges[1:]) / 2.0
    contours = ax.contour(
        xcenters,
        ycenters,
        histogram.T,
        levels=thresholds,
        colors="white",
        linewidths=1.0,
    )
    labels = {
        threshold: "/".join(f"{level:g}%" for level in grouped[threshold])
        for threshold in thresholds
    }
    ax.clabel(contours, contours.levels, fmt=labels, fontsize=8)
    return contours


def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("_") or "plot"


def validate_raster_resolution(fig, dpi: int | float) -> tuple[int, int]:
    """Return estimated pixel dimensions or reject an unsafe raster request."""

    try:
        dpi_value = float(dpi)
    except (TypeError, ValueError) as exc:
        raise InvalidPlotResolutionError("Raster export resolution must be a number.") from exc
    if not np.isfinite(dpi_value) or dpi_value <= 0:
        raise InvalidPlotResolutionError("Raster export resolution must be finite and positive.")
    if dpi_value > MAX_EXPORT_DPI:
        raise InvalidPlotResolutionError(
            f"Raster export resolution cannot exceed {MAX_EXPORT_DPI} DPI."
        )

    width = int(np.ceil(float(fig.get_figwidth()) * dpi_value))
    height = int(np.ceil(float(fig.get_figheight()) * dpi_value))
    pixels = width * height
    if pixels > MAX_RASTER_PIXELS:
        raise InvalidPlotResolutionError(
            f"The requested raster would be approximately {width:,} x {height:,} pixels "
            f"({pixels / 1_000_000:.1f} megapixels), above the "
            f"{MAX_RASTER_PIXELS / 1_000_000:.0f}-megapixel safety limit. Reduce the figure "
            "size or export resolution."
        )
    return width, height


def save_matplotlib_figure(
    fig,
    outdir: Path,
    name: str,
    formats: Sequence[str],
    *,
    dpi: int | float = DEFAULT_EXPORT_DPI,
) -> list[Path]:
    selected = [fmt for fmt in formats if fmt in {"png", "pdf"}]
    if not selected:
        return []
    if "png" in selected:
        validate_raster_resolution(fig, dpi)
    outdir.mkdir(parents=True, exist_ok=True)
    base = safe_filename(name)
    saved_paths: list[Path] = []
    for fmt in selected:
        path = outdir / f"{base}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(path)
    return saved_paths


def save_plotly_figure(fig, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{safe_filename(name)}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def matplotlib_figure_bytes(
    fig, file_format: str, *, dpi: int | float = DEFAULT_EXPORT_DPI
) -> bytes:
    """Render a Matplotlib figure for a browser download."""
    if file_format == "png":
        validate_raster_resolution(fig, dpi)
    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {
        "format": file_format,
        "bbox_inches": "tight",
        "dpi": dpi,
    }
    fig.savefig(buffer, **save_kwargs)
    return buffer.getvalue()


def plotly_figure_html(fig) -> bytes:
    """Render a Plotly figure as HTML for a browser download."""
    return fig.to_html(include_plotlyjs="cdn").encode("utf-8")


def build_plotly_3d(
    x,
    y,
    z,
    x_label,
    y_label,
    z_label,
    *,
    color=None,
    color_label=None,
    title="",
    marker_size=2.5,
    opacity=0.65,
    colorscale="Viridis",
    reversescale=False,
    aspectmode="data",
    color_min=None,
    color_max=None,
):
    plotly_colorscale = colorscale
    if colorscale.lower() == "coolwarm":
        cmap = colormaps["coolwarm"]
        plotly_colorscale = [
            [position, to_hex(cmap(position))] for position in np.linspace(0.0, 1.0, 11)
        ]
    marker = {
        "size": marker_size,
        "opacity": opacity,
        "colorscale": plotly_colorscale,
        "reversescale": reversescale,
    }
    if color is not None:
        marker.update(color=color, colorbar={"title": color_label or "color"})
        if color_min is not None:
            marker["cmin"] = color_min
        if color_max is not None:
            marker["cmax"] = color_max
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=marker)])
    fig.update_traces(
        hovertemplate=(
            f"{x_label}: %{{x:.5g}}<br>{y_label}: %{{y:.5g}}<br>{z_label}: %{{z:.5g}}<br>"
            + (f"{color_label}: %{{marker.color:.5g}}<br>" if color is not None else "")
            + "<extra></extra>"
        )
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": x_label,
            "yaxis_title": y_label,
            "zaxis_title": z_label,
            "aspectmode": aspectmode,
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 42},
    )
    return fig


MATPLOTLIB_CMAP_NAMES = {
    "viridis": "viridis",
    "plasma": "plasma",
    "inferno": "inferno",
    "magma": "magma",
    "cividis": "cividis",
    "greys": "Greys",
    "coolwarm": "coolwarm",
}


def matplotlib_cmap_name(colorscale: str, reverse: bool = False) -> str:
    """Return the case-sensitive Matplotlib name for a UI color scale."""

    name = MATPLOTLIB_CMAP_NAMES.get(colorscale.casefold(), colorscale)
    return f"{name}_r" if reverse else name


def create_static_figure(style: dict[str, Any]):
    """Create the Matplotlib figure used for a static ROOT Explorer plot."""

    return plt.subplots(
        figsize=(style["fig_width"], style["fig_height"]),
        dpi=float(style.get("display_dpi", 200)),
    )


def draw_histogram_1d(
    ax: Any,
    histogram: hist.Hist,
    y_label: str,
    *,
    reference_x: float | None,
    reference_label: str | None,
    flow_summary: Any | None = None,
    flow_mode: str = "ignore",
    log_x: bool = False,
) -> None:
    """Draw a prepared 1D histogram without recomputing its binning semantics."""

    histogram_counts = np.asarray(histogram.values(), dtype=float)
    histogram_edges = np.asarray(histogram.axes[0].edges, dtype=float)
    ax.bar(
        histogram_edges[:-1],
        histogram_counts,
        width=np.diff(histogram_edges),
        align="edge",
    )
    ax.set_xlabel(histogram.axes[0].label)
    ax.set_ylabel(y_label)
    if flow_mode == "separate" and flow_summary is not None:
        # Detached categorical bars: their horizontal positions have no physics meaning.
        width = float(histogram_edges[-1] - histogram_edges[0]) / len(histogram_counts)
        left, right = float(histogram_edges[0]), float(histogram_edges[-1])
        if log_x:
            if left <= 0:
                raise InvalidAxisScaleError("Log x-axis requires positive histogram edges.")
            ratio = (right / left) ** (1 / len(histogram_counts))
            positions = (left / ratio**1.8, right * ratio**0.8)
            widths = (positions[0] * (ratio - 1), positions[1] * (ratio - 1))
        else:
            positions = (left - 1.6 * width, right + 0.6 * width)
            widths = (width, width)
        heights = (flow_summary.underflow_sumw, flow_summary.overflow_sumw)
        for position, bar_width, height, label in zip(
            positions, widths, heights, ("UF", "OF"), strict=True
        ):
            ax.bar(
                position,
                height,
                width=bar_width,
                align="edge",
                color="tab:orange",
                hatch="//",
                edgecolor="black",
                linewidth=1,
            )
            ax.text(position + bar_width / 2, 0, label, ha="center", va="top")
    if reference_x is not None:
        ax.axvline(
            reference_x,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label=reference_label or f"x = {reference_x:g}",
        )
        ax.legend()


def draw_fit_overlay(ax: Any, histogram: hist.Hist, result: FitResult) -> None:
    """Display fitted bin predictions on the original histogram bin edges."""
    if not result.success or result.fit_range is None:
        return
    edges = np.asarray(histogram.axes[0].edges, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2
    lower, upper = result.fit_range
    selected = (centers >= lower) & (centers <= upper)
    indices = np.flatnonzero(selected)
    if not len(indices):
        return
    x0 = result.reference_x if result.reference_x is not None else (lower + upper) / 2
    selected_edges = edges[indices[0] : indices[-1] + 2]
    try:
        prediction = model_bin_integrals(
            result.model,
            selected_edges,
            result.parameters,
            x0=x0,
            expression=result.expression,
        )
    except (ValueError, FloatingPointError, OverflowError):
        return
    # A step representation remains valid for non-uniform bin widths.
    ax.stairs(
        prediction,
        selected_edges,
        color="tab:red",
        linewidth=2,
        label=f"{result.model} fit",
    )
    ax.legend()


def draw_histogram_2d(
    fig: Any,
    ax: Any,
    histogram: hist.Hist,
    *,
    log_scale: bool,
    colorbar_label: str,
) -> None:
    """Render a prepared 2D histogram, without reconstructing its binning."""
    if len(histogram.axes) != 2:
        raise ValueError("2D histogram rendering requires exactly two axes.")
    mesh = draw_hist2d_log(ax, histogram) if log_scale else draw_hist2d_linear(ax, histogram)
    ax.set_xlabel(histogram.axes[0].label)
    ax.set_ylabel(histogram.axes[1].label)
    fig.colorbar(mesh, ax=ax, label=colorbar_label)


def draw_scatter_2d(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    style: dict[str, Any],
) -> None:
    """Draw an uncoloured 2D scatter plot."""

    ax.scatter(x, y, s=style["marker_size_2d"], alpha=style["opacity_2d"])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def draw_colored_scatter_2d_linear(
    fig: Any,
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray,
    x_label: str,
    y_label: str,
    color_label: str | None,
    style: dict[str, Any],
    *,
    colorbar_extend: str,
) -> None:
    """Draw a 2D scatter plot with a linear colour scale."""

    cmin, cmax = scatter_color_scale_linear(style)
    scatter = ax.scatter(
        x,
        y,
        s=style["marker_size_2d"],
        alpha=style["opacity_2d"],
        c=color,
        cmap=matplotlib_cmap_name(
            style.get("colorscale", "Viridis"), style.get("reverse_colorscale", False)
        ),
        vmin=cmin,
        vmax=cmax,
    )
    fig.colorbar(scatter, ax=ax, label=color_label or "color", extend=colorbar_extend)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def draw_colored_scatter_2d_log(
    fig: Any,
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray,
    x_label: str,
    y_label: str,
    color_label: str | None,
    style: dict[str, Any],
    *,
    colorbar_extend: str,
) -> None:
    """Draw a 2D scatter plot with a logarithmic colour scale."""

    scatter = ax.scatter(
        x,
        y,
        s=style["marker_size_2d"],
        alpha=style["opacity_2d"],
        c=color,
        cmap=matplotlib_cmap_name(
            style.get("colorscale", "Viridis"), style.get("reverse_colorscale", False)
        ),
        norm=scatter_color_scale_log(color, style),
    )
    fig.colorbar(scatter, ax=ax, label=color_label or "color", extend=colorbar_extend)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def draw_kde_1d_overlay(
    ax: Any,
    x: np.ndarray,
    histogram_edges: np.ndarray,
    style: dict[str, Any],
    *,
    weights: np.ndarray | None,
) -> str | None:
    """Draw a selected 1D KDE overlay, returning a displayable warning on failure."""

    try:
        bandwidth = resolve_kde_bandwidth(
            style.get("kde_bandwidth", "scott"), style.get("kde_custom_factor", 1.0)
        )
        draw_kde_1d(ax, x, histogram_edges, weights=weights, bandwidth=bandwidth)
    except Exception as exc:
        return str(exc) if isinstance(exc, KDEError) else f"KDE could not be calculated: {exc}"
    return None


def draw_kde_2d_overlay(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    style: dict[str, Any],
    *,
    weights: np.ndarray | None,
) -> str | None:
    """Draw a selected 2D KDE overlay, returning a displayable warning on failure."""

    try:
        bandwidth = resolve_kde_bandwidth(
            style.get("kde_bandwidth", "scott"), style.get("kde_custom_factor", 1.0)
        )
        draw_kde_2d(ax, x, y, xedges, yedges, weights=weights, bandwidth=bandwidth)
    except Exception as exc:
        return str(exc) if isinstance(exc, KDEError) else f"KDE could not be calculated: {exc}"
    return None


def draw_histogram_contours(
    ax: Any,
    histogram: hist.Hist,
    style: dict[str, Any],
) -> str | None:
    """Draw selected contours from a prepared histogram's visible bins."""
    try:
        draw_highest_density_contours(
            ax,
            np.asarray(histogram.values(), dtype=float),
            np.asarray(histogram.axes[0].edges, dtype=float),
            np.asarray(histogram.axes[1].edges, dtype=float),
            style.get("contour_levels", []),
        )
    except ContourConfigurationError as exc:
        return str(exc)
    return None


def finish_static_axes(ax: Any, plot_type: str, title: str, style: dict[str, Any]) -> None:
    """Apply static-plot title, aspect-ratio, viewport, and grid settings."""

    ax.set_title(title or plot_type)
    if style.get("equal_aspect", False):
        ax.set_aspect("equal", adjustable="box")
    apply_axis_limits(ax, style)
    if style.get("grid", False):
        ax.grid(alpha=0.3)


def render_static_plot(
    fig,
    ax,
    plot_type: str,
    x: np.ndarray,
    y: np.ndarray | None,
    x_label: str,
    y_label: str,
    style: dict[str, Any],
    *,
    title: str,
    histogram_1d: hist.Hist | None = None,
    histogram_2d: hist.Hist | None = None,
    weights: np.ndarray | None = None,
    weight_branch: str | None = None,
    color: np.ndarray | None = None,
    color_label: str | None = None,
    reference_x: float | None = None,
    reference_label: str | None = None,
    colorbar_extend: str = "neither",
    fit_result: FitResult | None = None,
    histogram_flow_summary: Any | None = None,
    histogram_flow_mode: str = "ignore",
):
    warnings: list[str] = []
    if plot_type == "1D histogram":
        if histogram_1d is None:
            raise ValueError("1D histogram rendering requires a prepared histogram.")
        draw_histogram_1d(
            ax,
            histogram_1d,
            y_label,
            reference_x=reference_x,
            reference_label=reference_label,
            flow_summary=histogram_flow_summary,
            flow_mode=histogram_flow_mode,
            log_x=bool(style.get("log_x", False)),
        )
        if fit_result is not None and fit_result.success:
            draw_fit_overlay(ax, histogram_1d, fit_result)
        histogram_counts = np.asarray(histogram_1d.values(), dtype=float)
        histogram_edges = np.asarray(histogram_1d.axes[0].edges, dtype=float)
        if style.get("show_kde", False):
            kde_x = x
            kde_weights = weights
            if histogram_flow_summary is not None:
                inside = (x >= histogram_edges[0]) & (x <= histogram_edges[-1])
                kde_x = x[inside]
                kde_weights = weights[inside] if weights is not None else None
            warning = draw_kde_1d_overlay(ax, kde_x, histogram_edges, style, weights=kde_weights)
            if warning is not None:
                warnings.append(warning)
        apply_log_axis_scales(ax, style, x=x, histogram_counts=histogram_counts)

    elif plot_type == "2D histogram":
        if histogram_2d is None:
            raise ValueError("2D histogram rendering requires a prepared histogram.")
        draw_histogram_2d(
            fig,
            ax,
            histogram_2d,
            log_scale=bool(style["log_z"]),
            colorbar_label="entries" if weights is None else f"sum weights ({weight_branch})",
        )
        if style.get("show_contours", False):
            warning = draw_histogram_contours(ax, histogram_2d, style)
            if warning is not None:
                warnings.append(warning)
        if style.get("show_kde", False):
            if y is None:
                raise ValueError("2D histogram KDE requires y observations.")
            warning = draw_kde_2d_overlay(
                ax,
                x,
                y,
                histogram_2d.axes[0].edges,
                histogram_2d.axes[1].edges,
                style,
                weights=weights,
            )
            if warning is not None:
                warnings.append(warning)
        if y is not None:
            apply_log_axis_scales(ax, style, x=x, y=y)

    elif plot_type == "2D scatter":
        if y is None:
            raise ValueError("2D scatter requires a y variable.")
        if color is not None:
            draw_scatter = (
                draw_colored_scatter_2d_log
                if style.get("log_color", False)
                else draw_colored_scatter_2d_linear
            )
            draw_scatter(
                fig,
                ax,
                x,
                y,
                color,
                x_label,
                y_label,
                color_label,
                style,
                colorbar_extend=colorbar_extend,
            )
        else:
            draw_scatter_2d(ax, x, y, x_label, y_label, style)
        apply_log_axis_scales(ax, style, x=x, y=y)

    else:
        raise ValueError(f"Unsupported static plot type: {plot_type}")

    finish_static_axes(ax, plot_type, title, style)
    return warnings
