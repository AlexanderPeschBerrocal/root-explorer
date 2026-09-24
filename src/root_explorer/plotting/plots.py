"""Plot construction, filtering, and export helpers."""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import colormaps
from matplotlib.colors import LogNorm, to_hex
from scipy.stats import gaussian_kde

from ..analysis.arrays import align_arrays


class InvalidAxisScaleError(ValueError):
    """Raised when selected values cannot be displayed on a requested axis scale."""


class InvalidColorScaleError(ValueError):
    """Raised when scatter colour normalization is invalid."""


class ContourConfigurationError(ValueError):
    """Raised when highest-density contours cannot be configured for a histogram."""


class KDEError(ValueError):
    """Raised when a KDE overlay cannot be constructed without misleading output."""


KDE_MAX_POINTS = 5_000
KDE_1D_GRID_SIZE = 512
KDE_2D_MAX_GRID_SIZE = 100


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
        _require_positive_axis_values(x, "x")
        ax.set_xscale("log")

    if not style.get("log_y", False):
        return

    if histogram_counts is not None:
        counts = np.asarray(histogram_counts, dtype=float)
        if not np.any(np.isfinite(counts) & (counts > 0)):
            raise InvalidAxisScaleError(
                "Cannot use a logarithmic y-axis: the histogram has no positive bin contents."
            )
    else:
        if y is None:
            raise InvalidAxisScaleError(
                "Cannot use a logarithmic y-axis: no y coordinates are available."
            )
        _require_positive_axis_values(y, "y")
    ax.set_yscale("log")


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


def scatter_color_scale(
    color: np.ndarray, style: dict[str, Any]
) -> tuple[LogNorm | None, float | None, float | None]:
    """Validate and resolve 2D-scatter colour normalization."""

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

    values = np.asarray(color, dtype=float)
    if not style.get("log_color", False):
        return None, cmin, cmax

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
    return LogNorm(vmin=vmin, vmax=vmax), None, None


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


def safe_hist2d(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    bins_x: int,
    bins_y: int,
    log_scale: bool,
    weights: np.ndarray | None = None,
):
    h, xedges, yedges = histogram2d_values(x, y, bins_x, bins_y, weights=weights)
    return draw_hist2d(ax, h, xedges, yedges, log_scale)


def histogram2d_values(
    x: np.ndarray,
    y: np.ndarray,
    bins_x: int,
    bins_y: int,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the shared histogram used by the color mesh and contours."""

    return np.histogram2d(x, y, bins=(bins_x, bins_y), weights=weights)


def draw_hist2d(ax, h, xedges, yedges, log_scale: bool):
    """Draw precomputed 2D histogram values."""

    positive = h[np.isfinite(h) & (h > 0)]
    norm = None
    if log_scale and len(positive) > 1 and np.nanmax(positive) > np.nanmin(positive):
        norm = LogNorm(vmin=float(np.nanmin(positive)), vmax=float(np.nanmax(positive)))
    return ax.pcolormesh(xedges, yedges, h.T, shading="auto", norm=norm)


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


def save_matplotlib_figure(fig, outdir: Path, name: str, formats: Sequence[str]) -> list[Path]:
    selected = [fmt for fmt in formats if fmt in {"png", "pdf"}]
    if not selected:
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    base = safe_filename(name)
    saved_paths: list[Path] = []
    for fmt in selected:
        path = outdir / f"{base}.{fmt}"
        if fmt == "png":
            fig.savefig(path, dpi=180, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        saved_paths.append(path)
    return saved_paths


def save_plotly_figure(fig, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{safe_filename(name)}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def matplotlib_figure_bytes(fig, file_format: str) -> bytes:
    """Render a Matplotlib figure for a browser download."""
    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {"format": file_format, "bbox_inches": "tight"}
    if file_format == "png":
        save_kwargs["dpi"] = 180
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
    weights: np.ndarray | None = None,
    weight_branch: str | None = None,
    color: np.ndarray | None = None,
    color_label: str | None = None,
    reference_x: float | None = None,
    reference_label: str | None = None,
):
    warnings: list[str] = []
    if plot_type == "1D histogram":
        histogram_counts, histogram_edges, _ = ax.hist(x, bins=style["bins_x"], weights=weights)
        if style.get("show_kde", False):
            try:
                bandwidth = resolve_kde_bandwidth(
                    style.get("kde_bandwidth", "scott"),
                    style.get("kde_custom_factor", 1.0),
                )
                draw_kde_1d(ax, x, histogram_edges, weights=weights, bandwidth=bandwidth)
            except Exception as exc:
                warnings.append(
                    str(exc) if isinstance(exc, KDEError) else f"KDE could not be calculated: {exc}"
                )
        ax.set_xlabel(x_label)
        ax.set_ylabel(
            y_label or ("entries" if weights is None else f"sum weights ({weight_branch})")
        )
        apply_log_axis_scales(ax, style, x=x, histogram_counts=histogram_counts)
        if reference_x is not None:
            ax.axvline(
                reference_x,
                color="tab:red",
                linestyle="--",
                linewidth=1.5,
                label=reference_label or f"x = {reference_x:g}",
            )
            ax.legend()

    elif plot_type == "2D histogram":
        if y is None:
            raise ValueError("2D histogram requires a y variable.")
        histogram, xedges, yedges = histogram2d_values(
            x, y, style["bins_x"], style["bins_y"], weights=weights
        )
        mesh = draw_hist2d(ax, histogram, xedges, yedges, style["log_z"])
        if style.get("show_contours", False):
            try:
                draw_highest_density_contours(
                    ax, histogram, xedges, yedges, style.get("contour_levels", [])
                )
            except ContourConfigurationError as exc:
                warnings.append(str(exc))
        if style.get("show_kde", False):
            try:
                bandwidth = resolve_kde_bandwidth(
                    style.get("kde_bandwidth", "scott"),
                    style.get("kde_custom_factor", 1.0),
                )
                draw_kde_2d(
                    ax,
                    x,
                    y,
                    xedges,
                    yedges,
                    weights=weights,
                    bandwidth=bandwidth,
                )
            except Exception as exc:
                warnings.append(
                    str(exc) if isinstance(exc, KDEError) else f"KDE could not be calculated: {exc}"
                )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(
            mesh, ax=ax, label="entries" if weights is None else f"sum weights ({weight_branch})"
        )
        apply_log_axis_scales(ax, style, x=x, y=y)

    elif plot_type == "2D scatter":
        if y is None:
            raise ValueError("2D scatter requires a y variable.")
        if color is not None:
            color_norm, cmin, cmax = scatter_color_scale(color, style)
            scatter = ax.scatter(
                x,
                y,
                s=style["marker_size_2d"],
                alpha=style["opacity_2d"],
                c=color,
                cmap=matplotlib_cmap_name(
                    style.get("colorscale", "Viridis"), style.get("reverse_colorscale", False)
                ),
                norm=color_norm,
                vmin=cmin,
                vmax=cmax,
            )
            fig.colorbar(scatter, ax=ax, label=color_label or "color")
        else:
            ax.scatter(x, y, s=style["marker_size_2d"], alpha=style["opacity_2d"])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        apply_log_axis_scales(ax, style, x=x, y=y)

    else:
        raise ValueError(f"Unsupported static plot type: {plot_type}")

    ax.set_title(title or plot_type)
    if style.get("equal_aspect", False):
        ax.set_aspect("equal", adjustable="box")
    if style.get("grid", False):
        ax.grid(alpha=0.3)
    return warnings
