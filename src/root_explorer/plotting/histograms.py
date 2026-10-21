"""Shared static rendering of already-binned, persisted ROOT histograms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LogNorm

from ..analysis.fitting import FitConfig
from ..io.histograms import PersistedHistogram, read_histogram
from .plots import (
    InvalidAxisScaleError,
    InvalidColorScaleError,
    apply_axis_limits,
    apply_log_histogram_y_axis,
    create_static_figure,
    draw_histogram_2d,
)


@dataclass(frozen=True)
class HistogramPlotSpec:
    root_path: str
    object_path: str
    style: dict[str, Any]
    labels: dict[str, str]
    fit_config: FitConfig | None = None


def histogram_errors(data: PersistedHistogram) -> np.ndarray | None:
    """Retrieve uncertainties without inventing Poisson errors for profiles."""
    variance = data.profile_variances
    if variance is None:
        variance = data.histogram.variances()
    if variance is None:
        return None
    variances = np.asarray(variance, dtype=float)
    return np.sqrt(np.clip(variances, 0, None))


def visible_profile_occupancy(data: PersistedHistogram) -> np.ndarray:
    """Return bin occupancy for plotting, excluding retained flow bins.

    For a counting histogram all visible bins remain drawable, including
    genuinely empty count bins. For a profile, ROOT's effective entries (not
    the mean itself) determine which visible bins contain measurements.
    """
    if data.profile_entries is None:
        return np.ones(data.histogram.values().shape, dtype=bool)
    visible = data.profile_entries[tuple(slice(1, -1) for _ in data.histogram.axes)]
    return np.isfinite(visible) & (visible > 0)


def render_histogram_data(
    data: PersistedHistogram, style: dict[str, Any], labels: dict[str, str]
) -> Any:
    """Draw prepared histogram bins directly; never reconstruct pseudo-events."""
    h = data.histogram
    fig, ax = create_static_figure(style)
    try:
        if h.ndim == 1:
            edges = np.asarray(h.axes[0].edges, dtype=float)
            values = np.asarray(h.values(), dtype=float)
            occupied = visible_profile_occupancy(data)
            # NaNs split step paths without changing the original hist.Hist.
            # The mask is based on entries, not on whether the mean is zero.
            display_values = np.where(occupied, values, np.nan)
            mode = str(style.get("histogram_mode", "Error bars" if data.is_profile else "Step"))
            if mode not in {"Step", "Error bars", "Step + error bars", "Filled"}:
                raise ValueError(f"Unknown histogram drawing mode: {mode}")
            if mode in {"Step", "Step + error bars", "Filled"}:
                ax.stairs(display_values, edges, fill=(mode == "Filled"), color="tab:blue")
            if mode in {"Error bars", "Step + error bars"}:
                errors = histogram_errors(data)
                centers = (edges[:-1] + edges[1:]) / 2
                ax.errorbar(
                    centers[occupied],
                    values[occupied],
                    yerr=None if errors is None else errors[occupied],
                    fmt="o",
                    color="tab:blue",
                    markersize=4,
                    capsize=2,
                    linestyle="none",
                )
            if style.get("log_x", False):
                if not np.all(np.isfinite(edges)) or np.any(edges <= 0):
                    raise InvalidAxisScaleError("Log x-axis requires positive histogram edges.")
                ax.set_xscale("log")
            if style.get("log_y", False):
                apply_log_histogram_y_axis(ax, display_values)
            ax.set_xlabel(labels.get("custom_x") or data.x_label)
            ax.set_ylabel(labels.get("custom_y") or data.y_label)
        elif h.ndim == 2:
            values = np.asarray(h.values(), dtype=float)
            occupied = visible_profile_occupancy(data)
            log_z = bool(style.get("log_z", False))
            positive_bins = np.isfinite(values) & occupied & (values > 0)
            if log_z and not np.any(positive_bins):
                raise InvalidColorScaleError("Log colour scale requires positive bin contents.")
            # Delegate mesh creation and colourbar layout to the shared renderer.
            draw_histogram_2d(
                fig,
                ax,
                h,
                log_scale=log_z,
                colorbar_label=labels.get("custom_z") or data.colorbar_label,
            )
            cmap = str(style.get("cmap", "viridis"))
            mesh = ax.collections[0]
            mesh.set_cmap(colormaps[cmap])
            if data.is_profile or log_z:
                # pcolormesh expects y-major bin order. Mask only for plotting;
                # zero-valued means with positive occupancy remain visible.
                invalid = ~np.isfinite(values) | ~occupied
                if log_z:
                    invalid |= values <= 0
                display_values = np.ma.array(values.T, mask=invalid.T).ravel()
                mesh.set_array(display_values)
                if not log_z:
                    # Recompute the colour range without empty profile cells.
                    mesh.autoscale()
            if log_z:
                positive = values[positive_bins]
                lo, hi = float(np.min(positive)), float(np.max(positive))
                if lo == hi:
                    hi = np.nextafter(hi, np.inf)
                mesh.set_norm(LogNorm(vmin=lo, vmax=hi))
            if style.get("log_x", False):
                if np.any(np.asarray(h.axes[0].edges) <= 0):
                    raise InvalidAxisScaleError("Log x-axis requires positive histogram edges.")
                ax.set_xscale("log")
            if style.get("log_y", False):
                if np.any(np.asarray(h.axes[1].edges) <= 0):
                    raise InvalidAxisScaleError("Log y-axis requires positive histogram edges.")
                ax.set_yscale("log")
            ax.set_xlabel(labels.get("custom_x") or data.x_label)
            ax.set_ylabel(labels.get("custom_y") or data.y_label)
        else:
            raise ValueError("Only 1D and 2D persisted histograms can be rendered.")
        ax.set_title(labels.get("title") or data.title or "ROOT histogram")
        apply_axis_limits(ax, style)
        if style.get("grid", False):
            ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig
    except Exception:
        plt.close(fig)
        raise


def render_histogram_plot(spec: HistogramPlotSpec) -> Any:
    data = read_histogram(spec.root_path, spec.object_path)
    from ..analysis.fitting import fit_histogram_1d

    figure = render_histogram_data(data, spec.style, spec.labels)
    if spec.fit_config is not None:
        from .plots import draw_fit_overlay

        result = fit_histogram_1d(data.histogram, spec.fit_config)
        if result.success:
            draw_fit_overlay(figure.axes[0], data.histogram, result)
            figure.tight_layout()
    return figure
