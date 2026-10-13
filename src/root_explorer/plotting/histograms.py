"""Shared static rendering of already-binned, persisted ROOT histograms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LogNorm

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


def histogram_errors(data: PersistedHistogram) -> np.ndarray | None:
    """Retrieve uncertainties without inventing Poisson errors for profiles."""
    variance = data.profile_variances
    if variance is None:
        variance = data.histogram.variances()
    if variance is None:
        return None
    variances = np.asarray(variance, dtype=float)
    return np.sqrt(np.clip(variances, 0, None))


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
            mode = str(style.get("histogram_mode", "Error bars" if data.is_profile else "Step"))
            if mode not in {"Step", "Error bars", "Step + error bars", "Filled"}:
                raise ValueError(f"Unknown histogram drawing mode: {mode}")
            if mode in {"Step", "Step + error bars", "Filled"}:
                ax.stairs(values, edges, fill=(mode == "Filled"))
            if mode in {"Error bars", "Step + error bars"}:
                errors = histogram_errors(data)
                centers = (edges[:-1] + edges[1:]) / 2
                ax.errorbar(
                    centers,
                    values,
                    yerr=errors,
                    fmt="o",
                    markersize=4,
                    capsize=2,
                    linestyle="none",
                )
            if style.get("log_x", False):
                if not np.all(np.isfinite(edges)) or np.any(edges <= 0):
                    raise InvalidAxisScaleError("Log x-axis requires positive histogram edges.")
                ax.set_xscale("log")
            if style.get("log_y", False):
                apply_log_histogram_y_axis(ax, values)
            ax.set_xlabel(labels.get("custom_x") or data.x_label)
            ax.set_ylabel(labels.get("custom_y") or data.y_label)
        elif h.ndim == 2:
            values = np.asarray(h.values(), dtype=float)
            log_z = bool(style.get("log_z", False))
            if log_z and not np.any(np.isfinite(values) & (values > 0)):
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
            if log_z:
                positive = values[np.isfinite(values) & (values > 0)]
                lo, hi = float(np.min(positive)), float(np.max(positive))
                if lo == hi:
                    hi = np.nextafter(hi, np.inf)
                mesh.set_norm(LogNorm(vmin=lo, vmax=hi))
                mesh.set_array(
                    np.ma.masked_where(~np.isfinite(values.T) | (values.T <= 0), values.T).ravel()
                )
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
    return render_histogram_data(
        read_histogram(spec.root_path, spec.object_path), spec.style, spec.labels
    )
