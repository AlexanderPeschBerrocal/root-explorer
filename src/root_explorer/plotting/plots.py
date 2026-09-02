"""Plot construction, filtering, and export helpers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional
from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import LogNorm

from ..analysis.arrays import align_arrays


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

    for cut, (_, lo, hi) in zip(aligned_cuts, cuts):
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


def parse_optional_float(text: str) -> float | None:
    stripped = str(text).strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def describe_array(name: str, arr: np.ndarray) -> dict[str, Any]:
    if len(arr) == 0:
        return {
            "variable": name,
            "n": 0,
            "min": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "max": np.nan,
        }
    values = arr.astype(float, copy=False)
    return {
        "variable": name,
        "n": int(len(values)),
        "min": float(np.nanmin(values)),
        "median": float(np.nanmedian(values)),
        "mean": float(np.nanmean(values)),
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
    h, xedges, yedges = np.histogram2d(x, y, bins=(bins_x, bins_y), weights=weights)
    positive = h[np.isfinite(h) & (h > 0)]
    norm = None
    if log_scale and len(positive) > 1 and np.nanmax(positive) > np.nanmin(positive):
        norm = LogNorm(vmin=float(np.nanmin(positive)), vmax=float(np.nanmax(positive)))
    return ax.pcolormesh(xedges, yedges, h.T, shading="auto", norm=norm)


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
    marker = dict(
        size=marker_size, opacity=opacity, colorscale=colorscale, reversescale=reversescale
    )
    if color is not None:
        marker.update(color=color, colorbar=dict(title=color_label or "color"))
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
        scene=dict(
            xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label, aspectmode=aspectmode
        ),
        margin=dict(l=0, r=0, b=0, t=42),
    )
    return fig


def matplotlib_cmap_name(colorscale: str, reverse: bool = False) -> str:
    name = colorscale.lower()
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
):
    if plot_type == "1D histogram":
        ax.hist(x, bins=style["bins_x"], weights=weights)
        ax.set_xlabel(x_label)
        ax.set_ylabel(
            y_label or ("entries" if weights is None else f"weighted entries ({weight_branch})")
        )
        if style.get("log_y", False):
            ax.set_yscale("log")

    elif plot_type == "2D histogram":
        if y is None:
            raise ValueError("2D histogram requires a y variable.")
        mesh = safe_hist2d(
            ax, x, y, style["bins_x"], style["bins_y"], style["log_z"], weights=weights
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(
            mesh, ax=ax, label="entries" if weights is None else f"sum weights ({weight_branch})"
        )

    elif plot_type == "2D scatter":
        if y is None:
            raise ValueError("2D scatter requires a y variable.")
        if color is not None:
            cmin = parse_optional_float(style.get("color_min_text", ""))
            cmax = parse_optional_float(style.get("color_max_text", ""))
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
            fig.colorbar(scatter, ax=ax, label=color_label or "color")
        else:
            ax.scatter(x, y, s=style["marker_size_2d"], alpha=style["opacity_2d"])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    else:
        raise ValueError(f"Unsupported static plot type: {plot_type}")

    ax.set_title(title or plot_type)
    if style.get("equal_aspect", False):
        ax.set_aspect("equal", adjustable="box")
    if style.get("grid", False):
        ax.grid(alpha=0.3)
