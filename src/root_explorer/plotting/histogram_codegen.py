"""Standalone code generation for persisted ROOT histogram plots."""

from __future__ import annotations

from ..analysis.fitting import FitConfig
from ..io.histograms import read_histogram
from .histograms import HistogramPlotSpec, fit_persisted_histogram, render_histogram_data
from .sourcegen import bundle_python_source, python_source_literal


def generate_histogram_plot_code(spec: HistogramPlotSpec) -> str:
    """Vendor only selected ROOT histogram I/O and rendering helpers."""
    roots = [HistogramPlotSpec, read_histogram, render_histogram_data]
    if spec.fit_config is not None:
        from .plots import draw_fit_overlay

        roots.extend([FitConfig, fit_persisted_histogram, draw_fit_overlay])
    helpers = bundle_python_source(*roots).rstrip()
    return (
        '"""Standalone ROOT histogram exported by ROOT Explorer."""\n\n'
        f"{helpers}\n\n"
        f"PLOT = {python_source_literal(spec)}\n\n\n"
        "def main() -> None:\n"
        "    data = read_histogram(PLOT.root_path, PLOT.object_path)\n"
        + (
            "    result = fit_persisted_histogram(data, PLOT.fit_config)\n"
            "    if not result.success: print(f'Fit warning: {result.message}')\n"
            "    figure = render_histogram_data(data, PLOT.style, PLOT.labels)\n"
            "    if result.success: draw_fit_overlay(figure.axes[0], data.histogram, result)\n"
            if spec.fit_config is not None
            else "    render_histogram_data(data, PLOT.style, PLOT.labels)\n"
        )
        + "    plt.show()\n\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )
