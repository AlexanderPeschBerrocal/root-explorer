"""Standalone code generation for persisted ROOT histogram plots."""

from __future__ import annotations

from ..io.histograms import read_histogram
from .histograms import HistogramPlotSpec, render_histogram_data
from .sourcegen import bundle_python_source, python_source_literal


def generate_histogram_plot_code(spec: HistogramPlotSpec) -> str:
    """Vendor only selected ROOT histogram I/O and rendering helpers."""
    helpers = bundle_python_source(
        HistogramPlotSpec, read_histogram, render_histogram_data
    ).rstrip()
    return (
        '"""Standalone ROOT histogram exported by ROOT Explorer."""\n\n'
        f"{helpers}\n\n"
        f"PLOT = {python_source_literal(spec)}\n\n\n"
        "def main() -> None:\n"
        "    data = read_histogram(PLOT.root_path, PLOT.object_path)\n"
        "    render_histogram_data(data, PLOT.style, PLOT.labels)\n"
        "    plt.show()\n\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )
