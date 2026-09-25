"""Standalone Python source generation for static ROOT Explorer plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .pipeline import (
    STATIC_PLOT_TYPES,
    StaticPlotSpec,
    prepare_static_plot,
    render_prepared_static_plot,
)
from .sourcegen import bundle_python_source, python_source_literal

if TYPE_CHECKING:
    from ..app.plotting import PlotRequest


def generate_static_plot_code(request: PlotRequest) -> str:
    """Return a standalone script assembled from the canonical plotting pipeline."""

    if request.plot_type not in STATIC_PLOT_TYPES:
        raise ValueError(f"Plot-code download is not available for {request.plot_type!r}.")

    spec = request.to_static_spec()
    helpers = bundle_python_source(
        StaticPlotSpec,
        prepare_static_plot,
        render_prepared_static_plot,
    ).rstrip()
    configuration = python_source_literal(spec)

    return (
        '"""Standalone plot exported by ROOT Explorer.\n\n'
        "Edit PLOT.root_path in the configuration below if the ROOT file is moved.\n"
        '"""\n\n'
        f"{helpers}\n\n"
        "# Plot configuration\n"
        f"PLOT = {configuration}\n\n\n"
        "def main() -> None:\n"
        "    prepared = prepare_static_plot(PLOT)\n"
        "    figure, plot_warnings = render_prepared_static_plot(PLOT, prepared)\n"
        "    for warning in plot_warnings:\n"
        "        print(f\"ROOT Explorer warning: {warning}\")\n"
        "    plt.show()\n\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )
