"""Standalone source generation for selected persisted ROOT graphs."""

from __future__ import annotations

from ..io.graphs import read_graph
from .graphs import GraphPlotSpec, render_graph_data
from .sourcegen import bundle_python_source, literal_dataclass_types, python_source_literal


def generate_graph_plot_code(spec: GraphPlotSpec) -> str:
    """Emit a runnable source-vendored graph script without app imports."""
    helpers = bundle_python_source(
        GraphPlotSpec, read_graph, render_graph_data, *literal_dataclass_types(spec)
    ).rstrip()
    config = python_source_literal(spec)
    return (
        '"""Standalone ROOT graph exported by ROOT Explorer."""\n\n'
        f"{helpers}\n\n"
        f"PLOT = {config}\n\n\n"
        "def main() -> None:\n"
        "    graph = read_graph(PLOT.root_path, PLOT.object_path)\n"
        "    fig = render_graph_data(graph, PLOT.style, PLOT.labels)\n"
        "    plt.show()\n\n\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )
