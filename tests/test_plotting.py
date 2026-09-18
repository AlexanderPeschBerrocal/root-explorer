from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from root_explorer.analysis.arrays import ArrayShapeMismatchError
from root_explorer.plotting.plots import (
    apply_range_cuts,
    build_plotly_3d,
    describe_array,
    downsample_arrays,
    matplotlib_figure_bytes,
    parse_optional_float,
    plotly_figure_html,
    render_static_plot,
    safe_filename,
    save_matplotlib_figure,
    save_plotly_figure,
)


def test_filters_downsampling_and_descriptions() -> None:
    arrays = [np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])]
    cut = np.array([0.0, 2.0, np.nan])
    filtered = apply_range_cuts(arrays, [(cut, 1.0, None)])
    assert [array.tolist() for array in filtered] == [[2.0], [20.0]]

    sampled_one = downsample_arrays([np.arange(20), np.arange(20) * 2], 5)
    sampled_two = downsample_arrays([np.arange(20), np.arange(20) * 2], 5)
    assert [array.tolist() for array in sampled_one] == [array.tolist() for array in sampled_two]
    assert np.all(sampled_one[1] == sampled_one[0] * 2)
    assert describe_array("x", np.array([]))["n"] == 0
    assert parse_optional_float(" 2.5 ") == 2.5
    assert parse_optional_float("not-a-number") is None

    with pytest.raises(ArrayShapeMismatchError):
        apply_range_cuts([np.array([1.0]), np.array([2.0, 3.0])], [])


def test_static_and_plotly_figures_export(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    render_static_plot(
        fig,
        ax,
        "2D scatter",
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        "x",
        "y",
        {"marker_size_2d": 3.0, "opacity_2d": 0.5, "grid": True},
        title="scatter",
    )
    assert ax.get_title() == "scatter"
    assert save_matplotlib_figure(fig, tmp_path, "unsafe / title", ["png", "pdf"])
    assert (tmp_path / "unsafe___title.png").is_file()
    assert matplotlib_figure_bytes(fig, "png").startswith(b"\x89PNG")
    plt.close(fig)

    interactive = build_plotly_3d(
        [1],
        [2],
        [3],
        "x",
        "y",
        "z",
        color=[4],
        color_label="energy",
        title="event",
        colorscale="coolwarm",
    )
    assert interactive.layout.scene.xaxis.title.text == "x"
    assert interactive.data[0].marker.color[0] == 4
    assert len(interactive.data[0].marker.colorscale) == 11
    html_path = save_plotly_figure(interactive, tmp_path, "event plot")
    assert html_path.name == "event_plot.html"
    assert b"plotly" in plotly_figure_html(interactive).lower()
    assert safe_filename("../bad name") == ".._bad_name"
