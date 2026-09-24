from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from root_explorer.analysis.arrays import ArrayShapeMismatchError
from root_explorer.plotting.plots import (
    ContourConfigurationError,
    InvalidAxisScaleError,
    apply_range_cuts,
    build_plotly_3d,
    describe_array,
    downsample_arrays,
    draw_highest_density_contours,
    highest_density_contour_thresholds,
    matplotlib_cmap_name,
    matplotlib_figure_bytes,
    parse_contour_levels,
    parse_optional_float,
    plotly_figure_html,
    render_static_plot,
    safe_filename,
    save_matplotlib_figure,
    save_plotly_figure,
)


def test_contour_levels_are_validated_sorted_and_deduplicated() -> None:
    assert parse_contour_levels("95, 68 95, 99") == [68.0, 95.0, 99.0]

    for invalid in ["", "0", "100", "68, nope", [np.nan]]:
        with pytest.raises(ContourConfigurationError):
            parse_contour_levels(invalid)


def test_highest_density_thresholds_use_cumulative_bin_content() -> None:
    histogram = np.array([[4.0, 3.0], [2.0, 1.0]])

    thresholds = highest_density_contour_thresholds(histogram, [90, 50, 80])

    assert thresholds == {3.0: (50.0,), 2.0: (80.0, 90.0)}


def test_weighted_contour_thresholds_and_invalid_bin_contents() -> None:
    weighted = np.array([[5.5, 2.5], [1.5, 0.5]])
    assert highest_density_contour_thresholds(weighted, [60, 90]) == {
        2.5: (60.0,),
        1.5: (90.0,),
    }

    with pytest.raises(ContourConfigurationError, match="negative bin contents"):
        highest_density_contour_thresholds(np.array([[2.0, -0.1]]), [68])
    with pytest.raises(ContourConfigurationError, match="no positive bin contents"):
        highest_density_contour_thresholds(np.zeros((2, 2)), [68])


def test_highest_density_contours_create_contour_artists() -> None:
    figure, axes = plt.subplots()
    histogram = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 8.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    edges = np.arange(4, dtype=float)

    contours = draw_highest_density_contours(axes, histogram, edges, edges, [90])

    assert contours.levels.tolist() == [1.0]
    assert any(segment.size for segments in contours.allsegs for segment in segments)
    plt.close(figure)


def test_negative_weighted_bins_warn_without_hiding_the_histogram() -> None:
    figure, axes = plt.subplots()

    warnings = render_static_plot(
        figure,
        axes,
        "2D histogram",
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        "x",
        "y",
        {
            "bins_x": 2,
            "bins_y": 2,
            "log_z": False,
            "show_contours": True,
            "contour_levels": [68.0],
            "grid": False,
        },
        title="signed weights",
        weights=np.array([-1.0, 2.0]),
    )

    assert warnings == [
        "Contours are unavailable because signed weights produced negative bin contents."
    ]
    assert axes.collections
    plt.close(figure)


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


def test_weighted_histogram_uses_weighted_default_y_label() -> None:
    fig, ax = plt.subplots()
    render_static_plot(
        fig,
        ax,
        "1D histogram",
        np.array([1.0, 2.0]),
        None,
        "pt",
        "",
        {"bins_x": 10, "log_y": False, "grid": False},
        title="weighted",
        weights=np.array([2.0, 3.0]),
        weight_branch="genWeight",
    )

    assert ax.get_ylabel() == "sum weights (genWeight)"
    plt.close(fig)


def test_logarithmic_axes_for_histogram_and_scatter() -> None:
    histogram_figure, histogram_axes = plt.subplots()
    render_static_plot(
        histogram_figure,
        histogram_axes,
        "1D histogram",
        np.array([1.0, 2.0, 4.0]),
        None,
        "x",
        "entries",
        {"bins_x": 3, "log_x": True, "log_y": True, "grid": False},
        title="log histogram",
    )
    assert histogram_axes.get_xscale() == "log"
    assert histogram_axes.get_yscale() == "log"
    plt.close(histogram_figure)

    scatter_figure, scatter_axes = plt.subplots()
    render_static_plot(
        scatter_figure,
        scatter_axes,
        "2D scatter",
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        "x",
        "y",
        {
            "marker_size_2d": 3.0,
            "opacity_2d": 0.5,
            "log_x": True,
            "log_y": True,
            "grid": False,
        },
        title="log scatter",
    )
    assert scatter_axes.get_xscale() == "log"
    assert scatter_axes.get_yscale() == "log"
    plt.close(scatter_figure)


@pytest.mark.parametrize(
    ("x", "y", "style", "expected_axis"),
    [
        (np.array([0.0, 1.0]), np.array([1.0, 2.0]), {"log_x": True}, "x-axis"),
        (np.array([1.0, 2.0]), np.array([-1.0, 2.0]), {"log_y": True}, "y-axis"),
    ],
)
def test_scatter_rejects_non_positive_log_coordinates(x, y, style, expected_axis) -> None:
    figure, axes = plt.subplots()
    scatter_style = {
        "marker_size_2d": 3.0,
        "opacity_2d": 0.5,
        "grid": False,
        **style,
    }

    with pytest.raises(InvalidAxisScaleError, match=expected_axis):
        render_static_plot(
            figure,
            axes,
            "2D scatter",
            x,
            y,
            "x",
            "y",
            scatter_style,
            title="invalid log scatter",
        )
    plt.close(figure)


def test_histogram_log_y_requires_a_positive_bin_and_log_z_is_independent() -> None:
    invalid_x_figure, invalid_x_axes = plt.subplots()
    with pytest.raises(InvalidAxisScaleError, match="x-axis"):
        render_static_plot(
            invalid_x_figure,
            invalid_x_axes,
            "1D histogram",
            np.array([0.0, 1.0]),
            None,
            "x",
            "entries",
            {"bins_x": 2, "log_x": True, "grid": False},
            title="invalid log-x histogram",
        )
    plt.close(invalid_x_figure)

    histogram_figure, histogram_axes = plt.subplots()
    with pytest.raises(InvalidAxisScaleError, match="no positive bin contents"):
        render_static_plot(
            histogram_figure,
            histogram_axes,
            "1D histogram",
            np.array([1.0, 2.0]),
            None,
            "x",
            "weighted entries",
            {"bins_x": 2, "log_y": True, "grid": False},
            title="invalid weighted log histogram",
            weights=np.array([-1.0, -1.0]),
        )
    plt.close(histogram_figure)

    histogram_2d_figure, histogram_2d_axes = plt.subplots()
    render_static_plot(
        histogram_2d_figure,
        histogram_2d_axes,
        "2D histogram",
        np.array([1.0, 2.0, 2.0]),
        np.array([1.0, 2.0, 2.0]),
        "x",
        "y",
        {
            "bins_x": 2,
            "bins_y": 2,
            "log_x": False,
            "log_y": False,
            "log_z": True,
            "grid": False,
        },
        title="log color only",
    )
    assert histogram_2d_axes.get_xscale() == "linear"
    assert histogram_2d_axes.get_yscale() == "linear"
    plt.close(histogram_2d_figure)


@pytest.mark.parametrize(
    ("reverse", "expected_name"),
    [(False, "Greys"), (True, "Greys_r")],
)
def test_greys_colormap_for_static_scatter(reverse: bool, expected_name: str) -> None:
    assert matplotlib_cmap_name("Greys", reverse) == expected_name

    fig, ax = plt.subplots()
    render_static_plot(
        fig,
        ax,
        "2D scatter",
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        "x",
        "y",
        {
            "marker_size_2d": 3.0,
            "opacity_2d": 0.5,
            "colorscale": "Greys",
            "reverse_colorscale": reverse,
            "grid": False,
        },
        title="Greys scatter",
        color=np.array([0.0, 1.0]),
    )

    assert ax.collections[0].cmap.name == expected_name
    plt.close(fig)
