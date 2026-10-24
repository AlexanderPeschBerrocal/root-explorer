from __future__ import annotations

import struct

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, to_rgba

from root_explorer.analysis.arrays import ArrayShapeMismatchError
from root_explorer.analysis.fitting import FitResult
from root_explorer.analysis.histograms import build_histogram_1d, build_histogram_2d
from root_explorer.plotting.plots import (
    KDE_MAX_POINTS,
    ContourConfigurationError,
    InvalidAxisLimitError,
    InvalidAxisScaleError,
    InvalidColorScaleError,
    InvalidLabelError,
    InvalidPlotResolutionError,
    KDEError,
    apply_range_cuts,
    build_plotly_3d,
    calculate_color_clipping,
    calculate_correlations,
    describe_array,
    downsample_arrays,
    draw_highest_density_contours,
    draw_histogram_2d,
    draw_kde_1d,
    draw_kde_2d,
    highest_density_contour_thresholds,
    matplotlib_cmap_name,
    matplotlib_figure_bytes,
    parse_contour_levels,
    parse_optional_float,
    plotly_figure_html,
    render_static_plot,
    resolve_kde_bandwidth,
    safe_filename,
    validate_plot_labels,
    validate_raster_resolution,
)


def test_kde_bandwidth_options_and_validation() -> None:
    assert resolve_kde_bandwidth("Scott") == "scott"
    assert resolve_kde_bandwidth("SILVERMAN") == "silverman"
    assert resolve_kde_bandwidth("custom", 0.4) == 0.4

    for method, factor in [
        ("unknown", 1.0),
        ("custom", 0.0),
        ("custom", np.nan),
        ("custom", "invalid"),
    ]:
        with pytest.raises(KDEError):
            resolve_kde_bandwidth(method, factor)


@pytest.mark.parametrize("bandwidth", ["scott", "silverman", 0.5])
def test_1d_kde_is_scaled_to_histogram_units(bandwidth: str | float) -> None:
    rng = np.random.default_rng(17)
    finite_values = rng.normal(size=1_000)
    finite_weights = rng.uniform(0.5, 2.0, size=len(finite_values))
    values = np.append(finite_values, np.nan)
    weights = np.append(finite_weights, 1_000.0)
    edges = np.linspace(-5.0, 5.0, 41)
    figure, axes = plt.subplots()

    line = draw_kde_1d(axes, values, edges, weights=weights, bandwidth=bandwidth)

    represented_weight = np.trapezoid(line.get_ydata(), line.get_xdata()) / np.diff(edges)[0]
    assert represented_weight == pytest.approx(finite_weights.sum(), rel=0.01)
    assert line.get_label() == "KDE"
    plt.close(figure)


def test_unweighted_1d_kde_is_scaled_to_entry_counts() -> None:
    rng = np.random.default_rng(23)
    values = rng.normal(size=500)
    edges = np.linspace(-5.0, 5.0, 26)
    figure, axes = plt.subplots()

    line = draw_kde_1d(axes, values, edges)

    represented_entries = np.trapezoid(line.get_ydata(), line.get_xdata()) / np.diff(edges)[0]
    assert represented_entries == pytest.approx(len(values), rel=0.01)
    plt.close(figure)


def test_1d_kde_is_shown_in_legend_without_other_overlays() -> None:
    values = np.linspace(-2.0, 2.0, 100)
    figure, axes = plt.subplots()
    try:
        warnings = render_static_plot(
            figure,
            axes,
            "1D histogram",
            values,
            None,
            "x",
            "entries",
            {
                "show_kde": True,
                "kde_bandwidth": "scott",
                "log_y": False,
                "grid": False,
            },
            title="KDE legend",
            histogram_1d=build_histogram_1d(values, 10, label="x"),
        )

        assert not warnings
        assert [text.get_text() for text in axes.get_legend().get_texts()] == ["KDE"]
    finally:
        plt.close(figure)


def test_2d_weighted_kde_creates_contour_lines() -> None:
    rng = np.random.default_rng(29)
    x = rng.normal(size=500)
    y = 0.4 * x + rng.normal(size=500)
    weights = rng.uniform(0.1, 2.0, size=len(x))
    edges = np.linspace(-5.0, 5.0, 31)
    figure, axes = plt.subplots()

    contours = draw_kde_2d(axes, x, y, edges, edges, weights=weights, bandwidth="silverman")

    assert any(segment.size for segments in contours.allsegs for segment in segments)
    plt.close(figure)


def test_kde_sampling_is_deterministic_and_bounded(monkeypatch) -> None:
    captured: list[np.ndarray] = []

    class FakeKDE:
        def __init__(self, dataset, **_kwargs):
            captured.append(np.asarray(dataset).copy())

        def __call__(self, grid):
            return np.ones(np.asarray(grid).shape[-1])

    monkeypatch.setattr("root_explorer.plotting.plots.gaussian_kde", FakeKDE)
    values = np.linspace(-2.0, 2.0, KDE_MAX_POINTS + 100)
    edges = np.linspace(-3.0, 3.0, 20)

    for _ in range(2):
        figure, axes = plt.subplots()
        draw_kde_1d(axes, values, edges)
        plt.close(figure)

    assert captured[0].shape == (KDE_MAX_POINTS,)
    assert np.array_equal(captured[0], captured[1])


@pytest.mark.parametrize(
    ("values", "weights", "message"),
    [
        (np.array([1.0]), None, "at least 2"),
        (np.ones(3), None, "constant variable"),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, -1.0, 1.0]), "negative values"),
        (np.array([1.0, 2.0, 3.0]), np.zeros(3), "weight is zero"),
    ],
)
def test_pathological_1d_kde_samples_fail_clearly(values, weights, message) -> None:
    figure, axes = plt.subplots()
    with pytest.raises(KDEError, match=message):
        draw_kde_1d(axes, values, np.linspace(0.0, 4.0, 5), weights=weights)
    plt.close(figure)


def test_kde_failure_warns_without_hiding_histogram() -> None:
    figure, axes = plt.subplots()

    warnings = render_static_plot(
        figure,
        axes,
        "1D histogram",
        np.array([1.0, 2.0, 3.0]),
        None,
        "x",
        "entries",
        {
            "bins_x": 3,
            "show_kde": True,
            "kde_bandwidth": "scott",
            "log_y": False,
            "grid": False,
        },
        title="signed weights",
        histogram_1d=build_histogram_1d(
            np.array([1.0, 2.0, 3.0]),
            3,
            weights=np.array([1.0, -1.0, 2.0]),
            label="x",
        ),
        weights=np.array([1.0, -1.0, 2.0]),
    )

    assert warnings == ["KDE is unavailable because the selected weights contain negative values."]
    assert axes.patches
    plt.close(figure)


def test_singular_2d_kde_warns_without_hiding_histogram() -> None:
    figure, axes = plt.subplots()
    x = np.arange(5, dtype=float)

    warnings = render_static_plot(
        figure,
        axes,
        "2D histogram",
        x,
        2.0 * x,
        "x",
        "y",
        {
            "bins_x": 5,
            "bins_y": 5,
            "log_z": False,
            "show_kde": True,
            "kde_bandwidth": "scott",
            "grid": False,
        },
        title="singular KDE",
        histogram_2d=build_histogram_2d(x, 2.0 * x, 5, 5, x_label="x", y_label="y"),
    )

    assert len(warnings) == 1
    assert "KDE could not be calculated" in warnings[0]
    assert axes.collections
    plt.close(figure)


def test_correlations_cover_linear_rank_and_non_finite_behaviour() -> None:
    positive = calculate_correlations(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    negative = calculate_correlations(np.array([1.0, 2.0, 3.0]), np.array([6.0, 4.0, 2.0]))
    nonlinear = calculate_correlations(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 4.0, 9.0, 16.0])
    )
    filtered = calculate_correlations(
        np.array([1.0, np.nan, 2.0, 3.0]), np.array([2.0, 100.0, 4.0, np.inf])
    )

    assert positive.pearson_r == pytest.approx(1.0)
    assert positive.spearman_rho == pytest.approx(1.0)
    assert negative.pearson_r == pytest.approx(-1.0)
    assert negative.spearman_rho == pytest.approx(-1.0)
    assert nonlinear.pearson_r < nonlinear.spearman_rho
    assert nonlinear.spearman_rho == pytest.approx(1.0)
    assert filtered.n_pairs == 2
    assert filtered.pearson_r == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (np.array([1.0]), np.array([2.0])),
        (np.array([1.0, 1.0]), np.array([2.0, 3.0])),
        (np.array([1.0, 2.0]), np.array([3.0, 3.0])),
    ],
)
def test_undefined_correlations_return_nan(x: np.ndarray, y: np.ndarray) -> None:
    result = calculate_correlations(x, y)

    assert np.isnan(result.pearson_r)
    assert np.isnan(result.spearman_rho)


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
        histogram_2d=build_histogram_2d(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            2,
            2,
            weights=np.array([-1.0, 2.0]),
            x_label="x",
            y_label="y",
        ),
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


def test_static_and_plotly_figures_export() -> None:
    fig, ax = plt.subplots(figsize=(2.0, 1.0))
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
    png_150 = matplotlib_figure_bytes(fig, "png", dpi=150)
    png_300 = matplotlib_figure_bytes(fig, "png", dpi=300)
    assert png_150.startswith(b"\x89PNG")
    assert matplotlib_figure_bytes(fig, "pdf", dpi=300).startswith(b"%PDF")
    width_150, height_150 = struct.unpack(">II", png_150[16:24])
    width_300, height_300 = struct.unpack(">II", png_300[16:24])
    assert width_300 == pytest.approx(2 * width_150, abs=2)
    assert height_300 == pytest.approx(2 * height_150, abs=2)
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
    assert b"plotly" in plotly_figure_html(interactive).lower()
    assert safe_filename("../bad name") == ".._bad_name"


def test_raster_resolution_validation_rejects_oversized_or_invalid_exports() -> None:
    figure, _ = plt.subplots(figsize=(16.0, 12.0))

    with pytest.raises(InvalidPlotResolutionError, match="safety limit"):
        validate_raster_resolution(figure, 600)
    with pytest.raises(InvalidPlotResolutionError, match="cannot exceed 600 DPI"):
        validate_raster_resolution(figure, 601)
    with pytest.raises(InvalidPlotResolutionError, match="finite and positive"):
        validate_raster_resolution(figure, 0)

    plt.close(figure)


def test_weighted_histogram_uses_prepared_y_label() -> None:
    fig, ax = plt.subplots()
    values = np.array([1.0, 2.0])
    weights = np.array([2.0, 3.0])
    render_static_plot(
        fig,
        ax,
        "1D histogram",
        values,
        None,
        "pt",
        "sum weights (genWeight)",
        {"bins_x": 10, "log_y": False, "grid": False},
        title="weighted",
        histogram_1d=build_histogram_1d(values, 10, weights=weights, label="pt"),
        weights=weights,
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
        histogram_1d=build_histogram_1d(np.array([1.0, 2.0, 4.0]), 3, label="x"),
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


def test_explicit_axis_limits_support_complete_and_partial_bounds() -> None:
    figure, axes = plt.subplots()
    render_static_plot(
        figure,
        axes,
        "2D scatter",
        np.array([1.0, 2.0, 3.0]),
        np.array([10.0, 20.0, 30.0]),
        "x",
        "y",
        {
            "marker_size_2d": 3.0,
            "opacity_2d": 0.5,
            "log_x": False,
            "log_y": False,
            "x_min_text": "0.5",
            "x_max_text": "4.5",
            "y_min_text": "15",
            "y_max_text": "",
            "grid": False,
        },
        title="limited scatter",
    )

    assert axes.get_xlim() == pytest.approx((0.5, 4.5))
    assert axes.get_ylim()[0] == pytest.approx(15.0)
    assert axes.get_ylim()[1] > 15.0
    assert len(axes.collections[0].get_offsets()) == 3
    plt.close(figure)


@pytest.mark.parametrize(
    ("style", "message"),
    [
        ({"x_min_text": "not-a-number"}, "x-axis minimum must be a finite number"),
        ({"x_max_text": "inf"}, "x-axis maximum must be a finite number"),
        (
            {"x_min_text": "2", "x_max_text": "1"},
            "x-axis minimum must be smaller than x-axis maximum",
        ),
        (
            {"y_min_text": "3", "y_max_text": "3"},
            "y-axis minimum must be smaller than y-axis maximum",
        ),
        (
            {"log_x": True, "x_min_text": "0"},
            "x-axis minimum must be strictly positive",
        ),
        (
            {"log_y": True, "y_max_text": "-1"},
            "y-axis maximum must be strictly positive",
        ),
    ],
)
def test_invalid_explicit_axis_limits_are_rejected(style, message) -> None:
    figure, axes = plt.subplots()
    plot_style = {
        "marker_size_2d": 3.0,
        "opacity_2d": 0.5,
        "grid": False,
        **style,
    }

    with pytest.raises(InvalidAxisLimitError, match=message):
        render_static_plot(
            figure,
            axes,
            "2D scatter",
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            "x",
            "y",
            plot_style,
            title="invalid limits",
        )
    plt.close(figure)


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
            histogram_1d=build_histogram_1d(np.array([0.0, 1.0]), 2, label="x"),
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
            histogram_1d=build_histogram_1d(
                np.array([1.0, 2.0]),
                2,
                weights=np.array([-1.0, -1.0]),
                label="x",
            ),
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
        histogram_2d=build_histogram_2d(
            np.array([1.0, 2.0, 2.0]),
            np.array([1.0, 2.0, 2.0]),
            2,
            2,
            x_label="x",
            y_label="y",
        ),
    )
    assert histogram_2d_axes.get_xscale() == "linear"
    assert histogram_2d_axes.get_yscale() == "linear"
    plt.close(histogram_2d_figure)


def test_2d_histogram_log_color_rejects_no_positive_bins() -> None:
    figure, axes = plt.subplots()
    histogram = build_histogram_2d(
        np.array([], dtype=float),
        np.array([], dtype=float),
        2,
        2,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        x_label="x",
        y_label="y",
    )

    with pytest.raises(InvalidColorScaleError, match="positive bin contents"):
        render_static_plot(
            figure,
            axes,
            "2D histogram",
            np.array([], dtype=float),
            np.array([], dtype=float),
            "x",
            "y",
            {"bins_x": 2, "bins_y": 2, "log_z": True, "grid": False},
            title="empty log color histogram",
            histogram_2d=histogram,
        )
    plt.close(figure)


@pytest.mark.parametrize(
    "coordinates",
    [np.array([0.25]), np.array([0.25, 0.75])],
    ids=["one-positive-bin", "equal-positive-bins"],
)
def test_2d_histogram_log_color_remains_logarithmic_for_equal_bins(
    coordinates: np.ndarray,
) -> None:
    figure, axes = plt.subplots()
    histogram = build_histogram_2d(
        coordinates,
        coordinates,
        2,
        2,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        x_label="x",
        y_label="y",
    )

    render_static_plot(
        figure,
        axes,
        "2D histogram",
        coordinates,
        coordinates,
        "x",
        "y",
        {"bins_x": 2, "bins_y": 2, "log_z": True, "grid": False},
        title="equal-bin log color histogram",
        histogram_2d=histogram,
    )

    norm = axes.collections[0].norm
    assert isinstance(norm, LogNorm)
    assert 0 < norm.vmin <= 1.0
    assert norm.vmax > norm.vmin
    assert norm.vmax >= 1.0
    plt.close(figure)


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


def test_1d_reference_line_is_drawn_at_requested_value() -> None:
    figure, axes = plt.subplots()
    render_static_plot(
        figure,
        axes,
        "1D histogram",
        np.array([1.0, 2.0, 3.0]),
        None,
        "x",
        "entries",
        {"bins_x": 3, "log_y": False, "grid": False},
        title="reference",
        histogram_1d=build_histogram_1d(np.array([1.0, 2.0, 3.0]), 3, label="x"),
        reference_x=2.0,
        reference_label="threshold = 2",
    )
    assert axes.lines[-1].get_xdata()[0] == pytest.approx(2.0)
    assert axes.lines[-1].get_label() == "threshold = 2"
    assert to_rgba(axes.lines[-1].get_color()) == to_rgba("tab:green")
    plt.close(figure)


def test_1d_histogram_overlays_use_distinct_semantic_colors() -> None:
    from root_explorer.analysis.histograms import summarise_histogram_flow

    values = np.array([-2.0, 0.0, 1.5, 2.0, 2.5, 4.0, 8.0, 12.0])
    histogram = build_histogram_1d(values, 4, value_range=(1.0, 10.0), label="x")
    flow_summary = summarise_histogram_flow(histogram, values, weighted=False)
    fit_result = FitResult(
        model="Gaussian",
        success=True,
        parameters={"A": 3.0, "mu": 3.0, "sigma": 1.0},
        uncertainties={"A": None, "mu": None, "sigma": None},
        statistic=0.0,
        ndof=1,
        bins_used=4,
        bins_excluded=0,
        objective="Ordinary least squares",
        fit_range=(1.0, 10.0),
        message="Fit converged.",
    )
    figure, axes = plt.subplots()
    try:
        warnings = render_static_plot(
            figure,
            axes,
            "1D histogram",
            values,
            None,
            "x",
            "entries",
            {
                "show_kde": True,
                "kde_bandwidth": "scott",
                "log_y": False,
                "grid": False,
            },
            title="semantic colors",
            histogram_1d=histogram,
            reference_x=3.0,
            reference_label="threshold = 3",
            fit_result=fit_result,
            histogram_flow_summary=flow_summary,
            histogram_flow_mode="separate",
        )

        assert not warnings
        reference = next(line for line in axes.lines if line.get_label() == "threshold = 3")
        kde = next(line for line in axes.lines if line.get_label() == "KDE")
        fit = next(patch for patch in axes.patches if patch.get_label() == "Gaussian fit")
        flow = [patch for patch in axes.patches if patch.get_hatch() == "//"]
        assert len(flow) == 2
        colors = {
            "reference": to_rgba(reference.get_color()),
            "fit": to_rgba(fit.get_edgecolor()),
            "kde": to_rgba(kde.get_color()),
            "flow": to_rgba(flow[0].get_facecolor()),
        }
        assert colors == {
            "reference": to_rgba("tab:green"),
            "fit": to_rgba("tab:red"),
            "kde": to_rgba("tab:orange"),
            "flow": to_rgba("tab:purple"),
        }
        assert len(set(colors.values())) == len(colors)
        assert [text.get_text() for text in axes.get_legend().get_texts()] == [
            "threshold = 3",
            "Gaussian fit",
            "KDE",
        ]
    finally:
        plt.close(figure)


def test_scatter_color_limits_and_log_normalization() -> None:
    linear_figure, linear_axes = plt.subplots()
    render_static_plot(
        linear_figure,
        linear_axes,
        "2D scatter",
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        "x",
        "y",
        {
            "marker_size_2d": 3.0,
            "opacity_2d": 0.5,
            "color_min_text": "2",
            "color_max_text": "50",
            "grid": False,
        },
        title="linear color",
        color=np.array([1.0, 10.0, 100.0]),
    )
    assert linear_axes.collections[0].norm.vmin == pytest.approx(2.0)
    assert linear_axes.collections[0].norm.vmax == pytest.approx(50.0)
    plt.close(linear_figure)

    log_figure, log_axes = plt.subplots()
    render_static_plot(
        log_figure,
        log_axes,
        "2D scatter",
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        "x",
        "y",
        {
            "marker_size_2d": 3.0,
            "opacity_2d": 0.5,
            "log_color": True,
            "color_min_text": "1",
            "color_max_text": "100",
            "grid": False,
        },
        title="log color",
        color=np.array([1.0, 10.0, 100.0]),
    )
    assert isinstance(log_axes.collections[0].norm, LogNorm)
    assert log_axes.collections[0].norm.vmin == pytest.approx(1.0)
    assert log_axes.collections[0].norm.vmax == pytest.approx(100.0)
    plt.close(log_figure)


def test_scatter_rejects_invalid_color_limits() -> None:
    figure, axes = plt.subplots()
    with pytest.raises(InvalidColorScaleError, match="smaller"):
        render_static_plot(
            figure,
            axes,
            "2D scatter",
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            "x",
            "y",
            {
                "marker_size_2d": 3.0,
                "opacity_2d": 0.5,
                "color_min_text": "10",
                "color_max_text": "2",
                "grid": False,
            },
            title="bad color",
            color=np.array([1.0, 2.0]),
        )
    plt.close(figure)


def test_scatter_color_clipping_counts_weights_and_extension() -> None:
    result = calculate_color_clipping(
        np.array([0.0, 2.0, 5.0, 12.0]),
        {"color_min_text": "1", "color_max_text": "10"},
        np.array([0.5, 1.0, 2.0, 4.0]),
    )
    assert result.below_count == 1
    assert result.above_count == 1
    assert result.below_weight == pytest.approx(0.5)
    assert result.above_weight == pytest.approx(4.0)
    assert result.extend == "both"


def test_scatter_colorbar_marks_only_actual_clipping(monkeypatch) -> None:
    figure, axes = plt.subplots()
    captured: dict[str, object] = {}
    original_colorbar = figure.colorbar

    def capture_colorbar(*args, **kwargs):
        captured.update(kwargs)
        return original_colorbar(*args, **kwargs)

    monkeypatch.setattr(figure, "colorbar", capture_colorbar)
    render_static_plot(
        figure,
        axes,
        "2D scatter",
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        "x",
        "y",
        {
            "marker_size_2d": 3.0,
            "opacity_2d": 0.5,
            "color_min_text": "2",
            "color_max_text": "50",
            "grid": False,
        },
        title="clipped color",
        color=np.array([1.0, 10.0, 100.0]),
        colorbar_extend="both",
    )
    assert captured["extend"] == "both"
    plt.close(figure)


def test_2d_renderer_requires_prepared_histogram() -> None:
    figure, axes = plt.subplots()
    try:
        with pytest.raises(ValueError, match="requires a prepared histogram"):
            render_static_plot(
                figure,
                axes,
                "2D histogram",
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                "x",
                "y",
                {"log_z": False, "grid": False},
                title="missing prepared histogram",
            )
    finally:
        plt.close(figure)


@pytest.mark.parametrize("log_x", [False, True])
@pytest.mark.parametrize("log_y", [False, True])
def test_separate_flow_labels_remain_visible_on_log_axes(log_x, log_y) -> None:
    """UF/OF are categorical labels, so their y positions must not be data-space zero."""

    from root_explorer.analysis.histograms import summarise_histogram_flow

    values = np.array([-2.0, 0.0, 1.5, 2.0, 4.0, 8.0, 12.0])
    histogram = build_histogram_1d(values, 4, value_range=(1.0, 10.0))
    summary = summarise_histogram_flow(histogram, values, weighted=False)
    fig, ax = plt.subplots()
    try:
        warnings = render_static_plot(
            fig,
            ax,
            "1D histogram",
            values,
            None,
            "x",
            "entries",
            {"log_x": log_x, "log_y": log_y, "grid": False},
            title="flow labels",
            histogram_1d=histogram,
            histogram_flow_summary=summary,
            histogram_flow_mode="separate",
        )
        assert not warnings
        assert ax.get_xscale() == ("log" if log_x else "linear")
        assert ax.get_yscale() == ("log" if log_y else "linear")
        assert [label.get_text() for label in ax.texts] == ["UF", "OF"]
        fig.canvas.draw()
        for label in ax.texts:
            assert label.get_position()[1] == pytest.approx(0.02)
            # The y coordinate is in axes coordinates, independent of the data scale.
            bounds = label.get_window_extent(fig.canvas.get_renderer())
            assert ax.bbox.y0 < bounds.y0 < bounds.y1 < ax.bbox.y1
    finally:
        plt.close(fig)


@pytest.mark.parametrize("mode", ["ignore", "fold", "separate"])
def test_histogram_log_x_accepts_nonpositive_flow_values(mode) -> None:
    """Selected underflow data cannot invalidate the positive nominal x range."""

    from root_explorer.analysis.histograms import (
        histogram_for_flow_mode,
        summarise_histogram_flow,
    )

    values = np.array([-5.0, 0.0, 2.0, 3.0, 7.0, 11.0])
    raw = build_histogram_1d(values, 4, value_range=(1.0, 10.0))
    shown = histogram_for_flow_mode(raw, mode)
    summary = summarise_histogram_flow(raw, values, weighted=False)
    fig, ax = plt.subplots()
    try:
        render_static_plot(
            fig,
            ax,
            "1D histogram",
            values,
            None,
            "x",
            "entries",
            {"log_x": True, "log_y": False, "grid": False},
            title="positive nominal range",
            histogram_1d=shown,
            histogram_flow_summary=summary,
            histogram_flow_mode=mode,
        )
        assert ax.get_xscale() == "log"
        assert summary.underflow_entries == 2
        assert summary.overflow_entries == 1
        assert raw.values().sum() == 3
        assert shown.values().sum() == (len(values) if mode == "fold" else 3)
    finally:
        plt.close(fig)


def _colour_test_histogram():
    # Bin contents: (0, 0) = 3, (0, 1) = 2, (1, 0) = 1, (1, 1) = 5.
    return build_histogram_2d(
        np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5]),
        np.array([0.5, 0.5, 0.5, 1.5, 0.5, 1.5]),
        2,
        2,
        weights=np.array([1.0, 1.0, 1.0, 2.0, 1.0, 5.0]),
        x_label="x",
        y_label="y",
    )


def test_2d_histogram_colour_map_limits_and_label() -> None:
    fig, ax = plt.subplots()
    try:
        mesh = draw_histogram_2d(
            fig,
            ax,
            _colour_test_histogram(),
            log_scale=False,
            colorbar_label="custom",
            style={
                "colorscale": "Greys",
                "reverse_colorscale": True,
                "color_min_text": "1.5",
                "color_max_text": "2",
            },
        )
        assert mesh.cmap.name == "Greys_r"
        assert (mesh.norm.vmin, mesh.norm.vmax) == (1.5, 2.0)
        assert fig.axes[1].get_ylabel() == "custom"
        assert mesh.colorbar.extend == "both"
    finally:
        plt.close(fig)


def test_2d_histogram_partial_colour_limit_keeps_automatic_other_side() -> None:
    fig, ax = plt.subplots()
    try:
        mesh = draw_histogram_2d(
            fig,
            ax,
            _colour_test_histogram(),
            log_scale=True,
            colorbar_label="entries",
            style={"color_max_text": "2"},
        )
        assert isinstance(mesh.norm, LogNorm)
        assert (mesh.norm.vmin, mesh.norm.vmax) == (1.0, 2.0)
        assert mesh.colorbar.extend == "max"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("log_scale", "style", "message"),
    [
        (False, {"color_min_text": "abc"}, "finite number"),
        (False, {"color_min_text": "3", "color_max_text": "1"}, "smaller than color max"),
        (False, {"color_min_text": "10"}, "automatic range"),
        (True, {"color_min_text": "0"}, "positive color min"),
    ],
)
def test_2d_histogram_rejects_invalid_colour_limits(log_scale, style, message) -> None:
    fig, ax = plt.subplots()
    try:
        with pytest.raises(InvalidColorScaleError, match=message):
            draw_histogram_2d(
                fig,
                ax,
                _colour_test_histogram(),
                log_scale=log_scale,
                colorbar_label="entries",
                style=style,
            )
    finally:
        plt.close(fig)


def test_hidden_2d_bins_are_blank_and_excluded_from_automatic_range() -> None:
    fig, ax = plt.subplots()
    try:
        hidden = np.array([[False, False], [False, True]])  # hide the bin with content 5
        mesh = draw_histogram_2d(
            fig,
            ax,
            _colour_test_histogram(),
            log_scale=False,
            colorbar_label="entries",
            hidden_bins=hidden,
        )
        np.testing.assert_array_equal(np.ma.getmaskarray(mesh.get_array()).reshape(2, 2), hidden.T)
        assert (mesh.norm.vmin, mesh.norm.vmax) == (1.0, 3.0)
    finally:
        plt.close(fig)


def test_static_plot_colourbar_label_override_for_histogram_and_scatter() -> None:
    fig, ax = plt.subplots()
    render_static_plot(
        fig,
        ax,
        "2D histogram",
        np.array([0.5, 1.5]),
        np.array([0.5, 1.5]),
        "x",
        "y",
        {"log_z": False, "grid": False},
        title="histogram",
        histogram_2d=_colour_test_histogram(),
        colorbar_label="custom histogram label",
    )
    assert fig.axes[1].get_ylabel() == "custom histogram label"
    plt.close(fig)

    fig, ax = plt.subplots()
    render_static_plot(
        fig,
        ax,
        "2D scatter",
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        "x",
        "y",
        {"marker_size_2d": 3.0, "opacity_2d": 0.5, "grid": False},
        title="scatter",
        color=np.array([0.0, 1.0]),
        color_label="pt",
        colorbar_label="custom scatter label",
    )
    assert fig.axes[1].get_ylabel() == "custom scatter label"
    plt.close(fig)


@pytest.mark.parametrize(
    "label",
    ["", "plain text", "$p_T$ [GeV]", r"$\Delta R(\mu, j)$", r"$\eta$ and $\phi$", "cost $5"],
)
def test_valid_and_plain_labels_are_accepted(label: str) -> None:
    validate_plot_labels({"title": label, "custom_x": None})


@pytest.mark.parametrize(
    ("label", "reason"),
    [
        (r"$\pt$", r"Unknown symbol: \pt"),
        ("$p_{T$", "missing closing }"),
        ("$x^$", "syntax error"),
    ],
)
def test_invalid_math_labels_are_reported_readably(label: str, reason: str) -> None:
    with pytest.raises(InvalidLabelError) as excinfo:
        validate_plot_labels({"title": "fine", "custom_x": label})
    message = str(excinfo.value)
    assert f"`{label}`" in message
    assert f"({reason})" in message
    assert "`$p_T$ [GeV]`" in message


def test_valid_math_label_renders_without_error() -> None:
    fig, ax = plt.subplots()
    try:
        ax.set_xlabel(r"$\frac{1}{N}\,\mathrm{d}N/\mathrm{d}p_T$")
        validate_plot_labels({"custom_x": ax.get_xlabel()})
        matplotlib_figure_bytes(fig, "png", dpi=72)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("lower", "upper", "expected"), [(1.0, None, (1.0, 10.0)), (None, 8.0, (2.0, 8.0))]
)
def test_plotly_partial_colour_limits_resolve_both_endpoints(lower, upper, expected):
    figure = build_plotly_3d(
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        "x",
        "y",
        "z",
        color=[2.0, 10.0],
        color_min=lower,
        color_max=upper,
    )
    marker = figure.data[0].marker
    assert (marker.cmin, marker.cmax) == expected
    assert marker.cauto is False


def test_constant_positive_scatter_colour_can_use_log_scale():
    from root_explorer.plotting.plots import draw_colored_scatter_2d_log

    figure, axes = plt.subplots()
    try:
        draw_colored_scatter_2d_log(
            figure,
            axes,
            np.array([1.0, 2.0]),
            np.array([2.0, 3.0]),
            np.array([5.0, 5.0]),
            "x",
            "y",
            "colour",
            {"marker_size_2d": 2.0, "opacity_2d": 0.8},
            colorbar_extend="neither",
        )
        figure.canvas.draw()
        norm = axes.collections[0].norm
        assert isinstance(norm, LogNorm)
        assert 0 < norm.vmin < 5 < norm.vmax
    finally:
        plt.close(figure)
