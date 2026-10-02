from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from root_explorer.app.plotting import PlotRequest
from root_explorer.app.state import DataContext, GeometryState
from root_explorer.models import VariableSpec
from root_explorer.plotting.codegen import build_static_plot_export_plan, generate_static_plot_code
from root_explorer.plotting.pipeline import prepare_static_plot, render_prepared_static_plot


def _request(
    root_path,
    *,
    plot_type: str,
    x_spec: VariableSpec,
    y_spec: VariableSpec | None = None,
    color_spec: VariableSpec | None = None,
    weight_branch: str | None = None,
    cut_specs: list[VariableSpec] | None = None,
    cut_bounds: list[tuple[float | None, float | None]] | None = None,
    style: dict[str, Any],
    labels: dict[str, Any] | None = None,
    threshold: float | None = None,
    quantile_percentile: float | None = None,
) -> PlotRequest:
    return PlotRequest(
        context=DataContext(str(root_path), "Events", 12, 0, 12, [], []),
        plot_type=plot_type,
        x_spec=x_spec,
        y_spec=y_spec,
        z_spec=None,
        color_spec=color_spec,
        weight_branch=weight_branch,
        cut_specs=cut_specs or [],
        cut_bounds=cut_bounds or [],
        style=style,
        labels=labels
        or {
            "title": "",
            "custom_x": "",
            "custom_y": "",
            "custom_z": "",
        },
        geometry=GeometryState(scene=None, options={}),
        event_index=None,
        preview=False,
        save_name="plot",
        threshold=threshold,
        quantile_percentile=quantile_percentile,
    )


@contextmanager
def _executed_generated_module(source: str) -> Generator[ModuleType, None, None]:
    """Execute generated code in a real, temporarily registered module.

    Executing into only a dictionary with a synthetic __name__ does not create a
    registered Python module. Generated exports use postponed annotations, and
    dataclasses may resolve those annotations through sys.modules[cls.__module__]
    while defining vendored classes. Registering a ModuleType mirrors normal Python
    script/import execution, while a non-__main__ name prevents main() from running
    eagerly so the parity tests can inspect the generated helpers first.
    """

    module_name = "standalone_root_explorer_export"
    module = ModuleType(module_name)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(source, "root_explorer_export.py", "exec"), module.__dict__)
        yield module
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


def _execute_export(request: PlotRequest):
    source = generate_static_plot_code(request)

    assert "from root_explorer" not in source
    assert "import root_explorer" not in source
    assert "import streamlit" not in source

    with _executed_generated_module(source) as module:
        namespace = module.__dict__
        namespace["plt"].show = lambda: None

        namespace["main"]()
        main_figure = namespace["plt"].gcf()
        assert main_figure.axes
        plt.close(main_figure)

        canonical_spec = request.to_static_spec()
        canonical_prepared = prepare_static_plot(canonical_spec)
        canonical_figure, _ = render_prepared_static_plot(canonical_spec, canonical_prepared)
        namespace["main"]()
        generated_figure = namespace["plt"].gcf()
        assert generated_figure.axes
        return canonical_prepared, canonical_figure, generated_figure


def _histogram_heights(figure) -> np.ndarray:
    return np.asarray([patch.get_height() for patch in figure.axes[0].patches], dtype=float)


def test_exported_basic_histogram_matches_canonical_pipeline(root_path) -> None:
    request = _request(
        root_path,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 20,
            "log_x": False,
            "log_y": False,
            "x_min_text": "10",
            "x_max_text": "80",
            "y_min_text": "0",
            "y_max_text": "20",
            "show_kde": False,
            "equal_aspect": False,
            "grid": True,
        },
        labels={
            "title": "Muon transverse momentum",
            "custom_x": "pT [GeV]",
            "custom_y": "",
            "custom_z": "",
        },
        quantile_percentile=50.0,
    )

    prepared, canonical_figure, generated_figure = _execute_export(request)
    try:
        np.testing.assert_allclose(
            _histogram_heights(generated_figure),
            _histogram_heights(canonical_figure),
        )
        assert generated_figure.axes[0].get_xlabel() == canonical_figure.axes[0].get_xlabel()
        assert generated_figure.axes[0].get_ylabel() == canonical_figure.axes[0].get_ylabel()
        assert generated_figure.axes[0].get_title() == canonical_figure.axes[0].get_title()
        assert generated_figure.dpi == canonical_figure.dpi == 200
        assert generated_figure.axes[0].get_xlim() == pytest.approx((10.0, 80.0))
        assert generated_figure.axes[0].get_ylim() == pytest.approx((0.0, 20.0))
        assert prepared.quantile_result is not None
        assert prepared.reference_x == pytest.approx(prepared.quantile_result.value)
        assert len(generated_figure.axes[0].lines) == len(canonical_figure.axes[0].lines) == 1
        np.testing.assert_allclose(
            generated_figure.axes[0].lines[0].get_xdata(),
            canonical_figure.axes[0].lines[0].get_xdata(),
        )
    finally:
        plt.close(canonical_figure)
        plt.close(generated_figure)


def test_exported_weighted_cut_histogram_matches_canonical_pipeline(root_path) -> None:
    request = _request(
        root_path,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        weight_branch="genWeight",
        cut_specs=[VariableSpec(mode="branch", branch="Muon_eta")],
        cut_bounds=[(-2.1, 2.1)],
        style={
            "fig_width": 11.0,
            "fig_height": 5.0,
            "bins_x": 37,
            "log_x": True,
            "log_y": False,
            "show_kde": False,
            "equal_aspect": False,
            "grid": False,
        },
        labels={
            "title": "Weighted muon pT",
            "custom_x": "Muon transverse momentum [GeV]",
            "custom_y": "Weighted candidates",
            "custom_z": "",
        },
    )

    prepared, canonical_figure, generated_figure = _execute_export(request)
    try:
        assert prepared.weights is not None
        np.testing.assert_allclose(
            _histogram_heights(generated_figure),
            _histogram_heights(canonical_figure),
        )
        assert np.sum(_histogram_heights(generated_figure)) == pytest.approx(
            np.sum(prepared.weights)
        )
        assert generated_figure.axes[0].get_xscale() == "log"
        assert generated_figure.axes[0].get_ylabel() == "Weighted candidates"
    finally:
        plt.close(canonical_figure)
        plt.close(generated_figure)


def test_exported_derived_variable_matches_canonical_pipeline(root_path) -> None:
    request = _request(
        root_path,
        plot_type="1D histogram",
        x_spec=VariableSpec(
            mode="magnitude_3d",
            branch_a="Muon_px",
            branch_b="Muon_py",
            branch_c="Muon_pz",
        ),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 24,
            "log_x": False,
            "log_y": False,
            "show_kde": False,
            "equal_aspect": False,
            "grid": False,
        },
    )

    prepared, canonical_figure, generated_figure = _execute_export(request)
    try:
        assert prepared.x_label == "mag(Muon_px,Muon_py,Muon_pz)"
        assert len(prepared.x) > 0
        np.testing.assert_allclose(
            _histogram_heights(generated_figure),
            _histogram_heights(canonical_figure),
        )
    finally:
        plt.close(canonical_figure)
        plt.close(generated_figure)


def test_exported_2d_histogram_with_contours_and_kde_matches_pipeline(root_path) -> None:
    request = _request(
        root_path,
        plot_type="2D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        y_spec=VariableSpec(mode="branch", branch="Muon_eta"),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 5,
            "bins_y": 5,
            "log_x": False,
            "log_y": False,
            "log_z": False,
            "show_contours": True,
            "contour_levels": [68.0, 95.0, 99.0],
            "show_kde": True,
            "kde_bandwidth": "custom",
            "kde_custom_factor": 0.7,
            "equal_aspect": False,
            "grid": False,
        },
    )

    _, canonical_figure, generated_figure = _execute_export(request)
    try:
        canonical_axes = canonical_figure.axes[0]
        generated_axes = generated_figure.axes[0]
        assert len(generated_axes.collections) == len(canonical_axes.collections)
        assert len(generated_axes.collections) > 1
        np.testing.assert_allclose(
            np.asarray(generated_axes.collections[0].get_array()),
            np.asarray(canonical_axes.collections[0].get_array()),
        )
        assert len(generated_figure.axes) == len(canonical_figure.axes) == 2
        assert generated_figure.axes[1].get_ylabel() == canonical_figure.axes[1].get_ylabel()
    finally:
        plt.close(canonical_figure)
        plt.close(generated_figure)


def test_exported_coloured_scatter_matches_canonical_pipeline(root_path) -> None:
    request = _request(
        root_path,
        plot_type="2D scatter",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        y_spec=VariableSpec(mode="branch", branch="Muon_eta"),
        color_spec=VariableSpec(mode="branch", branch="Muon_energy"),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "marker_size_2d": 4.0,
            "opacity_2d": 0.6,
            "max_points": 10,
            "colorscale": "Greys",
            "reverse_colorscale": True,
            "log_color": True,
            "color_min_text": "20",
            "color_max_text": "120",
            "log_x": False,
            "log_y": False,
            "equal_aspect": False,
            "grid": False,
        },
    )

    prepared, canonical_figure, generated_figure = _execute_export(request)
    try:
        assert prepared.color is not None
        assert prepared.color_clipping_result is not None
        canonical_scatter = canonical_figure.axes[0].collections[0]
        generated_scatter = generated_figure.axes[0].collections[0]
        np.testing.assert_allclose(
            np.asarray(generated_scatter.get_offsets()),
            np.asarray(canonical_scatter.get_offsets()),
        )
        np.testing.assert_allclose(
            np.asarray(generated_scatter.get_array()),
            np.asarray(canonical_scatter.get_array()),
        )
        assert generated_scatter.cmap.name == canonical_scatter.cmap.name == "Greys_r"
        assert generated_scatter.norm.vmin == pytest.approx(canonical_scatter.norm.vmin)
        assert generated_scatter.norm.vmax == pytest.approx(canonical_scatter.norm.vmax)
        assert len(generated_figure.axes) == len(canonical_figure.axes) == 2
    finally:
        plt.close(canonical_figure)
        plt.close(generated_figure)


def test_basic_histogram_export_omits_unselected_features() -> None:
    request = _request(
        "events.root",
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 20,
            "log_x": False,
            "log_y": False,
            "show_kde": False,
            "equal_aspect": False,
            "grid": False,
        },
    )

    source = generate_static_plot_code(request)

    compile(source, "basic_histogram_export.py", "exec")
    assert "def evaluate_branch_variable(" in source
    assert "def build_histogram_1d(" in source
    assert "import hist" in source
    assert "def evaluate_variable(" not in source
    assert "def evaluate_magnitude_3d_variable(" not in source
    assert "def draw_kde_1d(" not in source
    assert "def draw_kde_2d(" not in source
    assert "def draw_highest_density_contours(" not in source
    assert "def calculate_quantile_result(" not in source
    assert "def calculate_threshold_result(" not in source
    assert "def histogram_weights_for_values(" not in source
    assert "def draw_colored_scatter_2d_" not in source
    assert "gaussian_kde" not in source


def test_plain_2d_histogram_export_omits_optional_overlays() -> None:
    request = _request(
        "events.root",
        plot_type="2D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        y_spec=VariableSpec(mode="branch", branch="Muon_eta"),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 20,
            "bins_y": 20,
            "log_x": False,
            "log_y": False,
            "log_z": False,
            "show_kde": False,
            "show_contours": False,
            "equal_aspect": False,
            "grid": False,
        },
    )

    source = generate_static_plot_code(request)

    compile(source, "plain_2d_histogram_export.py", "exec")
    assert "def draw_histogram_2d_linear(" in source
    assert "def build_histogram_1d(" not in source
    assert "import hist" not in source
    assert "def draw_histogram_2d_log(" not in source
    assert "def draw_kde_2d(" not in source
    assert "def draw_highest_density_contours(" not in source
    assert "def draw_colored_scatter_2d_" not in source
    assert "gaussian_kde" not in source


def test_export_plan_adds_only_selected_overlays() -> None:
    base_style = {
        "fig_width": 8.0,
        "fig_height": 6.0,
        "bins_x": 20,
        "bins_y": 20,
        "log_x": False,
        "log_y": False,
        "log_z": False,
        "equal_aspect": False,
        "grid": False,
    }
    kde_request = _request(
        "events.root",
        plot_type="2D histogram",
        x_spec=VariableSpec(mode="branch", branch="x"),
        y_spec=VariableSpec(mode="branch", branch="y"),
        style={**base_style, "show_kde": True, "show_contours": False},
    )
    contour_request = _request(
        "events.root",
        plot_type="2D histogram",
        x_spec=VariableSpec(mode="branch", branch="x"),
        y_spec=VariableSpec(mode="branch", branch="y"),
        style={**base_style, "show_kde": False, "show_contours": True},
    )

    kde_plan = build_static_plot_export_plan(kde_request.to_static_spec())
    contour_plan = build_static_plot_export_plan(contour_request.to_static_spec())
    kde_source = generate_static_plot_code(kde_request)
    contour_source = generate_static_plot_code(contour_request)

    assert kde_plan.show_kde is True
    assert kde_plan.show_contours is False
    assert contour_plan.show_kde is False
    assert contour_plan.show_contours is True
    assert "from scipy.stats import gaussian_kde" in kde_source
    assert "def draw_kde_2d_overlay(" in kde_source
    assert "def draw_histogram_contours(" not in kde_source
    assert "def draw_histogram_contours(" in contour_source
    assert "from scipy.stats import gaussian_kde" not in contour_source
    assert "def draw_kde_2d_overlay(" not in contour_source


def test_coloured_scatter_export_excludes_histogram_features() -> None:
    request = _request(
        "events.root",
        plot_type="2D scatter",
        x_spec=VariableSpec(mode="branch", branch="x"),
        y_spec=VariableSpec(mode="branch", branch="y"),
        color_spec=VariableSpec(mode="branch", branch="color"),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "marker_size_2d": 2.0,
            "opacity_2d": 0.5,
            "max_points": 100,
            "colorscale": "Viridis",
            "reverse_colorscale": False,
            "log_color": False,
            "log_x": False,
            "log_y": False,
            "equal_aspect": False,
            "grid": False,
        },
    )

    source = generate_static_plot_code(request)

    assert "def draw_colored_scatter_2d_linear(" in source
    assert "def calculate_color_clipping(" in source
    assert "def build_histogram_1d(" not in source
    assert "import hist" not in source
    assert "def draw_histogram_1d(" not in source
    assert "def draw_histogram_2d_" not in source
    assert "def draw_kde_" not in source
    assert "def draw_highest_density_contours(" not in source


def test_export_configuration_allows_non_structural_edits_and_rejects_mode_changes(
    root_path,
) -> None:
    request = _request(
        root_path,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 20,
            "log_x": False,
            "log_y": False,
            "show_kde": False,
            "equal_aspect": False,
            "grid": False,
        },
    )

    with _executed_generated_module(generate_static_plot_code(request)) as module:
        namespace = module.__dict__
        namespace["plt"].show = lambda: None
        namespace["PLOT"] = replace(
            namespace["PLOT"],
            labels={"title": "Edited", "custom_x": "", "custom_y": ""},
            style={
                **namespace["PLOT"].style,
                "show_contours": True,
                "log_z": True,
                "log_color": True,
            },
        )
        namespace["main"]()
        plt.close(namespace["plt"].gcf())

        namespace["PLOT"] = replace(
            namespace["PLOT"],
            x_spec=replace(namespace["PLOT"].x_spec, mode="sum"),
        )
        with pytest.raises(ValueError, match="variable modes"):
            namespace["main"]()


def test_export_rejects_interactive_3d_request() -> None:
    request = PlotRequest(
        context=DataContext("events.root", "Events", 1, 0, 1, [], []),
        plot_type="3D scatter interactive",
        x_spec=VariableSpec(mode="branch", branch="x"),
        y_spec=VariableSpec(mode="branch", branch="y"),
        z_spec=VariableSpec(mode="branch", branch="z"),
        color_spec=None,
        weight_branch=None,
        cut_specs=[],
        cut_bounds=[],
        style={},
        labels={"title": "", "custom_x": "", "custom_y": "", "custom_z": ""},
        geometry=GeometryState(scene=None, options={}),
        event_index=None,
        preview=False,
        save_name="plot",
    )

    with pytest.raises(ValueError, match="not available"):
        generate_static_plot_code(request)
