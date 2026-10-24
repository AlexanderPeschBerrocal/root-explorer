from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from root_explorer.analysis.fitting import FitConfig, fit_histogram_1d
from root_explorer.analysis.variables import VARIABLE_EVALUATORS
from root_explorer.app.plotting import PlotRequest
from root_explorer.app.state import DataContext, GeometryState
from root_explorer.models import VariableSpec
from root_explorer.plotting.codegen import (
    build_static_plot_export_plan,
    fit_summary_lines,
    flow_summary_lines,
    generate_static_plot_code,
    normalisation_summary_lines,
    threshold_summary_lines,
)
from root_explorer.plotting.graph_codegen import generate_graph_plot_code
from root_explorer.plotting.graphs import GraphPlotSpec
from root_explorer.plotting.histogram_codegen import generate_histogram_plot_code
from root_explorer.plotting.histograms import HistogramPlotSpec
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
    fit_config: FitConfig | None = None,
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
        file_name="plot",
        threshold=threshold,
        quantile_percentile=quantile_percentile,
        fit_config=fit_config,
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
    return np.asarray(
        [patch.get_height() for patch in figure.axes[0].patches if hasattr(patch, "get_height")],
        dtype=float,
    )


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


def test_exported_2d_histogram_reproduces_colour_settings(root_path) -> None:
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
            "colorscale": "Magma",
            "reverse_colorscale": True,
            "color_min_text": "",
            "color_max_text": "2",
            "equal_aspect": False,
            "grid": False,
        },
        labels={
            "title": "",
            "custom_x": "",
            "custom_y": "",
            "custom_z": "",
            "custom_colorbar": "muon count",
        },
    )

    _, canonical_figure, generated_figure = _execute_export(request)
    try:
        for figure in (canonical_figure, generated_figure):
            mesh = figure.axes[0].collections[0]
            assert mesh.cmap.name == "magma_r"
            assert mesh.norm.vmax == 2.0
            assert figure.axes[1].get_ylabel() == "muon count"
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
    assert "def draw_histogram_2d(" in source
    assert "def build_histogram_2d(" in source
    assert "def build_histogram_1d(" not in source
    assert "import hist" in source
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
        file_name="plot",
    )

    with pytest.raises(ValueError, match="not available"):
        generate_static_plot_code(request)


def test_exported_weighted_log_2d_histogram_matches_pipeline(root_path) -> None:
    request = _request(
        root_path,
        plot_type="2D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        y_spec=VariableSpec(mode="branch", branch="Muon_eta"),
        weight_branch="genWeight",
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 7,
            "bins_y": 9,
            "log_x": False,
            "log_y": False,
            "log_z": True,
            "show_contours": False,
            "show_kde": False,
            "grid": False,
        },
    )
    prepared, canonical, generated = _execute_export(request)
    try:
        assert prepared.histogram_2d is not None
        assert prepared.histogram_2d.variances() is not None
        np.testing.assert_allclose(
            np.asarray(generated.axes[0].collections[0].get_array()),
            np.asarray(canonical.axes[0].collections[0].get_array()),
        )
        assert len(generated.axes) == len(canonical.axes) == 2
        assert generated.axes[1].get_ylabel() == canonical.axes[1].get_ylabel()
    finally:
        plt.close(canonical)
        plt.close(generated)


@pytest.mark.parametrize(
    "model", ["Gaussian", "Exponential", "Crystal Ball", "Double-sided Crystal Ball"]
)
def test_fit_enabled_export_executes_and_overlays(root_path, model) -> None:
    request = _request(
        root_path,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 25,
            "log_x": False,
            "log_y": False,
            "show_kde": False,
            "grid": False,
        },
        fit_config=FitConfig(model),
    )
    source = generate_static_plot_code(request)
    assert "def fit_histogram_1d(" in source
    assert "def draw_fit_overlay(" in source
    assert "import streamlit" not in source
    with _executed_generated_module(source) as module:
        module.plt.show = lambda: None
        module.main()
        generated = module.plt.gcf()
        canonical = prepare_static_plot(request.to_static_spec())
        expected = fit_histogram_1d(canonical.histogram_1d, request.fit_config)
        try:
            assert expected.success, expected.message
            generated_lines = [
                artist
                for artist in generated.axes[0].patches
                if artist.get_label() == f"{model} fit"
            ]
            # Matplotlib stairs are rendered as a StepPatch, not ordinary Line2D.
            assert generated_lines
            exported = module.fit_histogram_1d(canonical.histogram_1d, module.PLOT.fit_config)
            assert exported.success, exported.message
            for name in expected.parameters:
                assert exported.parameters[name] == pytest.approx(expected.parameters[name])
        finally:
            plt.close(generated)


def test_fit_disabled_export_remains_lightweight() -> None:
    request = _request(
        "events.root",
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        style={"fig_width": 8.0, "fig_height": 6.0, "bins_x": 20},
    )
    source = generate_static_plot_code(request)
    assert "def fit_histogram_1d(" not in source
    assert "def draw_fit_overlay(" not in source
    compile(source, "<plot-export>", "exec")


def test_weighted_fit_export_uses_requested_stored_variances(root_path) -> None:
    request = _request(
        root_path,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        weight_branch="genWeight",
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 20,
            "log_x": False,
            "log_y": False,
            "show_kde": False,
            "grid": False,
        },
        fit_config=FitConfig("Gaussian", use_variances=True),
    )
    source = generate_static_plot_code(request)
    with _executed_generated_module(source) as module:
        module.plt.show = lambda: None
        module.main()
        try:
            prepared = prepare_static_plot(request.to_static_spec())
            canonical = fit_histogram_1d(prepared.histogram_1d, request.fit_config)
            exported = module.fit_histogram_1d(prepared.histogram_1d, module.PLOT.fit_config)
            assert exported.objective == canonical.objective == "Variance-weighted least squares"
            assert exported.bins_excluded == canonical.bins_excluded
            assert exported.success == canonical.success
        finally:
            plt.close(module.plt.gcf())


def test_custom_fit_export_matches_canonical_backend(root_path):
    config = FitConfig(
        "Custom expression",
        expression="a*x + b",
        initial_values={"a": 1.0, "b": 2.0},
    )
    request = _request(
        root_path,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 24,
            "log_x": False,
            "log_y": False,
            "grid": False,
            "show_kde": False,
        },
        fit_config=config,
    )
    source = generate_static_plot_code(request)
    assert "import root_explorer" not in source
    assert "import streamlit" not in source
    assert "def discover_fit_parameters(" in source
    with _executed_generated_module(source) as module:
        module.plt.show = lambda: None
        module.main()
        figure = module.plt.gcf()
        try:
            prepared = prepare_static_plot(request.to_static_spec())
            canonical = fit_histogram_1d(prepared.histogram_1d, config)
            exported = module.fit_histogram_1d(prepared.histogram_1d, module.PLOT.fit_config)
            assert canonical.success, canonical.message
            assert exported.success, exported.message
            for name, value in canonical.parameters.items():
                assert exported.parameters[name] == pytest.approx(value, rel=1e-5)
            assert any(p.get_label() == "Custom expression fit" for p in figure.axes[0].patches)
        finally:
            plt.close(figure)


@pytest.mark.parametrize("mode", ["ignore", "fold", "separate"])
def test_exported_manual_range_flow_matches_canonical(root_path, mode):
    request = _request(
        root_path,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        weight_branch="genWeight",
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 12,
            "log_x": False,
            "log_y": False,
            "grid": False,
            "manual_histogram_limits": True,
            "hist_x_min_text": "15",
            "hist_x_max_text": "35",
            "hist_flow_mode": mode,
        },
    )
    prepared, canonical, generated = _execute_export(request)
    try:
        np.testing.assert_allclose(_histogram_heights(canonical), _histogram_heights(generated))
        assert prepared.histogram_flow_mode == mode
        assert len(canonical.axes[0].patches) == len(generated.axes[0].patches)
    finally:
        plt.close(canonical)
        plt.close(generated)


def test_log_y_histogram_with_only_separate_flow_matches_export(root_path):
    request = _request(
        root_path,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 5,
            "log_y": True,
            "manual_histogram_limits": True,
            "hist_x_min_text": "10000",
            "hist_x_max_text": "20000",
            "hist_flow_mode": "separate",
        },
    )
    prepared, canonical, exported = _execute_export(request)
    try:
        assert prepared.histogram_1d.values().sum() == 0
        assert prepared.histogram_flow_summary.underflow_entries > 0
        for figure in (canonical, exported):
            assert figure.axes[0].get_yscale() == "log"
            figure.canvas.draw()
        np.testing.assert_allclose(_histogram_heights(canonical), _histogram_heights(exported))
    finally:
        plt.close(canonical)
        plt.close(exported)


@pytest.mark.parametrize("mode", ["ignore", "fold", "separate"])
@pytest.mark.parametrize("log_y", [False, True])
def test_exported_positive_log_x_range_ignores_nonpositive_flow_values(root_path, mode, log_y):
    """Standalone exports and the app accept the same log-x flow configurations."""

    request = _request(
        root_path,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_eta"),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 8,
            "log_x": True,
            "log_y": log_y,
            "grid": False,
            "manual_histogram_limits": True,
            "hist_x_min_text": "0.05",
            "hist_x_max_text": "2.5",
            "hist_flow_mode": mode,
        },
    )
    prepared, canonical, exported = _execute_export(request)
    try:
        assert np.any(prepared.x <= 0)
        assert prepared.histogram_flow_summary is not None
        assert prepared.histogram_flow_summary.underflow_entries > 0
        assert canonical.axes[0].get_xscale() == exported.axes[0].get_xscale() == "log"
        expected_yscale = "log" if log_y else "linear"
        assert canonical.axes[0].get_yscale() == exported.axes[0].get_yscale() == expected_yscale
        np.testing.assert_allclose(_histogram_heights(canonical), _histogram_heights(exported))
        if mode == "separate":
            for figure in (canonical, exported):
                figure.canvas.draw()
                ax = figure.axes[0]
                assert [label.get_text() for label in ax.texts] == ["UF", "OF"]
                for label in ax.texts:
                    bbox = label.get_window_extent(figure.canvas.get_renderer())
                    assert ax.bbox.y0 < bbox.y0 < bbox.y1 < ax.bbox.y1
    finally:
        plt.close(canonical)
        plt.close(exported)


@pytest.mark.parametrize("mode", ["ignore", "fold"])
def test_exported_unit_area_density_matches_application(root_path, mode: str) -> None:
    request = _request(
        root_path,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 20,
            "show_kde": False,
            "log_x": False,
            "log_y": False,
            "grid": True,
            "manual_histogram_limits": True,
            "hist_x_min_text": "10",
            "hist_x_max_text": "80",
            "hist_flow_mode": mode,
            "normalise_unit_area": True,
        },
    )
    prepared, original_figure, exported_figure = _execute_export(request)
    try:
        assert prepared.normalisation_result is not None
        assert prepared.normalisation_result.valid
        np.testing.assert_allclose(
            _histogram_heights(exported_figure),
            _histogram_heights(original_figure),
        )
        assert exported_figure.axes[0].get_ylabel() == "Normalised density"
        widths = np.diff(prepared.histogram_1d.axes[0].edges)
        assert np.dot(_histogram_heights(exported_figure), widths) == pytest.approx(1.0)
    finally:
        plt.close(original_figure)
        plt.close(exported_figure)


def test_exported_density_fit_predictions_match_application(root_path) -> None:
    request = _request(
        root_path,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 20,
            "show_kde": False,
            "log_x": False,
            "log_y": False,
            "grid": False,
            "normalise_unit_area": True,
        },
        fit_config=FitConfig(model="Exponential"),
    )
    prepared, original_figure, exported_figure = _execute_export(request)
    # _execute_export renders its reference without a fit by default.
    # Re-render it with the same fit the standalone script computes.
    plt.close(original_figure)
    fit_result = fit_histogram_1d(prepared.display_histogram_1d, request.fit_config)
    assert fit_result.success
    original_figure, _ = render_prepared_static_plot(
        request.to_static_spec(), prepared, fit_result=fit_result
    )
    try:
        assert prepared.normalisation_result is not None
        np.testing.assert_allclose(
            _histogram_heights(exported_figure), _histogram_heights(original_figure)
        )
        original_fit = [
            p for p in original_figure.axes[0].patches if p.get_label() == "Exponential fit"
        ]
        exported_fit = [
            p for p in exported_figure.axes[0].patches if p.get_label() == "Exponential fit"
        ]
        assert len(original_fit) == len(exported_fit) == 1
        np.testing.assert_allclose(
            original_fit[0].get_data().values,
            exported_fit[0].get_data().values,
        )
    finally:
        plt.close(original_figure)
        plt.close(exported_figure)


def _full_histogram_request(root_path, **changes: Any) -> PlotRequest:
    """A 1D histogram export with every result the application reports."""

    options: dict[str, Any] = {
        "plot_type": "1D histogram",
        "x_spec": VariableSpec(mode="branch", branch="Muon_pt"),
        "style": {
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 20,
            "log_x": False,
            "log_y": False,
            "show_kde": False,
            "grid": False,
            "manual_histogram_limits": True,
            "hist_x_min_text": "10",
            "hist_x_max_text": "80",
            "hist_flow_mode": "fold",
            "normalise_unit_area": True,
        },
        "threshold": 30.0,
        "fit_config": FitConfig(
            "Gaussian",
            # Unbounded limits used to make the whole export unavailable.
            bounds={"sigma": (0.0, float("inf")), "mu": (float("-inf"), float("inf"))},
        ),
    }
    options.update(changes)
    return _request(root_path, **options)


def _run_export_script(source: str, tmp_path) -> str:
    script = tmp_path / "exported_plot.py"
    script.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-I", str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=tmp_path,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_exported_script_prints_the_results_the_application_shows(root_path, tmp_path) -> None:
    request = _full_histogram_request(root_path)
    source = generate_static_plot_code(request)
    assert "float('inf')" in source
    assert "float('-inf')" in source

    output = _run_export_script(source, tmp_path)

    prepared = prepare_static_plot(request.to_static_spec())
    assert prepared.threshold_result is not None
    assert prepared.histogram_flow_summary is not None
    assert prepared.normalisation_result is not None
    assert prepared.normalisation_result.valid
    fit_result = fit_histogram_1d(prepared.display_histogram_1d, request.fit_config)
    assert fit_result.success, fit_result.message
    bounds = tuple(float(edge) for edge in prepared.histogram_1d.axes[0].edges[[0, -1]])
    expected_sections = {
        "Unit-area normalisation": normalisation_summary_lines(
            prepared.normalisation_result, bounds, prepared.histogram_flow_mode
        ),
        "Histogram flow": flow_summary_lines(
            prepared.histogram_flow_summary, prepared.histogram_flow_mode
        ),
        "Threshold calculator": threshold_summary_lines(prepared.threshold_result),
        "Histogram fit": fit_summary_lines(fit_result),
    }
    for title, lines in expected_sections.items():
        block = "\n".join([title, *(f"  {line}" for line in lines)])
        assert block in output
    # The printed numbers are the application's, not only their labels.
    denominator = prepared.normalisation_result.denominator
    assert f"Normalisation sum of weights: {denominator:.6g}" in output
    for name, value in fit_result.parameters.items():
        uncertainty = fit_result.uncertainties[name]
        assert uncertainty is not None
        assert f"= {value:.6g} +/- {uncertainty:.6g}" in output
    assert f"fitted bins: {fit_result.bins_used}" in output
    threshold = prepared.threshold_result
    assert f"x <= 30: count {threshold.below_count}" in output
    output.encode("ascii")


def test_exported_script_prints_quantile_and_failed_fit(root_path, tmp_path) -> None:
    request = _full_histogram_request(
        root_path,
        threshold=None,
        quantile_percentile=50.0,
        # Too few bins for three free parameters: the fit fails.
        style={"fig_width": 8.0, "fig_height": 6.0, "bins_x": 2, "grid": False},
    )
    output = _run_export_script(generate_static_plot_code(request), tmp_path)

    prepared = prepare_static_plot(request.to_static_spec())
    assert prepared.quantile_result is not None
    assert prepared.quantile_result.value is not None
    assert f"  50% quantile: {prepared.quantile_result.value:.6g}" in output
    fit_result = fit_histogram_1d(prepared.display_histogram_1d, request.fit_config)
    assert not fit_result.success
    assert f"  Fit failed: {fit_result.message.strip()}" in output
    # Without manual histogram limits there is no flow and no normalisation.
    assert "Histogram flow" not in output
    assert "Unit-area normalisation" not in output


def test_weighted_separate_flow_export_reports_the_application_results(root_path, tmp_path) -> None:
    style = {**_full_histogram_request(root_path).style, "hist_flow_mode": "separate"}
    request = _full_histogram_request(
        root_path, weight_branch="genWeight", style=style, fit_config=None
    )
    output = _run_export_script(generate_static_plot_code(request), tmp_path)

    prepared = prepare_static_plot(request.to_static_spec())
    assert prepared.normalisation_result is not None
    assert prepared.normalisation_result.error is not None
    assert f"  {prepared.normalisation_result.error}" in output
    assert "Unit-area normalisation unavailable; showing unnormalised bin yields." in output
    assert prepared.histogram_flow_summary is not None
    assert prepared.histogram_flow_summary.weighted
    flow = "\n".join(
        f"  {line}" for line in flow_summary_lines(prepared.histogram_flow_summary, "separate")
    )
    assert flow in output
    assert "sum of squared weights" in flow
    assert prepared.threshold_result is not None
    assert prepared.threshold_result.weighted
    threshold = "\n".join(
        f"  {line}" for line in threshold_summary_lines(prepared.threshold_result)
    )
    assert threshold in output
    assert "weighted fraction" in threshold


def _exportable_plot_sources() -> dict[str, str]:
    """Generated scripts covering each plot type, feature and variable mode."""

    base_style = {
        "fig_width": 8.0,
        "fig_height": 6.0,
        "bins_x": 20,
        "bins_y": 20,
        "grid": False,
    }
    branch = VariableSpec(mode="branch", branch="Muon_pt")
    eta = VariableSpec(mode="branch", branch="Muon_eta")
    requests = {
        "histogram_plain": _request(
            "events.root", plot_type="1D histogram", x_spec=branch, style=base_style
        ),
        "histogram_all_features": _full_histogram_request(
            "events.root",
            cut_specs=[eta],
            cut_bounds=[(-2.0, 2.0)],
            style={
                **_full_histogram_request("events.root").style,
                "show_kde": True,
                "log_x": True,
                "log_y": True,
            },
        ),
        "histogram_quantile": _request(
            "events.root",
            plot_type="1D histogram",
            x_spec=branch,
            style=base_style,
            quantile_percentile=90.0,
            fit_config=FitConfig("Custom expression", expression="a*x + b"),
        ),
        "histogram_2d_all_features": _request(
            "events.root",
            plot_type="2D histogram",
            x_spec=branch,
            y_spec=eta,
            weight_branch="genWeight",
            style={
                **base_style,
                "show_kde": True,
                "show_contours": True,
                "log_x": True,
                "log_y": True,
                "log_z": True,
            },
        ),
        "scatter_plain": _request(
            "events.root",
            plot_type="2D scatter",
            x_spec=branch,
            y_spec=eta,
            style={**base_style, "log_x": True, "log_y": True},
        ),
    }
    for log_color in (False, True):
        requests[f"scatter_colour_log_{log_color}"] = _request(
            "events.root",
            plot_type="2D scatter",
            x_spec=branch,
            y_spec=eta,
            color_spec=VariableSpec(mode="branch", branch="Muon_energy"),
            weight_branch="genWeight",
            style={**base_style, "log_color": log_color},
        )
    for mode in VARIABLE_EVALUATORS:
        spec = VariableSpec(
            mode=mode,
            branch="Muon_pt",
            branch_a="Muon_px",
            branch_b="Muon_py",
            branch_c="Muon_pz",
            branch_d="Muon_mass",
        )
        requests[f"variable_{mode}"] = _request(
            "events.root", plot_type="1D histogram", x_spec=spec, style=base_style
        )

    sources = {name: generate_static_plot_code(request) for name, request in requests.items()}
    for name, fit_config in (("persisted_plain", None), ("persisted_fit", FitConfig("Gaussian"))):
        sources[name] = generate_histogram_plot_code(
            HistogramPlotSpec("file.root", "histogram;1", base_style, {}, fit_config=fit_config)
        )
    sources["graph"] = generate_graph_plot_code(
        GraphPlotSpec("file.root", "graph;1", {**base_style, "graph_mode": "Markers"}, {})
    )
    return sources


def test_generated_scripts_have_no_undefined_names(tmp_path) -> None:
    pytest.importorskip("ruff")
    for name, source in _exportable_plot_sources().items():
        (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--no-cache",
            "--select",
            "F821,F822,F823",
            "--output-format",
            "concise",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
