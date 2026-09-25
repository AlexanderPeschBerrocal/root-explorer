from __future__ import annotations

from dataclasses import asdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from root_explorer.app.plotting import PlotRequest
from root_explorer.app.state import DataContext, GeometryState
from root_explorer.models import VariableSpec
from root_explorer.plotting.codegen import generate_static_plot_code
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


def _result_payload(value):
    return None if value is None else asdict(value)


def _assert_optional_array_equal(actual, expected) -> None:
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def _assert_prepared_parity(actual, expected) -> None:
    _assert_optional_array_equal(actual.x, expected.x)
    _assert_optional_array_equal(actual.y, expected.y)
    _assert_optional_array_equal(actual.weights, expected.weights)
    _assert_optional_array_equal(actual.color, expected.color)
    assert actual.x_label == expected.x_label
    assert actual.y_label == expected.y_label
    assert actual.color_label == expected.color_label
    assert actual.title == expected.title
    if expected.reference_x is None:
        assert actual.reference_x is None
    else:
        assert actual.reference_x == pytest.approx(expected.reference_x)
    assert actual.reference_label == expected.reference_label
    assert actual.max_points == expected.max_points
    assert _result_payload(actual.threshold_result) == _result_payload(expected.threshold_result)
    assert _result_payload(actual.quantile_result) == _result_payload(expected.quantile_result)
    assert _result_payload(actual.color_clipping_result) == _result_payload(
        expected.color_clipping_result
    )


def _execute_export(request: PlotRequest):
    source = generate_static_plot_code(request)
    compile(source, "root_explorer_export.py", "exec")

    assert "from root_explorer" not in source
    assert "import root_explorer" not in source
    assert "import streamlit" not in source

    namespace = {"__name__": "standalone_root_explorer_export"}
    exec(source, namespace)
    namespace["plt"].show = lambda: None

    namespace["main"]()
    main_figure = namespace["plt"].gcf()
    assert main_figure.axes
    plt.close(main_figure)

    canonical_spec = request.to_static_spec()
    canonical_prepared = prepare_static_plot(canonical_spec)
    generated_prepared = namespace["prepare_static_plot"](namespace["PLOT"])
    _assert_prepared_parity(generated_prepared, canonical_prepared)

    canonical_figure, canonical_warnings = render_prepared_static_plot(
        canonical_spec, canonical_prepared
    )
    generated_figure, generated_warnings = namespace["render_prepared_static_plot"](
        namespace["PLOT"], generated_prepared
    )
    assert generated_warnings == canonical_warnings
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
