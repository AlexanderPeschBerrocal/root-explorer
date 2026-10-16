from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from root_explorer.models import VariableSpec
from root_explorer.plotting.pipeline import (
    StaticPlotSpec,
    prepare_static_plot,
    render_prepared_static_plot,
)


def test_static_pipeline_prepares_weighted_cut_histogram(root_path) -> None:
    spec = StaticPlotSpec(
        root_path=str(root_path),
        tree_name="Events",
        entry_start=2,
        entry_stop=10,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        y_spec=None,
        color_spec=None,
        weight_branch="genWeight",
        cut_specs=(VariableSpec(mode="branch", branch="Muon_eta"),),
        cut_bounds=((-2.1, 2.1),),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 20,
            "log_x": False,
            "log_y": False,
            "grid": False,
        },
        labels={
            "title": "Weighted muons",
            "custom_x": "Muon pT",
            "custom_y": "",
        },
        threshold=25.0,
    )

    prepared = prepare_static_plot(spec)

    assert len(prepared.x) == len(prepared.weights)
    assert prepared.x_label == "Muon pT"
    assert prepared.y_label == "sum weights (genWeight)"
    assert prepared.histogram_1d is not None
    assert prepared.histogram_1d.axes[0].label == "Muon pT"
    assert prepared.histogram_1d.values().sum() == pytest.approx(prepared.weights.sum())
    assert prepared.histogram_1d.variances() is not None
    assert prepared.reference_x == 25.0
    assert prepared.threshold_result is not None

    figure, warnings = render_prepared_static_plot(spec, prepared)
    assert warnings == []
    assert figure.dpi == 200
    assert figure.get_size_inches() == pytest.approx((8.0, 6.0))
    assert figure.axes[0].get_xlabel() == "Muon pT"
    assert figure.axes[0].get_title() == "Weighted muons"
    plt.close(figure)


def test_static_pipeline_rejects_missing_y_variable() -> None:
    spec = StaticPlotSpec(
        root_path="events.root",
        tree_name="Events",
        entry_start=0,
        entry_stop=1,
        plot_type="2D scatter",
        x_spec=VariableSpec(mode="branch", branch="x"),
        y_spec=None,
        color_spec=None,
        weight_branch=None,
        cut_specs=(),
        cut_bounds=(),
        style={"fig_width": 8.0, "fig_height": 6.0},
        labels={},
    )

    with pytest.raises(ValueError, match="requires a y variable"):
        prepare_static_plot(spec)


def test_static_pipeline_prepares_2d_histogram_for_render(root_path, monkeypatch) -> None:
    import numpy as np

    spec = StaticPlotSpec(
        root_path=str(root_path),
        tree_name="Events",
        entry_start=0,
        entry_stop=10,
        plot_type="2D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        y_spec=VariableSpec(mode="branch", branch="Muon_eta"),
        color_spec=None,
        weight_branch="genWeight",
        cut_specs=(),
        cut_bounds=(),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 8,
            "bins_y": 6,
            "log_x": False,
            "log_y": False,
            "log_z": False,
            "grid": False,
        },
        labels={"custom_x": "pT", "custom_y": "eta"},
    )
    prepared = prepare_static_plot(spec)
    histogram = prepared.histogram_2d
    assert histogram is not None
    assert prepared.histogram_1d is None
    assert histogram.axes[0].label == "pT"
    assert histogram.axes[1].label == "eta"
    values, xedges, yedges = np.histogram2d(
        prepared.x, prepared.y, bins=(8, 6), weights=prepared.weights
    )
    np.testing.assert_allclose(histogram.values(), values)
    np.testing.assert_allclose(histogram.axes[0].edges, xedges)
    np.testing.assert_allclose(histogram.axes[1].edges, yedges)
    assert histogram.variances() is not None

    def forbid_rebinning(*args, **kwargs):
        raise AssertionError("Rendering must not recompute histogram bins")

    monkeypatch.setattr(np, "histogram2d", forbid_rebinning)
    figure, warnings = render_prepared_static_plot(spec, prepared)
    assert warnings == []
    np.testing.assert_allclose(
        figure.axes[0].collections[0].get_array().reshape(-1), values.T.ravel()
    )
    plt.close(figure)


@pytest.mark.parametrize("mode", ["ignore", "fold", "separate"])
def test_pipeline_manual_histogram_bounds_and_flow(root_path, mode):
    spec = StaticPlotSpec(
        root_path=str(root_path),
        tree_name="Events",
        entry_start=0,
        entry_stop=12,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        y_spec=None,
        color_spec=None,
        weight_branch="genWeight",
        cut_specs=(),
        cut_bounds=(),
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 10,
            "log_x": False,
            "log_y": False,
            "grid": False,
            "manual_histogram_limits": True,
            "hist_x_min_text": "15",
            "hist_x_max_text": "30",
            "hist_flow_mode": mode,
        },
        labels={},
    )
    prepared = prepare_static_plot(spec)
    assert prepared.histogram_1d is not None
    assert prepared.display_histogram_1d is not None
    assert prepared.histogram_flow_summary is not None
    assert prepared.histogram_flow_mode == mode
    assert prepared.histogram_1d.axes[0].edges[[0, -1]].tolist() == pytest.approx([15.0, 30.0])
    summary = prepared.histogram_flow_summary
    assert summary.underflow_entries + summary.in_range_entries + summary.overflow_entries == len(
        prepared.x
    )
    if mode == "fold":
        assert prepared.display_histogram_1d.values().sum() == pytest.approx(prepared.weights.sum())
    figure, warnings = render_prepared_static_plot(spec, prepared)
    assert not warnings
    if mode == "separate":
        assert len(figure.axes[0].patches) == 12
    else:
        assert len(figure.axes[0].patches) == 10
    plt.close(figure)


@pytest.mark.parametrize("mode", ["ignore", "fold", "separate"])
def test_density_pipeline_flow_and_fit_invariance(root_path, mode):
    import numpy as np
    from dataclasses import replace
    from root_explorer.analysis.fitting import FitConfig, fit_histogram_1d

    style = {
        "fig_width": 8.0, "fig_height": 6.0, "bins_x": 10,
        "log_x": False, "log_y": False, "grid": False,
        "manual_histogram_limits": True, "hist_x_min_text": "15",
        "hist_x_max_text": "30", "hist_flow_mode": mode,
        "normalise_unit_area": True,
    }
    spec = StaticPlotSpec(
        root_path=str(root_path), tree_name="Events", entry_start=0, entry_stop=12,
        plot_type="1D histogram", x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        y_spec=None, color_spec=None, weight_branch=None, cut_specs=(), cut_bounds=(),
        style=style, labels={},
    )
    prepared = prepare_static_plot(spec)
    original = prepare_static_plot(replace(spec, style={**style, "normalise_unit_area": False}))
    assert prepared.normalisation_result is not None
    if mode == "separate":
        assert not prepared.normalisation_result.valid
        assert "separate categorical" in prepared.normalisation_result.error
    else:
        assert prepared.normalisation_result.valid
        density = prepared.normalisation_result.histogram
        widths = np.diff(density.axes[0].edges)
        assert float(np.dot(density.values(), widths)) == pytest.approx(1.0)
        assert prepared.y_label == "Normalised density"
        assert np.allclose(prepared.histogram_1d.values(flow=True), original.histogram_1d.values(flow=True))
    config = FitConfig(model="Exponential")
    raw_fit = fit_histogram_1d(original.display_histogram_1d, config)
    density_fit = fit_histogram_1d(prepared.display_histogram_1d, config)
    assert raw_fit.success == density_fit.success
    if raw_fit.success:
        assert raw_fit.parameters == pytest.approx(density_fit.parameters)
        assert raw_fit.statistic == pytest.approx(density_fit.statistic)


def test_normalised_histogram_is_unaffected_by_viewport_limits(root_path):
    from dataclasses import replace
    import numpy as np

    base = StaticPlotSpec(
        root_path=str(root_path), tree_name="Events", entry_start=0, entry_stop=12,
        plot_type="1D histogram", x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        y_spec=None, color_spec=None, weight_branch=None, cut_specs=(), cut_bounds=(),
        style={"bins_x": 10, "normalise_unit_area": True}, labels={},
    )
    original = prepare_static_plot(base)
    viewport = prepare_static_plot(replace(
        base, style={**base.style, "manual_axis_limits": True, "x_min_text": "20", "x_max_text": "40"}
    ))
    assert original.normalisation_result.valid
    np.testing.assert_allclose(
        original.normalisation_result.histogram.values(),
        viewport.normalisation_result.histogram.values(),
    )
