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
