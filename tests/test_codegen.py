from __future__ import annotations

from dataclasses import replace

import awkward as ak
import matplotlib
import numpy as np
import pytest
import uproot

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from root_explorer.app.plotting import PlotRequest
from root_explorer.app.state import DataContext, GeometryState
from root_explorer.models import VariableSpec
from root_explorer.plotting.codegen import generate_static_plot_code


def test_generated_weighted_histogram_code_compiles_and_runs(root_path) -> None:
    request = PlotRequest(
        context=DataContext(
            root_path=str(root_path),
            event_container_name="Events",
            n_entries=12,
            entry_start=2,
            entry_stop=10,
            field_info=[],
            fields=[],
        ),
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        y_spec=None,
        z_spec=None,
        color_spec=None,
        weight_branch="genWeight",
        cut_specs=[VariableSpec(mode="branch", branch="Muon_eta")],
        cut_bounds=[(-2.1, 2.1)],
        style={
            "fig_width": 11.0,
            "fig_height": 5.0,
            "bins_x": 37,
            "log_y": True,
            "grid": False,
        },
        labels={
            "title": "Weighted muon pT",
            "custom_x": "Muon transverse momentum [GeV]",
            "custom_y": "Weighted candidates",
            "custom_z": "",
            "save_directory": "root_explorer_plots",
            "save_name": "weighted_muon_pt",
            "save_png": True,
            "save_pdf": False,
        },
        geometry=GeometryState(scene=None, options={}),
        event_index=None,
        preview=False,
        save_name="weighted_muon_pt",
    )

    source = generate_static_plot_code(request)

    compile(source, "weighted_muon_pt.py", "exec")
    assert "from root_explorer" not in source
    assert "root_path =" in source
    assert "entry_start = 2" in source
    assert "entry_stop = 10" in source
    assert "'genWeight'" in source
    assert "mask &= cut_0 >= -2.1" in source
    assert "axes.hist(x, bins=37, weights=weights)" in source
    assert 'axes.set_yscale("log")' in source
    assert "figure = plt.figure(figsize=(11.0, 5.0))" in source
    assert "Muon transverse momentum [GeV]" in source

    namespace = {"__name__": "generated_weighted_muon_pt"}
    exec(source, namespace)
    namespace["plt"].show = lambda: None
    namespace["main"]()
    assert plt.get_fignums()
    plt.close("all")

    delta_eta_request = replace(
        request,
        x_spec=VariableSpec(mode="delta_eta", branch_a="Muon_eta", branch_b="Muon_eta"),
        cut_specs=[],
        cut_bounds=[],
        labels={**request.labels, "custom_x": "", "custom_y": ""},
    )
    delta_eta_source = generate_static_plot_code(delta_eta_request)
    compile(delta_eta_source, "delta_eta.py", "exec")
    assert "x = x_a - x_b" in delta_eta_source
    assert "Δη(Muon_eta,Muon_eta)" in delta_eta_source


def test_generated_code_distinguishes_event_weights_when_lengths_are_ambiguous(
    tmp_path,
) -> None:
    root_path = tmp_path / "ambiguous_counts.root"
    with uproot.recreate(root_path) as root_file:
        root_file.mktree(
            "Events",
            {
                "Muon_pt": ak.Array([[10.0, 20.0], [], [30.0]]),
                "genWeight": np.array([2.0, 100.0, 5.0]),
            },
        )

    request = PlotRequest(
        context=DataContext(str(root_path), "Events", 3, 0, 3, [], []),
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        y_spec=None,
        z_spec=None,
        color_spec=None,
        weight_branch="genWeight",
        cut_specs=[],
        cut_bounds=[],
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 3,
            "log_y": False,
            "grid": False,
        },
        labels={"title": "", "custom_x": "", "custom_y": "", "custom_z": ""},
        geometry=GeometryState(scene=None, options={}),
        event_index=None,
        preview=False,
        save_name="plot",
    )

    source = generate_static_plot_code(request)
    namespace = {"__name__": "generated_ambiguous_weights"}
    exec(source, namespace)
    namespace["plt"].show = lambda: None
    namespace["main"]()

    axes = namespace["plt"].gcf().axes[0]
    assert sum(patch.get_height() for patch in axes.patches) == pytest.approx(9.0)
    assert axes.get_ylabel() == "sum weights (genWeight)"
    plt.close("all")


def test_generated_code_rejects_interactive_3d_request() -> None:
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

    try:
        generate_static_plot_code(request)
    except ValueError as exc:
        assert "not available" in str(exc)
    else:
        raise AssertionError("Expected 3D code generation to be rejected.")


@pytest.mark.parametrize(
    ("reverse", "expected_name"),
    [(False, "Greys"), (True, "Greys_r")],
)
def test_generated_scatter_code_preserves_greys_case(reverse: bool, expected_name: str) -> None:
    request = PlotRequest(
        context=DataContext("events.root", "Events", 1, 0, 1, [], []),
        plot_type="2D scatter",
        x_spec=VariableSpec(mode="branch", branch="x"),
        y_spec=VariableSpec(mode="branch", branch="y"),
        z_spec=None,
        color_spec=VariableSpec(mode="branch", branch="energy"),
        weight_branch=None,
        cut_specs=[],
        cut_bounds=[],
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "marker_size_2d": 3.0,
            "opacity_2d": 0.5,
            "max_points": 100_000,
            "colorscale": "Greys",
            "reverse_colorscale": reverse,
            "equal_aspect": False,
            "grid": False,
        },
        labels={"title": "", "custom_x": "", "custom_y": "", "custom_z": ""},
        geometry=GeometryState(scene=None, options={}),
        event_index=None,
        preview=False,
        save_name="plot",
    )

    source = generate_static_plot_code(request)

    compile(source, "greys_scatter.py", "exec")
    assert f"cmap={expected_name!r}" in source
