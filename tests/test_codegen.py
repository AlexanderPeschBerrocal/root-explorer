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
            "log_x": True,
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
    assert 'axes.set_xscale("log")' in source
    assert 'axes.set_yscale("log")' in source
    assert "figure = plt.figure(figsize=(11.0, 5.0))" in source
    assert "Muon transverse momentum [GeV]" in source

    namespace = {"__name__": "generated_weighted_muon_pt"}
    exec(source, namespace)
    namespace["plt"].show = lambda: None
    namespace["main"]()
    assert plt.get_fignums()
    assert namespace["plt"].gcf().axes[0].get_xscale() == "log"
    assert namespace["plt"].gcf().axes[0].get_yscale() == "log"
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
            "show_kde": True,
            "kde_bandwidth": "silverman",
            "kde_custom_factor": 1.0,
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
    assert axes.lines[0].get_label() == "KDE"
    plt.close("all")


def test_generated_2d_histogram_code_reproduces_highest_density_contours(root_path) -> None:
    request = PlotRequest(
        context=DataContext(str(root_path), "Events", 12, 0, 12, [], []),
        plot_type="2D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        y_spec=VariableSpec(mode="branch", branch="Muon_eta"),
        z_spec=None,
        color_spec=None,
        weight_branch=None,
        cut_specs=[],
        cut_bounds=[],
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
        labels={"title": "", "custom_x": "", "custom_y": "", "custom_z": ""},
        geometry=GeometryState(scene=None, options={}),
        event_index=None,
        preview=False,
        save_name="plot",
    )

    source = generate_static_plot_code(request)

    compile(source, "contour_plot.py", "exec")
    assert "contour_percentages = [68.0, 95.0, 99.0]" in source
    assert "contours = axes.contour(" in source
    assert "from scipy.stats import gaussian_kde" in source
    assert "bandwidth = kde_bandwidth('custom', 0.7)" in source
    namespace = {"__name__": "generated_contour_plot"}
    exec(source, namespace)
    namespace["plt"].show = lambda: None
    namespace["main"]()
    assert len(namespace["plt"].gcf().axes[0].collections) > 1
    plt.close("all")


@pytest.mark.parametrize(
    ("spec", "formula"),
    [
        (VariableSpec(mode="absolute", branch_a="Muon_eta"), "x = np.abs(x_a)"),
        (
            VariableSpec(
                mode="magnitude_3d",
                branch_a="Muon_px",
                branch_b="Muon_py",
                branch_c="Muon_pz",
            ),
            "x = np.hypot(np.hypot(x_a, x_b), x_c)",
        ),
        (
            VariableSpec(
                mode="energy_massless",
                branch_a="Muon_pt",
                branch_b="Muon_eta",
                longitudinal_kind="eta",
            ),
            "x = x_a * np.cosh(x_b)",
        ),
        (
            VariableSpec(
                mode="energy_massive",
                branch_a="Muon_pt",
                branch_b="Muon_eta",
                branch_c="Muon_mass",
                longitudinal_kind="eta",
            ),
            "x = np.sqrt(x_c**2 + (x_a * np.cosh(x_b))**2)",
        ),
        (
            VariableSpec(
                mode="energy_massive",
                branch_a="Muon_pt",
                branch_b="Muon_eta",
                branch_c="Muon_mass",
                longitudinal_kind="y",
            ),
            "x = np.hypot(x_c, x_a) * np.cosh(x_b)",
        ),
        (
            VariableSpec(
                mode="invariant_mass",
                branch_a="Muon_energy",
                branch_b="Muon_px",
                branch_c="Muon_py",
                branch_d="Muon_pz",
            ),
            "x = np.sqrt(np.maximum(x_mass_squared, 0.0))",
        ),
    ],
)
def test_generated_code_reproduces_single_particle_quantities(root_path, spec, formula) -> None:
    request = PlotRequest(
        context=DataContext(str(root_path), "Events", 12, 0, 12, [], []),
        plot_type="1D histogram",
        x_spec=spec,
        y_spec=None,
        z_spec=None,
        color_spec=None,
        weight_branch=None,
        cut_specs=[],
        cut_bounds=[],
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 20,
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
    assert formula in source
    namespace = {"__name__": "generated_single_particle_quantity"}
    exec(source, namespace)
    namespace["plt"].show = lambda: None
    namespace["main"]()
    assert namespace["plt"].gcf().axes[0].patches
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


def test_generated_histogram_code_marks_quantile(root_path) -> None:
    request = PlotRequest(
        context=DataContext(str(root_path), "Events", 12, 0, 12, [], []),
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        y_spec=None,
        z_spec=None,
        color_spec=None,
        weight_branch=None,
        cut_specs=[],
        cut_bounds=[],
        style={
            "fig_width": 8.0,
            "fig_height": 6.0,
            "bins_x": 20,
            "log_y": False,
            "grid": False,
        },
        labels={"title": "", "custom_x": "", "custom_y": "", "custom_z": ""},
        geometry=GeometryState(scene=None, options={}),
        event_index=None,
        preview=False,
        save_name="plot",
        quantile_percentile=50.0,
    )
    source = generate_static_plot_code(request)
    assert "np.quantile(x, 0.5)" in source
    assert "50% quantile" in source


def test_generated_scatter_code_supports_log_color_scale() -> None:
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
            "colorscale": "Viridis",
            "reverse_colorscale": False,
            "log_color": True,
            "color_min_text": "1",
            "color_max_text": "100",
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
    assert "from matplotlib.colors import LogNorm" in source
    assert "color_norm = LogNorm" in source
