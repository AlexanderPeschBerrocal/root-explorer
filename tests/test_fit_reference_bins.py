"""Fit reference points and default bounds follow the bins actually fitted (#192)."""

from __future__ import annotations

import hist
import numpy as np
import pytest
from scipy.special import erf

from root_explorer.analysis.fitting import (
    FIT_MODELS,
    FitConfig,
    fit_histogram_1d,
    fit_model_formula,
    model_bin_integrals,
)
from root_explorer.plotting.codegen import fit_summary_lines


def _histogram(edges, contents) -> hist.Hist:
    histogram = hist.Hist(hist.axis.Variable(edges), storage=hist.storage.Double())
    histogram.view()[:] = contents
    return histogram


def _exponential_histogram() -> hist.Hist:
    edges = np.linspace(0.0, 10.0, 51)
    histogram = hist.Hist(hist.axis.Variable(edges), storage=hist.storage.Double())
    histogram.fill(np.random.default_rng(192).exponential(1.0, 50_000))
    return histogram


# --- Exponential reference point ---------------------------------------------------------


@pytest.mark.parametrize("typed_range", [(-5.0, 10.0), (0.0, 100.0), (0.0, 1000.0), (0.0, 2000.0)])
def test_typed_ranges_selecting_the_same_bins_give_identical_results(typed_range) -> None:
    histogram = _exponential_histogram()
    reference = fit_histogram_1d(histogram, FitConfig("Exponential", fit_range=(0.0, 10.0)))
    result = fit_histogram_1d(histogram, FitConfig("Exponential", fit_range=typed_range))
    # (0, 1000) and (0, 2000) used to fail; (0, 100) gave A = 8e-18.
    assert result.success, result.message
    assert result.fit_range == reference.fit_range == (0.0, 10.0)
    assert result.reference_x == reference.reference_x == 5.0
    assert result.parameters == pytest.approx(reference.parameters, rel=1e-9)
    assert result.uncertainties == pytest.approx(reference.uncertainties, rel=1e-6)


def test_exponential_amplitude_is_the_density_at_the_reference_point() -> None:
    edges = np.linspace(2.0, 6.0, 41)
    truth = {"A": 250.0, "k": -0.8}
    contents = model_bin_integrals("Exponential", edges, truth, x0=4.0)
    result = fit_histogram_1d(_histogram(edges, contents), FitConfig("Exponential"))
    assert result.reference_x == 4.0
    assert result.parameters == pytest.approx(truth, rel=1e-7)


# --- Default width bounds from the fitted bins -------------------------------------------


def test_distant_excluded_bin_does_not_constrain_the_width() -> None:
    # The issue's exact-Gaussian reproduction with independent analytic integrals.
    edges = np.linspace(-1.0, 1.0, 41)
    contents = 1000.0 * 0.1 * np.sqrt(np.pi / 2) * np.diff(erf((edges - 0.1) / (np.sqrt(2) * 0.1)))
    results = []
    for far_edge in (None, 1e6):
        all_edges = edges if far_edge is None else np.append(edges, far_edge)
        all_contents = contents if far_edge is None else np.append(contents, 0.0)
        result = fit_histogram_1d(
            _histogram(all_edges, all_contents), FitConfig("Gaussian", fit_range=(-1.0, 1.0))
        )
        assert result.success, result.message
        assert result.bins_used == 40
        assert result.parameters["sigma"] == pytest.approx(0.1, rel=1e-7)
        results.append(result)
    assert results[1].parameters == pytest.approx(results[0].parameters, rel=1e-9)
    assert results[1].statistic == pytest.approx(results[0].statistic, abs=1e-20)


@pytest.mark.parametrize("model", ["Gaussian", "Crystal Ball", "Double-sided Crystal Ball"])
def test_peak_models_ignore_excluded_bins_for_default_bounds(model) -> None:
    truth = {"A": 500.0, "mu": 0.1, "sigma": 0.05, "alpha": 1.5, "n": 3.0}
    truth.update(alpha_L=1.5, n_L=3.0, alpha_R=2.0, n_R=5.0)
    truth = {name: truth[name] for name in FIT_MODELS[model]}
    edges = np.linspace(-0.5, 0.5, 81)
    contents = model_bin_integrals(model, edges, truth, x0=0.0)
    config = FitConfig(model, fit_range=(-0.5, 0.5), initial_values=truth)
    plain = fit_histogram_1d(_histogram(edges, contents), config)
    padded = fit_histogram_1d(_histogram(np.append(edges, 1e7), np.append(contents, 0.0)), config)
    assert plain.success and padded.success, (plain.message, padded.message)
    assert padded.parameters == pytest.approx(plain.parameters, rel=1e-9)
    assert padded.parameters["sigma"] == pytest.approx(0.05, rel=1e-6)


def test_explicit_width_bounds_are_still_respected() -> None:
    edges = np.linspace(-1.0, 1.0, 41)
    contents = model_bin_integrals("Gaussian", edges, {"A": 1000.0, "mu": 0.1, "sigma": 0.1}, x0=0)
    for histogram in (
        _histogram(edges, contents),
        _histogram(np.append(edges, 1e6), np.append(contents, 0.0)),
    ):
        result = fit_histogram_1d(
            histogram,
            FitConfig("Gaussian", fit_range=(-1.0, 1.0), bounds={"sigma": (0.15, None)}),
        )
        assert result.success, result.message
        assert result.parameters["sigma"] == pytest.approx(0.15, abs=1e-9)


# --- The meaning of A is shown --------------------------------------------------------------


def test_model_function_is_described_for_every_built_in_model() -> None:
    histogram = _exponential_histogram()
    exponential = fit_histogram_1d(histogram, FitConfig("Exponential", fit_range=(0.0, 1000.0)))
    assert fit_model_formula(exponential) == (
        "A * exp(k * (x - x0)) with x0 = 5, the centre of the fitted bins"
    )
    assert "Model function: A * exp(k * (x - x0)) with x0 = 5" in "\n".join(
        fit_summary_lines(exponential)
    )
    for model in ("Gaussian", "Crystal Ball", "Double-sided Crystal Ball"):
        result = fit_histogram_1d(histogram, FitConfig(model))
        assert fit_model_formula(result).startswith("A * exp(-0.5 * ")
    custom = fit_histogram_1d(
        histogram, FitConfig("Custom expression", expression="a * exp(b * x)")
    )
    assert fit_model_formula(custom) is None  # the expression itself is shown


def test_app_and_export_show_the_model_function(tmp_path) -> None:
    from pathlib import Path

    import uproot
    from streamlit.testing.v1 import AppTest

    path = tmp_path / "decay.root"
    with uproot.recreate(path) as root_file:
        values = np.random.default_rng(192).exponential(1.0, 5_000)
        root_file.mktree("Events", {"t": values})
    project_root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(project_root / "streamlit_app.py", default_timeout=20).run()
    at.text_input(key="root_path").input(str(path)).run()
    at.selectbox(key="x_branch").select("t").run()
    at.checkbox(key="plot_fit_enabled").check().run()
    at.selectbox(key="plot_fit_model").select("Exponential").run()
    at.button(key="plot_make").click().run()
    assert not at.exception
    result = at.session_state["plot_result"]["fit_result"]
    assert result.success, result.message
    expected = f"Model function: {fit_model_formula(result)}"
    assert any(caption.value == expected for caption in at.caption)
    assert expected in fit_summary_lines(result)
