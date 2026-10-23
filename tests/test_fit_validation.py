"""Fit convergence, uncertainties and failure diagnostics (#153).

Parameter uncertainties are checked against a closed-form linear least-squares
reference that does not use the production fitting code. Optimiser failures are
simulated deterministically by monkeypatching the solver.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hist
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from root_explorer.analysis import fitting
from root_explorer.analysis.fitting import FitConfig, fit_histogram_1d, model_bin_integrals

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _histogram(edges, contents, *, variances=None) -> hist.Hist:
    axis = hist.axis.Variable(edges)
    if variances is None:
        histogram = hist.Hist(axis, storage=hist.storage.Double())
        histogram.view()[...] = contents
    else:
        histogram = hist.Hist(axis, storage=hist.storage.Weight())
        histogram.view().value[...] = contents
        histogram.view().variance[...] = variances
    return histogram


TRUE_PARAMETERS = {
    "Gaussian": {"A": 120.0, "mu": 0.8, "sigma": 1.4},
    "Exponential": {"A": 80.0, "k": -0.45},
    "Crystal Ball": {"A": 150.0, "mu": 0.5, "sigma": 1.1, "alpha": 1.2, "n": 4.0},
}
# Starting values away from the truth: shifted means, wrong widths and amplitudes.
OFFSET_GUESSES = {
    "Gaussian": {"A": 60.0, "mu": 2.0, "sigma": 2.5},
    "Exponential": {"A": 30.0, "k": -0.1},
    "Crystal Ball": {"A": 80.0, "mu": 1.5, "sigma": 2.0, "alpha": 2.5, "n": 2.0},
}


@pytest.mark.parametrize("model", list(TRUE_PARAMETERS))
@pytest.mark.parametrize("start", ["automatic", "offset"])
def test_fits_recover_parameters_from_nontrivial_starts(model, start) -> None:
    edges = np.linspace(-6.0, 7.0, 131)
    truth = TRUE_PARAMETERS[model]
    contents = model_bin_integrals(model, edges, truth, x0=(edges[0] + edges[-1]) / 2)
    initial = OFFSET_GUESSES[model] if start == "offset" else None

    result = fit_histogram_1d(_histogram(edges, contents), FitConfig(model, initial_values=initial))

    assert result.success, result.message
    for name, expected in truth.items():
        assert result.parameters[name] == pytest.approx(expected, rel=1e-4, abs=1e-4), name


# A straight line a + b*x. Each bin's predicted content is linear in (a, b):
#   integral over [l, r] = a * (r - l) + b * (r**2 - l**2) / 2
EDGES = np.array([0.0, 1.0, 2.0, 3.5, 4.0, 5.0, 6.5, 8.0])
CONTENTS = np.array([2.9, 5.2, 11.6, 4.1, 9.7, 18.5, 22.4])
VARIANCES = np.array([0.5, 1.0, 2.0, 0.8, 1.5, 3.0, 4.0])
DESIGN = np.column_stack([np.diff(EDGES), (EDGES[1:] ** 2 - EDGES[:-1] ** 2) / 2])


def _linear_reference(weights: np.ndarray, *, scale_by_residuals: bool):
    """Closed-form weighted least squares and its parameter covariance."""

    root = np.sqrt(weights)
    solution, *_ = np.linalg.lstsq(DESIGN * root[:, None], CONTENTS * root, rcond=None)
    residuals = (DESIGN @ solution - CONTENTS) * root
    statistic = float(residuals @ residuals)
    covariance = np.linalg.inv(DESIGN.T @ (DESIGN * weights[:, None]))
    if scale_by_residuals:
        covariance *= statistic / (len(CONTENTS) - 2)
    return solution, np.sqrt(np.diag(covariance)), statistic


@pytest.mark.parametrize("use_variances", [False, True], ids=["ordinary", "variance-weighted"])
def test_linear_fit_matches_closed_form_parameters_and_uncertainties(use_variances) -> None:
    histogram = _histogram(EDGES, CONTENTS, variances=VARIANCES)
    result = fit_histogram_1d(
        histogram,
        FitConfig("Custom expression", expression="a + b * x", use_variances=use_variances),
    )
    assert result.success, result.message

    weights = 1 / VARIANCES if use_variances else np.ones_like(CONTENTS)
    # Ordinary least squares scales the covariance by the residual variance;
    # stored variances are used as they are.
    solution, errors, statistic = _linear_reference(weights, scale_by_residuals=not use_variances)
    assert [result.parameters["a"], result.parameters["b"]] == pytest.approx(solution, rel=1e-6)
    assert [result.uncertainties["a"], result.uncertainties["b"]] == pytest.approx(errors, rel=1e-6)
    assert result.statistic == pytest.approx(statistic, rel=1e-6)
    assert result.ndof == len(CONTENTS) - 2

    # The two conventions give clearly different uncertainties for these data.
    _, other_errors, _ = _linear_reference(weights, scale_by_residuals=use_variances)
    assert not np.allclose(errors, other_errors, rtol=0.05)


def test_redundant_parameters_report_singular_covariance() -> None:
    # Only a + b is identifiable, so the covariance matrix is singular.
    result = fit_histogram_1d(
        _histogram(EDGES, CONTENTS), FitConfig("Custom expression", expression="(a + b) * x")
    )
    assert result.success
    assert result.uncertainties == {"a": None, "b": None}
    assert "Covariance is singular; uncertainties are unavailable." in result.message


def test_model_undefined_over_the_selected_bins_is_reported() -> None:
    edges = np.linspace(-5.0, -1.0, 9)
    result = fit_histogram_1d(
        _histogram(edges, np.ones(8)), FitConfig("Custom expression", expression="a * log(x)")
    )
    assert not result.success
    assert result.message == "Model is undefined or numerically unstable over the selected bins."
    assert result.bins_used == 8


def _gaussian_histogram() -> hist.Hist:
    edges = np.linspace(-4.0, 4.0, 41)
    contents = model_bin_integrals("Gaussian", edges, {"A": 50.0, "mu": 0.0, "sigma": 1.0}, x0=0.0)
    return _histogram(edges, contents)


def test_optimiser_exception_is_a_nonfatal_failure(monkeypatch) -> None:
    def raise_error(*args, **kwargs):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(fitting, "least_squares", raise_error)
    result = fit_histogram_1d(_gaussian_histogram(), FitConfig("Gaussian"))
    assert not result.success
    assert result.message == "Fit failed: solver exploded"
    assert result.bins_used == 40


def test_non_convergence_is_a_nonfatal_failure(monkeypatch) -> None:
    def not_converged(residual, guess, **kwargs):
        return SimpleNamespace(
            success=False, x=np.asarray(guess), message="maximum evaluations reached"
        )

    monkeypatch.setattr(fitting, "least_squares", not_converged)
    result = fit_histogram_1d(_gaussian_histogram(), FitConfig("Gaussian"))
    assert not result.success
    assert result.message == "Fit did not converge: maximum evaluations reached"
    assert result.parameters == {}


def test_failed_fit_keeps_the_histogram_in_the_app(monkeypatch, root_path) -> None:
    def raise_error(*args, **kwargs):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(fitting, "least_squares", raise_error)
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20).run()
    at.text_input(key="root_path").input(str(root_path)).run()
    at.selectbox(key="x_branch").select("Muon_pt").run()
    at.checkbox(key="plot_fit_enabled").check().run()
    at.button(key="plot_make").click().run()

    assert not at.exception
    result = at.session_state["plot_result"]
    assert not result["fit_result"].success
    assert result["figure"].axes[0].patches, "the histogram itself must still be drawn"
    assert "Histogram fit: Fit failed: solver exploded" in [warning.value for warning in at.warning]
