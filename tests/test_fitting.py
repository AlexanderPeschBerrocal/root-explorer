"""Regression tests for built-in binned histogram fits."""

import hist
import numpy as np
import pytest

from root_explorer.analysis.fitting import (
    FIT_MODELS,
    FitConfig,
    fit_histogram_1d,
    model_bin_integrals,
    model_density,
)


def _histogram(edges, contents, *, variances=None):
    axis = hist.axis.Variable(edges)
    if variances is None:
        h = hist.Hist(axis, storage=hist.storage.Double())
        h.view()[...] = contents
    else:
        h = hist.Hist(axis, storage=hist.storage.Weight())
        h.view().value[...] = contents
        h.view().variance[...] = variances
    return h


@pytest.mark.parametrize(
    ("model", "params"),
    [
        ("Gaussian", {"A": 200.0, "mu": 1.1, "sigma": 1.3}),
        ("Exponential", {"A": 100.0, "k": -0.3}),
        ("Crystal Ball", {"A": 200.0, "mu": 1.1, "sigma": 1.3, "alpha": 1.5, "n": 3.0}),
        (
            "Double-sided Crystal Ball",
            {
                "A": 200.0,
                "mu": 1.1,
                "sigma": 1.3,
                "alpha_L": 1.5,
                "n_L": 3.0,
                "alpha_R": 1.8,
                "n_R": 4.0,
            },
        ),
    ],
)
def test_binned_model_parameter_recovery(model, params):
    edges = np.linspace(-8, 9, 171)
    x0 = (edges[0] + edges[-1]) / 2
    contents = model_bin_integrals(model, edges, params, x0=x0)
    h = _histogram(edges, contents)
    outcome = fit_histogram_1d(h, FitConfig(model, initial_values=params))
    assert outcome.success, outcome.message
    for name, expected in params.items():
        assert outcome.parameters[name] == pytest.approx(expected, rel=1e-3, abs=1e-3)
    assert outcome.statistic == pytest.approx(0, abs=1e-5)
    assert outcome.ndof == len(contents) - len(FIT_MODELS[model])


def test_nonuniform_edges_integrate_gaussian_density():
    from scipy.integrate import quad

    edges = np.array([-4.0, -1.0, 0.2, 0.4, 3.0, 8.0])
    params = {"A": 12.0, "mu": 0.1, "sigma": 1.2}
    actual = model_bin_integrals("Gaussian", edges, params, x0=1)
    expected = [
        quad(lambda x: model_density("Gaussian", np.array(x), params, x0=1), left, right)[0]
        for left, right in zip(edges[:-1], edges[1:], strict=True)
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize("model", ["Crystal Ball", "Double-sided Crystal Ball"])
def test_crystal_ball_integrals_match_quadrature(model):
    from scipy.integrate import quad

    params = {"A": 9.0, "mu": 0.2, "sigma": 1.4}
    if model == "Crystal Ball":
        params.update(alpha=1.2, n=2.5)
    else:
        params.update(alpha_L=1.4, n_L=3.0, alpha_R=2.0, n_R=5.0)
    edges = np.array([-8.0, -2.5, -1.0, 0.5, 2.0, 4.0, 8.0])
    actual = model_bin_integrals(model, edges, params, x0=0)
    expected = [
        quad(lambda x: model_density(model, np.array(x), params, x0=0), a, b)[0]
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]
    np.testing.assert_allclose(actual, expected, rtol=2e-7)


def test_weighted_fit_uses_stored_variances_and_excludes_zero_variance_bins():
    edges = np.linspace(-5, 5, 41)
    params = {"A": 100.0, "mu": 0.3, "sigma": 1.4}
    values = model_bin_integrals("Gaussian", edges, params, x0=0)
    values[20] = 1e4
    variances = np.maximum(values.copy(), 1.0)
    variances[20] = 0.0
    h = _histogram(edges, values, variances=variances)
    result = fit_histogram_1d(h, FitConfig("Gaussian", use_variances=True))
    assert result.success, result.message
    assert result.bins_excluded == 1
    assert result.parameters["mu"] == pytest.approx(params["mu"], abs=0.05)
    assert result.objective == "Variance-weighted least squares"


def test_no_poisson_assumption_for_unweighted_histograms():
    edges = np.linspace(-4, 4, 31)
    values = np.zeros(30)
    values[10] = -2
    h = _histogram(edges, values)
    result = fit_histogram_1d(h, FitConfig("Gaussian"))
    assert result.objective == "Ordinary least squares"
    assert result.bins_excluded == 0


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (FitConfig("Gaussian", fit_range=(2, 2)), "Fit range"),
        (FitConfig("Gaussian", fit_range=(np.nan, 1)), "Fit range"),
        (FitConfig("Gaussian", fit_range=(0, 0.1)), "Insufficient"),
        (FitConfig("Gaussian", initial_values={"A": -10}), "outside"),
        (FitConfig("Gaussian", bounds={"sigma": (2, 1)}), "Invalid"),
        (FitConfig("unknown"), "Unknown"),
    ],
)
def test_invalid_fit_configuration_is_nonfatal(config, message):
    h = _histogram(np.linspace(-3, 3, 25), np.ones(24))
    result = fit_histogram_1d(h, config)
    assert not result.success
    assert message in result.message


def test_no_variances_is_explicit_failure():
    h = _histogram(np.linspace(-3, 3, 25), np.ones(24))
    result = fit_histogram_1d(h, FitConfig("Gaussian", use_variances=True))
    assert not result.success
    assert "variances" in result.message


def test_insufficient_positive_variances_is_nonfatal():
    h = _histogram(np.linspace(-3, 3, 25), np.ones(24), variances=np.zeros(24))
    result = fit_histogram_1d(h, FitConfig("Gaussian", use_variances=True))
    assert not result.success
    assert "Insufficient" in result.message


def test_weighted_excluded_bin_does_not_affect_auto_initialisation():
    edges = np.linspace(-5, 5, 41)
    params = {"A": 100.0, "mu": 0.3, "sigma": 1.4}
    values = model_bin_integrals("Gaussian", edges, params, x0=0)
    variances = np.maximum(values, 1.0)
    variances[20] = 0.0
    good = _histogram(edges, values.copy(), variances=variances)
    bad_values = values.copy()
    bad_values[20] = 1e12
    bad = _histogram(edges, bad_values, variances=variances)
    a = fit_histogram_1d(good, FitConfig("Gaussian", use_variances=True))
    b = fit_histogram_1d(bad, FitConfig("Gaussian", use_variances=True))
    assert a.success and b.success
    for parameter in params:
        assert a.parameters[parameter] == pytest.approx(b.parameters[parameter], rel=1e-6)
