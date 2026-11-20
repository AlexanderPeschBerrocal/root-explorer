"""Fits must not depend on the units of x (#162).

The reference fits below are independent of the production code: they integrate the
Gaussian with their own erf expression and solve in well-conditioned coordinates
(nanoseconds, or offsets in units of the width) where the default solver is reliable.
"""

from __future__ import annotations

import hist
import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.special import erf

from root_explorer.analysis.fitting import (
    FitConfig,
    FitResult,
    fit_histogram_1d,
    model_bin_integrals,
)


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


def _gaussian_counts(edges, amplitude, mu, sigma) -> np.ndarray:
    """Expected bin contents of A * exp(-(x - mu)**2 / (2 * sigma**2)), per bin."""
    t = (np.asarray(edges) - mu) / (sigma * np.sqrt(2))
    return amplitude * sigma * np.sqrt(np.pi / 2) * np.diff(erf(t))


def _poisson_peak(edges, entries, mu, sigma, seed) -> np.ndarray:
    expected = _gaussian_counts(edges, 1.0, mu, sigma)
    expected *= entries / expected.sum()
    return np.random.default_rng(seed).poisson(expected).astype(float)


def _reference_fit(edges, counts, variances, start):
    """Independent weighted Gaussian fit; returns (parameters, uncertainties)."""
    used = variances > 0
    error = np.sqrt(variances[used])

    def residual(p):
        return (_gaussian_counts(edges, *p)[used] - counts[used]) / error

    solution = least_squares(residual, start, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
    covariance = np.linalg.inv(solution.jac.T @ solution.jac)
    return solution.x, np.sqrt(np.diag(covariance))


def _values(result: FitResult, names=("A", "mu", "sigma")) -> tuple[np.ndarray, np.ndarray]:
    assert result.success, result.message
    errors = [result.uncertainties[name] for name in names]
    assert all(error is not None for error in errors), result.message
    return (
        np.array([result.parameters[name] for name in names]),
        np.array(errors, dtype=float),
    )


@pytest.mark.parametrize("use_variances", [True, False])
@pytest.mark.parametrize(
    ("model", "shape"),
    [("Gaussian", {}), ("Crystal Ball", {"alpha": 1.5, "n": 3.0})],
)
def test_same_data_in_gev_mev_and_kev_give_identical_physical_results(
    model, shape, use_variances
) -> None:
    # A J/psi-like peak in GeV, which previously lost its uncertainties. The data
    # follow the fitted model, so that Crystal Ball tail parameters are determined.
    edges_gev = np.linspace(3.097 - 0.25, 3.097 + 0.15, 81)
    truth = {"A": 1.0, "mu": 3.097, "sigma": 0.03, **shape}
    expected = model_bin_integrals(model, edges_gev, truth, x0=3.097)
    rng = np.random.default_rng(162)
    counts = rng.poisson(expected * 100_000 / expected.sum()).astype(float)
    variances = counts if use_variances else None
    names = ("A", "mu", "sigma", *shape)

    results = {}
    for factor in (1.0, 1e3, 1e6):
        result = fit_histogram_1d(
            _histogram(edges_gev * factor, counts, variances=variances),
            FitConfig(model, use_variances=use_variances),
        )
        results[factor] = _values(result, names)

    # A is a density per unit of x; mu and sigma carry the unit of x; shape
    # parameters are dimensionless.
    unit_power = np.array([-1.0, 1.0, 1.0, *([0.0] * len(shape))])
    gev_values, gev_errors = results[1.0]
    for factor, (values, errors) in results.items():
        conversion = factor**unit_power
        # Identical up to the solver's termination tolerance, a tiny fraction of the
        # statistical uncertainty.
        assert np.all(np.abs(values / conversion - gev_values) < 1e-2 * gev_errors)
        assert errors / conversion == pytest.approx(gev_errors, rel=1e-3)


@pytest.mark.parametrize(("use_variances", "entries"), [(True, 1_000), (False, 3_000)])
def test_narrow_d0_peak_in_gev_reports_uncertainties(use_variances, entries) -> None:
    edges = np.linspace(1.865 - 0.04, 1.865 + 0.04, 61)
    counts = _poisson_peak(edges, entries, 1.865, 0.008, seed=1865)
    result = fit_histogram_1d(
        _histogram(edges, counts, variances=counts if use_variances else None),
        FitConfig("Gaussian", use_variances=use_variances),
    )
    values, errors = _values(result)
    assert np.all(errors > 0)
    assert values[1] == pytest.approx(1.865, abs=3 * errors[1])
    assert "singular" not in result.message


@pytest.mark.parametrize(
    "initial_values",
    [None, {"A": 2e12, "mu": 24e-9, "sigma": 1.5e-9}],
    ids=["automatic", "user"],
)
def test_nanosecond_peak_with_x_in_seconds_matches_reference(initial_values) -> None:
    edges_ns = np.linspace(15.0, 35.0, 81)
    counts = _poisson_peak(edges_ns, 10_000, 25.0, 1.0, seed=25)
    reference, reference_errors = _reference_fit(edges_ns, counts, counts, [4000.0, 25.0, 1.0])

    result = fit_histogram_1d(
        _histogram(edges_ns * 1e-9, counts, variances=counts),
        FitConfig("Gaussian", use_variances=True, initial_values=initial_values),
    )
    values, errors = _values(result)
    to_ns = np.array([1e-9, 1e9, 1e9])
    # Far tighter than the acceptance criterion of one statistical sigma.
    assert np.all(np.abs(values * to_ns - reference) < 1e-3 * reference_errors)
    assert errors * to_ns == pytest.approx(reference_errors, rel=1e-3)


def test_peak_at_large_offset_is_unbiased_with_correct_uncertainties() -> None:
    mu, sigma = 1e6, 0.01
    offsets = np.linspace(-5.0, 5.0, 61)  # in units of sigma
    counts = _poisson_peak(offsets, 10_000, 0.0, 1.0, seed=6)
    reference, reference_errors = _reference_fit(offsets, counts, counts, [1500.0, 0.0, 1.0])

    result = fit_histogram_1d(
        _histogram(mu + sigma * offsets, counts, variances=counts),
        FitConfig("Gaussian", use_variances=True),
    )
    values, errors = _values(result)
    expected = np.array([reference[0] / sigma, mu + sigma * reference[1], sigma * reference[2]])
    expected_errors = reference_errors * np.array([1 / sigma, sigma, sigma])
    assert np.all(np.abs(values - expected) < 1e-3 * expected_errors)
    assert errors == pytest.approx(expected_errors, rel=1e-3)


def test_sub_picosecond_width_in_seconds_is_allowed() -> None:
    # The former absolute 1e-12 lower bound on sigma made this width impossible.
    edges = np.linspace(-5e-13, 5e-13, 61)
    counts = _poisson_peak(edges, 10_000, 0.0, 1e-13, seed=13)
    result = fit_histogram_1d(
        _histogram(edges, counts, variances=counts), FitConfig("Gaussian", use_variances=True)
    )
    values, errors = _values(result)
    assert values[2] == pytest.approx(1e-13, abs=3 * errors[2])


def test_custom_expression_peak_at_large_offset_matches_reference() -> None:
    # A custom parameter starting inside the fit range is treated as a position, so its
    # finite-difference step follows the range instead of |m| = 1e6.
    mu, sigma = 1e6, 0.01
    offsets = np.linspace(-5.0, 5.0, 61)
    counts = _poisson_peak(offsets, 10_000, 0.0, 1.0, seed=6)
    reference, reference_errors = _reference_fit(offsets, counts, counts, [1500.0, 0.0, 1.0])

    result = fit_histogram_1d(
        _histogram(mu + sigma * offsets, counts, variances=counts),
        FitConfig(
            "Custom expression",
            expression="N * exp(-0.5 * ((x - m) / s)**2)",
            initial_values={"N": 1.2e5, "m": mu + 0.004, "s": 0.008},
            use_variances=True,
        ),
    )
    values, errors = _values(result, names=("N", "m", "s"))
    expected = np.array([reference[0] / sigma, mu + sigma * reference[1], sigma * reference[2]])
    expected_errors = reference_errors * np.array([1 / sigma, sigma, sigma])
    assert np.all(np.abs(values - expected) < 1e-2 * expected_errors)
    assert errors == pytest.approx(expected_errors, rel=1e-3)


@pytest.mark.parametrize("unit", [1e-15, 1e-9, 1e6])
def test_unweighted_crystal_ball_reaches_the_same_minimum_in_any_unit(unit) -> None:
    # Scaling solver steps by the Jacobian once let the weakly constrained tail
    # exponent n run off to ~6000 in some units, ending in a worse minimum.
    truth = {"A": 1.0, "mu": 0.3, "sigma": 1.0, "alpha": 1.5, "n": 3.0}
    edges = np.linspace(-8.0, 5.0, 81)
    expected = model_bin_integrals("Crystal Ball", edges, truth, x0=-1.5)
    counts = np.random.default_rng(162).poisson(expected * 20_000 / expected.sum()).astype(float)
    start = {"A": 0.8 * 7600.0, "mu": 0.7, "sigma": 0.8, "alpha": 1.5, "n": 3.0}

    def fit(scale: float) -> FitResult:
        initial = {
            "A": start["A"] / scale,
            "mu": start["mu"] * scale,
            "sigma": start["sigma"] * scale,
            "alpha": start["alpha"],
            "n": start["n"],
        }
        return fit_histogram_1d(
            _histogram(edges * scale, counts), FitConfig("Crystal Ball", initial_values=initial)
        )

    natural, scaled = fit(1.0), fit(unit)
    assert natural.success and scaled.success
    assert scaled.statistic == pytest.approx(natural.statistic, rel=1e-6)
    assert scaled.parameters["n"] == pytest.approx(natural.parameters["n"], rel=1e-3)
    assert scaled.parameters["alpha"] == pytest.approx(natural.parameters["alpha"], rel=1e-3)
