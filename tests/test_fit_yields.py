"""Fits must not depend on the overall yield of the histogram (#199).

Scaling all bin contents by c must rescale the amplitude (and, for ordinary least
squares, the statistic by c² and the amplitude uncertainty by c) while leaving the
fitted shape unchanged. Small yields occur for normalised histograms and small weights.
"""

from __future__ import annotations

from pathlib import Path

import hist
import numpy as np
import pytest
import uproot
from scipy.special import erf

from root_explorer.analysis.fitting import FitConfig, fit_histogram_1d, model_bin_integrals
from root_explorer.io.histograms import read_histogram
from root_explorer.plotting.histograms import fit_persisted_histogram

EDGES = np.linspace(-1.0, 2.0, 41)


def _gaussian_contents(edges, amplitude=100.0, mu=0.3, sigma=0.7) -> np.ndarray:
    """Independent analytic bin integrals, as in the issue's reproduction."""
    return (
        amplitude * sigma * np.sqrt(np.pi / 2) * np.diff(erf((edges - mu) / (np.sqrt(2) * sigma)))
    )


def _histogram(contents, *, variances=None, edges=EDGES) -> hist.Hist:
    axis = hist.axis.Variable(edges)
    if variances is None:
        histogram = hist.Hist(axis, storage=hist.storage.Double())
        histogram.view()[:] = contents
    else:
        histogram = hist.Hist(axis, storage=hist.storage.Weight())
        histogram.view().value[:] = contents
        histogram.view().variance[:] = variances
    return histogram


@pytest.mark.parametrize("factor", [1e6, 1.0, 1e-3, 1e-5, 1e-6, 1e-12, 1e-30])
def test_exact_gaussian_is_recovered_at_any_yield(factor) -> None:
    # The issue's example: before the fix, 1e-5 gave sigma 0.629 with "Fit converged".
    result = fit_histogram_1d(_histogram(_gaussian_contents(EDGES) * factor), FitConfig("Gaussian"))
    assert result.success, result.message
    assert result.parameters["mu"] == pytest.approx(0.3, rel=1e-7)
    assert result.parameters["sigma"] == pytest.approx(0.7, rel=1e-7)
    assert result.parameters["A"] / factor == pytest.approx(100.0, rel=1e-7)


@pytest.mark.parametrize("factor", [1.0, 1e-12])
def test_exact_exponential_is_recovered_at_any_yield(factor) -> None:
    edges = np.linspace(0.0, 10.0, 51)
    contents = model_bin_integrals("Exponential", edges, {"A": 300.0, "k": -0.7}, x0=5.0)
    result = fit_histogram_1d(_histogram(contents * factor, edges=edges), FitConfig("Exponential"))
    assert result.success, result.message
    assert result.parameters["k"] == pytest.approx(-0.7, rel=1e-7)
    assert result.parameters["A"] / factor == pytest.approx(300.0, rel=1e-6)


def _noisy_counts() -> np.ndarray:
    expected = _gaussian_contents(EDGES, amplitude=2000.0)
    return np.random.default_rng(199).poisson(expected).astype(float)


@pytest.mark.parametrize("factor", [1e-6, 1e4])
def test_ordinary_least_squares_scales_statistic_and_uncertainties(factor) -> None:
    counts = _noisy_counts()
    reference = fit_histogram_1d(_histogram(counts), FitConfig("Gaussian"))
    scaled = fit_histogram_1d(_histogram(counts * factor), FitConfig("Gaussian"))
    assert reference.success and scaled.success
    for name in ("mu", "sigma"):
        assert scaled.parameters[name] == pytest.approx(reference.parameters[name], rel=1e-6)
        assert scaled.uncertainties[name] == pytest.approx(reference.uncertainties[name], rel=1e-5)
    assert scaled.parameters["A"] == pytest.approx(reference.parameters["A"] * factor, rel=1e-6)
    assert scaled.uncertainties["A"] == pytest.approx(
        reference.uncertainties["A"] * factor, rel=1e-5
    )
    assert scaled.statistic == pytest.approx(reference.statistic * factor**2, rel=1e-6)


@pytest.mark.parametrize("factor", [1e-6, 1e4])
def test_weighted_fit_is_unchanged_by_a_common_weight_scale(factor) -> None:
    # Weights scaled by c scale contents by c and variances by c²; chi² is unchanged.
    counts = _noisy_counts()
    config = FitConfig("Gaussian", use_variances=True)
    reference = fit_histogram_1d(_histogram(counts, variances=counts), config)
    scaled = fit_histogram_1d(_histogram(counts * factor, variances=counts * factor**2), config)
    assert reference.success and scaled.success
    for name in ("mu", "sigma"):
        assert scaled.parameters[name] == pytest.approx(reference.parameters[name], rel=1e-6)
        assert scaled.uncertainties[name] == pytest.approx(reference.uncertainties[name], rel=1e-5)
    assert scaled.parameters["A"] == pytest.approx(reference.parameters["A"] * factor, rel=1e-6)
    assert scaled.uncertainties["A"] == pytest.approx(
        reference.uncertainties["A"] * factor, rel=1e-5
    )
    assert scaled.statistic == pytest.approx(reference.statistic, rel=1e-6)


def test_small_yield_persisted_histogram_fits_the_true_shape(tmp_path: Path) -> None:
    # A stored TH1 with tiny contents, e.g. a cross-section in small units.
    contents = _gaussian_contents(EDGES)
    path = tmp_path / "small_yield.root"
    with uproot.recreate(path) as root_file:
        root_file["shape"] = (contents * 1e-7, EDGES)
    data = read_histogram(str(path), "shape")
    result = fit_persisted_histogram(data, FitConfig("Gaussian"))
    assert result.success, result.message
    assert result.parameters["mu"] == pytest.approx(0.3, rel=1e-6)
    assert result.parameters["sigma"] == pytest.approx(0.7, rel=1e-6)
