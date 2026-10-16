"""Binned least-squares fits for canonical one-dimensional histograms.

No assumption of Poisson statistics is made. The caller explicitly selects
ordinary least squares or the variances retained by hist.Hist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import hist
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

FIT_MODELS: dict[str, tuple[str, ...]] = {
    "Gaussian": ("A", "mu", "sigma"),
    "Exponential": ("A", "k"),
    "Crystal Ball": ("A", "mu", "sigma", "alpha", "n"),
    "Double-sided Crystal Ball": ("A", "mu", "sigma", "alpha_L", "n_L", "alpha_R", "n_R"),
}


@dataclass(frozen=True)
class FitConfig:
    """Serializable settings for one histogram fit.

    Missing initial values and bounds use model-dependent defaults.
    use_variances=True explicitly requests variance-weighted least squares.
    """

    model: str
    fit_range: tuple[float | None, float | None] = (None, None)
    initial_values: dict[str, float] | None = None
    bounds: dict[str, tuple[float | None, float | None]] | None = None
    use_variances: bool = False


@dataclass(frozen=True)
class FitResult:
    """Fit parameters, quality indicators and nonfatal diagnostics."""

    model: str
    success: bool
    parameters: dict[str, float]
    uncertainties: dict[str, float | None]
    statistic: float | None
    ndof: int | None
    bins_used: int
    bins_excluded: int
    objective: str
    fit_range: tuple[float, float] | None
    message: str
    reference_x: float | None = None


def _core_integral(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(np.pi / 2) * (erf(right / np.sqrt(2)) - erf(left / np.sqrt(2)))


def _crystal_ball_integral(
    left: np.ndarray,
    right: np.ndarray,
    alpha_left: float,
    n_left: float,
    alpha_right: float,
    n_right: float,
) -> np.ndarray:
    """Integrate an unnormalised double-sided Crystal Ball in standardised x.

    The tail formulae are anchored at the Gaussian transitions, avoiding
    overflow from separately forming the conventional (n / alpha)**n factor.
    """
    left_upper = np.minimum(right, -alpha_left)
    left_lower = np.minimum(left, -alpha_left)
    d_left = n_left / alpha_left
    b_left = d_left - alpha_left
    left_area = (
        np.exp(-0.5 * alpha_left**2)
        * d_left
        / (n_left - 1)
        * (
            ((b_left - left_upper) / d_left) ** (1 - n_left)
            - ((b_left - left_lower) / d_left) ** (1 - n_left)
        )
    )

    core_lower = np.clip(left, -alpha_left, alpha_right)
    core_upper = np.clip(right, -alpha_left, alpha_right)
    core_area = _core_integral(core_lower, core_upper)

    right_lower = np.maximum(left, alpha_right)
    right_upper = np.maximum(right, alpha_right)
    d_right = n_right / alpha_right
    b_right = d_right - alpha_right
    right_area = (
        np.exp(-0.5 * alpha_right**2)
        * d_right
        / (n_right - 1)
        * (
            ((b_right + right_lower) / d_right) ** (1 - n_right)
            - ((b_right + right_upper) / d_right) ** (1 - n_right)
        )
    )
    return left_area + core_area + right_area


def model_density(model: str, x: np.ndarray, parameters: dict[str, float], *, x0: float) -> np.ndarray:
    """Evaluate an unnormalised density for display at arbitrary coordinates."""
    x = np.asarray(x, dtype=float)
    amplitude = parameters["A"]
    if model == "Exponential":
        return amplitude * np.exp(parameters["k"] * (x - x0))

    t = (x - parameters["mu"]) / parameters["sigma"]
    core = np.exp(-0.5 * t**2)
    if model == "Gaussian":
        return amplitude * core
    if model == "Crystal Ball":
        alpha, n = parameters["alpha"], parameters["n"]
        d = n / alpha
        tail = np.exp(-alpha**2 / 2) * (d / (d - alpha - np.minimum(t, -alpha))) ** n
        return amplitude * np.where(t < -alpha, tail, core)
    if model == "Double-sided Crystal Ball":
        al, nl = parameters["alpha_L"], parameters["n_L"]
        ar, nr = parameters["alpha_R"], parameters["n_R"]
        dl, dr = nl / al, nr / ar
        left = np.exp(-al**2 / 2) * (dl / (dl - al - np.minimum(t, -al))) ** nl
        right = np.exp(-ar**2 / 2) * (dr / (dr - ar + np.maximum(t, ar))) ** nr
        return amplitude * np.where(t < -al, left, np.where(t > ar, right, core))
    raise ValueError(f"Unsupported fit model: {model}")


def model_bin_integrals(
    model: str,
    edges: np.ndarray,
    parameters: dict[str, float],
    *,
    x0: float,
) -> np.ndarray:
    """Predict exact bin contents for the four supported built-in models."""
    edges = np.asarray(edges, dtype=float)
    if model not in FIT_MODELS:
        raise ValueError(f"Unsupported fit model: {model}")
    left, right = edges[:-1], edges[1:]
    amplitude = parameters["A"]
    if model == "Exponential":
        k = parameters["k"]
        if abs(k) < 1e-10:
            return amplitude * (right - left)
        return amplitude * np.exp(k * (left - x0)) * np.expm1(k * (right - left)) / k

    mu, sigma = parameters["mu"], parameters["sigma"]
    t_left, t_right = (left - mu) / sigma, (right - mu) / sigma
    if model == "Gaussian":
        areas = _core_integral(t_left, t_right)
    elif model == "Crystal Ball":
        areas = _crystal_ball_integral(
            t_left, t_right, parameters["alpha"], parameters["n"], 1e6, 2.0
        )
    else:
        areas = _crystal_ball_integral(
            t_left, t_right, parameters["alpha_L"], parameters["n_L"],
            parameters["alpha_R"], parameters["n_R"],
        )
    return amplitude * sigma * areas


def _initial_guess(
    model: str, bin_edges: np.ndarray, values: np.ndarray, *, x0: float
) -> dict[str, float]:
    """Estimate parameters from the *selected* bins, never excluded outliers."""
    centers = 0.5 * (bin_edges[:, 0] + bin_edges[:, 1])
    widths = bin_edges[:, 1] - bin_edges[:, 0]
    positive = np.maximum(values, 0)
    scale = max(float(bin_edges[:, 1].max() - bin_edges[:, 0].min()), float(np.min(widths)))
    if positive.sum() > 0:
        mu = float(np.average(centers, weights=positive))
        sigma = float(np.sqrt(np.average((centers - mu) ** 2, weights=positive)))
    else:
        mu, sigma = float(np.mean(centers)), scale / 6
    sigma = max(sigma, scale / 100, float(np.min(widths)) / 2)
    density = values / widths
    amplitude = max(float(np.max(density)), 1e-6)

    if model == "Exponential":
        positive_bins = density > 0
        k = 0.0
        if np.count_nonzero(positive_bins) >= 2:
            try:
                k = float(np.polyfit(centers[positive_bins], np.log(density[positive_bins]), 1)[0])
            except (ValueError, np.linalg.LinAlgError):
                pass
        k = float(np.clip(k, -20 / scale, 20 / scale))
        intercepts = np.log(density[positive_bins]) - k * (centers[positive_bins] - x0)
        exponential_amplitude = float(np.exp(np.median(intercepts))) if intercepts.size else amplitude
        return {"A": max(exponential_amplitude, 1e-6), "k": k}

    guessed = {"A": amplitude, "mu": mu, "sigma": sigma}
    if model == "Crystal Ball":
        guessed.update(alpha=1.5, n=3.0)
    elif model == "Double-sided Crystal Ball":
        guessed.update(alpha_L=1.5, n_L=3.0, alpha_R=1.5, n_R=3.0)
    return guessed


def _default_bounds(name: str, edges: np.ndarray) -> tuple[float, float]:
    if name == "A":
        return (0.0, np.inf)
    if name == "sigma":
        return (max(float(np.ptp(edges)) * 1e-6, 1e-12), np.inf)
    if name.startswith("alpha"):
        return (1e-3, 100.0)
    if name.startswith("n"):
        return (1.0001, 1e4)
    return (-np.inf, np.inf)


def fit_histogram_1d(histogram: hist.Hist, config: FitConfig) -> FitResult:
    """Fit an existing histogram without rebinning or invoking Streamlit.

    Range selection includes full bins whose centres are inside the range.
    With use_variances=True, only bins with finite positive stored variances
    contribute; excluded bins are reported in FitResult.
    """
    objective = "Variance-weighted least squares" if config.use_variances else "Ordinary least squares"

    def failed(message: str, *, bins_used: int = 0, bins_excluded: int = 0) -> FitResult:
        return FitResult(config.model, False, {}, {}, None, None, bins_used, bins_excluded, objective, None, message)

    if config.model not in FIT_MODELS:
        return failed(f"Unknown fit model: {config.model}")
    if histogram.ndim != 1:
        return failed("Fitting is supported for 1D histograms only.")
    edges = np.asarray(histogram.axes[0].edges, dtype=float)
    values = np.asarray(histogram.values(), dtype=float)
    if len(values) == 0 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        return failed("The histogram does not contain valid bin edges.")
    if not np.all(np.isfinite(values)):
        return failed("Histogram bin contents must be finite.")
    lower = float(edges[0] if config.fit_range[0] is None else config.fit_range[0])
    upper = float(edges[-1] if config.fit_range[1] is None else config.fit_range[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return failed("Fit range must have finite bounds with minimum < maximum.")
    centers = (edges[:-1] + edges[1:]) / 2
    selected = (centers >= lower) & (centers <= upper)
    excluded = 0
    if config.use_variances:
        if histogram.storage_type is not hist.storage.Weight:
            return failed("Explicit stored sum(w²) variances are unavailable for this histogram.")
        variances = histogram.variances()
        if variances is None:
            return failed("This histogram has no stored bin variances.")
        variances = np.asarray(variances, dtype=float)
        usable = np.isfinite(variances) & (variances > 0)
        excluded = int(np.count_nonzero(selected & ~usable))
        selected &= usable

    indices = np.flatnonzero(selected)
    nparams = len(FIT_MODELS[config.model])
    if len(indices) <= nparams:
        return failed(
            f"Insufficient usable bins: {len(indices)} for {nparams} free parameters. "
            "Widen the fit range or reduce the number of parameters.",
            bins_used=len(indices),
            bins_excluded=excluded,
        )
    # Selection may contain gaps from excluded zero-variance bins. Evaluate
    # each original pair of bin edges, never inventing contiguous rebinned data.
    fit_left, fit_right = edges[indices], edges[indices + 1]
    selected_values = values[indices]
    # Vectorised evaluation: each pair is represented by a two-element array;
    # keeping the original boundaries avoids rebinning gaps.
    fit_edges = np.column_stack((fit_left, fit_right))
    x0 = float((lower + upper) / 2)
    guesses = _initial_guess(config.model, fit_edges, selected_values, x0=x0)
    allowed = set(FIT_MODELS[config.model])
    if config.initial_values and not set(config.initial_values).issubset(allowed):
        return failed("Unknown initial-value parameter.", bins_used=len(indices), bins_excluded=excluded)
    if config.bounds and not set(config.bounds).issubset(allowed):
        return failed("Unknown parameter bound.", bins_used=len(indices), bins_excluded=excluded)

    guess_values: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for name in FIT_MODELS[config.model]:
        low, high = _default_bounds(name, edges)
        if config.bounds and name in config.bounds:
            override_low, override_high = config.bounds[name]
            low = -np.inf if override_low is None else float(override_low)
            high = np.inf if override_high is None else float(override_high)
        value = float((config.initial_values or {}).get(name, guesses[name]))
        if not np.isfinite(value) or np.isnan(low) or np.isnan(high) or low >= high:
            return failed(f"Invalid initial value or bounds for {name}.", bins_used=len(indices), bins_excluded=excluded)
        if value < low or value > high:
            if config.initial_values and name in config.initial_values:
                return failed(f"Initial value for {name} lies outside its bounds.", bins_used=len(indices), bins_excluded=excluded)
            if np.isfinite(low) and np.isfinite(high):
                value = (low + high) / 2
            elif np.isfinite(low):
                value = low + max(1e-3, abs(low) * 0.01)
            else:
                value = high - max(1e-3, abs(high) * 0.01)
        # scipy.optimize.least_squares requires strict interior for typical models.
        if value == low:
            value = float(np.nextafter(low, np.inf))
        if value == high:
            value = float(np.nextafter(high, -np.inf))
        guess_values.append(value)
        lower_bounds.append(low)
        upper_bounds.append(high)

    sigma = np.sqrt(np.asarray(histogram.variances(), dtype=float)[indices]) if config.use_variances else np.ones(len(indices))

    def residual(params: np.ndarray) -> np.ndarray:
        params_dict = dict(zip(FIT_MODELS[config.model], params, strict=True))
        # Evaluate all selected original bins in one vectorised call; this also
        # works when zero-variance bins create gaps in the selection.
        left = fit_edges[:, 0]
        right = fit_edges[:, 1]
        if config.model == "Exponential":
            k = params_dict["k"]
            predictions = (params_dict["A"] * (right - left) if abs(k) < 1e-10 else
                           params_dict["A"] * np.exp(k * (left - x0)) * np.expm1(k * (right - left)) / k)
        else:
            # Two-endpoint arrays are integrated independently, without
            # interpreting the gaps between selected bins as additional bins.
            integral_left = model_bin_integrals(
                config.model, np.concatenate((left, [right[-1]])), params_dict, x0=x0
            )
            if np.any(right[:-1] != left[1:]):
                integral_left = np.array([
                    model_bin_integrals(config.model, pair, params_dict, x0=x0)[0]
                    for pair in fit_edges
                ])
            predictions = integral_left
        result = (predictions - selected_values) / sigma
        return np.nan_to_num(result, nan=1e100, posinf=1e100, neginf=-1e100)

    try:
        solution = least_squares(
            residual, guess_values, bounds=(lower_bounds, upper_bounds), max_nfev=3000
        )
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        return failed(f"Fit failed: {exc}", bins_used=len(indices), bins_excluded=excluded)

    if not solution.success or not np.all(np.isfinite(solution.x)):
        return failed(f"Fit did not converge: {solution.message}", bins_used=len(indices), bins_excluded=excluded)

    parameters = dict(zip(FIT_MODELS[config.model], (float(x) for x in solution.x), strict=True))
    statistic = float(np.sum(solution.fun**2))
    ndof = len(indices) - nparams
    uncertainties: dict[str, float | None] = dict.fromkeys(parameters)
    notes = []
    try:
        jac = np.asarray(solution.jac)
        if np.linalg.matrix_rank(jac) == nparams:
            inverse = np.linalg.inv(jac.T @ jac)
            scale = 1.0 if config.use_variances else statistic / ndof
            diagonal = np.diag(inverse) * scale
            if np.all(np.isfinite(diagonal)) and np.all(diagonal >= 0):
                uncertainties = dict(zip(parameters, np.sqrt(diagonal).tolist(), strict=True))
            else:
                notes.append("Covariance is not finite.")
        else:
            notes.append("Covariance is singular; uncertainties are unavailable.")
    except np.linalg.LinAlgError:
        notes.append("Covariance is singular; uncertainties are unavailable.")

    if excluded:
        notes.append(f"{excluded} bin(s) excluded because stored variances were not positive and finite.")
    return FitResult(
        config.model, True, parameters, uncertainties, statistic, ndof, len(indices), excluded,
        objective, (float(fit_left[0]), float(fit_right[-1])), "Fit converged. " + " ".join(notes),
        reference_x=x0,
    )
