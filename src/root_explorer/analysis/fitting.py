"""Binned least-squares fits for canonical one-dimensional histograms.

The fit itself assumes no Poisson statistics. The caller explicitly selects
ordinary least squares, the variances retained by hist.Hist, or explicit
variances such as ROOT's sqrt(N) errors of a persisted histogram without Sumw2.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

import hist
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

from .fit_expressions import discover_fit_parameters, evaluate_fit_expression

FIT_MODELS: dict[str, tuple[str, ...]] = {
    "Gaussian": ("A", "mu", "sigma"),
    "Exponential": ("A", "k"),
    "Crystal Ball": ("A", "mu", "sigma", "alpha", "n"),
    "Double-sided Crystal Ball": ("A", "mu", "sigma", "alpha_L", "n_L", "alpha_R", "n_R"),
}

# Smallest Jacobian singular value, relative to the largest, still treated as
# identifiable; set above the sqrt(eps) noise of two-point finite differences.
_SINGULAR_RTOL = 10 * np.sqrt(np.finfo(float).eps)


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
    expression: str | None = None


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
    expression: str | None = None


def _core_integral(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.sqrt(np.pi / 2) * (erf(right / np.sqrt(2)) - erf(left / np.sqrt(2))), dtype=float
    )


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
    return np.asarray(left_area + core_area + right_area, dtype=float)


def model_density(
    model: str,
    x: np.ndarray,
    parameters: dict[str, float],
    *,
    x0: float,
    expression: str | None = None,
) -> np.ndarray:
    """Evaluate an unnormalised density for display at arbitrary coordinates."""
    x = np.asarray(x, dtype=float)
    if model == "Custom expression":
        if expression is None:
            raise ValueError("Missing custom fit expression.")
        return evaluate_fit_expression(expression, x, parameters)
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
        tail = np.exp(-(alpha**2) / 2) * (d / (d - alpha - np.minimum(t, -alpha))) ** n
        return amplitude * np.where(t < -alpha, tail, core)
    if model == "Double-sided Crystal Ball":
        al, nl = parameters["alpha_L"], parameters["n_L"]
        ar, nr = parameters["alpha_R"], parameters["n_R"]
        dl, dr = nl / al, nr / ar
        left = np.exp(-(al**2) / 2) * (dl / (dl - al - np.minimum(t, -al))) ** nl
        right = np.exp(-(ar**2) / 2) * (dr / (dr - ar + np.maximum(t, ar))) ** nr
        return amplitude * np.where(t < -al, left, np.where(t > ar, right, core))
    raise ValueError(f"Unsupported fit model: {model}")


def _exponential_bin_integrals(
    left: np.ndarray, right: np.ndarray, amplitude: float, k: float, x0: float
) -> np.ndarray:
    """Integrate without treating a dimensionful small slope as exactly zero."""
    widths = right - left
    exponent = k * widths
    factor = np.divide(
        np.expm1(exponent), exponent, out=np.ones_like(exponent), where=exponent != 0
    )
    return np.asarray(amplitude * np.exp(k * (left - x0)) * widths * factor, dtype=float)


def model_bin_integrals(
    model: str,
    edges: np.ndarray,
    parameters: dict[str, float],
    *,
    x0: float,
    expression: str | None = None,
) -> np.ndarray:
    """Predict bin contents analytically or by quadrature for custom expressions."""
    edges = np.asarray(edges, dtype=float)
    if model == "Custom expression":
        if expression is None:
            raise ValueError("Missing custom fit expression.")
        # Integrate each original bin independently, including selections with gaps.
        nodes, weights = np.polynomial.legendre.leggauss(32)
        mid = (edges[:-1] + edges[1:]) / 2
        half = (edges[1:] - edges[:-1]) / 2
        coords = mid[:, None] + half[:, None] * nodes[None, :]
        values = evaluate_fit_expression(expression, coords, parameters)
        return np.asarray(half * np.sum(values * weights[None, :], axis=1), dtype=float)
    if model not in FIT_MODELS:
        raise ValueError(f"Unsupported fit model: {model}")
    left, right = edges[:-1], edges[1:]
    amplitude = parameters["A"]
    if model == "Exponential":
        return _exponential_bin_integrals(left, right, amplitude, parameters["k"], x0)

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
            t_left,
            t_right,
            parameters["alpha_L"],
            parameters["n_L"],
            parameters["alpha_R"],
            parameters["n_R"],
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
    # Positive, but without an absolute floor: a floor such as 1e-6 would start fits
    # to very small yields far from the data (#199).
    peak_density = float(np.max(density))
    amplitude = peak_density if peak_density > 0 else 1.0

    if model == "Exponential":
        positive_bins = density > 0
        k = 0.0
        if np.count_nonzero(positive_bins) >= 2:
            with suppress(ValueError, np.linalg.LinAlgError):
                k = float(np.polyfit(centers[positive_bins], np.log(density[positive_bins]), 1)[0])
        k = float(np.clip(k, -20 / scale, 20 / scale))
        intercepts = np.log(density[positive_bins]) - k * (centers[positive_bins] - x0)
        exponential_amplitude = (
            float(np.exp(np.median(intercepts))) if intercepts.size else amplitude
        )
        return {"A": exponential_amplitude if exponential_amplitude > 0 else amplitude, "k": k}

    guessed = {"A": amplitude, "mu": mu, "sigma": sigma}
    if model == "Crystal Ball":
        guessed.update(alpha=1.5, n=3.0)
    elif model == "Double-sided Crystal Ball":
        guessed.update(alpha_L=1.5, n_L=3.0, alpha_R=1.5, n_R=3.0)
    return guessed


def _parameter_scale(
    name: str, value: float, fit_range: tuple[float, float], *, custom: bool
) -> float:
    """Characteristic size of a parameter, used to solve in unit-free variables.

    The fit is solved for (p - p0) / scale, so finite-difference steps and the
    termination tests follow the parameter's physical scale instead of absolute
    numbers in the user's units (e.g. a width of 1e-9 when x is in seconds).
    """
    half_range = 0.5 * (fit_range[1] - fit_range[0])
    if custom:
        # A custom parameter starting inside the fit range, far from zero relative to
        # the range (a peak position at 1e6 in a range of width 0.1), is a position on
        # the x axis, which varies on the scale of the range rather than of |p|.
        if fit_range[0] <= value <= fit_range[1] and abs(value) > half_range:
            return half_range
        return abs(value) if value != 0 else 1.0
    if name == "mu":
        return half_range
    if name == "k":
        return 1.0 / half_range
    if name in ("A", "sigma") and value != 0:
        return abs(value)
    return max(1.0, abs(value))


def _default_bounds(name: str, edges: np.ndarray) -> tuple[float, float]:
    if name == "A":
        return (0.0, np.inf)
    if name == "sigma":
        # Relative to the histogram range only, so the bound does not depend on units.
        return (float(np.ptp(edges)) * 1e-6, np.inf)
    if name.startswith("alpha"):
        return (1e-3, 100.0)
    if name.startswith("n"):
        return (1.0001, 1e4)
    return (-np.inf, np.inf)


def fit_histogram_1d(
    histogram: hist.Hist,
    config: FitConfig,
    *,
    variances: np.ndarray | None = None,
    count_errors: bool = False,
) -> FitResult:
    """Fit an existing histogram without rebinning or invoking Streamlit.

    Range selection includes full bins whose centres are inside the range.
    With use_variances=True, only bins with finite positive stored variances
    contribute; excluded bins are reported in FitResult. Explicit visible-bin
    `variances` replace the storage variances, e.g. ROOT's sqrt(N) errors of a
    persisted histogram without Sumw2, so the fit uses the errors drawn.
    `count_errors` marks such sqrt(N) variances, which are only meaningful for
    unweighted counts, so the result can say that they were assumed.
    """
    count_errors = count_errors and config.use_variances
    objective = (
        "Variance-weighted least squares" if config.use_variances else "Ordinary least squares"
    )
    if count_errors:
        objective += " with √N errors"

    def failed(message: str, *, bins_used: int = 0, bins_excluded: int = 0) -> FitResult:
        return FitResult(
            config.model,
            False,
            {},
            {},
            None,
            None,
            bins_used,
            bins_excluded,
            objective,
            None,
            message,
        )

    expression = config.expression if config.model == "Custom expression" else None
    if config.model == "Custom expression":
        try:
            names = discover_fit_parameters(expression or "")
        except ValueError as exc:
            return failed(f"Invalid custom fit expression: {exc}")
    elif config.model in FIT_MODELS:
        if config.expression is not None:
            return failed("Custom expression is only supported for the Custom expression model.")
        names = FIT_MODELS[config.model]
    else:
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
    bin_variances = np.ones(len(values))
    if config.use_variances:
        if variances is None:
            # Unweighted event histograms carry no explicit sum(w²); hist would
            # otherwise silently substitute Poisson variances equal to the counts.
            if histogram.storage_type is not hist.storage.Weight:
                return failed(
                    "Explicit stored sum(w²) variances are unavailable for this histogram."
                )
            variances = histogram.variances()
            if variances is None:
                return failed("This histogram has no stored bin variances.")
        bin_variances = np.asarray(variances, dtype=float)
        if bin_variances.shape != values.shape:
            return failed("Bin variances do not match the histogram bins.")
        usable = np.isfinite(bin_variances) & (bin_variances > 0)
        excluded = int(np.count_nonzero(selected & ~usable))
        selected &= usable

    indices = np.flatnonzero(selected)
    nparams = len(names)
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
    guesses = (
        dict.fromkeys(names, 1.0)
        if expression is not None
        else _initial_guess(config.model, fit_edges, selected_values, x0=x0)
    )
    allowed = set(names)
    if config.initial_values and not set(config.initial_values).issubset(allowed):
        return failed(
            "Unknown initial-value parameter.", bins_used=len(indices), bins_excluded=excluded
        )
    if config.bounds and not set(config.bounds).issubset(allowed):
        return failed("Unknown parameter bound.", bins_used=len(indices), bins_excluded=excluded)

    guess_values: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for name in names:
        low, high = (-np.inf, np.inf) if expression is not None else _default_bounds(name, edges)
        if config.bounds and name in config.bounds:
            # A one-sided override keeps the default on the side left unset.
            override_low, override_high = config.bounds[name]
            low = low if override_low is None else float(override_low)
            high = high if override_high is None else float(override_high)
        value = float((config.initial_values or {}).get(name, guesses[name]))
        if not np.isnan(low) and not np.isnan(high) and low >= high:
            return failed(
                f"Bounds for {name} are empty: lower {low:g} is not below upper {high:g}. "
                "A blank bound keeps the model's default; enter inf or -inf to remove it.",
                bins_used=len(indices),
                bins_excluded=excluded,
            )
        if not np.isfinite(value) or np.isnan(low) or np.isnan(high):
            return failed(
                f"Invalid initial value or bounds for {name}.",
                bins_used=len(indices),
                bins_excluded=excluded,
            )
        if value < low or value > high:
            if config.initial_values and name in config.initial_values:
                return failed(
                    f"Initial value for {name} lies outside its bounds.",
                    bins_used=len(indices),
                    bins_excluded=excluded,
                )
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

    sigma = np.sqrt(bin_variances[indices])

    def residual(params: np.ndarray) -> np.ndarray:
        params_dict = dict(zip(names, params, strict=True))
        # Evaluate all selected original bins in one vectorised call; this also
        # works when zero-variance bins create gaps in the selection.
        left = fit_edges[:, 0]
        right = fit_edges[:, 1]
        if config.model == "Exponential":
            predictions = _exponential_bin_integrals(
                left, right, params_dict["A"], params_dict["k"], x0
            )
        else:
            # Two-endpoint arrays are integrated independently, without
            # interpreting the gaps between selected bins as additional bins.
            integral_left = model_bin_integrals(
                config.model,
                np.concatenate((left, [right[-1]])),
                params_dict,
                x0=x0,
                expression=expression,
            )
            if np.any(right[:-1] != left[1:]):
                integral_left = np.array(
                    [
                        model_bin_integrals(
                            config.model, pair, params_dict, x0=x0, expression=expression
                        )[0]
                        for pair in fit_edges
                    ]
                )
            predictions = integral_left
        return np.asarray((predictions - selected_values) / sigma, dtype=float)

    # Solve for u = (p - p0) / s with a characteristic scale s per parameter, so the
    # result does not depend on the units of x (see _parameter_scale). The residual
    # and custom expressions still receive physical parameters and physical x.
    start = np.asarray(guess_values, dtype=float)
    low_array = np.asarray(lower_bounds, dtype=float)
    high_array = np.asarray(upper_bounds, dtype=float)
    fitted_range = (float(fit_left[0]), float(fit_right[-1]))
    natural_scales = np.array(
        [
            _parameter_scale(name, value, fitted_range, custom=expression is not None)
            for name, value in zip(names, start, strict=True)
        ]
    )
    # A parameter far from zero relative to its scale (a peak at 1e6 with width 0.01) is
    # only resolved to eps * |p|, so a finite-difference step of sqrt(eps) * scale would
    # span just a few rounding units. Widening the variable's unit to sqrt(|p| * scale)
    # balances rounding against curvature in the steps; x_scale restores the natural
    # scale for the solver's trust region, so only the finite differences change.
    scales = natural_scales * np.sqrt(np.maximum(1.0, np.abs(start) / natural_scales))

    def physical(u: np.ndarray) -> np.ndarray:
        # Rounding in start + scales * u must not step outside the bounds.
        return np.asarray(np.clip(start + scales * u, low_array, high_array), dtype=float)

    # least_squares stops when the gradient of the cost falls below an absolute
    # tolerance. For small yields (a normalised histogram, small weights) the residuals
    # and their gradient are tiny from the start, so it would stop before the shape is
    # resolved (#199). Optimise residuals divided by the typical size of the data in
    # residual units instead: scaling all contents by c then leaves the optimisation
    # unchanged. The statistic and covariance below are converted back.
    residual_scale = float(np.sqrt(np.mean((selected_values / sigma) ** 2)))
    if not np.isfinite(residual_scale) or residual_scale <= 0:
        residual_scale = 1.0

    def scaled_residual(u: np.ndarray) -> np.ndarray:
        result = residual(physical(u)) / residual_scale
        # Undefined model values become a huge, finite penalty; applied after scaling
        # so that the penalty cannot shrink for large data.
        return np.asarray(
            np.nan_to_num(result, nan=1e100, posinf=1e100, neginf=-1e100), dtype=float
        )

    scaled_low = np.asarray((low_array - start) / scales, dtype=float)
    scaled_high = np.asarray((high_array - start) / scales, dtype=float)
    try:
        solution = least_squares(
            scaled_residual,
            np.zeros(nparams),
            bounds=(scaled_low, scaled_high),
            # Not "jac": scaling by the Jacobian lets weakly constrained parameters (a
            # large Crystal Ball n) take huge steps into flat regions of the chi-square.
            x_scale=natural_scales / scales,
            max_nfev=3000,
        )
    except (ValueError, RuntimeError, FloatingPointError, OverflowError, TypeError) as exc:
        return failed(f"Fit failed: {exc}", bins_used=len(indices), bins_excluded=excluded)

    if not solution.success or not np.all(np.isfinite(solution.x)):
        return failed(
            f"Fit did not converge: {solution.message}",
            bins_used=len(indices),
            bins_excluded=excluded,
        )

    parameters = dict(zip(names, (float(x) for x in physical(solution.x)), strict=True))
    statistic = residual_scale**2 * float(np.sum(solution.fun**2))
    if not np.isfinite(statistic) or float(np.sum(solution.fun**2)) > 1e150:
        return failed(
            "Model is undefined or numerically unstable over the selected bins.",
            bins_used=len(indices),
            bins_excluded=excluded,
        )
    ndof = len(indices) - nparams
    uncertainties: dict[str, float | None] = dict.fromkeys(parameters)
    notes = []
    try:
        # Normalise the Jacobian columns to unit length so that the rank test does not
        # depend on units; the physical column norms are divided out again below.
        jacobian = np.asarray(solution.jac, dtype=float)
        scaled_norms = np.linalg.norm(jacobian, axis=0)
        if not np.all(scaled_norms > 0):
            raise np.linalg.LinAlgError("A parameter does not affect the model.")
        # Undo both the parameter scaling and the residual normalisation.
        physical_norms = scaled_norms * residual_scale / scales
        _, singular_values, vt = np.linalg.svd(jacobian / scaled_norms, full_matrices=False)
        # The finite-difference Jacobian carries relative noise of order sqrt(eps), so
        # numpy's default rank tolerance would mistake that noise for an identifiable
        # direction and yield arbitrary or negative variances.
        if len(singular_values) == nparams and singular_values[-1] > (
            _SINGULAR_RTOL * singular_values[0]
        ):
            scale = 1.0 if config.use_variances else statistic / ndof
            diagonal = (
                np.sum((vt / singular_values[:, None]) ** 2, axis=0) / physical_norms**2 * scale
            )
            if np.all(np.isfinite(diagonal)) and np.all(diagonal >= 0):
                uncertainties = dict(zip(parameters, np.sqrt(diagonal).tolist(), strict=True))
            else:
                notes.append("Covariance is singular; uncertainties are unavailable.")
        else:
            notes.append("Covariance is singular; uncertainties are unavailable.")
    except np.linalg.LinAlgError:
        notes.append("Covariance is singular; uncertainties are unavailable.")

    if count_errors:
        notes.append(
            "The histogram stores no Sumw2, so ROOT's default bin errors √N of the bin "
            "contents are assumed. They are only meaningful if the bins hold unweighted counts."
        )
    if excluded:
        reason = (
            "their content is not positive, so √N gives no usable error"
            if count_errors
            else "stored variances were not positive and finite"
        )
        notes.append(f"{excluded} bin(s) excluded because {reason}.")
    return FitResult(
        config.model,
        True,
        parameters,
        uncertainties,
        statistic,
        ndof,
        len(indices),
        excluded,
        objective,
        (float(fit_left[0]), float(fit_right[-1])),
        "Fit converged. " + " ".join(notes),
        reference_x=x0,
        expression=expression,
    )
