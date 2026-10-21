"""Evaluation of direct, reduced, and derived variable specifications."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..models import VariableSpec
from .arrays import (
    branch_values,
    delta_phi,
    event_to_numpy,
    multiplicity_per_event,
    nth_per_event,
    read_aligned_branch_values,
    read_for_scope,
    reduce_per_event,
    safe_float_array,
)

VariableEvaluator = Callable[
    [VariableSpec, str, str, int, int | None, int | None], tuple[np.ndarray, str]
]


def evaluate_branch_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate one branch as flat plot values."""

    branch = _required_branch(spec)
    return safe_float_array(
        branch_values(path, tree_name, branch, entry_start, entry_stop, event_index)
    ), branch


def evaluate_multiplicity_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate the number of values in one branch per event."""

    branch = _required_branch(spec)
    arr = read_for_scope(path, tree_name, branch, entry_start, entry_stop, event_index)
    if event_index is None:
        return multiplicity_per_event(arr), f"N({branch})"
    return np.array([len(event_to_numpy(arr))], dtype=float), f"N({branch})"


def evaluate_nth_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate one indexed value per event."""

    branch = _required_branch(spec)
    arr = read_for_scope(path, tree_name, branch, entry_start, entry_stop, event_index)
    label = f"{branch}[{spec.index}]"
    if event_index is None:
        return nth_per_event(arr, spec.index), label
    values = np.asarray(event_to_numpy(arr)).reshape(-1)
    value = values[spec.index] if len(values) > spec.index else np.nan
    return np.asarray([value], dtype=float), label


def evaluate_reduced_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate a sum, mean, minimum, or maximum per event."""

    if spec.mode not in {"sum", "mean", "max", "min"}:
        raise ValueError(f"Unsupported reduction mode: {spec.mode}")
    branch = _required_branch(spec)
    arr = read_for_scope(path, tree_name, branch, entry_start, entry_stop, event_index)
    label = f"{spec.mode}({branch})"
    if event_index is None:
        return reduce_per_event(arr, spec.mode), label

    values = safe_float_array(event_to_numpy(arr))
    if len(values) == 0:
        empty_value = 0.0 if spec.mode == "sum" else np.nan
        return np.array([empty_value], dtype=float), label
    if spec.mode == "sum":
        reduced = np.nansum(values)
    elif spec.mode == "mean":
        reduced = np.nanmean(values)
    elif spec.mode == "max":
        reduced = np.nanmax(values)
    else:
        reduced = np.nanmin(values)
    return np.array([reduced], dtype=float), label


def evaluate_absolute_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate the absolute value of one branch."""

    (a,), branches = _read_derived_values(
        spec, 1, path, tree_name, entry_start, entry_stop, event_index
    )
    return np.abs(a), f"|{branches[0]}|"


def evaluate_r_xy_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate transverse radius from two branches."""

    (a, b), branches = _read_derived_values(
        spec, 2, path, tree_name, entry_start, entry_stop, event_index
    )
    return np.sqrt(a * a + b * b), f"r({branches[0]},{branches[1]})"


def evaluate_phi_xy_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate azimuth from x/y coordinates."""

    (a, b), branches = _read_derived_values(
        spec, 2, path, tree_name, entry_start, entry_stop, event_index
    )
    return np.arctan2(b, a), f"phi({branches[0]},{branches[1]})"


def evaluate_eta_xyz_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate pseudorapidity from Cartesian coordinates."""

    (a, b, c), branches = _read_derived_values(
        spec, 3, path, tree_name, entry_start, entry_stop, event_index
    )
    r = np.sqrt(a * a + b * b)
    out = np.full(len(r), np.nan, dtype=float)
    mask = np.isfinite(r) & np.isfinite(c) & (r > 0)
    out[mask] = np.arcsinh(c[mask] / r[mask])
    return out, f"eta({branches[0]},{branches[1]},{branches[2]})"


def evaluate_magnitude_3d_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate a three-dimensional magnitude."""

    (a, b, c), branches = _read_derived_values(
        spec, 3, path, tree_name, entry_start, entry_stop, event_index
    )
    return np.hypot(np.hypot(a, b), c), f"mag({branches[0]},{branches[1]},{branches[2]})"


def evaluate_energy_massless_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate massless energy from transverse momentum and eta/y."""

    _validate_longitudinal_kind(spec)
    (a, b), branches = _read_derived_values(
        spec, 2, path, tree_name, entry_start, entry_stop, event_index
    )
    return a * np.cosh(b), f"E_massless({branches[0]},{spec.longitudinal_kind}={branches[1]})"


def evaluate_energy_massive_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate massive energy from transverse momentum, eta/y, and mass."""

    _validate_longitudinal_kind(spec)
    (a, b, c), branches = _read_derived_values(
        spec, 3, path, tree_name, entry_start, entry_stop, event_index
    )
    if spec.longitudinal_kind == "eta":
        energy = np.sqrt(c * c + (a * np.cosh(b)) ** 2)
    else:
        energy = np.hypot(c, a) * np.cosh(b)
    return energy, f"E({branches[0]},{spec.longitudinal_kind}={branches[1]},{branches[2]})"


def evaluate_invariant_mass_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate invariant mass from energy and Cartesian momentum."""

    (a, b, c, d), branches = _read_derived_values(
        spec, 4, path, tree_name, entry_start, entry_stop, event_index
    )
    mass_squared = a * a - b * b - c * c - d * d
    return (
        np.sqrt(np.maximum(mass_squared, 0.0)),
        f"m({branches[0]},{branches[1]},{branches[2]},{branches[3]})",
    )


def evaluate_ratio_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate a safe ratio."""

    (a, b), branches = _read_derived_values(
        spec, 2, path, tree_name, entry_start, entry_stop, event_index
    )
    out = np.full(len(a), np.nan, dtype=float)
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    return out, f"{branches[0]}/{branches[1]}"


def evaluate_difference_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate a difference between two branches."""

    (a, b), branches = _read_derived_values(
        spec, 2, path, tree_name, entry_start, entry_stop, event_index
    )
    return a - b, f"{branches[0]}-{branches[1]}"


def evaluate_sum_ab_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate a sum between two branches."""

    (a, b), branches = _read_derived_values(
        spec, 2, path, tree_name, entry_start, entry_stop, event_index
    )
    return a + b, f"{branches[0]}+{branches[1]}"


def evaluate_product_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate a product between two branches."""

    (a, b), branches = _read_derived_values(
        spec, 2, path, tree_name, entry_start, entry_stop, event_index
    )
    return a * b, f"{branches[0]}*{branches[1]}"


def evaluate_delta_phi_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate wrapped azimuthal separation."""

    (a, b), branches = _read_derived_values(
        spec, 2, path, tree_name, entry_start, entry_stop, event_index
    )
    return delta_phi(a, b), f"Δφ({branches[0]},{branches[1]})"


def evaluate_delta_eta_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate pseudorapidity separation."""

    (a, b), branches = _read_derived_values(
        spec, 2, path, tree_name, entry_start, entry_stop, event_index
    )
    return a - b, f"Δη({branches[0]},{branches[1]})"


def evaluate_delta_r_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate angular separation in eta/phi."""

    (a, b, c, d), branches = _read_derived_values(
        spec, 4, path, tree_name, entry_start, entry_stop, event_index
    )
    return np.sqrt((a - c) ** 2 + delta_phi(b, d) ** 2), (
        f"ΔR({branches[0]},{branches[1]},{branches[2]},{branches[3]})"
    )


VARIABLE_EVALUATORS: dict[str, VariableEvaluator] = {
    "branch": evaluate_branch_variable,
    "multiplicity": evaluate_multiplicity_variable,
    "nth": evaluate_nth_variable,
    "sum": evaluate_reduced_variable,
    "mean": evaluate_reduced_variable,
    "max": evaluate_reduced_variable,
    "min": evaluate_reduced_variable,
    "absolute": evaluate_absolute_variable,
    "magnitude_3d": evaluate_magnitude_3d_variable,
    "r_xy": evaluate_r_xy_variable,
    "phi_xy": evaluate_phi_xy_variable,
    "eta_xyz": evaluate_eta_xyz_variable,
    "energy_massless": evaluate_energy_massless_variable,
    "energy_massive": evaluate_energy_massive_variable,
    "invariant_mass": evaluate_invariant_mass_variable,
    "ratio": evaluate_ratio_variable,
    "difference": evaluate_difference_variable,
    "sum_ab": evaluate_sum_ab_variable,
    "product": evaluate_product_variable,
    "delta_phi": evaluate_delta_phi_variable,
    "delta_eta": evaluate_delta_eta_variable,
    "delta_r": evaluate_delta_r_variable,
}


def variable_evaluator(spec: VariableSpec) -> VariableEvaluator:
    """Resolve the canonical evaluator for one selected variable mode."""

    try:
        return VARIABLE_EVALUATORS[spec.mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported variable mode: {spec.mode}") from exc


def evaluate_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate a variable specification for either a global range or one event."""

    return variable_evaluator(spec)(spec, path, tree_name, entry_start, entry_stop, event_index)


def _required_branch(spec: VariableSpec) -> str:
    if not spec.branch:
        raise ValueError("No field selected.")
    return spec.branch


def _read_derived_values(
    spec: VariableSpec,
    count: int,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None,
) -> tuple[list[np.ndarray], list[str]]:
    if not spec.branch_a:
        raise ValueError("This derived variable needs at least one field.")
    candidates = [spec.branch_a, spec.branch_b, spec.branch_c, spec.branch_d]
    branches = candidates[:count]
    if any(branch is None for branch in branches):
        raise ValueError(f"Derived variable {spec.mode!r} needs {count} fields.")
    resolved = [str(branch) for branch in branches]
    values = read_aligned_branch_values(
        resolved,
        path,
        tree_name,
        entry_start,
        entry_stop,
        event_index,
        context=f"derived variable {spec.mode!r}",
    )
    return values, resolved


def _validate_longitudinal_kind(spec: VariableSpec) -> None:
    if spec.longitudinal_kind not in {"eta", "y"}:
        raise ValueError("Energy longitudinal coordinate must be 'eta' or 'y'.")
