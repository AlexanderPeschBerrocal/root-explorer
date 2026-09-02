"""Evaluation of direct, reduced, and derived variable specifications."""

from __future__ import annotations

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


def evaluate_variable(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None = None,
) -> tuple[np.ndarray, str]:
    """Evaluate a variable specification for either a global range or one event."""

    mode = spec.mode

    if mode == "branch":
        if not spec.branch:
            raise ValueError("No branch selected.")
        return safe_float_array(
            branch_values(path, tree_name, spec.branch, entry_start, entry_stop, event_index)
        ), spec.branch

    if mode == "multiplicity":
        if not spec.branch:
            raise ValueError("No branch selected.")
        arr = read_for_scope(path, tree_name, spec.branch, entry_start, entry_stop, event_index)
        if event_index is None:
            return multiplicity_per_event(arr), f"N({spec.branch})"
        return np.array([len(event_to_numpy(arr))], dtype=float), f"N({spec.branch})"

    if mode == "nth":
        if not spec.branch:
            raise ValueError("No branch selected.")
        arr = read_for_scope(path, tree_name, spec.branch, entry_start, entry_stop, event_index)
        if event_index is None:
            return nth_per_event(arr, spec.index), f"{spec.branch}[{spec.index}]"
        values = event_to_numpy(arr)
        value = values[spec.index] if len(values) > spec.index else np.nan
        return np.asarray([value], dtype=float), f"{spec.branch}[{spec.index}]"

    if mode in {"sum", "mean", "max", "min"}:
        if not spec.branch:
            raise ValueError("No branch selected.")
        arr = read_for_scope(path, tree_name, spec.branch, entry_start, entry_stop, event_index)
        if event_index is None:
            return reduce_per_event(arr, mode), f"{mode}({spec.branch})"
        values = safe_float_array(event_to_numpy(arr))
        if len(values) == 0:
            empty_value = 0.0 if mode == "sum" else np.nan
            return np.array([empty_value], dtype=float), f"{mode}({spec.branch})"
        func = {"sum": np.nansum, "mean": np.nanmean, "max": np.nanmax, "min": np.nanmin}[mode]
        return np.array([func(values)], dtype=float), f"{mode}({spec.branch})"

    if mode in {
        "r_xy",
        "phi_xy",
        "eta_xyz",
        "ratio",
        "difference",
        "sum_ab",
        "product",
        "delta_phi",
        "delta_r",
    }:
        if not spec.branch_a or not spec.branch_b:
            raise ValueError("This derived variable needs at least two branches.")

        branches = [spec.branch_a, spec.branch_b]
        if mode == "eta_xyz":
            if not spec.branch_c:
                raise ValueError("eta needs x, y, and z branches.")
            branches.append(spec.branch_c)
        elif mode == "delta_r":
            if not spec.branch_c or not spec.branch_d:
                raise ValueError("delta-R needs eta1, phi1, eta2, and phi2 branches.")
            branches.extend([spec.branch_c, spec.branch_d])

        aligned_values = read_aligned_branch_values(
            branches,
            path,
            tree_name,
            entry_start,
            entry_stop,
            event_index,
            context=f"derived variable {mode!r}",
        )
        a, b = aligned_values[:2]

        if mode == "r_xy":
            return np.sqrt(a * a + b * b), f"r({spec.branch_a},{spec.branch_b})"
        if mode == "phi_xy":
            return np.arctan2(b, a), f"phi({spec.branch_a},{spec.branch_b})"
        if mode == "ratio":
            out = np.full(len(a), np.nan, dtype=float)
            mask = b != 0
            out[mask] = a[mask] / b[mask]
            return out, f"{spec.branch_a}/{spec.branch_b}"
        if mode == "difference":
            return a - b, f"{spec.branch_a}-{spec.branch_b}"
        if mode == "sum_ab":
            return a + b, f"{spec.branch_a}+{spec.branch_b}"
        if mode == "product":
            return a * b, f"{spec.branch_a}*{spec.branch_b}"
        if mode == "delta_phi":
            return delta_phi(a, b), f"Δφ({spec.branch_a},{spec.branch_b})"
        if mode == "eta_xyz":
            c = aligned_values[2]
            r = np.sqrt(a * a + b * b)
            out = np.full(len(r), np.nan, dtype=float)
            mask = np.isfinite(r) & np.isfinite(c) & (r > 0)
            out[mask] = np.arcsinh(c[mask] / r[mask])
            return out, f"eta({spec.branch_a},{spec.branch_b},{spec.branch_c})"
        if mode == "delta_r":
            c, d = aligned_values[2:4]
            return np.sqrt(
                (a - c) ** 2 + delta_phi(b, d) ** 2
            ), f"ΔR({spec.branch_a},{spec.branch_b},{spec.branch_c},{spec.branch_d})"

    raise ValueError(f"Unsupported variable mode: {mode}")
