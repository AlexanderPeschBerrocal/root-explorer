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
            raise ValueError("No field selected.")
        return safe_float_array(
            branch_values(path, tree_name, spec.branch, entry_start, entry_stop, event_index)
        ), spec.branch

    if mode == "multiplicity":
        if not spec.branch:
            raise ValueError("No field selected.")
        arr = read_for_scope(path, tree_name, spec.branch, entry_start, entry_stop, event_index)
        if event_index is None:
            return multiplicity_per_event(arr), f"N({spec.branch})"
        return np.array([len(event_to_numpy(arr))], dtype=float), f"N({spec.branch})"

    if mode == "nth":
        if not spec.branch:
            raise ValueError("No field selected.")
        arr = read_for_scope(path, tree_name, spec.branch, entry_start, entry_stop, event_index)
        if event_index is None:
            return nth_per_event(arr, spec.index), f"{spec.branch}[{spec.index}]"
        values = event_to_numpy(arr)
        value = values[spec.index] if len(values) > spec.index else np.nan
        return np.asarray([value], dtype=float), f"{spec.branch}[{spec.index}]"

    if mode in {"sum", "mean", "max", "min"}:
        if not spec.branch:
            raise ValueError("No field selected.")
        arr = read_for_scope(path, tree_name, spec.branch, entry_start, entry_stop, event_index)
        if event_index is None:
            return reduce_per_event(arr, mode), f"{mode}({spec.branch})"
        values = safe_float_array(event_to_numpy(arr))
        if len(values) == 0:
            empty_value = 0.0 if mode == "sum" else np.nan
            return np.array([empty_value], dtype=float), f"{mode}({spec.branch})"
        if mode == "sum":
            reduced = np.nansum(values)
        elif mode == "mean":
            reduced = np.nanmean(values)
        elif mode == "max":
            reduced = np.nanmax(values)
        else:
            reduced = np.nanmin(values)
        return np.array([reduced], dtype=float), f"{mode}({spec.branch})"

    if mode in {
        "absolute",
        "magnitude_3d",
        "r_xy",
        "phi_xy",
        "eta_xyz",
        "energy_massless",
        "energy_massive",
        "invariant_mass",
        "ratio",
        "difference",
        "sum_ab",
        "product",
        "delta_phi",
        "delta_eta",
        "delta_r",
    }:
        if not spec.branch_a:
            raise ValueError("This derived variable needs at least one field.")

        branches = [spec.branch_a]
        if mode != "absolute":
            if not spec.branch_b:
                raise ValueError("This derived variable needs at least two fields.")
            branches.append(spec.branch_b)
        if mode in {"eta_xyz", "magnitude_3d", "energy_massive"}:
            if not spec.branch_c:
                raise ValueError(f"Derived variable {mode!r} needs three fields.")
            branches.append(spec.branch_c)
        elif mode in {"delta_r", "invariant_mass"}:
            if not spec.branch_c or not spec.branch_d:
                raise ValueError(f"Derived variable {mode!r} needs four fields.")
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
        a = aligned_values[0]

        if mode == "absolute":
            return np.abs(a), f"|{spec.branch_a}|"

        b = aligned_values[1]

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
        if mode == "eta_xyz":
            c = aligned_values[2]
            r = np.sqrt(a * a + b * b)
            out = np.full(len(r), np.nan, dtype=float)
            mask = np.isfinite(r) & np.isfinite(c) & (r > 0)
            out[mask] = np.arcsinh(c[mask] / r[mask])
            return out, f"eta({spec.branch_a},{spec.branch_b},{spec.branch_c})"
        if mode == "magnitude_3d":
            c = aligned_values[2]
            return np.hypot(
                np.hypot(a, b), c
            ), f"mag({spec.branch_a},{spec.branch_b},{spec.branch_c})"
        if mode in {"energy_massless", "energy_massive"}:
            if spec.longitudinal_kind not in {"eta", "y"}:
                raise ValueError("Energy longitudinal coordinate must be 'eta' or 'y'.")
            if mode == "energy_massless":
                energy = a * np.cosh(b)
                return (
                    energy,
                    f"E_massless({spec.branch_a},{spec.longitudinal_kind}={spec.branch_b})",
                )
            c = aligned_values[2]
            if spec.longitudinal_kind == "eta":
                energy = np.sqrt(c * c + (a * np.cosh(b)) ** 2)
            else:
                energy = np.hypot(c, a) * np.cosh(b)
            return energy, (
                f"E({spec.branch_a},{spec.longitudinal_kind}={spec.branch_b},{spec.branch_c})"
            )
        if mode == "invariant_mass":
            c, d = aligned_values[2:4]
            mass_squared = a * a - b * b - c * c - d * d
            return np.sqrt(np.maximum(mass_squared, 0.0)), (
                f"m({spec.branch_a},{spec.branch_b},{spec.branch_c},{spec.branch_d})"
            )
        if mode == "delta_phi":
            return delta_phi(a, b), f"Δφ({spec.branch_a},{spec.branch_b})"
        if mode == "delta_eta":
            return a - b, f"Δη({spec.branch_a},{spec.branch_b})"
        if mode == "delta_r":
            c, d = aligned_values[2:4]
            return np.sqrt(
                (a - c) ** 2 + delta_phi(b, d) ** 2
            ), f"ΔR({spec.branch_a},{spec.branch_b},{spec.branch_c},{spec.branch_d})"

    raise ValueError(f"Unsupported variable mode: {mode}")
