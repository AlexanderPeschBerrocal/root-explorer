"""Awkward/NumPy conversion and safe variable-alignment helpers."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

import awkward as ak
import numpy as np

from ..io.root import read_branch, read_event_branch
from ..models import VariableSpec


def to_numpy_flat(arr: Any) -> np.ndarray:
    try:
        return ak.to_numpy(ak.flatten(arr, axis=None))
    except Exception:
        try:
            return np.asarray(arr)
        except Exception:
            return np.array([])


def event_to_numpy(arr: Any) -> np.ndarray:
    try:
        return ak.to_numpy(ak.flatten(arr, axis=None))
    except Exception:
        try:
            return np.asarray(arr)
        except Exception:
            return np.array([])


def safe_float_array(arr: Any) -> np.ndarray:
    out = np.asarray(arr)
    try:
        return out.astype(float, copy=False)
    except Exception:
        return out


class ArrayShapeMismatchError(ValueError):
    """Raised when plot inputs cannot be aligned without changing their meaning."""


def align_arrays(
    arrays: Sequence[np.ndarray],
    *,
    names: Sequence[str] | None = None,
    context: str = "arrays",
) -> list[np.ndarray]:
    """Validate that flat arrays have identical shapes; never truncate values."""

    normalized: list[np.ndarray] = []
    for arr in arrays:
        values = np.asarray(arr)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1:
            raise ArrayShapeMismatchError(
                f"Cannot align {context}: expected flat one-dimensional arrays, "
                f"but received shape {values.shape}."
            )
        normalized.append(values)

    if not normalized:
        return []

    labels = (
        list(names) if names is not None else [f"array {i + 1}" for i in range(len(normalized))]
    )
    if len(labels) != len(normalized):
        raise ValueError("The number of array names must match the number of arrays.")

    reference_shape = normalized[0].shape
    mismatches = [
        f"{label} has shape {values.shape}"
        for label, values in zip(labels, normalized)
        if values.shape != reference_shape
    ]
    if mismatches:
        reference = f"{labels[0]} has shape {reference_shape}"
        raise ArrayShapeMismatchError(
            f"Cannot align {context}: {reference}; " + "; ".join(mismatches) + ". "
            "No values were truncated. Choose variables with matching shapes or use "
            "a per-event reduction such as multiplicity, sum, mean, max, min, or nth."
        )

    return normalized


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    aligned = align_arrays(arrays)
    if not aligned:
        return np.array([], dtype=bool)
    n = len(aligned[0])
    mask = np.ones(n, dtype=bool)
    for arr in aligned:
        with contextlib.suppress(Exception):
            mask &= np.isfinite(arr.astype(float, copy=False))
    return mask


def apply_finite_filter(arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    aligned = align_arrays(arrays)
    mask = finite_mask(*aligned)
    return [a[mask] for a in aligned]


def multiplicity_per_event(arr: Any) -> np.ndarray:
    try:
        return ak.to_numpy(ak.num(arr, axis=1)).astype(float)
    except Exception:
        # Scalar event-level branch: multiplicity is one value per event.
        try:
            return np.ones(len(arr), dtype=float)
        except Exception:
            return np.array([], dtype=float)


def nth_per_event(arr: Any, index: int) -> np.ndarray:
    try:
        selected = ak.pad_none(arr, index + 1, axis=1)[:, index]
        return ak.to_numpy(ak.fill_none(selected, np.nan)).astype(float)
    except Exception:
        flat = to_numpy_flat(arr)
        return safe_float_array(flat)


def reduce_per_event(arr: Any, mode: str) -> np.ndarray:
    try:
        values = ak.values_astype(arr, np.float64)
        if mode == "sum":
            out = ak.sum(values, axis=1)
        elif mode == "mean":
            out = ak.mean(values, axis=1)
        elif mode == "max":
            out = ak.max(values, axis=1, mask_identity=False)
        elif mode == "min":
            out = ak.min(values, axis=1, mask_identity=False)
        else:
            raise ValueError(mode)
        return ak.to_numpy(out).astype(float)
    except Exception:
        return safe_float_array(to_numpy_flat(arr))


def delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))


def read_for_scope(
    path: str,
    tree_name: str,
    branch: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None,
):
    if event_index is None:
        return read_branch(path, tree_name, branch, entry_start, entry_stop)
    return read_event_branch(path, tree_name, branch, event_index)


def branch_values(
    path: str,
    tree_name: str,
    branch: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None,
) -> np.ndarray:
    arr = read_for_scope(path, tree_name, branch, entry_start, entry_stop, event_index)
    return event_to_numpy(arr) if event_index is not None else to_numpy_flat(arr)


def variable_reference_branch(spec: VariableSpec) -> str | None:
    if spec.mode in {"branch", "multiplicity", "nth", "sum", "mean", "max", "min"}:
        return spec.branch
    return spec.branch_a


def variable_is_per_event(spec: VariableSpec) -> bool:
    return spec.mode in {"multiplicity", "nth", "sum", "mean", "max", "min"}


def array_length(arr: Any) -> int:
    values = np.asarray(arr)
    if values.ndim == 0:
        return 1
    return len(values)


def raw_counts_per_event(arr: Any, event_index: int | None) -> np.ndarray:
    """Return the number of scalar values contributed by each event."""

    if event_index is not None:
        if value_is_scalar(arr):
            return np.ones(1, dtype=int)
        return np.asarray([array_length(event_to_numpy(arr))], dtype=int)

    try:
        values = ak.Array(arr)
        while ak.to_layout(values).purelist_depth > 2:
            # Flatten nested object substructure while preserving the outer event axis.
            values = ak.flatten(values, axis=2)
        return ak.to_numpy(ak.num(values, axis=1)).astype(int)
    except Exception:
        try:
            return np.ones(array_length(arr), dtype=int)
        except Exception:
            return np.array([], dtype=int)


def raw_list_shape_levels(arr: Any) -> list[np.ndarray]:
    """Return list lengths at every jagged depth without discarding boundaries."""

    try:
        values = ak.Array(arr)
        depth = ak.to_layout(values).purelist_depth
        return [
            ak.to_numpy(ak.flatten(ak.num(values, axis=axis), axis=None)).astype(int)
            for axis in range(1, depth)
        ]
    except Exception:
        return []


def validate_raw_event_shapes(
    arrays: Sequence[Any],
    names: Sequence[str],
    *,
    entry_start: int,
    event_index: int | None,
    context: str,
) -> None:
    """Validate scalar counts and nested list boundaries before flattening."""

    validate_event_counts(
        [raw_counts_per_event(arr, event_index) for arr in arrays],
        names,
        entry_start=entry_start,
        event_index=event_index,
        context=context,
    )
    if len(arrays) <= 1:
        return

    levels = [raw_list_shape_levels(arr) for arr in arrays]
    reference = levels[0]
    for name, candidate in zip(names[1:], levels[1:]):
        if len(candidate) != len(reference):
            raise ArrayShapeMismatchError(
                f"Cannot align {context}: {names[0]!r} and {name!r} have different "
                "nested list depths. Flattening them would discard collection boundaries. "
                "No plot was produced."
            )
        for depth, (reference_counts, candidate_counts) in enumerate(
            zip(reference, candidate), start=1
        ):
            if reference_counts.shape != candidate_counts.shape or not np.array_equal(
                reference_counts, candidate_counts
            ):
                raise ArrayShapeMismatchError(
                    f"Cannot align {context}: {names[0]!r} and {name!r} have different "
                    f"list boundaries at nested depth {depth}. No values were flattened or "
                    "truncated; choose branches with matching collection structure."
                )


def variable_spec_name(spec: VariableSpec) -> str:
    """Return a compact human-readable name for shape diagnostics."""

    if spec.label:
        return spec.label
    if spec.mode == "branch":
        return str(spec.branch or "branch")
    if spec.mode in {"multiplicity", "nth", "sum", "mean", "max", "min"}:
        return f"{spec.mode}({spec.branch or '?'})"
    branches = [spec.branch_a, spec.branch_b, spec.branch_c, spec.branch_d]
    return f"{spec.mode}({','.join(str(branch) for branch in branches if branch)})"


def validate_event_counts(
    counts: Sequence[np.ndarray],
    names: Sequence[str],
    *,
    entry_start: int,
    event_index: int | None,
    context: str,
) -> None:
    """Require identical per-event multiplicities before flattening arrays."""

    if len(counts) <= 1:
        return
    if len(counts) != len(names):
        raise ValueError("The number of count arrays and names must match.")

    normalized = [np.asarray(values, dtype=int).reshape(-1) for values in counts]
    reference = normalized[0]
    for name, candidate in zip(names[1:], normalized[1:]):
        if candidate.shape != reference.shape:
            raise ArrayShapeMismatchError(
                f"Cannot align {context}: {names[0]!r} covers {len(reference)} event(s), "
                f"but {name!r} covers {len(candidate)} event(s). No plot was produced."
            )
        mismatches = np.flatnonzero(candidate != reference)
        if len(mismatches):
            local_index = int(mismatches[0])
            absolute_index = (
                int(event_index) if event_index is not None else int(entry_start) + local_index
            )
            raise ArrayShapeMismatchError(
                f"Cannot align {context}: in event {absolute_index}, {names[0]!r} contributes "
                f"{int(reference[local_index])} value(s), while {name!r} contributes "
                f"{int(candidate[local_index])}. Flattening these branches would mix event "
                "boundaries. No plot was produced; choose matching collections or use "
                "per-event reductions."
            )


def variable_counts_per_event(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None,
) -> np.ndarray:
    """Return how many plotted values each event contributes for a variable."""

    if variable_is_per_event(spec):
        ref_branch = variable_reference_branch(spec)
        if not ref_branch:
            return np.array([], dtype=int)
        arr = read_for_scope(path, tree_name, ref_branch, entry_start, entry_stop, event_index)
        return np.ones(len(raw_counts_per_event(arr, event_index)), dtype=int)

    ref_branch = variable_reference_branch(spec)
    if not ref_branch:
        return np.array([], dtype=int)

    arr = read_for_scope(path, tree_name, ref_branch, entry_start, entry_stop, event_index)
    return raw_counts_per_event(arr, event_index)


def value_is_scalar(arr: Any) -> bool:
    try:
        return np.asarray(arr).ndim == 0
    except Exception:
        return False


def weight_is_per_event_array(arr: Any, event_index: int | None) -> bool:
    if event_index is not None:
        return value_is_scalar(arr)
    try:
        ak.num(arr, axis=1)
        return False
    except Exception:
        return True


def histogram_weights_for_values(
    path: str,
    tree_name: str,
    weight_branch: str,
    reference_spec: VariableSpec,
    value_count: int,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None,
) -> np.ndarray:
    """Read histogram weights and broadcast per-event weights over object values."""

    raw_weights = read_for_scope(
        path, tree_name, weight_branch, entry_start, entry_stop, event_index
    )
    weights = safe_float_array(
        event_to_numpy(raw_weights) if event_index is not None else to_numpy_flat(raw_weights)
    )
    reference_counts = variable_counts_per_event(
        reference_spec, path, tree_name, entry_start, entry_stop, event_index
    )
    reference_name = variable_spec_name(reference_spec)
    expected_values = int(np.sum(reference_counts))
    if expected_values != value_count:
        raise ArrayShapeMismatchError(
            f"Cannot align weights with {reference_name!r}: its event structure predicts "
            f"{expected_values} value(s), but evaluation produced {value_count}. No plot was produced."
        )

    if weight_is_per_event_array(raw_weights, event_index):
        if array_length(weights) == len(reference_counts):
            return np.repeat(np.asarray(weights).reshape(-1), reference_counts)
        if array_length(weights) == 1:
            return np.repeat(np.asarray(weights).reshape(-1), value_count)
        raise ArrayShapeMismatchError(
            f"Cannot align weight branch {weight_branch!r} with {reference_name!r}: "
            f"there are {array_length(weights)} event weight(s) for {len(reference_counts)} "
            "selected event(s). No plot was produced."
        )

    weight_counts = raw_counts_per_event(raw_weights, event_index)
    validate_event_counts(
        [reference_counts, weight_counts],
        [reference_name, weight_branch],
        entry_start=entry_start,
        event_index=event_index,
        context="object-level histogram weights",
    )
    reference_branch = variable_reference_branch(reference_spec)
    if reference_branch:
        raw_reference = read_for_scope(
            path, tree_name, reference_branch, entry_start, entry_stop, event_index
        )
        validate_raw_event_shapes(
            [raw_reference, raw_weights],
            [reference_name, weight_branch],
            entry_start=entry_start,
            event_index=event_index,
            context="object-level histogram weights",
        )
    validated_weights = align_arrays(
        [weights],
        names=[weight_branch],
        context="object-level histogram weights",
    )[0]
    if len(validated_weights) != value_count:
        raise ArrayShapeMismatchError(
            f"Cannot align weight branch {weight_branch!r} with {reference_name!r}: "
            f"it produced {len(validated_weights)} value(s), while the reference variable "
            f"produced {value_count}. No plot was produced."
        )
    return validated_weights


def variable_is_event_level_array(
    spec: VariableSpec,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None,
) -> bool:
    if variable_is_per_event(spec):
        return True

    ref_branch = variable_reference_branch(spec)
    if not ref_branch:
        return False

    arr = read_for_scope(path, tree_name, ref_branch, entry_start, entry_stop, event_index)
    if event_index is not None:
        return value_is_scalar(arr)

    try:
        ak.num(arr, axis=1)
        return False
    except Exception:
        return True


def validate_variable_event_shapes(
    specs: Sequence[VariableSpec],
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None,
    *,
    context: str,
) -> None:
    """Require plot-axis variables to contribute equally within every event."""

    selected = [spec for spec in specs if spec is not None]
    names = [variable_spec_name(spec) for spec in selected]
    counts = [
        variable_counts_per_event(spec, path, tree_name, entry_start, entry_stop, event_index)
        for spec in selected
    ]
    validate_event_counts(
        counts,
        names,
        entry_start=entry_start,
        event_index=event_index,
        context=context,
    )

    object_specs = [spec for spec in selected if not variable_is_per_event(spec)]
    if len(object_specs) > 1:
        reference_branches = [variable_reference_branch(spec) for spec in object_specs]
        if all(reference_branches):
            raw_arrays = [
                read_for_scope(path, tree_name, str(branch), entry_start, entry_stop, event_index)
                for branch in reference_branches
            ]
            validate_raw_event_shapes(
                raw_arrays,
                [variable_spec_name(spec) for spec in object_specs],
                entry_start=entry_start,
                event_index=event_index,
                context=context,
            )


def read_aligned_branch_values(
    branches: Sequence[str],
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None,
    *,
    context: str,
) -> list[np.ndarray]:
    """Read related branches and validate event multiplicities before flattening."""

    raw_arrays = [
        read_for_scope(path, tree_name, branch, entry_start, entry_stop, event_index)
        for branch in branches
    ]
    validate_raw_event_shapes(
        raw_arrays,
        list(branches),
        entry_start=entry_start,
        event_index=event_index,
        context=context,
    )
    values = [
        safe_float_array(event_to_numpy(arr) if event_index is not None else to_numpy_flat(arr))
        for arr in raw_arrays
    ]
    return align_arrays(values, names=list(branches), context=context)


def values_for_reference_shape(
    values: np.ndarray,
    reference_spec: VariableSpec,
    value_count: int,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None,
    *,
    values_are_per_event: bool = False,
    values_name: str = "secondary variable",
) -> np.ndarray:
    """Broadcast event-level values to match flattened object-level plot values."""

    counts = variable_counts_per_event(
        reference_spec, path, tree_name, entry_start, entry_stop, event_index
    )
    reference_name = variable_spec_name(reference_spec)
    expected_values = int(np.sum(counts))
    if expected_values != value_count:
        raise ArrayShapeMismatchError(
            f"Cannot align values with {reference_name!r}: its event structure predicts "
            f"{expected_values} value(s), but evaluation produced {value_count}. No plot was produced."
        )

    if values_are_per_event and len(counts) and array_length(values) == len(counts):
        return np.repeat(np.asarray(values).reshape(-1), counts)

    if array_length(values) == value_count:
        return align_arrays(
            [values],
            names=[values_name],
            context="plot values",
        )[0]

    if array_length(values) == 1 and value_count != 1 and values_are_per_event:
        return np.repeat(np.asarray(values).reshape(-1), value_count)

    raise ArrayShapeMismatchError(
        f"Cannot align {values_name!r} with {reference_name!r}: it produced "
        f"{array_length(values)} value(s), while the reference variable produced "
        f"{value_count}. No plot was produced."
    )


def variable_values_for_values(
    spec: VariableSpec,
    reference_spec: VariableSpec,
    value_count: int,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None,
) -> tuple[np.ndarray, str]:
    from .variables import evaluate_variable

    values, label = evaluate_variable(spec, path, tree_name, entry_start, entry_stop, event_index)
    values_are_per_event = variable_is_event_level_array(
        spec, path, tree_name, entry_start, entry_stop, event_index
    )
    if not values_are_per_event:
        validate_event_counts(
            [
                variable_counts_per_event(
                    reference_spec, path, tree_name, entry_start, entry_stop, event_index
                ),
                variable_counts_per_event(
                    spec, path, tree_name, entry_start, entry_stop, event_index
                ),
            ],
            [variable_spec_name(reference_spec), label],
            entry_start=entry_start,
            event_index=event_index,
            context="object-level plot variables",
        )
        reference_branch = variable_reference_branch(reference_spec)
        value_branch = variable_reference_branch(spec)
        if reference_branch and value_branch:
            validate_raw_event_shapes(
                [
                    read_for_scope(
                        path, tree_name, reference_branch, entry_start, entry_stop, event_index
                    ),
                    read_for_scope(
                        path, tree_name, value_branch, entry_start, entry_stop, event_index
                    ),
                ],
                [variable_spec_name(reference_spec), label],
                entry_start=entry_start,
                event_index=event_index,
                context="object-level plot variables",
            )
    values = values_for_reference_shape(
        values,
        reference_spec,
        value_count,
        path,
        tree_name,
        entry_start,
        entry_stop,
        event_index,
        values_are_per_event=values_are_per_event,
        values_name=label,
    )
    return values, label


def cut_values_for_values(
    spec: VariableSpec,
    reference_spec: VariableSpec,
    value_count: int,
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None,
) -> tuple[np.ndarray, str]:
    return variable_values_for_values(
        spec,
        reference_spec,
        value_count,
        path,
        tree_name,
        entry_start,
        entry_stop,
        event_index,
    )
