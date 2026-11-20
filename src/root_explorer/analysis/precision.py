"""Detect integer values that float64 cannot represent exactly (#166).

Branch values are converted to float64 early, because missing values are encoded as
NaN. Integers with a magnitude above 2**53 (about 9.0e15) then lose precision:
neighbouring values become equal, so cuts, differences and rankings on them can be
silently wrong. Typical physics variables are far below this limit; nanosecond
timestamps, bit-packed time words and 64-bit detector IDs or hash keys are not.
These helpers detect the affected inputs so the application can warn about them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import awkward as ak
import numpy as np

from ..models import VariableSpec
from .arrays import read_for_scope

FLOAT64_EXACT_INTEGER_LIMIT = 2**53

_SINGLE_BRANCH_MODES = frozenset({"branch", "nth", "sum", "mean", "max", "min"})


def largest_inexact_integer(values: Any) -> int | None:
    """Return the largest magnitude of integer ``values`` if it exceeds 2**53.

    Returns ``None`` for non-integer data, for integers that float64 represents
    exactly and for data that cannot be inspected (records, strings). Missing values
    are ignored.
    """

    try:
        if isinstance(values, ak.Array):
            array = np.asarray(ak.to_numpy(ak.flatten(values, axis=None)))
        else:
            array = np.asarray(values).ravel()
    except Exception:
        return None
    if array.dtype.kind not in "iu" or array.size == 0:
        return None
    # Python integers avoid overflow in abs() of the smallest int64.
    largest = max(int(array.max()), -int(array.min()))
    return largest if largest > FLOAT64_EXACT_INTEGER_LIMIT else None


def variable_value_branches(spec: VariableSpec | None) -> list[str]:
    """Return the branches whose values ``spec`` uses (not merely counts)."""

    if spec is None or spec.mode == "multiplicity":
        return []
    if spec.mode in _SINGLE_BRANCH_MODES:
        return [spec.branch] if spec.branch else []
    return [
        branch for branch in (spec.branch_a, spec.branch_b, spec.branch_c, spec.branch_d) if branch
    ]


def integer_precision_warning(branch: str, largest: int) -> str:
    """Describe the precision loss of ``branch`` for the user."""

    spacing = float(np.spacing(float(largest)))
    return (
        f"`{branch}` holds integers up to {largest:.3g} in magnitude, beyond 2⁵³ "
        f"(about 9.0e15). They are converted to floating point, where values less than "
        f"{spacing:g} apart become equal at this size, so cuts, differences and rankings "
        f"using `{branch}` can be wrong. See Numeric precision in the user guide."
    )


def integer_precision_warnings(
    specs: Sequence[VariableSpec | None],
    extra_branches: Iterable[str | None],
    path: str,
    tree_name: str,
    entry_start: int,
    entry_stop: int | None,
    event_index: int | None,
) -> list[str]:
    """Return one warning per branch used by ``specs`` that exceeds 2**53.

    Branches are read through the same cached reader as the plotted data, so the
    check costs little after the data were read. A branch that cannot be read is
    skipped; its read error is reported where the data are used.
    """

    branches = dict.fromkeys(
        [branch for spec in specs for branch in variable_value_branches(spec)]
        + [branch for branch in extra_branches if branch]
    )
    warnings = []
    for branch in branches:
        try:
            values = read_for_scope(path, tree_name, branch, entry_start, entry_stop, event_index)
        except Exception:
            continue
        largest = largest_inexact_integer(values)
        if largest is not None:
            warnings.append(integer_precision_warning(branch, largest))
    return warnings


def is_inexact_integer_bound(text: str) -> bool:
    """Return whether ``text`` is an integer that float64 cannot represent exactly."""

    stripped = str(text).strip()
    try:
        integer = int(stripped)
    except ValueError:
        return False  # not an integer literal, e.g. 1.5 or 1e18
    try:
        return int(float(stripped)) != integer
    except OverflowError:
        return True  # too large even for float64
