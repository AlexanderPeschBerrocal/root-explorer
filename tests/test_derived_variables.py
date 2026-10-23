"""Discriminating numerical cases for derived variables (#153).

Every expected value is written out by hand or computed independently with the
``math`` module; none comes from the production helpers. Each case is chosen so
that a typical mistake changes the result: reversed operands, an absolute instead
of a signed difference, a missing zero-denominator guard, or an unwrapped angle.
"""

from __future__ import annotations

import math
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot

from root_explorer.analysis.arrays import delta_phi
from root_explorer.analysis.variables import evaluate_variable
from root_explorer.models import VariableSpec
from root_explorer.plotting.pipeline import StaticPlotSpec, prepare_static_plot

PI = math.pi
N_EVENTS = 6

SCALARS: dict[str, list[float]] = {
    # Ratio: ordinary, nonzero/0, 0/nonzero, nonzero/0, 0/0, negative.
    "num": [6.0, -3.0, 0.0, 5.0, 0.0, 2.0],
    "den": [3.0, 0.0, 4.0, 0.0, 0.0, -4.0],
    # Signed delta eta: positive, negative, zero and a sign change.
    "eta1": [1.0, -0.5, 2.0, 0.0, -1.25, 0.5],
    "eta2": [0.25, 1.5, 2.0, 0.0, 0.75, -0.5],
    # Delta phi: back-to-back, wrapping across +-pi in both directions, equal angles,
    # back-to-back the other way round, and a small ordinary difference.
    "phi1": [0.0, 3.0, -3.0, 1.0, PI / 2, 1.0],
    "phi2": [PI, -3.0, 3.0, 1.0, -PI / 2, 0.5],
    # Azimuth from x/y in all four quadrants and on the positive y and negative x axes.
    "x": [1.0, -1.0, -1.0, 1.0, 0.0, -1.0],
    "y": [1.0, 1.0, -1.0, -1.0, 2.0, 0.0],
    # Pseudorapidity: z = 0, zero transverse radius with z != 0 and z = 0, and +-|z|.
    "ex": [3.0, 0.0, 0.0, 0.0, 1.0, -1.0],
    "ey": [4.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    "ez": [0.0, 5.0, 0.0, -2.0, 1.0, 1.0],
}

NAN = math.nan
EXPECTED_RATIO = [2.0, NAN, 0.0, NAN, NAN, -0.5]
EXPECTED_DELTA_ETA = [0.75, -2.0, 0.0, 0.0, -2.0, 1.0]
# Documented interval (-pi, pi]: back-to-back pairs give +pi in both orders.
EXPECTED_DELTA_PHI = [PI, 6.0 - 2 * PI, 2 * PI - 6.0, 0.0, PI, 0.5]
EXPECTED_PHI_XY = [PI / 4, 3 * PI / 4, -3 * PI / 4, -PI / 4, PI / 2, PI]
# Documented as undefined at zero transverse radius.
EXPECTED_ETA_XYZ = [0.0, NAN, NAN, math.asinh(-1.0), math.asinh(1.0), math.asinh(1.0)]
EXPECTED_DELTA_R = [
    math.hypot(deta, dphi)
    for deta, dphi in zip(EXPECTED_DELTA_ETA, EXPECTED_DELTA_PHI, strict=True)
]


def _write(path: Path, kind: str) -> Path:
    events = {name: np.asarray(values) for name, values in SCALARS.items()}
    # A jagged copy in a second container exercises object-level and single-event
    # evaluation: the six values are split into events of 2, 0, 3 and 1 objects.
    objects = {
        name: ak.unflatten(np.asarray(values), [2, 0, 3, 1]) for name, values in SCALARS.items()
    }
    with uproot.recreate(path) as root_file:
        for container, fields in [("Events", events), ("Objects", objects)]:
            if kind == "ttree":
                root_file.mktree(container, fields)
            else:
                root_file.mkrntuple(container, fields)
    return path


@pytest.fixture(params=["ttree", "rntuple"])
def derived_path(request, tmp_path) -> Path:
    return _write(tmp_path / f"derived_{request.param}.root", request.param)


def _evaluate(path: Path, mode: str, *branches: str, objects: bool = False) -> np.ndarray:
    names = dict(zip(("branch_a", "branch_b", "branch_c", "branch_d"), branches, strict=False))
    spec = VariableSpec(mode=mode, **names)
    container, n_entries = ("Objects", 4) if objects else ("Events", N_EVENTS)
    values, _ = evaluate_variable(spec, str(path), container, 0, n_entries)
    return np.asarray(values, dtype=float)


CASES = [
    ("ratio", ("num", "den"), EXPECTED_RATIO),
    ("delta_eta", ("eta1", "eta2"), EXPECTED_DELTA_ETA),
    ("delta_phi", ("phi1", "phi2"), EXPECTED_DELTA_PHI),
    ("phi_xy", ("x", "y"), EXPECTED_PHI_XY),
    ("eta_xyz", ("ex", "ey", "ez"), EXPECTED_ETA_XYZ),
    ("delta_r", ("eta1", "phi1", "eta2", "phi2"), EXPECTED_DELTA_R),
]


@pytest.mark.parametrize(("mode", "branches", "expected"), CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("objects", [False, True], ids=["event-level", "object-level"])
def test_derived_variables_match_independent_values(
    derived_path, mode, branches, expected, objects
) -> None:
    np.testing.assert_allclose(
        _evaluate(derived_path, mode, *branches, objects=objects), expected, rtol=0, atol=1e-12
    )


def test_operand_order_and_sign_are_respected(derived_path) -> None:
    forward = _evaluate(derived_path, "delta_eta", "eta1", "eta2")
    backward = _evaluate(derived_path, "delta_eta", "eta2", "eta1")
    np.testing.assert_allclose(backward, -forward, atol=1e-12)
    assert np.any(forward > 0) and np.any(forward < 0)

    # Wrapped delta phi is antisymmetric except at the +pi boundary.
    forward_phi = _evaluate(derived_path, "delta_phi", "phi1", "phi2")
    backward_phi = _evaluate(derived_path, "delta_phi", "phi2", "phi1")
    interior = np.abs(forward_phi) < PI
    np.testing.assert_allclose(backward_phi[interior], -forward_phi[interior], atol=1e-12)
    np.testing.assert_allclose(backward_phi[~interior], PI, atol=1e-12)


def test_delta_phi_stays_in_the_documented_interval() -> None:
    # Differences that are odd multiples of pi in both directions, and far outside.
    phi1 = np.array([0.0, -PI, PI, 3 * PI, -3 * PI, 7.0, -7.0])
    phi2 = np.zeros_like(phi1)
    phi2[0] = PI
    result = delta_phi(phi1, phi2)
    assert np.all((result > -PI) & (result <= PI))
    np.testing.assert_allclose(result, [PI, PI, PI, PI, PI, 7.0 - 2 * PI, 2 * PI - 7.0], atol=1e-12)


@pytest.mark.parametrize(("mode", "branches", "expected"), CASES, ids=[c[0] for c in CASES])
def test_single_event_evaluation_matches_the_event_slice(derived_path, mode, branches, expected):
    # Event 2 holds the objects at positions 2, 3 and 4 of the flat values.
    names = dict(zip(("branch_a", "branch_b", "branch_c", "branch_d"), branches, strict=False))
    spec = VariableSpec(mode=mode, **names)
    values, _ = evaluate_variable(spec, str(derived_path), "Objects", 0, 4, event_index=2)
    np.testing.assert_allclose(np.asarray(values, dtype=float), expected[2:5], atol=1e-12)


def test_undefined_values_are_excluded_from_plots_and_counted(derived_path) -> None:
    spec = StaticPlotSpec(
        root_path=str(derived_path),
        tree_name="Events",
        entry_start=0,
        entry_stop=N_EVENTS,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="ratio", branch_a="num", branch_b="den"),
        y_spec=None,
        color_spec=None,
        weight_branch=None,
        cut_specs=(),
        cut_bounds=(),
        style={"bins_x": 4},
        labels={},
    )
    prepared = prepare_static_plot(spec)
    assert prepared.x.tolist() == [2.0, 0.0, -0.5]
    assert prepared.excluded_count == 3
