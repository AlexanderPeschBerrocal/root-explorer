"""The Plotter's derived-variable controls fill the right VariableSpec slots (#170).

The same VariableSpec drives the plot and the exported script, so a swapped slot
(for example ΔR's φ₁ and η₂) would silently plot the wrong physics in both. Each
test selects a different branch for every labelled control, makes a plot through
the real app, and checks the slots and the plotted values against plain NumPy.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import uproot
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N = 50


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


@pytest.fixture(scope="module")
def branches(tmp_path_factory) -> tuple[Path, dict[str, np.ndarray]]:
    """Five scalar branches with distinct values; f0 is large enough to be an energy."""
    rng = np.random.default_rng(170)
    values = {
        "f0": rng.uniform(50.0, 60.0, N),
        "f1": rng.uniform(-3.0, 3.0, N),
        "f2": rng.uniform(0.5, 3.0, N),
        "f3": rng.uniform(-3.0, -0.5, N),
        "f4": rng.uniform(1.0, 2.0, N),
    }
    path = tmp_path_factory.mktemp("wiring") / "wiring.root"
    with uproot.recreate(path) as root_file:
        root_file.mktree("Events", values)
    return path, values


# mode -> (control key suffix -> branch, expected spec slots, NumPy formula)
Formula = Callable[[dict[str, np.ndarray]], np.ndarray]
CASES: dict[str, tuple[dict[str, str], dict[str, str], Formula]] = {
    "absolute": ({"a": "f3"}, {"branch_a": "f3"}, lambda v: np.abs(v["f3"])),
    "sum_ab": (
        {"a": "f1", "b": "f2"},
        {"branch_a": "f1", "branch_b": "f2"},
        lambda v: v["f1"] + v["f2"],
    ),
    "difference": (
        {"a": "f1", "b": "f2"},
        {"branch_a": "f1", "branch_b": "f2"},
        lambda v: v["f1"] - v["f2"],
    ),
    "ratio": (
        {"a": "f1", "b": "f2"},
        {"branch_a": "f1", "branch_b": "f2"},
        lambda v: v["f1"] / v["f2"],
    ),
    "product": (
        {"a": "f1", "b": "f3"},
        {"branch_a": "f1", "branch_b": "f3"},
        lambda v: v["f1"] * v["f3"],
    ),
    "r_xy": (
        {"a": "f1", "b": "f3"},
        {"branch_a": "f1", "branch_b": "f3"},
        lambda v: np.hypot(v["f1"], v["f3"]),
    ),
    "phi_xy": (
        {"a": "f1", "b": "f3"},
        {"branch_a": "f1", "branch_b": "f3"},
        lambda v: np.arctan2(v["f3"], v["f1"]),
    ),
    "eta_xyz": (
        {"x": "f1", "y": "f3", "z": "f2"},
        {"branch_a": "f1", "branch_b": "f3", "branch_c": "f2"},
        lambda v: np.arcsinh(v["f2"] / np.hypot(v["f1"], v["f3"])),
    ),
    "magnitude_3d": (
        {"a": "f1", "b": "f2", "c": "f3"},
        {"branch_a": "f1", "branch_b": "f2", "branch_c": "f3"},
        lambda v: np.sqrt(v["f1"] ** 2 + v["f2"] ** 2 + v["f3"] ** 2),
    ),
    "energy_massless": (
        {"pt": "f2", "longitudinal": "f1"},
        {"branch_a": "f2", "branch_b": "f1"},
        lambda v: v["f2"] * np.cosh(v["f1"]),
    ),
    "energy_massive": (
        {"pt": "f2", "longitudinal": "f1", "mass": "f4"},
        {"branch_a": "f2", "branch_b": "f1", "branch_c": "f4"},
        lambda v: np.sqrt(v["f4"] ** 2 + (v["f2"] * np.cosh(v["f1"])) ** 2),
    ),
    "invariant_mass": (
        {"e": "f0", "px": "f1", "py": "f2", "pz": "f3"},
        {"branch_a": "f0", "branch_b": "f1", "branch_c": "f2", "branch_d": "f3"},
        lambda v: np.sqrt(v["f0"] ** 2 - v["f1"] ** 2 - v["f2"] ** 2 - v["f3"] ** 2),
    ),
    "delta_phi": (
        {"phi1": "f1", "phi2": "f3"},
        {"branch_a": "f1", "branch_b": "f3"},
        lambda v: _wrap(v["f1"] - v["f3"]),
    ),
    "delta_eta": (
        {"eta1": "f1", "eta2": "f3"},
        {"branch_a": "f1", "branch_b": "f3"},
        lambda v: v["f1"] - v["f3"],
    ),
    "delta_r": (
        {"eta1": "f1", "phi1": "f2", "eta2": "f3", "phi2": "f4"},
        {"branch_a": "f1", "branch_b": "f2", "branch_c": "f3", "branch_d": "f4"},
        lambda v: np.hypot(v["f1"] - v["f3"], _wrap(v["f2"] - v["f4"])),
    ),
}


@pytest.mark.parametrize("mode", list(CASES))
def test_derived_variable_controls_fill_the_documented_slots(branches, mode) -> None:
    path, values = branches
    controls, slots, formula = CASES[mode]
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=30).run()
    at.text_input(key="root_path").input(str(path)).run()
    at.selectbox(key="x_mode").select(mode).run()
    for suffix, branch in controls.items():
        at.selectbox(key=f"x_{suffix}").select(branch).run()
    at.button(key="plot_make").click().run()
    assert not at.exception

    result = at.session_state["plot_result"]
    spec = result["settings"].x_spec
    assert spec.mode == mode
    for slot, branch in slots.items():
        assert getattr(spec, slot) == branch, (mode, slot)
    # Unused slots stay empty, so no stale branch can enter the evaluation.
    for slot in {"branch_a", "branch_b", "branch_c", "branch_d"} - set(slots):
        assert getattr(spec, slot) is None, (mode, slot)

    expected = formula(values)
    summary = result["summary"].iloc[0]
    assert summary["n"] == N
    assert summary["mean"] == pytest.approx(float(np.mean(expected)), rel=1e-9)
    assert summary["min"] == pytest.approx(float(np.min(expected)), rel=1e-9)
    assert summary["max"] == pytest.approx(float(np.max(expected)), rel=1e-9)


def test_every_derived_mode_is_covered() -> None:
    from root_explorer.analysis.variables import VARIABLE_EVALUATORS

    single_branch = {"branch", "multiplicity", "nth", "sum", "mean", "max", "min"}
    assert set(VARIABLE_EVALUATORS) - single_branch == set(CASES)
