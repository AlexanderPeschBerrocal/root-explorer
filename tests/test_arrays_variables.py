from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
import uproot

from root_explorer.analysis.arrays import (
    ArrayShapeMismatchError,
    align_arrays,
    apply_finite_filter,
    histogram_weights_for_values,
    multiplicity_per_event,
    nth_per_event,
    reduce_per_event,
    validate_raw_event_shapes,
    variable_values_for_values,
)
from root_explorer.analysis.variables import evaluate_variable
from root_explorer.io.root import read_field
from root_explorer.models import VariableSpec


def test_jagged_array_helpers_and_finite_filter() -> None:
    values = ak.Array([[1.0, 2.0], [], [4.0]])

    np.testing.assert_array_equal(multiplicity_per_event(values), [2.0, 0.0, 1.0])
    np.testing.assert_allclose(nth_per_event(values, 1), [2.0, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(reduce_per_event(values, "sum"), [3.0, 0.0, 4.0])
    np.testing.assert_allclose(reduce_per_event(values, "mean"), [1.5, np.nan, 4.0], equal_nan=True)

    filtered = apply_finite_filter([np.array([1.0, np.nan, 3.0]), np.array([4.0, 5.0, np.inf])])
    assert [array.tolist() for array in filtered] == [[1.0], [4.0]]


@pytest.mark.parametrize("values", [np.array([1.0, 2.0, 3.0]), ak.Array([1.0, 2.0, 3.0])])
def test_nth_per_event_treats_scalar_values_as_single_item_collections(values) -> None:
    np.testing.assert_allclose(nth_per_event(values, 0), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        nth_per_event(values, 1),
        [np.nan, np.nan, np.nan],
        equal_nan=True,
    )


def test_alignment_reports_shape_and_jagged_boundary_errors() -> None:
    with pytest.raises(ArrayShapeMismatchError, match="No values were truncated"):
        align_arrays([np.array([1.0, 2.0]), np.array([3.0])], names=["x", "y"])

    with pytest.raises(ArrayShapeMismatchError, match="list boundaries"):
        validate_raw_event_shapes(
            [ak.Array([[[1.0, 2.0], [3.0]], [[4.0]]]), ak.Array([[[5.0], [6.0, 7.0]], [[8.0]]])],
            ["left", "right"],
            entry_start=0,
            event_index=None,
            context="test variables",
        )


@pytest.mark.parametrize("fixture_name", ["root_path", "rntuple_path"])
def test_variable_evaluation_and_weight_broadcasting(request, fixture_name: str) -> None:
    root_path = request.getfixturevalue(fixture_name)
    muon_pt = VariableSpec(mode="branch", branch="Muon_pt")
    pt, label = evaluate_variable(muon_pt, str(root_path), "Events", 0, 12)
    radial, radial_label = evaluate_variable(
        VariableSpec(mode="r_xy", branch_a="Muon_px", branch_b="Muon_py"),
        str(root_path),
        "Events",
        0,
        12,
    )
    assert label == "Muon_pt"
    assert radial_label == "r(Muon_px,Muon_py)"
    np.testing.assert_allclose(radial, pt, rtol=1e-5)

    counts = multiplicity_per_event(read_field(str(root_path), "Events", "Muon_pt", 0, 12)).astype(
        int
    )
    event_weights = histogram_weights_for_values(
        str(root_path), "Events", "genWeight", muon_pt, len(pt), 0, 12, None
    )
    raw_event_weights = read_field(str(root_path), "Events", "genWeight", 0, 12)
    np.testing.assert_allclose(event_weights, np.repeat(np.asarray(raw_event_weights), counts))

    object_weights = histogram_weights_for_values(
        str(root_path), "Events", "Muon_mass", muon_pt, len(pt), 0, 12, None
    )
    assert len(object_weights) == len(pt)

    met_values, met_label = variable_values_for_values(
        VariableSpec(mode="branch", branch="MET_pt"),
        muon_pt,
        len(pt),
        str(root_path),
        "Events",
        0,
        12,
        None,
    )
    assert met_label == "MET_pt"
    assert len(met_values) == len(pt)

    scalar_nth_zero, scalar_nth_zero_label = evaluate_variable(
        VariableSpec(mode="nth", branch="MET_pt", index=0),
        str(root_path),
        "Events",
        0,
        12,
    )
    scalar_nth_missing, _ = evaluate_variable(
        VariableSpec(mode="nth", branch="MET_pt", index=7),
        str(root_path),
        "Events",
        0,
        12,
    )
    raw_met = np.asarray(read_field(str(root_path), "Events", "MET_pt", 0, 12), dtype=float)
    assert scalar_nth_zero_label == "MET_pt[0]"
    np.testing.assert_allclose(scalar_nth_zero, raw_met)
    assert np.isnan(scalar_nth_missing).all()


@pytest.mark.parametrize("fixture_name", ["root_path", "rntuple_path"])
def test_event_nth_variable_uses_scalar_only_at_index_zero(request, fixture_name: str) -> None:
    root_path = request.getfixturevalue(fixture_name)
    raw_met = np.asarray(read_field(str(root_path), "Events", "MET_pt", 0, 1), dtype=float)

    scalar_nth_zero, _ = evaluate_variable(
        VariableSpec(mode="nth", branch="MET_pt", index=0),
        str(root_path),
        "Events",
        0,
        12,
        event_index=0,
    )
    scalar_nth_missing, _ = evaluate_variable(
        VariableSpec(mode="nth", branch="MET_pt", index=7),
        str(root_path),
        "Events",
        0,
        12,
        event_index=0,
    )

    np.testing.assert_allclose(scalar_nth_zero, raw_met)
    assert np.isnan(scalar_nth_missing).all()


@pytest.mark.parametrize(
    ("mode", "reducer"),
    [("sum", np.nansum), ("mean", np.nanmean), ("max", np.nanmax), ("min", np.nanmin)],
)
@pytest.mark.parametrize("fixture_name", ["root_path", "rntuple_path"])
def test_event_reductions_preserve_numpy_results(request, fixture_name: str, mode, reducer) -> None:
    root_path = request.getfixturevalue(fixture_name)
    counts = np.asarray(read_field(str(root_path), "Events", "nMuon", 0, 12))
    event_index = int(np.flatnonzero(counts)[0])
    event_values = np.asarray(
        read_field(str(root_path), "Events", "Muon_pt", event_index, event_index + 1)[0]
    )

    values, label = evaluate_variable(
        VariableSpec(mode=mode, branch="Muon_pt"),
        str(root_path),
        "Events",
        0,
        12,
        event_index,
    )

    assert label == f"{mode}(Muon_pt)"
    np.testing.assert_allclose(values, [reducer(event_values)])


@pytest.mark.parametrize("fixture_name", ["root_path", "rntuple_path"])
def test_ratio_of_a_field_with_itself_is_one(request, fixture_name: str) -> None:
    root_path = request.getfixturevalue(fixture_name)
    ratio, label = evaluate_variable(
        VariableSpec(mode="ratio", branch_a="Muon_pt", branch_b="Muon_pt"),
        str(root_path),
        "Events",
        0,
        12,
    )
    assert label == "Muon_pt/Muon_pt"
    np.testing.assert_allclose(ratio, np.ones(len(ratio)))


@pytest.mark.parametrize("fixture_name", ["root_path", "rntuple_path"])
def test_delta_eta_label_and_identical_operands(request, fixture_name: str) -> None:
    root_path = request.getfixturevalue(fixture_name)
    delta_eta, label = evaluate_variable(
        VariableSpec(mode="delta_eta", branch_a="Muon_eta", branch_b="Muon_eta"),
        str(root_path),
        "Events",
        0,
        12,
    )

    assert label == "Δη(Muon_eta,Muon_eta)"
    np.testing.assert_allclose(delta_eta, np.zeros(len(delta_eta)))


@pytest.mark.parametrize("fixture_name", ["root_path", "rntuple_path"])
def test_single_particle_derived_quantities(request, fixture_name: str) -> None:
    root_path = str(request.getfixturevalue(fixture_name))

    def values(branch: str) -> np.ndarray:
        return evaluate_variable(
            VariableSpec(mode="branch", branch=branch), root_path, "Events", 0, 12
        )[0]

    px = values("Muon_px")
    py = values("Muon_py")
    pz = values("Muon_pz")
    pt = values("Muon_pt")
    eta = values("Muon_eta")
    mass = values("Muon_mass")
    energy = values("Muon_energy")
    met_phi = values("MET_phi")

    cases = [
        (
            VariableSpec(mode="absolute", branch_a="Muon_eta"),
            np.abs(eta),
            "|Muon_eta|",
        ),
        (
            VariableSpec(mode="absolute", branch_a="MET_phi"),
            np.abs(met_phi),
            "|MET_phi|",
        ),
        (
            VariableSpec(
                mode="magnitude_3d",
                branch_a="Muon_px",
                branch_b="Muon_py",
                branch_c="Muon_pz",
            ),
            np.hypot(np.hypot(px, py), pz),
            "mag(Muon_px,Muon_py,Muon_pz)",
        ),
        (
            VariableSpec(
                mode="energy_massless",
                branch_a="Muon_pt",
                branch_b="Muon_eta",
                longitudinal_kind="eta",
            ),
            pt * np.cosh(eta),
            "E_massless(Muon_pt,eta=Muon_eta)",
        ),
        (
            VariableSpec(
                mode="energy_massless",
                branch_a="Muon_pt",
                branch_b="Muon_eta",
                longitudinal_kind="y",
            ),
            pt * np.cosh(eta),
            "E_massless(Muon_pt,y=Muon_eta)",
        ),
        (
            VariableSpec(
                mode="energy_massive",
                branch_a="Muon_pt",
                branch_b="Muon_eta",
                branch_c="Muon_mass",
                longitudinal_kind="eta",
            ),
            np.sqrt(mass**2 + (pt * np.cosh(eta)) ** 2),
            "E(Muon_pt,eta=Muon_eta,Muon_mass)",
        ),
        (
            VariableSpec(
                mode="energy_massive",
                branch_a="Muon_pt",
                branch_b="Muon_eta",
                branch_c="Muon_mass",
                longitudinal_kind="y",
            ),
            np.hypot(mass, pt) * np.cosh(eta),
            "E(Muon_pt,y=Muon_eta,Muon_mass)",
        ),
        (
            VariableSpec(
                mode="invariant_mass",
                branch_a="Muon_energy",
                branch_b="Muon_px",
                branch_c="Muon_py",
                branch_d="Muon_pz",
            ),
            np.sqrt(np.maximum(energy**2 - px**2 - py**2 - pz**2, 0.0)),
            "m(Muon_energy,Muon_px,Muon_py,Muon_pz)",
        ),
    ]

    for spec, expected, expected_label in cases:
        actual, label = evaluate_variable(spec, root_path, "Events", 0, 12)
        assert label == expected_label
        np.testing.assert_allclose(actual, expected)


def test_invariant_mass_clamps_tiny_negative_mass_squared(tmp_path) -> None:
    root_path = tmp_path / "roundoff.root"
    with uproot.recreate(root_path) as root_file:
        root_file.mktree(
            "Events",
            {
                "energy": ak.Array([[1.0]]),
                "px": ak.Array([[np.nextafter(1.0, 2.0)]]),
                "py": ak.Array([[0.0]]),
                "pz": ak.Array([[0.0]]),
            },
        )

    result, _ = evaluate_variable(
        VariableSpec(
            mode="invariant_mass",
            branch_a="energy",
            branch_b="px",
            branch_c="py",
            branch_d="pz",
        ),
        str(root_path),
        "Events",
        0,
        1,
    )

    np.testing.assert_array_equal(result, [0.0])


@pytest.mark.parametrize("fixture_name", ["root_path", "rntuple_path"])
def test_new_derived_quantities_reject_mismatched_collections(request, fixture_name: str) -> None:
    root_path = str(request.getfixturevalue(fixture_name))

    with pytest.raises(ArrayShapeMismatchError, match=r"event|boundaries"):
        evaluate_variable(
            VariableSpec(
                mode="magnitude_3d",
                branch_a="Muon_px",
                branch_b="Jet_py",
                branch_c="Muon_pz",
            ),
            root_path,
            "Events",
            0,
            12,
        )
