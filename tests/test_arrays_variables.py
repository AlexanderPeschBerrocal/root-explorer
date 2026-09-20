from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

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
def test_derived_ratio_handles_zero_denominators(request, fixture_name: str) -> None:
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
def test_delta_eta_is_signed_difference(request, fixture_name: str) -> None:
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
