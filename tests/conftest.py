from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import streamlit as st
import uproot

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_collection(
    rng: np.random.Generator,
    counts: np.ndarray,
    *,
    mass: float,
    pdg_id: int | None = None,
) -> dict[str, ak.Array]:
    """Create deterministic jagged particle branches for the test TTree."""

    n_particles = int(np.sum(counts))
    px = rng.normal(0.0, 35.0, n_particles).astype(np.float32)
    py = rng.normal(0.0, 35.0, n_particles).astype(np.float32)
    pz = rng.normal(0.0, 60.0, n_particles).astype(np.float32)
    pt = np.hypot(px, py).astype(np.float32)
    phi = np.arctan2(py, px).astype(np.float32)
    eta = np.arcsinh(pz / np.maximum(pt, 1e-6)).astype(np.float32)

    if pdg_id is None:
        mass_values = rng.uniform(2.0, 35.0, n_particles).astype(np.float32)
        charge = rng.integers(-1, 2, n_particles, dtype=np.int8)
        pdg_values = np.zeros(n_particles, dtype=np.int32)
    else:
        mass_values = np.full(n_particles, mass, dtype=np.float32)
        charge = rng.choice(np.array([-1, 1], dtype=np.int8), n_particles)
        pdg_values = (-pdg_id * charge).astype(np.int32)

    energy = np.sqrt(px**2 + py**2 + pz**2 + mass_values**2).astype(np.float32)
    return {
        "px": ak.unflatten(px, counts),
        "py": ak.unflatten(py, counts),
        "pz": ak.unflatten(pz, counts),
        "pt": ak.unflatten(pt, counts),
        "eta": ak.unflatten(eta, counts),
        "phi": ak.unflatten(phi, counts),
        "mass": ak.unflatten(mass_values, counts),
        "energy": ak.unflatten(energy, counts),
        "charge": ak.unflatten(charge, counts),
        "pdgId": ak.unflatten(pdg_values, counts),
    }


def _generate_event_fields(*, n_events: int, seed: int) -> dict[str, object]:
    """Create deterministic NanoAOD-like fields for either ROOT event format."""

    rng = np.random.default_rng(seed)
    muon_counts = rng.poisson(1.5, n_events).astype(np.int32)
    jet_counts = rng.poisson(4.0, n_events).astype(np.int32)
    muons = _make_collection(rng, muon_counts, mass=0.1057, pdg_id=13)
    jets = _make_collection(rng, jet_counts, mass=0.0)

    branches: dict[str, object] = {
        "run": np.ones(n_events, dtype=np.uint32),
        "event": np.arange(1, n_events + 1, dtype=np.uint64),
        "nMuon": muon_counts,
        "nJet": jet_counts,
        "MET_pt": rng.exponential(40.0, n_events).astype(np.float32),
        "MET_phi": rng.uniform(-np.pi, np.pi, n_events).astype(np.float32),
        "genWeight": (
            rng.choice(np.array([-1.0, 1.0]), n_events, p=[0.1, 0.9])
            * rng.uniform(0.5, 1.5, n_events)
        ).astype(np.float32),
    }
    branches.update({f"Muon_{field}": values for field, values in muons.items()})
    branches.update({f"Jet_{field}": values for field, values in jets.items()})

    return branches


def _generate_test_root(output_path: Path, *, n_events: int, seed: int) -> Path:
    """Write an isolated NanoAOD-like TTree."""

    with uproot.recreate(output_path) as root_file:
        root_file.mktree("Events", _generate_event_fields(n_events=n_events, seed=seed))
    return output_path


def _generate_test_rntuple(output_path: Path, *, n_events: int, seed: int) -> Path:
    """Write top-level and nested RNTuples with identical event data."""

    fields = _generate_event_fields(n_events=n_events, seed=seed)
    fields["Muon"] = ak.zip({"pt": fields["Muon_pt"], "eta": fields["Muon_eta"]})
    with uproot.recreate(output_path) as root_file:
        root_file.mkrntuple("Events", fields)
        root_file.mkdir("nested").mkrntuple("Events", fields)
    return output_path


def _generate_mixed_event_containers(output_path: Path, *, n_events: int, seed: int) -> Path:
    """Write a ROOT file with nested TTree and RNTuple event containers."""

    fields = _generate_event_fields(n_events=n_events, seed=seed)
    with uproot.recreate(output_path) as root_file:
        root_file.mkdir("trees").mktree("Events", fields)
        root_file.mkdir("ntuples").mkrntuple("Events", fields)
    return output_path


@pytest.fixture(autouse=True)
def clear_streamlit_caches() -> None:
    """Prevent cached reads and scenes from leaking between temporary fixtures."""

    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def root_path(tmp_path: Path) -> Path:
    return _generate_test_root(tmp_path / "test_nanoaod.root", n_events=12, seed=20250831)


@pytest.fixture
def rntuple_path(tmp_path: Path) -> Path:
    return _generate_test_rntuple(
        tmp_path / "test_nanoaod_rntuple.root", n_events=12, seed=20250831
    )


@pytest.fixture
def mixed_event_containers_path(tmp_path: Path) -> Path:
    return _generate_mixed_event_containers(
        tmp_path / "test_mixed_event_containers.root", n_events=12, seed=20250831
    )


@pytest.fixture
def tiny_gdml_path(tmp_path: Path) -> Path:
    """Create the smallest useful world-plus-detector GDML scene."""

    path = tmp_path / "tiny_detector.gdml"
    path.write_text(
        """<?xml version="1.0"?>
<gdml>
  <solids>
    <box name="world_box" x="20" y="20" z="20" lunit="mm"/>
    <box name="sensor_box" x="2" y="4" z="6" lunit="mm"/>
  </solids>
  <structure>
    <volume name="World">
      <materialref ref="Galactic"/>
      <solidref ref="world_box"/>
      <physvol name="sensor_placement">
        <volumeref ref="Sensor"/>
        <position x="5" y="0" z="0" unit="mm"/>
      </physvol>
    </volume>
    <volume name="Sensor">
      <materialref ref="G4_Si"/>
      <solidref ref="sensor_box"/>
    </volume>
  </structure>
  <setup name="Default" version="1.0"><world ref="World"/></setup>
</gdml>
""",
        encoding="utf-8",
    )
    return path
