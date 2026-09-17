from __future__ import annotations

import sys

import awkward as ak
import numpy as np
import pytest

from root_explorer.app.results import collection_summary
from root_explorer.io import root as root_io


class _FakeBranch:
    typename = "float32"
    interpretation = "fake interpretation"


class _FakeTree:
    def __init__(self, names: list[str]) -> None:
        self._branches = {name: _FakeBranch() for name in names}

    def __iter__(self):
        # Match Uproot's TTree behavior: direct iteration yields branch objects.
        return iter(self._branches.values())

    def keys(self) -> list[str]:
        return list(self._branches)

    def typenames(self) -> dict[str, str]:
        return dict.fromkeys(self._branches, "float32")

    def __getitem__(self, name: str) -> _FakeBranch:
        return self._branches[name]


class _FakeFile:
    def __init__(self, tree: _FakeTree) -> None:
        self.tree = tree

    def __getitem__(self, name: str) -> _FakeTree:
        assert name == "Events"
        return self.tree


def test_get_field_info_keeps_field_names_as_strings(monkeypatch) -> None:
    tree = _FakeTree(["run", "Muon_pt", "Muon_eta"])
    monkeypatch.setattr(root_io.uproot, "open", lambda path: _FakeFile(tree))

    info = root_io.get_field_info("fake.root", "Events")

    assert [field.name for field in info] == ["run", "Muon_pt", "Muon_eta"]
    assert [field.field for field in info] == ["run", "pt", "eta"]
    summary = collection_summary(info)
    assert summary.loc[summary["collection"] == "Muon", "example fields"].item() == "pt, eta"


@pytest.mark.parametrize(
    ("fixture_name", "expected_containers"),
    [
        ("root_path", ["Events"]),
        ("rntuple_path", ["Events", "nested/Events"]),
    ],
)
def test_event_container_metadata_reads_and_helper_choices(
    request, fixture_name: str, expected_containers: list[str]
) -> None:
    path = request.getfixturevalue(fixture_name)
    root_file = root_io.open_root_file(str(path))
    field_info = root_io.get_field_info(str(path), "Events")
    muon_pt = root_io.read_field(str(path), "Events", "Muon_pt", 0, 4)
    event_muon_pt = root_io.read_event_field(str(path), "Events", "Muon_pt", 0)

    assert root_io.walk_event_containers(root_file) == expected_containers
    assert root_io.get_entry_count(str(path), "Events") == 12
    assert {field.name for field in field_info} >= {"event", "Muon_pt", "Jet_pt", "genWeight"}
    assert "Muon_pt" in root_io.numeric_fields(field_info)
    assert root_io.field_choices(["Muon_pt", "Jet_pt"], "muON") == ["Muon_pt"]
    assert root_io.choose_default(["Jet_pt", "Muon_pt"], ["muon"]) == 1
    assert isinstance(muon_pt, ak.Array)
    np.testing.assert_allclose(np.asarray(event_muon_pt), np.asarray(muon_pt[0]))


def test_rntuple_nested_field_reads_like_its_flat_equivalent(rntuple_path) -> None:
    nested_muon_pt = root_io.read_field(str(rntuple_path), "Events", "Muon.pt", 0, 4)
    flat_muon_pt = root_io.read_field(str(rntuple_path), "Events", "Muon_pt", 0, 4)
    field_info = root_io.get_field_info(str(rntuple_path), "Events")

    assert "Muon.pt" in {field.name for field in field_info}
    assert "Muon.pt" in root_io.numeric_fields(field_info)
    assert ak.almost_equal(nested_muon_pt, flat_muon_pt)


def test_mixed_nested_event_containers_are_discovered(mixed_event_containers_path) -> None:
    root_file = root_io.open_root_file(str(mixed_event_containers_path))

    assert root_io.walk_event_containers(root_file) == ["ntuples/Events", "trees/Events"]
    assert root_io.is_event_container(root_file["ntuples/Events"])
    assert root_io.is_event_container(root_file["trees/Events"])


def test_root_name_helpers_and_cli_path(monkeypatch) -> None:
    assert root_io.split_collection("Electron.eta") == ("Electron", "eta")
    assert root_io.split_collection("eventNumber") == ("event-level", "eventNumber")
    assert root_io.is_probably_numeric_typename("std::vector<float>")
    assert not root_io.is_probably_numeric_typename("std::string")

    monkeypatch.setattr(sys, "argv", ["streamlit_app.py", "input.root", "--server.port", "8501"])
    assert root_io.parse_cli_rootfile() == "input.root"
