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

    def __init__(self, *, title: str = "", description: str | None = None) -> None:
        self.title = title
        if description is not None:
            self.description = description


class _FakeTree:
    def __init__(self, names: list[str], branches: dict[str, _FakeBranch] | None = None) -> None:
        self._branches = branches or {name: _FakeBranch() for name in names}

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
    # The fake path does not exist; give it a fixed source version.
    monkeypatch.setattr(root_io, "source_version", lambda path: ("local", path, 0, 0))

    info = root_io.get_field_info("fake.root", "Events")

    assert [field.name for field in info] == ["run", "Muon_pt", "Muon_eta"]
    assert [field.field for field in info] == ["run", "pt", "eta"]
    summary = collection_summary(info)
    assert summary.loc[summary["collection"] == "Muon", "example fields"].item() == "pt, eta"


def test_get_field_info_exposes_useful_source_documentation(monkeypatch) -> None:
    tree = _FakeTree(
        [],
        {
            "Muon_pt": _FakeBranch(title="Muon transverse momentum [GeV]"),
            "Muon_eta": _FakeBranch(title="Muon_eta[nMuon]/F"),
            "covariance": _FakeBranch(title="covariance[3][3]/D"),
            "run": _FakeBranch(title="run/i"),
            "event": _FakeBranch(description="Unique event identifier"),
            "empty": _FakeBranch(description=""),
        },
    )
    monkeypatch.setattr(root_io.uproot, "open", lambda path: _FakeFile(tree))
    # The fake path does not exist; give it a fixed source version.
    monkeypatch.setattr(root_io, "source_version", lambda path: ("local", path, 0, 0))

    info = {field.name: field for field in root_io.get_field_info("fake.root", "Events")}

    assert info["Muon_pt"].description == "Muon transverse momentum [GeV]"
    assert info["event"].description == "Unique event identifier"
    assert info["Muon_eta"].description is None
    assert info["covariance"].description is None
    assert info["run"].description is None
    assert info["empty"].description is None


class _MetadataOnlyRootFile:
    """Fake ROOT directory that fails if discovery tries to load object payloads."""

    def classnames(self, *, recursive: bool, cycle: bool) -> dict[str, str]:
        assert recursive
        assert cycle
        return {
            "Events;1": "TTree",
            "Analysis;1": "TDirectoryFile",
            "Analysis/h_mass;2": "TH1D",
            "Analysis/note;1": "TObjString",
        }

    def items(self, *args, **kwargs):
        raise AssertionError("catalogue discovery must not call items()")

    def __getitem__(self, key):
        raise AssertionError(f"catalogue discovery must not load {key}")


@pytest.mark.parametrize(
    ("classname", "expected_kind"),
    [
        ("TDirectoryFile", "directory"),
        ("TTree", "ttree"),
        ("TNtuple", "ttree"),
        ("ROOT::RNTuple", "rntuple"),
        ("TH1D", "histogram"),
        ("TH2F", "histogram"),
        ("TProfile", "histogram"),
        ("TGraphAsymmErrors", "graph"),
        ("RooWorkspace", "workspace"),
        ("TObjString", "other"),
    ],
)
def test_root_object_classification_is_deterministic(classname: str, expected_kind: str) -> None:
    assert root_io.classify_root_object(classname) == expected_kind


def test_root_catalogue_discovery_does_not_load_payload_objects() -> None:
    root_file = _MetadataOnlyRootFile()

    catalogue = root_io.build_root_catalogue(root_file)

    by_path = {info.path: info for info in catalogue}
    assert by_path["Events"].kind == "ttree"
    assert by_path["Analysis"].kind == "directory"
    assert by_path["Analysis/h_mass"].kind == "histogram"
    assert by_path["Analysis/h_mass"].cycle == 2
    assert by_path["Analysis/note"].kind == "other"
    assert root_io.walk_event_containers(root_file) == ["Events"]


def test_nontrivial_root_catalogue_resolves_hierarchy_cycles_and_duplicate_names(
    catalogue_root_path,
) -> None:
    catalogue = root_io.get_root_catalogue(str(catalogue_root_path))
    latest = root_io.latest_root_objects(catalogue)
    by_path = {info.path: info for info in latest}

    assert by_path["Events"].kind == "ttree"
    assert by_path["Analysis"].kind == "directory"
    assert by_path["Analysis/nested"].kind == "directory"
    assert by_path["Analysis/nested/Events"].kind == "rntuple"
    assert by_path["Analysis/h_mass"].kind == "histogram"
    assert by_path["Control/h_mass"].kind == "histogram"
    assert by_path["metadata"].kind == "other"

    assert by_path["Analysis/duplicate"].parent == "Analysis"
    assert by_path["Control/duplicate"].parent == "Control"
    assert by_path["Analysis/duplicate"].name == by_path["Control/duplicate"].name == "duplicate"

    repeated = [info for info in catalogue if info.path == "Analysis/repeated"]
    assert len(repeated) == 2
    assert [info.cycle for info in repeated] == [1, 2]
    assert by_path["Analysis/repeated"].cycle == 2
    assert by_path["Analysis/repeated"].qualified_path == "Analysis/repeated;2"

    assert [info.path for info in root_io.event_container_infos(catalogue)] == [
        "Analysis/nested/Events",
        "Events",
    ]


@pytest.mark.parametrize(
    ("form", "expected"),
    [
        (ak.forms.NumpyForm("float32"), ("scalar", "float32", ())),
        (
            ak.forms.RegularForm(ak.forms.NumpyForm("int16"), 3),
            ("regular", "int16", (3,)),
        ),
        (
            ak.forms.ListOffsetForm("i64", ak.forms.NumpyForm("float64")),
            ("jagged", "float64", ()),
        ),
        (
            ak.forms.ListOffsetForm(
                "i64", ak.forms.ListOffsetForm("i64", ak.forms.NumpyForm("float64"))
            ),
            ("nested/record", "float64", ()),
        ),
        (
            ak.forms.RecordForm(
                [ak.forms.NumpyForm("float32"), ak.forms.NumpyForm("float32")],
                ["x", "y"],
            ),
            ("nested/record", "record", ()),
        ),
        (
            ak.forms.NumpyForm("uint8", parameters={"__array__": "char"}),
            ("unsupported", "string", ()),
        ),
        (ak.forms.EmptyForm(), ("unknown", "unknown", ())),
    ],
)
def test_classify_field_form_covers_supported_shape_categories(form, expected) -> None:
    assert root_io.classify_field_form(form) == expected


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
    assert all(field.description is None for field in field_info)
    by_name = {field.name: field for field in field_info}
    assert (by_name["event"].structure, by_name["event"].dtype) == ("scalar", "uint64")
    assert (by_name["Muon_pt"].structure, by_name["Muon_pt"].dtype) == (
        "jagged",
        "float32",
    )
    assert "Muon_pt" in root_io.numeric_fields(field_info)
    assert root_io.field_choices(["Muon_pt", "Jet_pt"], "muON") == ["Muon_pt"]
    assert root_io.choose_default(["Jet_pt", "Muon_pt"], ["muon"]) == 1
    assert isinstance(muon_pt, ak.Array)
    np.testing.assert_allclose(np.asarray(event_muon_pt), np.asarray(muon_pt[0]))


@pytest.mark.parametrize(
    ("fixture_name", "container_name", "expected_kind"),
    [
        ("root_path", "Events", "TTree"),
        ("rntuple_path", "Events", "RNTuple"),
    ],
)
def test_event_container_kind(
    request, fixture_name: str, container_name: str, expected_kind: str
) -> None:
    path = request.getfixturevalue(fixture_name)
    root_file = root_io.open_root_file(str(path))

    assert root_io.event_container_kind(root_file[container_name]) == expected_kind


def test_rntuple_nested_field_reads_like_its_flat_equivalent(rntuple_path) -> None:
    nested_muon_pt = root_io.read_field(str(rntuple_path), "Events", "Muon.pt", 0, 4)
    flat_muon_pt = root_io.read_field(str(rntuple_path), "Events", "Muon_pt", 0, 4)
    field_info = root_io.get_field_info(str(rntuple_path), "Events")

    assert "Muon.pt" in {field.name for field in field_info}
    assert "Muon.pt" in root_io.numeric_fields(field_info)
    by_name = {field.name: field for field in field_info}
    assert by_name["Muon"].structure == "nested/record"
    assert (by_name["Muon.pt"].structure, by_name["Muon.pt"].dtype) == (
        "jagged",
        "float32",
    )
    assert ak.almost_equal(nested_muon_pt, flat_muon_pt)


def test_mixed_nested_event_containers_are_discovered(mixed_event_containers_path) -> None:
    root_file = root_io.open_root_file(str(mixed_event_containers_path))

    assert root_io.walk_event_containers(root_file) == ["ntuples/Events", "trees/Events"]
    assert root_io.is_event_container(root_file["ntuples/Events"])
    assert root_io.is_event_container(root_file["trees/Events"])


def test_fixed_size_fields_are_classified_consistently_for_ttree_and_rntuple(tmp_path) -> None:
    path = tmp_path / "fixed_shapes.root"
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    with root_io.uproot.recreate(path) as root_file:
        tree = root_file.mktree("Tree", {"coordinates": np.dtype((np.float32, (3,)))})
        tree.extend({"coordinates": values})
        root_file.mkrntuple("Ntuple", {"coordinates": values})

    tree_field = root_io.get_field_info(str(path), "Tree")[0]
    ntuple_field = root_io.get_field_info(str(path), "Ntuple")[0]

    assert (tree_field.structure, tree_field.dtype, tree_field.fixed_shape) == (
        "regular",
        "float32",
        (3,),
    )
    assert (ntuple_field.structure, ntuple_field.dtype, ntuple_field.fixed_shape) == (
        "regular",
        "float32",
        (3,),
    )


def test_root_name_helpers_and_cli_path(monkeypatch) -> None:
    assert root_io.split_collection("Electron.eta") == ("Electron", "eta")
    assert root_io.split_collection("eventNumber") == ("event-level", "eventNumber")
    assert root_io.is_probably_numeric_typename("std::vector<float>")
    assert not root_io.is_probably_numeric_typename("std::string")

    monkeypatch.setattr(sys, "argv", ["streamlit_app.py", "input.root", "--server.port", "8501"])
    assert root_io.parse_cli_rootfile() == "input.root"
