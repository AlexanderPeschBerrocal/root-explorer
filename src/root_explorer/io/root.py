"""Metadata-first ROOT catalogue and event-container access helpers."""

from __future__ import annotations

import argparse
import re
from collections.abc import Collection, Sequence
from typing import Any

import awkward as ak
import numpy as np
import streamlit as st
import uproot

from ..models import FieldInfo, FieldStructure, RootObjectInfo, RootObjectKind
from .source import SourceVersion, source_version

_ROOT_LEAFLIST_RE = re.compile(r"(?:[^:/]+(?:\[[^\]]+\])*/[BbSsIiFfDdLlOoGgCc](?::|$))+")


def split_root_cycle(key: str) -> tuple[str, int | None]:
    """Split a ROOT key into its cycle-free path and optional cycle number."""

    path, separator, cycle_text = key.rpartition(";")
    if separator and cycle_text.isdigit():
        return path, int(cycle_text)
    return key, None


def classify_root_object(classname: str) -> RootObjectKind:
    """Classify a ROOT classname without deserialising the corresponding object."""

    if classname.startswith("TDirectory"):
        return "directory"
    if classname == "TTree" or classname.startswith("TNtuple"):
        return "ttree"
    if classname == "RNTuple" or classname.endswith("::RNTuple"):
        return "rntuple"
    if classname.startswith(("TH1", "TH2", "TH3", "TProfile")):
        return "histogram"
    if classname.startswith("TGraph"):
        return "graph"
    if classname == "RooWorkspace" or classname.endswith("::RooWorkspace"):
        return "workspace"
    return "other"


def _root_object_sort_key(info: RootObjectInfo) -> tuple[str, str, int]:
    return (info.path.casefold(), info.path, -1 if info.cycle is None else info.cycle)


def build_root_catalogue(root_file: Any) -> tuple[RootObjectInfo, ...]:
    """Return metadata for all keyed ROOT objects without loading their payloads."""

    catalogue: list[RootObjectInfo] = []
    for key, classname in root_file.classnames(recursive=True, cycle=True).items():
        path, cycle = split_root_cycle(str(key))
        parent, separator, name = path.rpartition("/")
        if not separator:
            parent = ""
            name = path
        catalogue.append(
            RootObjectInfo(
                path=path,
                name=name,
                classname=str(classname),
                parent=parent,
                cycle=cycle,
                kind=classify_root_object(str(classname)),
            )
        )
    return tuple(sorted(catalogue, key=_root_object_sort_key))


def get_root_catalogue(path: str) -> tuple[RootObjectInfo, ...]:
    """Open a ROOT file and return its metadata-only object catalogue."""

    return _cached_get_root_catalogue(path, source_version(path))


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_get_root_catalogue(path: str, version: SourceVersion) -> tuple[RootObjectInfo, ...]:
    """Cached implementation; ``version`` only keys the cache (see ``io.source``)."""

    with uproot.open(path) as root_file:
        return build_root_catalogue(root_file)


def latest_root_objects(catalogue: Sequence[RootObjectInfo]) -> tuple[RootObjectInfo, ...]:
    """Return only the highest ROOT cycle for every cycle-free object path."""

    latest: dict[str, RootObjectInfo] = {}
    for info in catalogue:
        previous = latest.get(info.path)
        previous_cycle = -1 if previous is None or previous.cycle is None else previous.cycle
        current_cycle = -1 if info.cycle is None else info.cycle
        if previous is None or current_cycle >= previous_cycle:
            latest[info.path] = info
    return tuple(sorted(latest.values(), key=_root_object_sort_key))


def filter_root_objects(
    catalogue: Sequence[RootObjectInfo],
    *,
    kinds: Collection[RootObjectKind] | None = None,
    latest_only: bool = True,
) -> tuple[RootObjectInfo, ...]:
    """Filter catalogue entries by kind, optionally keeping only the latest cycles."""

    objects = latest_root_objects(catalogue) if latest_only else tuple(catalogue)
    if kinds is None:
        return objects
    return tuple(info for info in objects if info.kind in kinds)


def event_container_infos(
    catalogue: Sequence[RootObjectInfo], *, latest_only: bool = True
) -> tuple[RootObjectInfo, ...]:
    """Return TTree and RNTuple catalogue entries."""

    return filter_root_objects(
        catalogue,
        kinds={"ttree", "rntuple"},
        latest_only=latest_only,
    )


def event_container_kind_from_info(info: RootObjectInfo) -> str:
    """Return the user-facing event-container type from catalogue metadata."""

    if info.kind == "rntuple":
        return "RNTuple"
    if info.kind == "ttree":
        return "TTree"
    raise TypeError(f"ROOT object is not an event container: {info.classname}")


def parse_cli_rootfile() -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("rootfile", nargs="?")
    args, _ = parser.parse_known_args()
    return args.rootfile  # type: ignore[no-any-return]


def open_root_file(path: str) -> Any:
    """Return a shared Uproot handle for the current version of ``path``."""

    return _cached_open_root_file(path, source_version(path))


@st.cache_resource(show_spinner=False, max_entries=8)
def _cached_open_root_file(path: str, version: SourceVersion) -> Any:
    """Cached implementation; ``version`` only keys the cache (see ``io.source``).

    Every source version opens a new handle, so the number of retained handles is
    bounded to avoid keeping many superseded files open.
    """

    return uproot.open(path)


def is_event_container(obj: Any) -> bool:
    """Return whether ``obj`` is a readable TTree or RNTuple."""

    return isinstance(
        obj,
        (
            uproot.behaviors.TTree.TTree,
            uproot.behaviors.RNTuple.RNTuple,
        ),
    )


def event_container_kind(obj: Any) -> str:
    """Return the ROOT storage type for a known event container."""

    if isinstance(obj, uproot.behaviors.RNTuple.RNTuple):
        return "RNTuple"
    if isinstance(obj, uproot.behaviors.TTree.TTree):
        return "TTree"
    raise TypeError(f"Unsupported event container type: {type(obj).__name__}")


def walk_event_containers(root_file: Any) -> list[str]:
    """Return readable TTree and RNTuple paths from the shared ROOT catalogue."""

    return [info.path for info in event_container_infos(build_root_catalogue(root_file))]


def get_entry_count(path: str, container_name: str) -> int:
    """Return the number of entries in an event container."""

    return _cached_get_entry_count(path, source_version(path), container_name)


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_get_entry_count(path: str, version: SourceVersion, container_name: str) -> int:
    """Cached implementation; ``version`` only keys the cache (see ``io.source``)."""

    return int(uproot.open(path)[container_name].num_entries)


def split_collection(field_name: str) -> tuple[str, str]:
    """Infer a HEP-style collection and field from a field name.

    Examples
    --------
    Muon_pt      -> (Muon, pt)
    Electron.eta -> (Electron, eta)
    eventNumber  -> (event-level, eventNumber)
    """

    if "." in field_name:
        head, tail = field_name.split(".", 1)
        return head or "event-level", tail or field_name
    if "_" in field_name:
        head, tail = field_name.split("_", 1)
        if head and tail:
            return head, tail
    return "event-level", field_name


def source_field_description(field_object: Any, field_name: str) -> str | None:
    """Return useful source-provided documentation for a field when available.

    RNTuple fields expose a dedicated ``description`` attribute. TTree branches
    generally use ``title`` for their ROOT leaf-list declaration, but files may
    replace it with human-readable documentation. Leaf-list implementation
    strings are deliberately excluded because typename and interpretation are
    already displayed separately.
    """

    for attribute in ("description", "title"):
        try:
            raw_value = getattr(field_object, attribute)
        except Exception:
            continue
        try:
            if isinstance(raw_value, bytes):
                value = raw_value.decode(errors="replace").strip()
            else:
                value = str(raw_value).strip() if raw_value is not None else ""
        except Exception:
            continue
        if not value or value == field_name or _ROOT_LEAFLIST_RE.fullmatch(value):
            continue
        return value
    return None


def _source_field_form(field_object: Any) -> ak.forms.Form | None:
    """Return one field's Awkward form using schema metadata only."""

    interpretation = getattr(field_object, "interpretation", None)
    if interpretation is not None and hasattr(interpretation, "awkward_form"):
        try:
            return interpretation.awkward_form(field_object.file)
        except Exception:
            pass

    if hasattr(field_object, "to_akform"):
        try:
            form, _ = field_object.to_akform()
            if isinstance(form, ak.forms.RecordForm) and len(form.contents) == 1:
                return form.contents[0]
            return form
        except Exception:
            pass
    return None


def _form_is_string(form: ak.forms.Form) -> bool:
    return form.parameter("__array__") in {"string", "bytestring", "char", "byte"}


def classify_field_form(
    form: ak.forms.Form,
) -> tuple[FieldStructure, str, tuple[int, ...]]:
    """Classify a schema form without reading field values."""

    if _form_is_string(form):
        return "unsupported", "string", ()

    if isinstance(
        form,
        (
            ak.forms.IndexedForm,
            ak.forms.IndexedOptionForm,
            ak.forms.ByteMaskedForm,
            ak.forms.BitMaskedForm,
            ak.forms.UnmaskedForm,
        ),
    ):
        return classify_field_form(form.content)

    if isinstance(form, ak.forms.NumpyForm):
        dtype = form.primitive
        try:
            numeric = np.issubdtype(np.dtype(dtype), np.number) or np.dtype(dtype) == np.bool_
        except TypeError:
            numeric = False
        if not numeric:
            return "unsupported", dtype, ()
        numpy_shape = tuple(int(size) for size in form.inner_shape)
        return ("regular" if numpy_shape else "scalar"), dtype, numpy_shape

    if isinstance(form, ak.forms.RegularForm):
        regular_shape: list[int] = []
        current: ak.forms.Form = form
        while isinstance(current, ak.forms.RegularForm):
            regular_shape.append(int(current.size))
            current = current.content
        structure, dtype, inner_shape = classify_field_form(current)
        if structure == "scalar":
            return "regular", dtype, (*regular_shape, *inner_shape)
        if structure == "regular":
            return "regular", dtype, (*regular_shape, *inner_shape)
        return "nested/record", dtype, ()

    if isinstance(form, (ak.forms.ListForm, ak.forms.ListOffsetForm)):
        structure, dtype, _ = classify_field_form(form.content)
        if structure == "scalar":
            return "jagged", dtype, ()
        return "nested/record", dtype, ()

    if isinstance(form, ak.forms.RecordForm):
        if len(form.contents) == 1:
            return classify_field_form(form.contents[0])
        return "nested/record", "record", ()

    if isinstance(form, ak.forms.EmptyForm):
        return "unknown", "unknown", ()
    return "unsupported", type(form).__name__.removesuffix("Form").casefold(), ()


def _fallback_field_structure(
    field_object: Any, typename: str
) -> tuple[FieldStructure, str, tuple[int, ...]]:
    """Classify unusual fields conservatively when no Awkward form is exposed."""

    lower = typename.casefold()
    if any(token in lower for token in ("string", "char*")):
        return "unsupported", "string", ()
    if getattr(field_object, "fields", None):
        return "nested/record", "record", ()
    if "vector<" in lower or typename.endswith("[]"):
        return "jagged", "unknown", ()
    fixed_sizes = tuple(int(size) for size in re.findall(r"\[(\d+)\]", typename))
    if fixed_sizes:
        return "regular", "unknown", fixed_sizes
    if is_probably_numeric_typename(typename):
        return "scalar", typename, ()
    return "unknown", "unknown", ()


def field_structure_metadata(
    field_object: Any, typename: str
) -> tuple[FieldStructure, str, tuple[int, ...]]:
    """Return structural category, logical dtype, and any fixed dimensions."""

    if getattr(field_object, "fields", None):
        return "nested/record", "record", ()
    form = _source_field_form(field_object)
    if form is not None:
        return classify_field_form(form)
    return _fallback_field_structure(field_object, typename)


def get_field_info(path: str, container_name: str) -> list[FieldInfo]:
    """Return display metadata for the fields in one event container."""

    return _cached_get_field_info(path, source_version(path), container_name)


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_get_field_info(
    path: str, version: SourceVersion, container_name: str
) -> list[FieldInfo]:
    """Cached implementation; ``version`` only keys the cache (see ``io.source``)."""

    f = uproot.open(path)
    container = f[container_name]
    field_names = container.keys()
    try:
        typenames = dict(container.typenames())
    except Exception:
        typenames = {}

    info: list[FieldInfo] = []
    for name in field_names:
        description = None
        structure: FieldStructure = "unknown"
        dtype = "unknown"
        fixed_shape: tuple[int, ...] = ()
        try:
            field_object = container[name]
            typename = str(typenames.get(name, getattr(field_object, "typename", "unknown")))
            interpretation = str(getattr(field_object, "interpretation", "RField"))
            description = source_field_description(field_object, name)
            structure, dtype, fixed_shape = field_structure_metadata(field_object, typename)
        except Exception:
            typename = "unknown"
            interpretation = "unknown"
        collection, field = split_collection(name)
        info.append(
            FieldInfo(
                name,
                typename,
                interpretation,
                collection,
                field,
                description,
                structure,
                dtype,
                fixed_shape,
            )
        )
    return info


def read_field_uncached(
    path: str, container_name: str, field_name: str, entry_start: int, entry_stop: int | None
) -> ak.Array:
    """Read one field without framework-specific caching."""

    f = uproot.open(path)
    container = f[container_name]
    arr: ak.Array = container[field_name].array(
        library="ak", entry_start=entry_start, entry_stop=entry_stop
    )
    return arr


def read_field(
    path: str, container_name: str, field_name: str, entry_start: int, entry_stop: int | None
) -> ak.Array:
    return _cached_read_field(
        path, source_version(path), container_name, field_name, entry_start, entry_stop
    )


@st.cache_data(show_spinner=True, max_entries=64)
def _cached_read_field(
    path: str,
    version: SourceVersion,
    container_name: str,
    field_name: str,
    entry_start: int,
    entry_stop: int | None,
) -> ak.Array:
    """Cached implementation; ``version`` only keys the cache (see ``io.source``)."""

    return read_field_uncached(path, container_name, field_name, entry_start, entry_stop)


def read_event_field_uncached(
    path: str, container_name: str, field_name: str, event_index: int
) -> Any:
    """Read one event field without framework-specific caching."""

    f = uproot.open(path)
    container = f[container_name]
    arr = container[field_name].array(
        library="ak", entry_start=event_index, entry_stop=event_index + 1
    )
    return arr[0]


def read_event_field(path: str, container_name: str, field_name: str, event_index: int) -> Any:
    return _cached_read_event_field(
        path, source_version(path), container_name, field_name, event_index
    )


@st.cache_data(show_spinner=True, max_entries=128)
def _cached_read_event_field(
    path: str, version: SourceVersion, container_name: str, field_name: str, event_index: int
) -> Any:
    """Cached implementation; ``version`` only keys the cache (see ``io.source``)."""

    return read_event_field_uncached(path, container_name, field_name, event_index)


def is_probably_numeric_typename(typename: str) -> bool:
    lower = typename.lower()
    return any(
        token in lower for token in ["int", "float", "double", "bool", "char", "short", "long"]
    )


def numeric_fields(field_info: list[FieldInfo]) -> list[str]:
    numeric = [field.name for field in field_info if is_probably_numeric_typename(field.typename)]
    return numeric or [field.name for field in field_info]


def field_choices(fields: list[str], query: str = "") -> list[str]:
    if not query.strip():
        return fields
    q = query.casefold()
    return [field for field in fields if q in field.casefold()]


def choose_default(options: Sequence[str], preferred: Sequence[str]) -> int:
    for item in preferred:
        if item in options:
            return list(options).index(item)
    for item in preferred:
        for i, option in enumerate(options):
            if item.casefold() in option.casefold():
                return i
    return 0
