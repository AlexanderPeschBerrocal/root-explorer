"""Format-neutral ROOT event-container access for the application and analysis helpers."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import awkward as ak
import streamlit as st
import uproot

from ..models import FieldInfo


def parse_cli_rootfile() -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("rootfile", nargs="?")
    args, _ = parser.parse_known_args()
    return args.rootfile  # type: ignore[no-any-return]


@st.cache_resource(show_spinner=False)
def open_root_file(path: str) -> Any:
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
    """Return readable TTree and RNTuple paths, including nested folders."""

    names: list[str] = []
    for key, obj in root_file.items(recursive=True):
        if is_event_container(obj):
            names.append(key.split(";")[0])
    return sorted(set(names))


@st.cache_data(show_spinner=False)
def get_entry_count(path: str, container_name: str) -> int:
    """Return the number of entries in an event container."""

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


@st.cache_data(show_spinner=False)
def get_field_info(path: str, container_name: str) -> list[FieldInfo]:
    """Return display metadata for the fields in one event container."""

    f = uproot.open(path)
    container = f[container_name]
    field_names = container.keys()
    try:
        typenames = dict(container.typenames())
    except Exception:
        typenames = {}

    info: list[FieldInfo] = []
    for name in field_names:
        try:
            field_object = container[name]
            typename = str(typenames.get(name, getattr(field_object, "typename", "unknown")))
            interpretation = str(getattr(field_object, "interpretation", "RField"))
        except Exception:
            typename = "unknown"
            interpretation = "unknown"
        collection, field = split_collection(name)
        info.append(FieldInfo(name, typename, interpretation, collection, field))
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


@st.cache_data(show_spinner=True, max_entries=64)
def read_field(
    path: str, container_name: str, field_name: str, entry_start: int, entry_stop: int | None
) -> ak.Array:
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


@st.cache_data(show_spinner=True, max_entries=128)
def read_event_field(path: str, container_name: str, field_name: str, event_index: int) -> Any:
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
