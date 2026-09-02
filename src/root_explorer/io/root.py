"""ROOT file and branch access used by the application and analysis helpers."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import streamlit as st
import uproot

from ..models import BranchInfo


def parse_cli_rootfile() -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("rootfile", nargs="?")
    args, _ = parser.parse_known_args()
    return args.rootfile


@st.cache_resource(show_spinner=False)
def open_root_file(path: str):
    return uproot.open(path)


def is_ttree(obj: Any) -> bool:
    return isinstance(obj, uproot.behaviors.TTree.TTree)


def walk_ttrees(root_file: Any) -> list[str]:
    """Return all readable TTree paths in a ROOT file, including nested folders."""

    names: list[str] = []
    for key, obj in root_file.items(recursive=True):
        try:
            if is_ttree(obj):
                names.append(key.split(";")[0])
        except Exception:
            continue
    return sorted(set(names))


def split_collection(branch_name: str) -> tuple[str, str]:
    """Infer a HEP-style collection and field from a branch name.

    Examples
    --------
    Muon_pt      -> (Muon, pt)
    Electron.eta -> (Electron, eta)
    eventNumber  -> (event-level, eventNumber)
    """

    if "." in branch_name:
        head, tail = branch_name.split(".", 1)
        return head or "event-level", tail or branch_name
    if "_" in branch_name:
        head, tail = branch_name.split("_", 1)
        if head and tail:
            return head, tail
    return "event-level", branch_name


@st.cache_data(show_spinner=False)
def get_branch_info(path: str, tree_name: str) -> list[BranchInfo]:
    f = uproot.open(path)
    t = f[tree_name]
    info: list[BranchInfo] = []
    for name in t.keys():
        try:
            branch = t[name]
            typename = str(getattr(branch, "typename", "unknown"))
            interpretation = str(getattr(branch, "interpretation", "unknown"))
        except Exception:
            typename = "unknown"
            interpretation = "unknown"
        collection, field = split_collection(name)
        info.append(BranchInfo(name, typename, interpretation, collection, field))
    return info


@st.cache_data(show_spinner=True, max_entries=64)
def read_branch(path: str, tree_name: str, branch: str, entry_start: int, entry_stop: int | None):
    f = uproot.open(path)
    t = f[tree_name]
    return t[branch].array(library="ak", entry_start=entry_start, entry_stop=entry_stop)


@st.cache_data(show_spinner=True, max_entries=128)
def read_event_branch(path: str, tree_name: str, branch: str, event_index: int):
    f = uproot.open(path)
    t = f[tree_name]
    arr = t[branch].array(library="ak", entry_start=event_index, entry_stop=event_index + 1)
    return arr[0]


def is_probably_numeric_typename(typename: str) -> bool:
    lower = typename.lower()
    return any(
        token in lower for token in ["int", "float", "double", "bool", "char", "short", "long"]
    )


def numeric_branches(branch_info: list[BranchInfo]) -> list[str]:
    numeric = [b.name for b in branch_info if is_probably_numeric_typename(b.typename)]
    return numeric or [b.name for b in branch_info]


def branch_choices(branches: list[str], query: str = "") -> list[str]:
    if not query.strip():
        return branches
    q = query.casefold()
    return [b for b in branches if q in b.casefold()]


def choose_default(options: Sequence[str], preferred: Sequence[str]) -> int:
    for item in preferred:
        if item in options:
            return list(options).index(item)
    for item in preferred:
        for i, option in enumerate(options):
            if item.casefold() in option.casefold():
                return i
    return 0
