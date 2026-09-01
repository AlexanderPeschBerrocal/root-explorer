"""Small data models shared by ROOT Explorer components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BranchInfo:
    """Small summary of one TTree branch."""

    name: str
    typename: str
    interpretation: str
    collection: str
    field: str


@dataclass(frozen=True)
class VariableSpec:
    """A user-selected variable definition."""

    mode: str
    branch: Optional[str] = None
    index: int = 0
    branch_a: Optional[str] = None
    branch_b: Optional[str] = None
    branch_c: Optional[str] = None
    branch_d: Optional[str] = None
    label: str = ""
