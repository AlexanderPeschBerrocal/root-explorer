"""Small data models shared by ROOT Explorer components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldInfo:
    """Small summary of one field in an event container."""

    name: str
    typename: str
    interpretation: str
    collection: str
    field: str
    description: str | None = None


@dataclass(frozen=True)
class VariableSpec:
    """A user-selected variable definition."""

    mode: str
    branch: str | None = None
    index: int = 0
    branch_a: str | None = None
    branch_b: str | None = None
    branch_c: str | None = None
    branch_d: str | None = None
    label: str = ""
    longitudinal_kind: str = "eta"
