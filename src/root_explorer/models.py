"""Small data models shared by ROOT Explorer components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FieldStructure = Literal[
    "scalar",
    "regular",
    "jagged",
    "nested/record",
    "unknown",
    "unsupported",
]


RootObjectKind = Literal[
    "directory",
    "ttree",
    "rntuple",
    "histogram",
    "graph",
    "workspace",
    "other",
]


@dataclass(frozen=True)
class RootObjectInfo:
    """Metadata-only description of one keyed object in a ROOT file."""

    path: str
    name: str
    classname: str
    parent: str
    cycle: int | None
    kind: RootObjectKind

    @property
    def qualified_path(self) -> str:
        """Return the ROOT path including its cycle when one is available."""

        if self.cycle is None:
            return self.path
        return f"{self.path};{self.cycle}"


@dataclass(frozen=True)
class FieldInfo:
    """Small summary of one field in an event container."""

    name: str
    typename: str
    interpretation: str
    collection: str
    field: str
    description: str | None = None
    structure: FieldStructure = "unknown"
    dtype: str = "unknown"
    fixed_shape: tuple[int, ...] = ()


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
