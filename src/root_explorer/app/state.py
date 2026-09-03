"""Small state objects passed between Streamlit UI sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import BranchInfo


@dataclass(frozen=True)
class DataContext:
    """The selected ROOT data and entry range for one Streamlit rerun."""

    root_path: str
    tree_name: str
    n_entries: int
    entry_start: int
    entry_stop: int
    branch_info: list[BranchInfo]
    branches: list[str]

    @property
    def cache_key(self) -> tuple[str, str, int, int]:
        return (self.root_path, self.tree_name, self.entry_start, self.entry_stop)


@dataclass(frozen=True)
class GeometryState:
    """Optional geometry data shared by the geometry and plot tabs."""

    scene: dict[str, Any] | None
    options: dict[str, Any]
