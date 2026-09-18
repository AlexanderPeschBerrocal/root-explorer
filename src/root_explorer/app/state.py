"""Small state objects passed between Streamlit UI sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import FieldInfo


@dataclass(frozen=True)
class DataContext:
    """The selected ROOT event data and entry range for one Streamlit rerun."""

    root_path: str
    event_container_name: str
    n_entries: int
    entry_start: int
    entry_stop: int
    field_info: list[FieldInfo]
    fields: list[str]
    container_kind: str = "TTree"

    @property
    def cache_key(self) -> tuple[str, str, int, int]:
        return (self.root_path, self.event_container_name, self.entry_start, self.entry_stop)

    @property
    def data_item_name(self) -> str:
        """Return the user-facing ROOT term for a selected data item."""

        return "field" if self.container_kind == "RNTuple" else "branch"

    @property
    def data_item_plural(self) -> str:
        return "fields" if self.data_item_name == "field" else "branches"


@dataclass(frozen=True)
class GeometryState:
    """Optional geometry data shared by the geometry and plot tabs."""

    scene: dict[str, Any] | None
    options: dict[str, Any]
