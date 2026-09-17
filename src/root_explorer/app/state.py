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

    @property
    def cache_key(self) -> tuple[str, str, int, int]:
        return (self.root_path, self.event_container_name, self.entry_start, self.entry_stop)


@dataclass(frozen=True)
class GeometryState:
    """Optional geometry data shared by the geometry and plot tabs."""

    scene: dict[str, Any] | None
    options: dict[str, Any]
