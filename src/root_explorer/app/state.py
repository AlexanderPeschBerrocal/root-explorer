"""Small state objects passed between Streamlit UI sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..io.source import SourceVersion
from ..models import FieldInfo, RootObjectInfo


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
    source_version: SourceVersion | None = None

    @property
    def cache_key(self) -> tuple[Any, ...]:
        # Including the source version discards retained plot and ranking results
        # when the selected file is replaced at the same path.
        return (
            self.root_path,
            self.event_container_name,
            self.entry_start,
            self.entry_stop,
            self.source_version,
        )

    @property
    def data_item_name(self) -> str:
        """Return the user-facing ROOT term for a selected data item."""

        return "field" if self.container_kind == "RNTuple" else "branch"

    @property
    def data_item_plural(self) -> str:
        return "fields" if self.data_item_name == "field" else "branches"


EventDataStatus = Literal[
    "available", "no_event_containers", "empty_container", "unreadable_container"
]


@dataclass(frozen=True)
class RootFileContext:
    """Selected ROOT file metadata plus optional event-data context.

    ``data`` is ``None`` whenever the file offers no usable event data; the
    catalogue remains available so the ROOT object browser can still be shown.
    ``event_data_status`` records why event data is unavailable.
    """

    root_path: str
    catalogue: tuple[RootObjectInfo, ...]
    data: DataContext | None
    event_data_status: EventDataStatus = "available"


@dataclass(frozen=True)
class GeometryState:
    """Optional geometry data shared by the geometry and plot tabs."""

    scene: dict[str, Any] | None
    options: dict[str, Any]
    source: tuple[str, SourceVersion] | None = None
