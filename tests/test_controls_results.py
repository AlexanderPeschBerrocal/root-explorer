from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from root_explorer.app.controls import (
    COLOR_SCALES,
    needs_y_variable,
    needs_z_variable,
    selected_static_formats,
    supports_histogram_weights,
    supports_point_color,
)
from root_explorer.app.results import collection_summary, store_uploaded_file
from root_explorer.app.state import DataContext
from root_explorer.models import FieldInfo


class _Upload:
    name = "unsafe file.root"

    def getbuffer(self):
        return BytesIO(b"test root bytes").getbuffer()


def test_control_predicates_and_static_formats() -> None:
    assert COLOR_SCALES[-2:] == ["Greys", "coolwarm"]
    assert needs_y_variable("2D scatter")
    assert not needs_y_variable("1D histogram")
    assert needs_z_variable("3D scatter interactive")
    assert supports_point_color("2D scatter")
    assert not supports_point_color("1D histogram")
    assert supports_histogram_weights("2D histogram")
    assert selected_static_formats({"save_png": False, "save_pdf": True}) == ["pdf"]


def test_collection_summary_and_uploaded_file_storage() -> None:
    summary = collection_summary(
        [
            FieldInfo("Muon_pt", "float", "", "Muon", "pt"),
            FieldInfo("Muon_eta", "float", "", "Muon", "eta"),
            FieldInfo("event", "uint", "", "event-level", "event"),
        ]
    )
    assert summary.iloc[0].to_dict() == {
        "collection": "Muon",
        "fields": 2,
        "example fields": "pt, eta",
    }
    stored = store_uploaded_file(_Upload())
    digest = hashlib.sha256(b"test root bytes").hexdigest()[:12]
    assert Path(stored).name == f"unsafe_file_{digest}.root"
    assert Path(stored).read_bytes() == b"test root bytes"


def test_data_context_cache_key_uses_the_selected_range() -> None:
    context = DataContext(
        root_path="events.root",
        event_container_name="Events",
        n_entries=100,
        entry_start=10,
        entry_stop=50,
        field_info=[],
        fields=[],
    )

    assert context.cache_key == ("events.root", "Events", 10, 50)
