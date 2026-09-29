from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd

from root_explorer.app import tables
from root_explorer.app.controls import (
    COLOR_SCALES,
    field_option_labels,
    needs_y_variable,
    needs_z_variable,
    selected_static_formats,
    supports_histogram_weights,
    supports_point_color,
)
from root_explorer.app.results import (
    collection_summary,
    color_clipping_summary,
    store_uploaded_file,
)
from root_explorer.app.state import DataContext
from root_explorer.models import FieldInfo
from root_explorer.plotting.plots import ColorClippingResult


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


def test_field_option_labels_keep_names_first_and_show_shape_and_dtype() -> None:
    labels = field_option_labels(
        [
            FieldInfo(
                "event",
                "uint64_t",
                "",
                "event-level",
                "event",
                structure="scalar",
                dtype="uint64",
            ),
            FieldInfo(
                "Muon_cov",
                "float[3][3]",
                "",
                "Muon",
                "cov",
                structure="regular",
                dtype="float32",
                fixed_shape=(3, 3),
            ),
            FieldInfo(
                "Muon_pt",
                "float[]",
                "",
                "Muon",
                "pt",
                structure="jagged",
                dtype="float32",
            ),
        ]
    )

    assert labels == {
        "event": "event  [scalar] [uint64]",
        "Muon_cov": "Muon_cov  [fixed 3x3] [float32]",
        "Muon_pt": "Muon_pt  [jagged] [float32]",
    }


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


def test_color_clipping_summary_handles_counts_and_optional_weights() -> None:
    unweighted = ColorClippingResult(1.0, 10.0, 2, 3)
    assert color_clipping_summary(unweighted) == (
        "Color clipping: 2 points below color min 1 · 3 points above color max 10"
    )

    weighted = ColorClippingResult(
        1.0,
        10.0,
        2,
        3,
        below_weight=-0.25,
        above_weight=4.5,
    )
    assert color_clipping_summary(weighted) == (
        "Color clipping: 2 points below color min 1 (sum of weights: -0.25) · "
        "3 points above color max 10 (sum of weights: 4.5)"
    )

    assert color_clipping_summary(ColorClippingResult(1.0, 10.0, 0, 0)) is None


def test_render_dataframe_centres_cells_and_uses_shared_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def capture_dataframe(dataframe, **kwargs) -> None:
        captured["dataframe"] = dataframe
        captured.update(kwargs)

    monkeypatch.setattr(tables.st, "dataframe", capture_dataframe)
    dataframe = pd.DataFrame({"variable": ["Muon_pt"], "n": [1]})

    tables.render_dataframe(dataframe)

    assert captured["dataframe"] is dataframe
    assert captured["width"] == "stretch"
    assert captured["height"] == "auto"
    assert captured["hide_index"] is True
    column_config = captured["column_config"]
    assert set(column_config) == {"variable", "n"}
    assert all(config["alignment"] == "center" for config in column_config.values())

    tables.render_dataframe(dataframe, height=220)
    assert captured["height"] == 220
