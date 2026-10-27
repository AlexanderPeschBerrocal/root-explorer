from __future__ import annotations

import gc
import hashlib
import weakref
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from root_explorer.analysis.fitting import FitResult
from root_explorer.analysis.histograms import HistogramFlowSummary
from root_explorer.analysis.statistics import QuantileResult, ThresholdResult
from root_explorer.app import results, tables
from root_explorer.app.controls import (
    COLOR_SCALES,
    field_option_labels,
    needs_y_variable,
    needs_z_variable,
    supports_histogram_weights,
    supports_point_color,
)
from root_explorer.app.results import (
    collection_summary,
    color_clipping_summary,
    discard_plot_result,
    store_plot_result,
    store_uploaded_file,
)
from root_explorer.app.state import DataContext
from root_explorer.models import FieldInfo
from root_explorer.plotting.plots import ColorClippingResult, matplotlib_cmap_name


class _Upload:
    name = "unsafe file.root"

    def getbuffer(self):
        return BytesIO(b"test root bytes").getbuffer()


def test_cached_matplotlib_results_do_not_retain_pyplot_managers(monkeypatch) -> None:
    session_state: dict[str, object] = {}
    monkeypatch.setattr(results.st, "session_state", session_state)
    baseline = set(plt.get_fignums())

    first = plt.figure()
    store_plot_result("plot_result", {"kind": "matplotlib", "figure": first})

    assert session_state["plot_result"]["figure"] is first  # type: ignore[index]
    assert set(plt.get_fignums()) == baseline
    assert results.matplotlib_figure_bytes(first, "png").startswith(b"\x89PNG")
    first_reference = weakref.ref(first)

    second = plt.figure()
    store_plot_result("plot_result", {"kind": "matplotlib", "figure": second})
    del first
    gc.collect()

    assert session_state["plot_result"]["figure"] is second  # type: ignore[index]
    assert set(plt.get_fignums()) == baseline
    assert first_reference() is None

    second_reference = weakref.ref(second)
    discard_plot_result("plot_result")
    del second
    gc.collect()
    assert "plot_result" not in session_state
    assert set(plt.get_fignums()) == baseline
    assert second_reference() is None


def test_plotly_results_are_not_closed(monkeypatch) -> None:
    session_state: dict[str, object] = {}
    monkeypatch.setattr(results.st, "session_state", session_state)
    figure = object()

    store_plot_result("plot_result", {"kind": "plotly", "figure": figure})
    discard_plot_result("plot_result")

    assert "plot_result" not in session_state


def test_control_predicates() -> None:
    assert COLOR_SCALES[-2:] == ["Greys", "Coolwarm"]
    # Every displayed name must resolve to a real Matplotlib colour map.
    assert all(matplotlib_cmap_name(name) in plt.colormaps() for name in COLOR_SCALES)
    assert needs_y_variable("2D scatter")
    assert not needs_y_variable("1D histogram")
    assert needs_z_variable("3D scatter interactive")
    assert supports_point_color("2D scatter")
    assert not supports_point_color("1D histogram")
    assert supports_histogram_weights("2D histogram")


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
    digest = hashlib.sha256(b"test root bytes").hexdigest()[:32]
    assert Path(stored).name == f"unsafe_file_{digest}.root"
    assert Path(stored).read_bytes() == b"test root bytes"


def test_data_context_cache_key_uses_the_selected_range_and_source_version() -> None:
    version = ("local", "/data/events.root", 123, 456)
    context = DataContext(
        root_path="events.root",
        event_container_name="Events",
        n_entries=100,
        entry_start=10,
        entry_stop=50,
        field_info=[],
        fields=[],
        source_version=version,
    )

    assert context.cache_key == ("events.root", "Events", 10, 50, version)
    replaced = DataContext(
        **{**context.__dict__, "source_version": ("local", "/data/events.root", 124, 456)}
    )
    assert replaced.cache_key != context.cache_key


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


def test_cached_plot_renders_primary_figure_before_secondary_analysis(monkeypatch) -> None:
    """The summary is followed by the figure, not lengthy secondary tables."""

    class StopBeforeExport(Exception):
        pass

    calls: list[str] = []
    plot_result = {
        "kind": "matplotlib",
        "plot_type": "1D histogram",
        "figure": object(),
        "summary": pd.DataFrame({"variable": ["x"], "n": [3]}),
        "warnings": ["KDE was unavailable."],
        "histogram_flow_mode": "fold",
        "histogram_flow_summary": HistogramFlowSummary(
            lower=0.0,
            upper=1.0,
            underflow_entries=1,
            in_range_entries=1,
            overflow_entries=1,
            underflow_sumw=1.0,
            in_range_sumw=1.0,
            overflow_sumw=1.0,
            underflow_sumw2=1.0,
            in_range_sumw2=1.0,
            overflow_sumw2=1.0,
            weighted=False,
        ),
        "threshold_result": ThresholdResult(
            threshold=0.5,
            total_count=3,
            below_count=2,
            above_count=1,
            below_fraction=2 / 3,
            above_fraction=1 / 3,
            weighted=False,
        ),
        "quantile_result": QuantileResult(50.0, 0.5, 3, False),
        "fit_result": FitResult(
            model="Gaussian",
            success=True,
            parameters={"mean": 0.5},
            uncertainties={"mean": 0.1},
            statistic=1.0,
            ndof=3,
            bins_used=5,
            bins_excluded=0,
            objective="Ordinary least squares",
            fit_range=None,
            message="Fit converged.",
        ),
        "preview": pd.DataFrame({"x": [0.5]}),
        "file_name": "example",
    }

    def show_table(dataframe, **kwargs) -> None:
        if "variable" in dataframe.columns:
            calls.append("summary")
        elif "Region" in dataframe.columns:
            calls.append("flow")
        elif "selection" in dataframe.columns:
            calls.append("threshold")
        elif "percentile (%)" in dataframe.columns:
            calls.append("quantile")
        elif "Parameter" in dataframe.columns:
            calls.append("fit")
        elif "x" in dataframe.columns:
            calls.append("preview")

    def stop_export(*args, **kwargs):
        calls.append("export")
        raise StopBeforeExport

    monkeypatch.setattr(results.st, "session_state", {"test_plot_result": plot_result})
    monkeypatch.setattr(results, "render_dataframe", show_table)
    monkeypatch.setattr(results.st, "caption", lambda message: calls.append("caption:" + message))
    monkeypatch.setattr(results.st, "markdown", lambda message, **kwargs: calls.append(message))
    monkeypatch.setattr(results.st, "warning", lambda message: calls.append("warning:" + message))
    monkeypatch.setattr(results.st, "pyplot", lambda *args, **kwargs: calls.append("figure"))
    monkeypatch.setattr(results, "export_controls", stop_export)

    with pytest.raises(StopBeforeExport):
        results.show_cached_plot_result(
            "test_plot_result",
            export_prefix="plot",
            download_key="plot_export",
            make_plot_label="Make plot",
            make_plot_shortcut="P",
            empty_message="No plot",
        )

    assert (
        calls.index("summary")
        < next(i for i, call in enumerate(calls) if call.startswith("caption:Summary statistics"))
        < calls.index("figure")
    )
    assert calls.index("figure") < calls.index("warning:KDE was unavailable.")
    assert calls.index("figure") < calls.index("flow")
    assert calls.index("flow") < calls.index("threshold") < calls.index("quantile")
    assert calls.index("quantile") < calls.index("fit") < calls.index("preview")
    assert calls.index("preview") < calls.index("export")
    assert not any("statistics above" in call for call in calls)
