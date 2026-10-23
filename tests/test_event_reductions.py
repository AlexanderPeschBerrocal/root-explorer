"""Per-event reductions as point colours and cuts (#42).

Event-level values are repeated over the plotted objects of their event, so an
event-level cut keeps or rejects all plotted objects of an event together.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pytest
import uproot
from streamlit.testing.v1 import AppTest
from test_codegen import _execute_export, _request

from root_explorer.analysis.arrays import ArrayShapeMismatchError
from root_explorer.models import VariableSpec
from root_explorer.plotting.pipeline import StaticPlotSpec, prepare_static_plot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_EVENTS = 3
REDUCTION_LABELS = {
    "Multiplicity per event",
    "nth object per event",
    "Sum per event",
    "Mean per event",
    "Max per event",
    "Min per event",
}


@pytest.fixture
def events_path(tmp_path) -> Path:
    """Three events; the second has jets but no muons."""

    path = tmp_path / "reductions.root"
    with uproot.recreate(path) as root_file:
        root_file["Events"] = {
            "Muon_pt": ak.Array([[60.0, 20.0], [], [30.0]]),
            "Muon_eta": ak.Array([[0.1, 0.2], [], [0.3]]),
            "Jet_pt": ak.Array([[100.0], [80.0, 40.0], [55.0]]),
            "Jet_eta": ak.Array([[1.0], [2.0, 3.0], [4.0]]),
            "MET_pt": np.array([10.0, 20.0, 30.0]),
        }
    return path


def _branch(name: str) -> VariableSpec:
    return VariableSpec(mode="branch", branch=name)


def _reduction(mode: str, branch: str) -> VariableSpec:
    return VariableSpec(mode=mode, branch=branch)


def _prepare(path: Path, x: VariableSpec, y: VariableSpec | None = None, **options):
    spec = StaticPlotSpec(
        root_path=str(path),
        tree_name="Events",
        entry_start=0,
        entry_stop=N_EVENTS,
        plot_type="2D scatter" if y is not None else "1D histogram",
        x_spec=x,
        y_spec=y,
        color_spec=options.get("color_spec"),
        weight_branch=None,
        cut_specs=tuple(options.get("cut_specs", ())),
        cut_bounds=tuple(options.get("cut_bounds", ())),
        style={"bins_x": 5, "max_points": 1_000},
        labels={},
    )
    return prepare_static_plot(spec)


def test_object_points_are_coloured_by_their_events_multiplicity(events_path) -> None:
    prepared = _prepare(
        events_path,
        _branch("Muon_pt"),
        _branch("Muon_eta"),
        color_spec=_reduction("multiplicity", "Muon_pt"),
    )
    assert prepared.x.tolist() == [60.0, 20.0, 30.0]
    assert prepared.color.tolist() == [2.0, 2.0, 1.0]
    assert prepared.excluded_count == 0


def test_colour_from_another_collection_keeps_events_with_empty_sums(events_path) -> None:
    prepared = _prepare(
        events_path,
        _branch("Jet_pt"),
        _branch("Jet_eta"),
        color_spec=_reduction("sum", "Muon_pt"),
    )
    # The muon-less event has a sum of 0, so its two jets stay visible.
    assert prepared.x.tolist() == [100.0, 80.0, 40.0, 55.0]
    assert prepared.color.tolist() == [80.0, 0.0, 0.0, 30.0]
    assert prepared.excluded_count == 0


def test_colour_reduction_without_value_excludes_the_events_objects(events_path) -> None:
    prepared = _prepare(
        events_path,
        _branch("Jet_pt"),
        _branch("Jet_eta"),
        color_spec=_reduction("max", "Muon_pt"),
    )
    assert prepared.x.tolist() == [100.0, 55.0]
    assert prepared.color.tolist() == [60.0, 30.0]
    assert prepared.excluded_count == 2


def test_event_level_reduction_cut_selects_whole_events(events_path) -> None:
    prepared = _prepare(
        events_path,
        _branch("Jet_pt"),
        cut_specs=[_reduction("max", "Muon_pt")],
        cut_bounds=[(25.0, None)],
    )
    # Events 0 and 2 pass; event 1 has no muons and fails the cut with both jets.
    assert prepared.x.tolist() == [100.0, 55.0]
    assert prepared.excluded_count == 2

    prepared = _prepare(
        events_path,
        _branch("Jet_pt"),
        cut_specs=[_reduction("multiplicity", "Muon_pt")],
        cut_bounds=[(None, 0.0)],
    )
    assert prepared.x.tolist() == [80.0, 40.0]
    assert prepared.excluded_count == 0


def test_object_level_cut_on_event_level_plot_still_fails(events_path) -> None:
    with pytest.raises(ArrayShapeMismatchError):
        _prepare(
            events_path,
            _reduction("max", "Muon_pt"),
            cut_specs=[_branch("Jet_pt")],
            cut_bounds=[(50.0, None)],
        )


def test_export_reproduces_reduction_colours_and_cuts(events_path) -> None:
    request = _request(
        events_path,
        plot_type="2D scatter",
        x_spec=_branch("Jet_pt"),
        y_spec=_branch("Jet_eta"),
        color_spec=_reduction("sum", "Muon_pt"),
        cut_specs=[_reduction("multiplicity", "Muon_pt")],
        cut_bounds=[(1.0, None)],
        style={
            "fig_width": 6.0,
            "fig_height": 4.0,
            "marker_size_2d": 2.0,
            "opacity_2d": 0.5,
            "max_points": 1_000,
            "colorscale": "Viridis",
            "reverse_colorscale": False,
            "log_color": False,
        },
    )
    request = replace(
        request, context=replace(request.context, n_entries=N_EVENTS, entry_stop=N_EVENTS)
    )
    prepared, canonical, generated = _execute_export(request)
    try:
        assert prepared.x.tolist() == [100.0, 55.0]
        assert prepared.color.tolist() == [80.0, 30.0]
        canonical_points = canonical.axes[0].collections[0]
        generated_points = generated.axes[0].collections[0]
        np.testing.assert_array_equal(
            generated_points.get_offsets(), canonical_points.get_offsets()
        )
        np.testing.assert_array_equal(generated_points.get_array(), canonical_points.get_array())
    finally:
        plt.close(canonical)
        plt.close(generated)


def _app(path: Path) -> AppTest:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20).run()
    at.text_input(key="root_path").input(str(path)).run()
    return at


def test_plotter_offers_reductions_for_colour_and_cuts(events_path) -> None:
    at = _app(events_path)
    at.selectbox(key="plot_type").select("2D scatter").run()
    at.checkbox(key="plot_color_points").check().run()
    at.number_input(key="plot_n_cuts").set_value(1).run()

    assert set(at.selectbox(key="color_mode").options) >= REDUCTION_LABELS
    assert set(at.selectbox(key="plot_cut_0_mode").options) >= REDUCTION_LABELS


def test_single_event_plot_keeps_reductions_hidden(events_path) -> None:
    at = _app(events_path)
    at.selectbox(key="event_plot_type").select("2D scatter").run()
    at.checkbox(key="event_color_points").check().run()
    at.number_input(key="event_n_cuts").set_value(1).run()

    for key in ["event_x_mode", "event_color_mode", "event_cut_0_mode"]:
        assert not REDUCTION_LABELS & set(at.selectbox(key=key).options)


@pytest.mark.parametrize("plot_type", ["2D scatter", "3D scatter interactive"])
def test_app_explains_objects_excluded_by_a_colour_reduction(events_path, plot_type) -> None:
    at = _app(events_path)
    at.selectbox(key="plot_type").select(plot_type).run()
    at.selectbox(key="x_branch").select("Jet_pt")
    at.selectbox(key="y_branch").select("Jet_eta")
    if plot_type == "3D scatter interactive":
        at.selectbox(key="z_branch").select("Jet_pt")
    at.run()
    at.checkbox(key="plot_color_points").check().run()
    at.selectbox(key="color_mode").select("max").run()
    at.selectbox(key="color_branch").select("Muon_pt").run()
    assert any(
        caption.value
        == "Events with an empty collection have no max, so their objects are not plotted."
        for caption in at.caption
    )

    at.button(key="plot_make").click().run()
    assert not at.exception
    note = at.session_state["plot_result"]["exclusion_note"]
    assert note.startswith("2 values were excluded")
    assert "too short for a colour or cut reduction" in note


def test_cut_caption_names_the_nth_index(events_path) -> None:
    at = _app(events_path)
    at.number_input(key="plot_n_cuts").set_value(1).run()
    at.selectbox(key="plot_cut_0_mode").select("nth").run()
    at.selectbox(key="plot_cut_0_branch").select("Muon_pt").run()
    at.number_input(key="plot_cut_0_index").set_value(1).run()
    assert any(
        caption.value == "Events with fewer than 2 objects have no value, so they fail this cut."
        for caption in at.caption
    )

    # A scalar branch is its own maximum, so no caption is needed.
    at.selectbox(key="plot_cut_0_mode").select("max").run()
    at.selectbox(key="plot_cut_0_branch").select("MET_pt").run()
    assert not any("fail this cut" in caption.value for caption in at.caption)
