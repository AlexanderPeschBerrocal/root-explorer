"""Collapsed style and label settings and the scatter sampling caption (#90)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import uproot
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_EVENTS = 1_500


@pytest.fixture
def scalar_path(tmp_path) -> Path:
    """More events than the smallest scatter limit of 1,000 points."""

    path = tmp_path / "scalars.root"
    rng = np.random.default_rng(1)
    with uproot.recreate(path) as root_file:
        tree = root_file.mktree("Events", {"a": "f8", "b": "f8"})
        tree.extend({"a": rng.uniform(1, 10, N_EVENTS), "b": rng.uniform(1, 10, N_EVENTS)})
    return path


def _app(path: Path) -> AppTest:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20).run()
    return at.text_input(key="root_path").input(str(path)).run()


def _expanders(tab) -> dict[str, object]:
    return {expander.label: expander for expander in tab.expander}


@pytest.mark.parametrize(("tab_index", "prefix"), [(0, "plot"), (1, "event")])
def test_style_and_labels_are_collapsed_and_analysis_stays_visible(
    scalar_path, tab_index, prefix
) -> None:
    at = _app(scalar_path)
    expanders = _expanders(at.tabs[tab_index])
    style, labels = expanders["Style"], expanders["Labels"]
    assert not style.proto.expanded
    assert not labels.proto.expanded

    style_keys = {widget.key for widget in [*style.slider, *style.checkbox]}
    assert {f"{prefix}_log_x", f"{prefix}_log_y", f"{prefix}_grid"} <= style_keys
    assert f"{prefix}_title" in {widget.key for widget in labels.text_input}
    # Settings that change the plotted data are not inside the collapsed sections.
    assert f"{prefix}_bins" not in style_keys
    assert f"{prefix}_use_weights" not in style_keys
    assert at.slider(key=f"{prefix}_bins")
    assert at.checkbox(key=f"{prefix}_use_weights")


def test_values_in_collapsed_sections_reach_the_plot_and_code(scalar_path) -> None:
    at = _app(scalar_path)
    at.checkbox(key="plot_log_y").check()
    at.text_input(key="plot_title").input("Collapsed title")
    at.run()
    at.button(key="plot_make").click().run()
    assert not at.exception

    result = at.session_state["plot_result"]
    axes = result["figure"].axes[0]
    assert axes.get_title() == "Collapsed title"
    assert axes.get_yscale() == "log"
    assert "'log_y': True" in result["code"]
    assert "'title': 'Collapsed title'" in result["code"]


def _scatter(at: AppTest, plot_type: str, suffix: str, max_points: int) -> dict:
    at.selectbox(key="plot_type").select(plot_type).run()
    at.selectbox(key="x_branch").select("a")
    at.selectbox(key="y_branch").select("b")
    if plot_type == "3D scatter interactive":
        at.selectbox(key="z_branch").select("a")
    at.run()
    at.number_input(key=f"plot_max_points_{suffix}").set_value(max_points).run()
    at.button(key="plot_make").click().run()
    assert not at.exception
    return at.session_state["plot_result"]


@pytest.mark.parametrize(
    ("plot_type", "suffix"), [("2D scatter", "2d"), ("3D scatter interactive", "3d")]
)
def test_scatter_plots_state_when_only_a_sample_is_drawn(scalar_path, plot_type, suffix) -> None:
    at = _app(scalar_path)
    result = _scatter(at, plot_type, suffix, 1_000)
    note = (
        f"Showing a fixed random sample of 1,000 of {N_EVENTS:,} points. Summary statistics "
        "use all points. Increase **Maximum scatter points** under **Style** to draw more."
    )
    assert result["sampling_note"] == note
    assert note in [caption.value for caption in at.caption]
    # The summary still describes every point.
    assert result["summary"].iloc[0]["n"] == N_EVENTS

    result = _scatter(at, plot_type, suffix, 2_000)
    assert result["sampling_note"] is None
    assert not any("random sample" in caption.value for caption in at.caption)
