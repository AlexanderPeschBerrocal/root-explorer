"""Adaptive 36/64 or equal-thirds layout of the plot tabs (#71)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import uproot
from streamlit.testing.v1 import AppTest

from root_explorer.app.layout import (
    MIN_COLUMN_WIDTH_REM,
    column_min_width_rem,
    three_column_min_width_rem,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LONG_BRANCH = "Electron_mvaFall17V2noIso_WP90_with_a_deliberately_long_name"


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ([], MIN_COLUMN_WIDTH_REM),
        (["MET_pt", "Muon_eta"], MIN_COLUMN_WIDTH_REM),
        # 60 characters need ceil(60 * 0.45 + 4) = 31 rem.
        (["x" * 60, "y"], 31.0),
    ],
)
def test_column_width_fits_the_longest_option_label(labels, expected) -> None:
    assert column_min_width_rem(labels) == expected


def test_equal_thirds_need_three_columns_and_both_gaps() -> None:
    # Three 22 rem columns plus the 4 rem column gap and the 2 rem group gap.
    assert three_column_min_width_rem(["MET_pt"]) == 72.0


def _layout_styles(path: Path) -> dict[str, str]:
    at = AppTest.from_file(PROJECT_ROOT / "streamlit_app.py", default_timeout=20).run()
    at.text_input(key="root_path").input(str(path)).run()
    assert not at.exception
    styles = {}
    for element in at.get("html"):
        match = re.search(r"\.st-key-(\w+)_layout \{ container-type", element.proto.body)
        if match:
            styles[match.group(1)] = element.proto.body
    return styles


def _threshold(css: str) -> float:
    return float(re.search(r"@container \(min-width: ([\d.]+)rem\)", css).group(1))


def test_plot_tabs_switch_from_36_64_to_equal_thirds(root_path) -> None:
    styles = _layout_styles(root_path)
    assert set(styles) == {"plot", "event"}
    for prefix, css in styles.items():
        # Narrow: the familiar 36/64 split with stacked control groups.
        assert f".st-key-{prefix}_layout > :nth-child(1) {{ flex: 36 1 0 !important; }}" in css
        assert f".st-key-{prefix}_layout > :nth-child(2) {{ flex: 64 1 0 !important; }}" in css
        assert f".st-key-{prefix}_control_groups > * {{ flex: 1 1 100%" in css
        # Wide: the output takes a third and the two groups share the rest equally.
        assert _threshold(css) == three_column_min_width_rem(["MET_pt"])
        assert "flex: 0 0 calc((100% - 6rem) / 3) !important;" in css
        assert f".st-key-{prefix}_control_groups > * {{ flex: 1 1 0 !important; }}" in css
        # Phones stack the controls above the output, like st.columns.
        assert "@media (max-width: 640px)" in css


def test_long_branch_names_raise_the_equal_thirds_threshold(tmp_path) -> None:
    path = tmp_path / "long_names.root"
    with uproot.recreate(path) as root_file:
        root_file["Events"] = {LONG_BRANCH: np.arange(5.0), "MET_pt": np.arange(5.0)}

    expected = three_column_min_width_rem([LONG_BRANCH])
    assert expected > three_column_min_width_rem(["MET_pt"])
    assert {prefix: _threshold(css) for prefix, css in _layout_styles(path).items()} == {
        "plot": expected,
        "event": expected,
    }
