"""Explicit histogram bounds are used without automatic binning first (#200)."""

from __future__ import annotations

import numpy as np
import pytest
import uproot

from root_explorer.analysis.histograms import (
    InvalidHistogramRangeError,
    build_histogram_1d,
    resolve_histogram_range,
)
from root_explorer.models import VariableSpec
from root_explorer.plotting.pipeline import StaticPlotSpec, prepare_static_plot


def _style(minimum: str, maximum: str) -> dict[str, object]:
    return {
        "manual_histogram_limits": True,
        "hist_x_min_text": minimum,
        "hist_x_max_text": maximum,
    }


@pytest.mark.parametrize(
    ("values", "minimum", "maximum"),
    [
        # The issue's examples: the data span is too small for 100 automatic bins.
        (np.array([1.0, np.nextafter(1.0, 2.0)]), "0", "2"),
        (np.array([0.0, np.nextafter(0.0, 1.0)]), "-1", "1"),
        (np.array([1e20, 1e20]), "0", "2e20"),
        (np.array([1.0, 1.0]), "0", "2"),  # already worked
    ],
)
def test_explicit_bounds_give_the_requested_histogram(values, minimum, maximum) -> None:
    value_range = resolve_histogram_range(values, 100, _style(minimum, maximum))
    assert value_range == (float(minimum), float(maximum))
    histogram = build_histogram_1d(values, 100, value_range=value_range)
    assert len(histogram.axes[0]) == 100
    assert histogram.values().sum() == 2


def test_one_sided_wider_range_is_not_rejected() -> None:
    values = np.array([1.0, np.nextafter(1.0, 2.0)])
    value_range = resolve_histogram_range(values, 100, _style("0", ""))
    assert value_range == (0.0, float(values[1]))
    assert build_histogram_1d(values, 100, value_range=value_range).values().sum() == 2


@pytest.mark.parametrize(
    "values",
    [np.random.default_rng(200).normal(size=1_000), np.array([3.0, 3.0]), np.array([])],
    ids=["spread", "constant", "empty"],
)
def test_missing_endpoints_match_numpy_automatic_edges(values) -> None:
    automatic = np.histogram_bin_edges(values, bins=50)
    assert resolve_histogram_range(values, 50, _style("", "1e9"))[0] == automatic[0]
    assert resolve_histogram_range(values, 50, _style("-1e9", ""))[1] == automatic[-1]


def test_without_explicit_bounds_nothing_changes() -> None:
    values = np.array([1.0, 2.0])
    assert resolve_histogram_range(values, 10, {"manual_histogram_limits": False}) is None
    assert resolve_histogram_range(values, 10, _style("", "")) is None
    # A fully automatic range that is too narrow for the bins is still explained.
    tiny = np.array([1.0, np.nextafter(1.0, 2.0)])
    with pytest.raises(InvalidHistogramRangeError, match="Cannot create 100 distinct"):
        build_histogram_1d(tiny, 100)


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    [
        ("abc", "2", "must be finite numbers"),
        ("0", "inf", "must be finite numbers"),
        ("2", "1", "minimum must be smaller"),
        ("5", "", "minimum must be smaller"),  # above the data maximum
    ],
)
def test_invalid_explicit_bounds_are_still_rejected(minimum, maximum, message) -> None:
    with pytest.raises(InvalidHistogramRangeError, match=message):
        resolve_histogram_range(np.array([1.0, 2.0]), 10, _style(minimum, maximum))


def test_plot_pipeline_accepts_the_explicit_range(tmp_path) -> None:
    path = tmp_path / "tiny_span.root"
    with uproot.recreate(path) as root_file:
        root_file.mktree("Events", {"x": np.array([1.0, np.nextafter(1.0, 2.0)])})
    spec = StaticPlotSpec(
        root_path=str(path),
        tree_name="Events",
        entry_start=0,
        entry_stop=2,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="x"),
        y_spec=None,
        color_spec=None,
        weight_branch=None,
        cut_specs=(),
        cut_bounds=(),
        style={"fig_width": 8.0, "fig_height": 6.0, "bins_x": 100, **_style("0", "2")},
        labels={"title": "", "custom_x": "", "custom_y": ""},
    )
    prepared = prepare_static_plot(spec)
    assert tuple(prepared.histogram_1d.axes[0].edges[[0, -1]]) == (0.0, 2.0)
    assert prepared.histogram_1d.values().sum() == 2
