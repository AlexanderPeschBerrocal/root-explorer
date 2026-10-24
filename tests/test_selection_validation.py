"""Invalid field structures and cut bounds cannot silently alter a selection."""

from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot
from streamlit.testing.v1 import AppTest

from root_explorer.analysis.arrays import ArrayShapeMismatchError, nth_per_event, reduce_per_event
from root_explorer.io.root import get_field_info, numeric_fields
from root_explorer.plotting.plots import InvalidCutError, apply_range_cuts


@pytest.mark.parametrize("mode", ["sum", "mean", "min", "max", "nth"])
def test_nested_reductions_do_not_return_multiple_values_per_event(mode):
    values = ak.Array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]])
    with pytest.raises(ArrayShapeMismatchError, match="nested lists"):
        if mode == "nth":
            nth_per_event(values, 0)
        else:
            reduce_per_event(values, mode)


@pytest.mark.parametrize("kind", ["ttree", "rntuple"])
def test_string_only_container_remains_browseable(tmp_path, kind):
    path = tmp_path / "strings.root"
    with uproot.recreate(path) as root:
        fields = {"text": ak.Array(["123", "456"])}
        if kind == "ttree":
            root.mktree("Events", fields)
        else:
            root.mkrntuple("Events", fields)
    assert numeric_fields(get_field_info(str(path), "Events")) == []
    at = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "streamlit_app.py", default_timeout=20
    ).run()
    at.text_input(key="root_path").set_value(str(path)).run()
    assert not at.exception
    assert any("no supported numeric fields" in item.value for item in at.info)
    assert any("text" in table.value.to_string() for table in at.dataframe)
    assert not any(
        button.key in {"plot_make", "event_make", "rank_compute"} for button in at.button
    )


@pytest.mark.parametrize(("low", "high"), [(np.nan, None), (None, np.nan), (3.0, 2.0)])
def test_invalid_cut_bounds_are_rejected(low, high):
    with pytest.raises(InvalidCutError):
        apply_range_cuts([np.array([1.0, 2.0])], [(np.array([1.0, 2.0]), low, high)])


@pytest.mark.parametrize(("low", "high"), [(-np.inf, None), (None, np.inf), (-np.inf, np.inf)])
def test_infinite_cut_bounds_are_open_bounds(low, high):
    values = np.array([1.0, 2.0])
    (selected,) = apply_range_cuts([values], [(values, low, high)])
    np.testing.assert_array_equal(selected, values)


def test_malformed_cut_does_not_become_an_open_bound(root_path):
    at = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "streamlit_app.py", default_timeout=20
    ).run()
    at.text_input(key="root_path").set_value(str(root_path)).run()
    at.button(key="plot_make").click().run()
    at.number_input(key="plot_n_cuts").set_value(1).run()
    at.text_input(key="plot_cut_0_min").set_value("not a number").run()
    at.button(key="plot_make").click().run()
    assert not at.exception
    assert "plot_result" not in at.session_state
    assert any("bounds must be numbers or blank" in item.value for item in at.warning)
