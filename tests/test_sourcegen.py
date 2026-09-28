from __future__ import annotations

from dataclasses import dataclass

import pytest

from root_explorer.analysis.statistics import calculate_quantile_result
from root_explorer.models import VariableSpec
from root_explorer.plotting.pipeline import StaticPlotSpec
from root_explorer.plotting.sourcegen import (
    SourceBundleError,
    bundle_python_source,
    python_source_literal,
)


@dataclass(frozen=True)
class ExampleConfig:
    label: str
    bounds: tuple[float | None, float | None]


def test_python_source_literal_serialises_nested_dataclasses() -> None:
    source = python_source_literal(ExampleConfig("pt", (1.0, None)))
    assert source == "ExampleConfig(label='pt', bounds=(1.0, None))"


def test_python_source_literal_serialises_static_plot_spec() -> None:
    spec = StaticPlotSpec(
        root_path="events.root",
        tree_name="Events",
        entry_start=0,
        entry_stop=10,
        plot_type="1D histogram",
        x_spec=VariableSpec(mode="branch", branch="Muon_pt"),
        y_spec=None,
        color_spec=None,
        weight_branch=None,
        cut_specs=(),
        cut_bounds=(),
        style={"fig_width": 8.0, "fig_height": 6.0, "bins_x": 20},
        labels={"title": "", "custom_x": "", "custom_y": ""},
    )

    source = python_source_literal(spec)

    assert source.startswith("StaticPlotSpec(")
    assert "VariableSpec(mode='branch', branch='Muon_pt'" in source
    assert "root_path='events.root'" in source


def test_bundle_python_source_vendors_internal_dependencies() -> None:
    source = bundle_python_source(calculate_quantile_result)

    compile(source, "standalone_helpers.py", "exec")
    assert "def calculate_quantile_result(" in source
    assert "def align_arrays(" in source
    assert "class ArrayShapeMismatchError(" in source
    assert "import numpy as np" in source
    assert "from root_explorer" not in source


def test_bundle_python_source_rejects_non_root_object() -> None:
    with pytest.raises(SourceBundleError, match="external module"):
        bundle_python_source(pytest.raises)
