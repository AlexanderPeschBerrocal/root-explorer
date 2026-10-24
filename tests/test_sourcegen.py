from __future__ import annotations

import importlib
import inspect
import math
import sys
import textwrap
from dataclasses import dataclass

import pytest

from root_explorer.analysis.statistics import calculate_quantile_result
from root_explorer.models import VariableSpec
from root_explorer.plotting import sourcegen
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


def test_python_source_literal_serialises_non_finite_floats() -> None:
    source = python_source_literal({"sigma": (0.0, math.inf), "mu": (-math.inf, math.nan)})

    assert source == "{'sigma': (0.0, float('inf')), 'mu': (float('-inf'), float('nan'))}"
    value = eval(source, {"__builtins__": {"float": float}})
    assert value["sigma"] == (0.0, math.inf)
    assert value["mu"][0] == -math.inf
    assert math.isnan(value["mu"][1])


def test_bundle_python_source_reports_missing_source_as_bundle_error(monkeypatch) -> None:
    def no_source(obj):
        raise OSError("could not get source code")

    monkeypatch.setattr(sourcegen.inspect, "getsource", no_source)
    with pytest.raises(SourceBundleError, match="Cannot inspect the source"):
        bundle_python_source(calculate_quantile_result)


@pytest.fixture
def fake_internal_modules(tmp_path, monkeypatch):
    """Internal-looking helper modules with module-level globals."""

    (tmp_path / "root_explorer_fake_constants.py").write_text(
        "LIMIT = 3\nSHARED = 'shared'\n", encoding="utf-8"
    )
    (tmp_path / "root_explorer_fake_helpers.py").write_text(
        textwrap.dedent(
            """
            import re

            from root_explorer_fake_constants import LIMIT, SHARED

            PATTERN = re.compile("x")
            SCALE = float("inf")


            def uses_constants(value):
                return value < LIMIT * SCALE, SHARED


            def uses_pattern(value):
                return PATTERN.match(value)
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield importlib.import_module("root_explorer_fake_helpers")
    for name in ("root_explorer_fake_helpers", "root_explorer_fake_constants"):
        sys.modules.pop(name, None)


def test_bundle_python_source_emits_literal_constants_of_internal_modules(
    fake_internal_modules,
) -> None:
    source = bundle_python_source(fake_internal_modules.uses_constants)

    assert "LIMIT = 3" in source
    assert "SHARED = 'shared'" in source
    assert "SCALE = float('inf')" in source
    assert "root_explorer_fake_constants" not in source
    namespace: dict[str, object] = {}
    exec(compile(source, "bundle.py", "exec"), namespace)
    assert namespace["uses_constants"](2) == (True, "shared")


def test_bundle_python_source_rejects_non_literal_globals(fake_internal_modules) -> None:
    with pytest.raises(SourceBundleError, match="'PATTERN' of type Pattern"):
        bundle_python_source(fake_internal_modules.uses_pattern)


def test_bundle_python_source_rejects_unreconstructable_external_callables(
    fake_internal_modules, monkeypatch
) -> None:
    # A callable from outside the package that is not imported at module level.
    monkeypatch.setattr(fake_internal_modules, "PATTERN", inspect.getsource, raising=True)
    with pytest.raises(SourceBundleError, match="Cannot reconstruct import"):
        bundle_python_source(fake_internal_modules.uses_pattern)


def test_bundle_python_source_ignores_globals_used_only_in_annotations(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "root_explorer_fake_annotations.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            from collections.abc import Callable

            Scale = Callable[[float], float]


            def doubled(value: float, *factors: Scale) -> Scale:
                result: Scale = lambda x: 2 * x * value
                return result
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        module = importlib.import_module("root_explorer_fake_annotations")
        # The typing alias appears only in annotations, which the generated
        # script never evaluates, so it neither needs nor blocks bundling.
        source = bundle_python_source(module.doubled)
    finally:
        sys.modules.pop("root_explorer_fake_annotations", None)
    assert "\nScale =" not in source
    namespace: dict[str, object] = {}
    exec(compile(source, "bundle.py", "exec"), namespace)
    assert namespace["doubled"](3.0)(1.0) == 6.0
