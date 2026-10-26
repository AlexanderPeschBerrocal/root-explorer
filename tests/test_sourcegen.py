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


def test_bundle_python_source_ignores_locals_that_shadow_module_globals(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "root_explorer_fake_shadow.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations


            def shadow(items):
                annotations = [item * 2 for item in items]
                return annotations
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        module = importlib.import_module("root_explorer_fake_shadow")
        # ``annotations`` is a module global (a __future__ feature) that cannot be
        # bundled, but inside ``shadow`` it is a local variable.
        source = bundle_python_source(module.shadow)
    finally:
        sys.modules.pop("root_explorer_fake_shadow", None)
    namespace: dict[str, object] = {}
    exec(compile(source, "bundle.py", "exec"), namespace)
    assert namespace["shadow"]([1, 2]) == [2, 4]


def test_bundle_python_source_keeps_globals_shadowed_only_in_other_scopes(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "root_explorer_fake_scopes.py").write_text(
        textwrap.dedent(
            """
            SCALE = 2.0
            OFFSET = 1.0


            def scaled(values):
                def identity(SCALE):
                    return SCALE

                shifted = [OFFSET for OFFSET in values]
                return [identity(value) * SCALE + OFFSET for value in shifted]


            class Holder:
                OFFSET = 100.0

                def parameter(self, SCALE):
                    return SCALE

                def tripled(self):
                    return SCALE * 3 + OFFSET
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        module = importlib.import_module("root_explorer_fake_scopes")
        # Inner functions, comprehensions, other methods and class bodies bind
        # SCALE or OFFSET locally, but the outer code still reads the globals.
        function_source = bundle_python_source(module.scaled)
        class_source = bundle_python_source(module.Holder)
    finally:
        sys.modules.pop("root_explorer_fake_scopes", None)
    for source in (function_source, class_source):
        assert "SCALE = 2.0" in source
        assert "OFFSET = 1.0" in source
    function_namespace: dict[str, object] = {}
    exec(compile(function_source, "bundle.py", "exec"), function_namespace)
    assert function_namespace["scaled"]([1.0, 2.0]) == [3.0, 5.0]
    class_namespace: dict[str, object] = {}
    exec(compile(class_source, "bundle.py", "exec"), class_namespace)
    assert class_namespace["Holder"]().tripled() == 7.0


def test_bundle_python_source_ignores_names_bound_in_enclosing_function(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "root_explorer_fake_closure.py").write_text(
        textwrap.dedent(
            """
            import threading

            LOCK = threading.Lock()


            def outer(values):
                LOCK = 2
                if any((total := value) > 1 for value in values):
                    pass

                def inner():
                    return LOCK * total

                return inner()
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        module = importlib.import_module("root_explorer_fake_closure")
        # LOCK and total are closure variables of ``outer``. The module-level lock
        # cannot be bundled and must not be needed.
        source = bundle_python_source(module.outer)
    finally:
        sys.modules.pop("root_explorer_fake_closure", None)
    namespace: dict[str, object] = {}
    exec(compile(source, "bundle.py", "exec"), namespace)
    assert namespace["outer"]([1, 3]) == 6
