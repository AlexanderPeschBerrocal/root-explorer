"""Utilities for vendoring canonical ROOT Explorer Python helpers into exports."""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import math
import textwrap
from dataclasses import fields, is_dataclass
from types import ModuleType
from typing import Any


class SourceBundleError(ValueError):
    """Raised when a Python object cannot be safely represented in an export bundle."""


def python_source_literal(value: Any) -> str:
    """Return readable Python source for supported immutable configuration values."""

    if value is None or isinstance(value, (bool, int, str)):
        return repr(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SourceBundleError("Non-finite floats cannot be serialised as plot configuration.")
        return repr(value)
    if is_dataclass(value) and not isinstance(value, type):
        arguments = ", ".join(
            f"{field.name}={python_source_literal(getattr(value, field.name))}"
            for field in fields(value)
        )
        return f"{type(value).__name__}({arguments})"
    if isinstance(value, tuple):
        items = ", ".join(python_source_literal(item) for item in value)
        suffix = "," if len(value) == 1 else ""
        return f"({items}{suffix})"
    if isinstance(value, list):
        return "[" + ", ".join(python_source_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{python_source_literal(key)}: {python_source_literal(item)}"
                for key, item in value.items()
            )
            + "}"
        )
    if isinstance(value, set):
        if not value:
            return "set()"
        return "{" + ", ".join(sorted(python_source_literal(item) for item in value)) + "}"
    if isinstance(value, frozenset):
        items = ", ".join(sorted(python_source_literal(item) for item in value))
        return f"frozenset({{{items}}})"

    raise SourceBundleError(
        f"Unsupported configuration value {value!r} of type {type(value).__name__}."
    )


def bundle_python_source(
    *roots: Any,
    package_prefix: str = "root_explorer",
    forbidden_module_prefixes: tuple[str, ...] = ("root_explorer.app", "streamlit"),
) -> str:
    """Bundle canonical helper source and external imports needed by the roots.

    Internal ROOT Explorer dependencies are recursively vendored. External dependencies
    retain their original import statements. The result intentionally contains no
    root_explorer imports so it can be embedded in a standalone script.
    """

    if not roots:
        raise SourceBundleError("At least one root callable or class is required.")

    definitions: list[str] = []
    constants: list[str] = []
    imports: set[str] = set()
    visited: set[tuple[str, str]] = set()
    emitted_names: dict[str, tuple[str, str]] = {}

    def visit(obj: Any) -> None:
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            raise SourceBundleError(f"Cannot bundle unsupported root object {obj!r}.")

        module_name = getattr(obj, "__module__", "")
        object_name = getattr(obj, "__name__", "")
        if not module_name.startswith(package_prefix):
            raise SourceBundleError(
                f"Cannot vendor {object_name!r} from external module {module_name!r}."
            )
        if any(module_name.startswith(prefix) for prefix in forbidden_module_prefixes):
            raise SourceBundleError(
                f"Export helper {object_name!r} depends on forbidden module {module_name!r}."
            )

        key = (module_name, object_name)
        if key in visited:
            return
        visited.add(key)
        _claim_name(object_name, key, emitted_names)

        module = inspect.getmodule(obj)
        if module is None:
            raise SourceBundleError(f"Cannot resolve source module for {object_name!r}.")

        overrides = getattr(module, "__standalone_source_overrides__", {})
        replacement = overrides.get(object_name)
        if replacement is not None:
            visit(replacement)
            definitions.append(f"{object_name} = {replacement.__name__}")
            return

        source = textwrap.dedent(inspect.getsource(obj))
        tree = ast.parse(source)
        import_map = _module_import_map(module)
        source = _vendor_local_internal_imports(
            source,
            tree,
            module,
            package_prefix=package_prefix,
            visit=visit,
        )

        for name in _referenced_global_names(tree, module):
            dependency = module.__dict__[name]
            dependency_module = getattr(dependency, "__module__", "")

            if inspect.isfunction(dependency) or inspect.isclass(dependency):
                if dependency_module.startswith(package_prefix):
                    visit(dependency)
                else:
                    import_source = import_map.get(name)
                    if import_source is not None:
                        imports.add(import_source)
                continue

            if isinstance(dependency, ModuleType):
                import_source = import_map.get(name)
                if import_source is None:
                    raise SourceBundleError(
                        f"Cannot reconstruct import for module global {name!r} in {module_name!r}."
                    )
                if any(dependency.__name__.startswith(prefix) for prefix in forbidden_module_prefixes):
                    raise SourceBundleError(
                        f"Export helper {object_name!r} depends on forbidden module "
                        f"{dependency.__name__!r}."
                    )
                imports.add(import_source)
                continue

            import_source = import_map.get(name)
            if import_source is not None:
                if not _is_internal_import(import_source, package_prefix):
                    imports.add(import_source)
                continue

            if _is_source_constant(dependency):
                constant_key = (module_name, name)
                if constant_key not in visited:
                    visited.add(constant_key)
                    _claim_name(name, constant_key, emitted_names)
                    constants.append(f"{name} = {python_source_literal(dependency)}")
                continue

        definitions.append(source.rstrip())

    for root in roots:
        visit(root)

    sections = ["from __future__ import annotations"]
    if imports:
        sections.append("\n".join(sorted(imports)))
    if constants:
        sections.append("\n".join(constants))
    if definitions:
        sections.append("\n\n\n".join(definitions))
    return "\n\n".join(sections).rstrip() + "\n"


def _vendor_local_internal_imports(
    source: str,
    tree: ast.AST,
    module: ModuleType,
    *,
    package_prefix: str,
    visit,
) -> str:
    """Vendor function-local internal imports and remove them from emitted source."""

    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        relative_name = "." * node.level + (node.module or "")
        target_name = importlib.util.resolve_name(relative_name, module.__package__ or "")
        if not target_name.startswith(package_prefix):
            continue
        target_module = importlib.import_module(target_name)
        for alias in node.names:
            if alias.name == "*":
                raise SourceBundleError("Wildcard internal imports cannot be bundled.")
            if alias.asname is not None and alias.asname != alias.name:
                raise SourceBundleError(
                    f"Aliased internal import {alias.name!r} cannot be bundled yet."
                )
            dependency = getattr(target_module, alias.name)
            if not (inspect.isfunction(dependency) or inspect.isclass(dependency)):
                raise SourceBundleError(
                    f"Function-local internal import {alias.name!r} is not a callable/class."
                )
            visit(dependency)
        ranges.append((node.lineno, node.end_lineno or node.lineno))

    if not ranges:
        return source

    lines = source.splitlines()
    for start, end in ranges:
        for index in range(start - 1, end):
            indentation = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
            lines[index] = indentation + "pass  # internal helper vendored above"
    return "\n".join(lines)


def _claim_name(
    name: str,
    key: tuple[str, str],
    emitted_names: dict[str, tuple[str, str]],
) -> None:
    previous = emitted_names.get(name)
    if previous is not None and previous != key:
        raise SourceBundleError(
            f"Standalone bundle name collision for {name!r}: {previous[0]!r} and {key[0]!r}."
        )
    emitted_names[name] = key


def _referenced_global_names(tree: ast.AST, module: ModuleType) -> set[str]:
    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return {
        name
        for name in loaded
        if name in module.__dict__ and not hasattr(builtins, name)
    }


def _module_import_map(module: ModuleType) -> dict[str, str]:
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError) as exc:
        raise SourceBundleError(f"Cannot inspect module {module.__name__!r}.") from exc

    tree = ast.parse(source)
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            segment = ast.get_source_segment(source, node)
            if segment is None:
                continue
            for alias in node.names:
                result[alias.asname or alias.name.split(".", 1)[0]] = segment
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            segment = ast.get_source_segment(source, node)
            if segment is None:
                continue
            for alias in node.names:
                result[alias.asname or alias.name] = segment
    return result


def _is_internal_import(source: str, package_prefix: str) -> bool:
    stripped = source.strip()
    return stripped.startswith("from .") or stripped.startswith(f"from {package_prefix}")


def _is_source_constant(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (tuple, list, set, frozenset)):
        return all(_is_source_constant(item) for item in value)
    if isinstance(value, dict):
        return all(
            _is_source_constant(key) and _is_source_constant(item)
            for key, item in value.items()
        )
    return False
