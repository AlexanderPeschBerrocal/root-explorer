"""Utilities for vendoring canonical ROOT Explorer Python helpers into exports."""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import math
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass
from importlib.util import resolve_name
from types import ModuleType
from typing import Any


class SourceBundleError(ValueError):
    """Raised when a Python object cannot be safely represented in an export bundle."""


def python_source_literal(value: Any) -> str:
    """Return readable Python source for supported immutable configuration values."""

    if value is None or isinstance(value, (bool, int, str)):
        return repr(value)
    if isinstance(value, float):
        # repr() would give the undefined names inf and nan.
        if math.isnan(value):
            return "float('nan')"
        if math.isinf(value):
            return "float('inf')" if value > 0 else "float('-inf')"
        return repr(value)
    if is_dataclass(value) and not isinstance(value, type):
        arguments = ", ".join(
            f"{item_field.name}={python_source_literal(getattr(value, item_field.name))}"
            for item_field in fields(value)
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


def literal_dataclass_types(value: Any) -> list[type]:
    """Return the dataclasses whose constructors ``python_source_literal(value)`` emits.

    They must be bundled even when no helper refers to them outside annotations.
    """

    found: list[type] = []

    def visit(item: Any) -> None:
        if is_dataclass(item) and not isinstance(item, type):
            if type(item) not in found:
                found.append(type(item))
            for item_field in fields(item):
                visit(getattr(item, item_field.name))
        elif isinstance(item, dict):
            for key, entry in item.items():
                visit(key)
                visit(entry)
        elif isinstance(item, (tuple, list, set, frozenset)):
            for entry in item:
                visit(entry)

    visit(value)
    return found


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

        try:
            source = textwrap.dedent(inspect.getsource(obj))
        except (OSError, TypeError) as exc:
            # For example, an installation that ships only bytecode.
            raise SourceBundleError(f"Cannot inspect the source of {object_name!r}.") from exc
        tree = ast.parse(source)
        import_map = _module_import_map(module)
        source = _vendor_local_internal_imports(
            source,
            tree,
            module,
            package_prefix=package_prefix,
            visit=visit,
        )

        # Generated scripts use postponed annotations, so a name that appears only
        # in annotations is bundled when possible but never blocks the export.
        annotation_only = _annotation_only_names(tree)
        for name in _referenced_global_names(tree, module):
            dependency = module.__dict__[name]
            dependency_module = getattr(dependency, "__module__", "")

            if inspect.isfunction(dependency) or inspect.isclass(dependency):
                if dependency_module.startswith(package_prefix):
                    visit(dependency)
                else:
                    import_source = import_map.get(name)
                    if import_source is None:
                        if name in annotation_only:
                            continue
                        raise SourceBundleError(
                            f"Cannot reconstruct import for global {name!r} in {module_name!r}."
                        )
                    imports.add(import_source)
                continue

            if isinstance(dependency, ModuleType):
                import_source = import_map.get(name)
                if name in annotation_only and (
                    import_source is None
                    or _is_internal_import(import_source, package_prefix)
                    or dependency.__name__.startswith(forbidden_module_prefixes)
                ):
                    continue
                if import_source is None:
                    raise SourceBundleError(
                        f"Cannot reconstruct import for module global {name!r} in {module_name!r}."
                    )
                if any(
                    dependency.__name__.startswith(prefix) for prefix in forbidden_module_prefixes
                ):
                    raise SourceBundleError(
                        f"Export helper {object_name!r} depends on forbidden module "
                        f"{dependency.__name__!r}."
                    )
                if _is_internal_import(import_source, package_prefix):
                    raise SourceBundleError(
                        f"Export helper {object_name!r} imports internal module "
                        f"{dependency.__name__!r}, which cannot be bundled."
                    )
                imports.add(import_source)
                continue

            import_source = import_map.get(name)
            if import_source is not None and not _is_internal_import(import_source, package_prefix):
                imports.add(import_source)
                continue

            # Literal constants, whether defined in this module or imported from another
            # internal module, are emitted under their own name. Keying them by value lets
            # several modules share one constant while differing values still collide.
            if _is_source_constant(dependency):
                literal = python_source_literal(dependency)
                constant_key = (f"constant {literal}", name)
                if constant_key not in visited:
                    visited.add(constant_key)
                    _claim_name(name, constant_key, emitted_names)
                    constants.append(f"{name} = {literal}")
                continue

            if name in annotation_only:
                continue
            # Anything else would leave an undefined name in the standalone script.
            raise SourceBundleError(
                f"Export helper {object_name!r} uses global {name!r} of type "
                f"{type(dependency).__name__}, which cannot be bundled."
            )

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
    visit: Callable[[Any], None],
) -> str:
    """Vendor function-local internal imports and remove them from emitted source."""

    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        relative_name = "." * node.level + (node.module or "")
        target_name = resolve_name(relative_name, module.__package__ or "")
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


def _annotation_nodes(tree: ast.AST) -> set[int]:
    """Return the ids of nodes inside annotations, which are never evaluated."""

    annotations: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            parameters = [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                *(argument for argument in (arguments.vararg, arguments.kwarg) if argument),
            ]
            annotations.extend(p.annotation for p in parameters if p.annotation is not None)
            if node.returns is not None:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
    return {id(child) for annotation in annotations for child in ast.walk(annotation)}


_COMPREHENSION_NODES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_SCOPE_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    *_COMPREHENSION_NODES,
)


@dataclass
class _Scope:
    node: ast.AST
    parent: _Scope | None
    bound: set[str] = field(default_factory=set)
    declared_global: set[str] = field(default_factory=set)
    loads: set[str] = field(default_factory=set)


def _function_arguments(arguments: ast.arguments) -> list[ast.arg]:
    return [
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
        *(argument for argument in (arguments.vararg, arguments.kwarg) if argument),
    ]


def _scope_body(node: ast.AST) -> list[ast.AST]:
    """Return the parts of a scope node that are evaluated inside that scope."""

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return [*_function_arguments(node.args), *node.body]
    if isinstance(node, ast.Lambda):
        return [*_function_arguments(node.args), node.body]
    if isinstance(node, ast.ClassDef):
        return list(node.body)
    if isinstance(node, _COMPREHENSION_NODES):
        elements: list[ast.AST] = (
            [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        )
        for index, generator in enumerate(node.generators):
            elements.extend([generator.target, *generator.ifs])
            if index:
                elements.append(generator.iter)
        return elements
    return list(ast.iter_child_nodes(node))


def _enclosing_parts(node: ast.AST) -> list[ast.AST]:
    """Return the parts of a scope node that are evaluated in the enclosing scope."""

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parts: list[ast.AST] = [*node.decorator_list, *node.args.defaults]
        parts.extend(default for default in node.args.kw_defaults if default is not None)
        parts.extend(
            argument.annotation
            for argument in _function_arguments(node.args)
            if argument.annotation is not None
        )
        if node.returns is not None:
            parts.append(node.returns)
        return parts
    if isinstance(node, ast.Lambda):
        return [
            *node.args.defaults,
            *(default for default in node.args.kw_defaults if default is not None),
        ]
    if isinstance(node, ast.ClassDef):
        return [*node.decorator_list, *node.bases, *node.keywords]
    if isinstance(node, _COMPREHENSION_NODES):
        return [node.generators[0].iter]
    return []


def _collect_scope(scope: _Scope, nodes: list[ast.AST], scopes: list[_Scope]) -> None:
    """Record the names bound, declared global and loaded directly in ``scope``."""

    stack = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPE_NODES):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope.bound.add(node.name)
            stack.extend(_enclosing_parts(node))
            child = _Scope(node, scope)
            scopes.append(child)
            _collect_scope(child, _scope_body(node), scopes)
            continue
        if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            # An assignment expression in a comprehension binds in the enclosing scope.
            target_scope = scope
            while target_scope.parent is not None and isinstance(
                target_scope.node, _COMPREHENSION_NODES
            ):
                target_scope = target_scope.parent
            target_scope.bound.add(node.target.id)
            stack.append(node.value)
            continue
        if isinstance(node, ast.arg):
            # Annotations belong to the enclosing scope and are collected there.
            scope.bound.add(node.arg)
            continue
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                scope.loads.add(node.id)
            else:
                scope.bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            scope.bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            scope.bound.update(
                (alias.asname or alias.name).split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.Global):
            scope.declared_global.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            scope.bound.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            scope.bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            scope.bound.add(node.rest)
        stack.extend(ast.iter_child_nodes(node))


def _resolves_to_global(scope: _Scope, name: str) -> bool:
    """Return whether ``name`` loaded in ``scope`` is read from the module globals."""

    current: _Scope | None = scope
    while current is not None and current.parent is not None:
        # Names bound in a class body are not visible inside its nested scopes.
        if current is scope or not isinstance(current.node, ast.ClassDef):
            if name in current.declared_global:
                return True
            if name in current.bound:
                return False
        current = current.parent
    return True


def _referenced_global_names(tree: ast.AST, module: ModuleType) -> set[str]:
    """Return module globals that the bundled source reads, following Python's scope rules."""

    root = _Scope(tree, None)
    scopes = [root]
    _collect_scope(root, _scope_body(tree), scopes)
    loaded = {name for scope in scopes for name in scope.loads if _resolves_to_global(scope, name)}
    return {name for name in loaded if name in module.__dict__ and not hasattr(builtins, name)}


def _annotation_only_names(tree: ast.AST) -> set[str]:
    """Return names that the source loads only inside annotations."""

    in_annotations = _annotation_nodes(tree)
    names = [node for node in ast.walk(tree) if isinstance(node, ast.Name)]
    evaluated = {node.id for node in names if id(node) not in in_annotations}
    return {node.id for node in names if node.id not in evaluated}


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
    return stripped.startswith(("from .", f"from {package_prefix}"))


def _is_source_constant(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (tuple, list, set, frozenset)):
        return all(_is_source_constant(item) for item in value)
    if isinstance(value, dict):
        return all(
            _is_source_constant(key) and _is_source_constant(item) for key, item in value.items()
        )
    return False
