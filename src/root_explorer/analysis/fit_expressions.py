"""Restricted, NumPy-vectorised mathematical expressions for histogram fits.

Only AST nodes explicitly accepted by the validator are ever interpreted.
User-provided source is never compiled or executed as Python.
"""

from __future__ import annotations

import ast
from functools import lru_cache

import numpy as np

MAX_EXPRESSION_LENGTH = 512
MAX_EXPRESSION_NODES = 128
MAX_EXPRESSION_DEPTH = 24
MAX_FIT_PARAMETERS = 12
FIT_FUNCTION_NAMES = (
    "exp", "log", "sqrt", "abs", "sin", "cos", "tan", "sinh", "cosh", "tanh"
)


def _fit_functions() -> dict[str, object]:
    """Keep the whitelist explicit and reconstructible in standalone exports."""
    return {
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
    }


def _check_expression_node(node: ast.AST, parameters: list[str], depth: int = 0) -> None:
    if depth > MAX_EXPRESSION_DEPTH:
        raise ValueError("Expression nesting is too deep.")
    if isinstance(node, ast.Expression):
        _check_expression_node(node.body, parameters, depth + 1)
    elif isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Only finite numeric literals are allowed.")
        if not np.isfinite(value) or abs(value) > 1e100:
            raise ValueError("Numeric literal is too large or non-finite.")
    elif isinstance(node, ast.Name):
        if node.id.startswith("_"):
            raise ValueError("Identifiers beginning with '_' are not allowed.")
        if node.id in FIT_FUNCTION_NAMES:
            raise ValueError(f"Function {node.id!r} must be called with one argument.")
        if node.id not in ("x", "pi", "e") and node.id not in parameters:
            parameters.append(node.id)
            if len(parameters) > MAX_FIT_PARAMETERS:
                raise ValueError(f"At most {MAX_FIT_PARAMETERS} fit parameters are supported.")
    elif isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
    ):
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
            exponent = node.right.value
            if isinstance(exponent, (int, float)) and abs(exponent) > 64:
                raise ValueError("Literal exponents above 64 are not supported.")
        _check_expression_node(node.left, parameters, depth + 1)
        _check_expression_node(node.right, parameters, depth + 1)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _check_expression_node(node.operand, parameters, depth + 1)
    elif isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in FIT_FUNCTION_NAMES
            or len(node.args) != 1
            or node.keywords
        ):
            raise ValueError("Only whitelisted mathematical functions with one argument are allowed.")
        _check_expression_node(node.args[0], parameters, depth + 1)
    else:
        raise ValueError(f"Unsupported expression syntax: {type(node).__name__}.")


@lru_cache(maxsize=128)
def _parsed_expression(expression: str) -> tuple[ast.Expression, tuple[str, ...]]:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Enter a non-empty mathematical expression.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Expression exceeds {MAX_EXPRESSION_LENGTH} characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid mathematical expression syntax.") from exc
    if len(list(ast.walk(tree))) > MAX_EXPRESSION_NODES:
        raise ValueError("Expression is too complex.")
    parameters: list[str] = []
    _check_expression_node(tree, parameters)
    if not parameters:
        raise ValueError("Expression must contain at least one fit parameter besides x.")
    return tree, tuple(parameters)


def discover_fit_parameters(expression: str) -> tuple[str, ...]:
    """Return first-occurrence-order free parameters from a validated expression."""
    return _parsed_expression(expression)[1]


def _evaluate_expression_node(
    node: ast.AST, x: np.ndarray, parameters: dict[str, float]
) -> np.ndarray | float:
    if isinstance(node, ast.Expression):
        return _evaluate_expression_node(node.body, x, parameters)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == "x":
            return x
        if node.id == "pi":
            return np.pi
        if node.id == "e":
            return np.e
        return parameters[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_expression_node(node.operand, x, parameters)
        return +operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _evaluate_expression_node(node.left, x, parameters)
        right = _evaluate_expression_node(node.right, x, parameters)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return np.divide(left, right)
        if isinstance(node.op, ast.Pow):
            return np.power(left, right)
    if isinstance(node, ast.Call):
        argument = _evaluate_expression_node(node.args[0], x, parameters)
        return _fit_functions()[node.func.id](argument)
    # This should be unreachable: _parsed_expression validated every node.
    raise ValueError("Unsupported expression node.")


def evaluate_fit_expression(
    expression: str, x: np.ndarray, parameters: dict[str, float]
) -> np.ndarray:
    """Evaluate only validated arithmetic and whitelisted functions on NumPy arrays."""
    tree, names = _parsed_expression(expression)
    if set(parameters) != set(names):
        raise ValueError("Expression parameters do not match the supplied fit parameters.")
    coordinates = np.asarray(x, dtype=float)
    with np.errstate(all="ignore"):
        values = np.asarray(_evaluate_expression_node(tree, coordinates, parameters), dtype=float)
    return np.broadcast_to(values, coordinates.shape)
