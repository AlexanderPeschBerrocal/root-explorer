"""Restricted custom fit expressions, numeric integration and failure isolation."""

from __future__ import annotations

import hist
import numpy as np
import pytest
from scipy.integrate import quad

from root_explorer.analysis.fit_expressions import (
    discover_fit_parameters,
    evaluate_fit_expression,
)
from root_explorer.analysis.fitting import FitConfig, fit_histogram_1d, model_bin_integrals


@pytest.mark.parametrize(
    ("expression", "names"),
    [
        ("a*x**2 + b*x + c", ("a", "b", "c")),
        ("A * exp(-0.5 * ((x - mu)/sigma)**2) + c", ("A", "mu", "sigma", "c")),
        ("a*sin(x) + b*cos(x) + pi + e", ("a", "b")),
        ("p * log(x) + q * sqrt(x)", ("p", "q")),
        ("a + a*x + a*x**2", ("a",)),
    ],
)
def test_parameter_discovery(expression, names):
    assert discover_fit_parameters(expression) == names


@pytest.mark.parametrize(
    "expression",
    [
        '__import__("os").system("echo nope")',
        "open('/etc/passwd')",
        "np.exp(x) + a",
        "x.__class__ + a",
        "a[0] + x",
        "(lambda x: x)(1) + a",
        "[x for x in (1,2)] + a",
        "a if x else b",
        "a < x",
        "(a := x)",
        "sum([x]) + a",
        "__builtins__ + x",
        "a; x",
        "exp(x, a)",
        "exp(x=x) + a",
        "a ** 100000",
        "a + True",
        "x + pi",
        "a + " + "x*" * 300 + "x",
    ],
)
def test_unsafe_or_invalid_expressions_rejected(expression):
    with pytest.raises(ValueError):
        discover_fit_parameters(expression)


def test_numeric_evaluation_broadcasts_constants():
    x = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        evaluate_fit_expression("a + sin(x)", x, {"a": 3.0}),
        3 + np.sin(x),
    )
    np.testing.assert_allclose(evaluate_fit_expression("a", x, {"a": 3.0}), [3, 3, 3])


def test_nonuniform_numeric_bin_integration():
    edges = np.array([0.2, 0.8, 1.9, 3.0, 6.0])
    params = {"A": 12.0, "tau": 2.0, "c": 0.4}
    expression = "A * exp(-x / tau) + c"
    integral = model_bin_integrals("Custom expression", edges, params, x0=0, expression=expression)
    expected = [
        quad(lambda x: params["A"] * np.exp(-x / params["tau"]) + params["c"], a, b)[0]
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]
    np.testing.assert_allclose(integral, expected, rtol=1e-12)


def test_custom_fit_recovers_parameters():
    edges = np.linspace(0, 8, 81)
    expr = "A * exp(-x / tau) + c"
    truth = {"A": 70.0, "tau": 1.6, "c": 2.0}
    values = model_bin_integrals("Custom expression", edges, truth, x0=0, expression=expr)
    h = hist.Hist(hist.axis.Variable(edges), storage=hist.storage.Weight())
    h.view().value = values
    h.view().variance = np.ones_like(values)
    config = FitConfig(
        "Custom expression",
        expression=expr,
        initial_values={"A": 60, "tau": 2.0, "c": 1.0},
        bounds={"tau": (0.01, None)},
        use_variances=True,
    )
    fitted = fit_histogram_1d(h, config)
    assert fitted.success, fitted.message
    assert fitted.expression == expr
    for key, expected in truth.items():
        assert fitted.parameters[key] == pytest.approx(expected, rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("expr", ["", "np.exp(x)", "__import__('os')", "x + 1"])
def test_invalid_fit_does_not_raise(expr):
    h = hist.Hist(hist.axis.Regular(25, 0, 5))
    result = fit_histogram_1d(h, FitConfig("Custom expression", expression=expr))
    assert not result.success
    assert "expression" in result.message.lower()


def test_custom_rejects_unknown_parameter_options():
    h = hist.Hist(hist.axis.Regular(25, 0, 5))
    result = fit_histogram_1d(
        h, FitConfig("Custom expression", expression="a*x + b", initial_values={"evil": 1})
    )
    assert not result.success
    assert "parameter" in result.message.lower()
