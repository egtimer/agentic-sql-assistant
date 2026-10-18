"""Tool for evaluating mathematical expressions safely."""

import ast
import logging
import math
import operator
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SAFE_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
}

MAX_EXPRESSION_LENGTH = 500
MAX_POWER = 1000


class CalculatorResult(BaseModel):
    """Result of a mathematical calculation."""

    expression: str = Field(description="The input expression")
    result: float | int = Field(description="Calculated result")
    result_formatted: str = Field(description="Human-readable formatted result")
    success: bool = Field(description="Whether the calculation succeeded")
    error: str | None = Field(default=None, description="Error message if calculation failed")


def _eval_node(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node using only safe operations."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")

        left = _eval_node(node.left)
        right = _eval_node(node.right)

        if op_type == ast.Pow and isinstance(right, (int, float)) and right > MAX_POWER:
            raise ValueError(f"Exponent too large: {right} (max {MAX_POWER})")

        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ValueError("Division by zero")

        bin_result: float | int = SAFE_OPERATORS[op_type](left, right)
        return bin_result

    if isinstance(node, ast.UnaryOp):
        unary_type = type(node.op)
        if unary_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {unary_type.__name__}")
        unary_result: float | int = SAFE_OPERATORS[unary_type](_eval_node(node.operand))
        return unary_result

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only named function calls are supported")

        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Function not allowed: {func_name}")

        args = [_eval_node(arg) for arg in node.args]
        call_result: float | int = SAFE_FUNCTIONS[func_name](*args)
        return call_result

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def _format_result(value: float | int) -> str:
    """Format a numeric result for display."""
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return f"{value:,}"


def calculate(expression: str) -> CalculatorResult:
    """Evaluate a mathematical expression safely.

    Supports basic arithmetic (+, -, *, /, //, %, **) and functions
    (abs, round, min, max, sum, sqrt, ceil, floor, log, log10).

    Args:
        expression: Mathematical expression to evaluate (e.g., "(234500 - 198000) / 198000 * 100").

    Returns:
        CalculatorResult with the computed value.
    """
    expression = expression.strip()

    if not expression:
        return CalculatorResult(
            expression=expression,
            result=0,
            result_formatted="0",
            success=False,
            error="Empty expression",
        )

    if len(expression) > MAX_EXPRESSION_LENGTH:
        return CalculatorResult(
            expression=expression[:50] + "...",
            result=0,
            result_formatted="0",
            success=False,
            error=f"Expression too long (max {MAX_EXPRESSION_LENGTH} characters)",
        )

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)

        logger.info("Calculated: %s = %s", expression[:80], result)

        return CalculatorResult(
            expression=expression,
            result=result,
            result_formatted=_format_result(result),
            success=True,
        )

    except SyntaxError:
        return CalculatorResult(
            expression=expression,
            result=0,
            result_formatted="0",
            success=False,
            error=f"Invalid mathematical expression: {expression[:80]}",
        )
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        return CalculatorResult(
            expression=expression,
            result=0,
            result_formatted="0",
            success=False,
            error=str(e),
        )
