"""Tests for the calculator tool."""

import pytest

from src.tools.calculator import calculate


class TestCalculatorBasic:
    def test_addition(self) -> None:
        result = calculate("2 + 3")
        assert result.success is True
        assert result.result == 5

    def test_subtraction(self) -> None:
        result = calculate("100 - 42")
        assert result.success is True
        assert result.result == 58

    def test_multiplication(self) -> None:
        result = calculate("7 * 8")
        assert result.success is True
        assert result.result == 56

    def test_division(self) -> None:
        result = calculate("100 / 4")
        assert result.success is True
        assert result.result == 25.0

    def test_floor_division(self) -> None:
        result = calculate("17 // 3")
        assert result.success is True
        assert result.result == 5

    def test_modulo(self) -> None:
        result = calculate("17 % 5")
        assert result.success is True
        assert result.result == 2

    def test_power(self) -> None:
        result = calculate("2 ** 10")
        assert result.success is True
        assert result.result == 1024

    def test_negative_numbers(self) -> None:
        result = calculate("-5 + 3")
        assert result.success is True
        assert result.result == -2

    def test_complex_expression(self) -> None:
        result = calculate("(234500 - 198000) / 198000 * 100")
        assert result.success is True
        assert result.result == pytest.approx(18.4343, rel=1e-3)

    def test_nested_parentheses(self) -> None:
        result = calculate("((2 + 3) * (4 + 5)) / 3")
        assert result.success is True
        assert result.result == 15.0

    def test_floating_point(self) -> None:
        result = calculate("0.1 + 0.2")
        assert result.success is True
        assert result.result == pytest.approx(0.3)


class TestCalculatorFunctions:
    def test_abs(self) -> None:
        result = calculate("abs(-42)")
        assert result.success is True
        assert result.result == 42

    def test_round(self) -> None:
        result = calculate("round(3.14159, 2)")
        assert result.success is True
        assert result.result == 3.14

    def test_sqrt(self) -> None:
        result = calculate("sqrt(144)")
        assert result.success is True
        assert result.result == 12.0

    def test_min_max(self) -> None:
        result_min = calculate("min(3, 1, 4, 1, 5)")
        result_max = calculate("max(3, 1, 4, 1, 5)")
        assert result_min.result == 1
        assert result_max.result == 5

    def test_ceil_floor(self) -> None:
        assert calculate("ceil(4.2)").result == 5
        assert calculate("floor(4.8)").result == 4

    def test_log(self) -> None:
        result = calculate("log10(1000)")
        assert result.success is True
        assert result.result == pytest.approx(3.0)


class TestCalculatorFormatting:
    def test_integer_result_formatted(self) -> None:
        result = calculate("2 + 3")
        assert result.result_formatted == "5"

    def test_large_integer_formatted(self) -> None:
        result = calculate("1000 * 1000")
        assert result.result_formatted == "1,000,000"

    def test_float_result_formatted(self) -> None:
        result = calculate("10 / 3")
        assert "3.333" in result.result_formatted

    def test_whole_float_formatted_as_integer(self) -> None:
        result = calculate("10.0 / 2.0")
        assert result.result_formatted == "5"


class TestCalculatorValidation:
    def test_empty_expression(self) -> None:
        result = calculate("")
        assert result.success is False
        assert "Empty" in result.error

    def test_invalid_syntax(self) -> None:
        result = calculate("2 +* 3")
        assert result.success is False
        assert "Invalid" in result.error

    def test_division_by_zero(self) -> None:
        result = calculate("1 / 0")
        assert result.success is False
        assert "Division by zero" in result.error

    def test_forbidden_function(self) -> None:
        result = calculate("__import__('os').system('ls')")
        assert result.success is False

    def test_power_too_large(self) -> None:
        result = calculate("2 ** 10000")
        assert result.success is False
        assert "too large" in result.error

    def test_expression_too_long(self) -> None:
        result = calculate("1 + " * 200 + "1")
        assert result.success is False
        assert "too long" in result.error

    def test_string_not_allowed(self) -> None:
        result = calculate("'hello' + 'world'")
        assert result.success is False
