"""Tests for the SQL query tool."""

import sqlite3
from pathlib import Path

import pytest

from src.tools.sql_query import sql_query


@pytest.fixture
def sample_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database with test data."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL
        );
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            amount REAL NOT NULL,
            quarter TEXT NOT NULL,
            region TEXT NOT NULL
        );

        INSERT INTO products VALUES (1, 'Widget A', 'Electronics', 99.99);
        INSERT INTO products VALUES (2, 'Widget B', 'Office', 149.99);
        INSERT INTO products VALUES (3, 'Widget C', 'Software', 29.99);

        INSERT INTO sales VALUES (1, 1, 10, 999.90, 'Q1', 'Norte');
        INSERT INTO sales VALUES (2, 1, 5, 499.95, 'Q2', 'Norte');
        INSERT INTO sales VALUES (3, 2, 3, 449.97, 'Q3', 'Centro');
        INSERT INTO sales VALUES (4, 3, 20, 599.80, 'Q3', 'Sur');
        INSERT INTO sales VALUES (5, 2, 7, 1049.93, 'Q4', 'Norte');
    """)
    conn.close()
    return db_path


class TestSQLQueryBasic:
    def test_select_all_products(self, sample_db: Path) -> None:
        result = sql_query("SELECT * FROM products", db_path=sample_db)
        assert result.success is True
        assert result.row_count == 3
        assert result.columns == ["id", "name", "category", "price"]

    def test_select_with_where(self, sample_db: Path) -> None:
        result = sql_query(
            "SELECT name, price FROM products WHERE category = 'Electronics'",
            db_path=sample_db,
        )
        assert result.success is True
        assert result.row_count == 1
        assert result.rows[0][0] == "Widget A"

    def test_aggregate_query(self, sample_db: Path) -> None:
        result = sql_query(
            "SELECT SUM(amount) FROM sales WHERE quarter = 'Q3'",
            db_path=sample_db,
        )
        assert result.success is True
        assert result.row_count == 1
        assert result.rows[0][0] == pytest.approx(1049.77, rel=1e-2)

    def test_join_query(self, sample_db: Path) -> None:
        result = sql_query(
            "SELECT p.name, SUM(s.amount) as total "
            "FROM sales s JOIN products p ON s.product_id = p.id "
            "GROUP BY p.name ORDER BY total DESC",
            db_path=sample_db,
        )
        assert result.success is True
        assert result.row_count == 3

    def test_max_rows_limit(self, sample_db: Path) -> None:
        result = sql_query("SELECT * FROM sales", db_path=sample_db, max_rows=2)
        assert result.success is True
        assert result.row_count == 2


class TestSQLQueryValidation:
    def test_reject_insert(self, sample_db: Path) -> None:
        result = sql_query(
            "INSERT INTO products VALUES (99, 'Hack', 'Bad', 0)",
            db_path=sample_db,
        )
        assert result.success is False
        assert "Write operations" in result.error

    def test_reject_update(self, sample_db: Path) -> None:
        result = sql_query("UPDATE products SET price = 0", db_path=sample_db)
        assert result.success is False
        assert "Write operations" in result.error

    def test_reject_delete(self, sample_db: Path) -> None:
        result = sql_query("DELETE FROM products", db_path=sample_db)
        assert result.success is False
        assert "Write operations" in result.error

    def test_reject_drop(self, sample_db: Path) -> None:
        result = sql_query("DROP TABLE products", db_path=sample_db)
        assert result.success is False
        assert "Write operations" in result.error

    def test_reject_multiple_statements(self, sample_db: Path) -> None:
        result = sql_query(
            "SELECT 1; SELECT 2",
            db_path=sample_db,
        )
        assert result.success is False
        assert "Multiple statements" in result.error

    def test_reject_empty_query(self, sample_db: Path) -> None:
        result = sql_query("", db_path=sample_db)
        assert result.success is False
        assert "Empty query" in result.error


class TestSQLQueryErrors:
    def test_nonexistent_database(self, tmp_path: Path) -> None:
        result = sql_query("SELECT 1", db_path=tmp_path / "nonexistent.db")
        assert result.success is False
        assert "Database not found" in result.error

    def test_invalid_sql_syntax(self, sample_db: Path) -> None:
        result = sql_query("SELEC * FORM products", db_path=sample_db)
        assert result.success is False
        assert result.error is not None

    def test_nonexistent_table(self, sample_db: Path) -> None:
        result = sql_query("SELECT * FROM nonexistent", db_path=sample_db)
        assert result.success is False
        assert result.error is not None
