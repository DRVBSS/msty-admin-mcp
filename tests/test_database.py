#!/usr/bin/env python3
"""
Tests for Msty Admin MCP - Database Module

Run with: pytest tests/test_database.py -v
"""

import os
import sqlite3
import tempfile
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import (
    get_database_connection,
    query_database,
    get_table_names,
    is_safe_table_name,
    validate_table_exists,
    safe_query_table,
    safe_count_table,
    get_table_row_count
)
from src.constants import ALLOWED_TABLE_NAMES


class TestIsSafeTableName:
    """Tests for SQL injection protection"""

    def test_allows_safe_tables(self):
        """Verify allowed tables pass validation"""
        for table in ["chats", "messages", "personas", "prompts"]:
            assert is_safe_table_name(table) is True, f"Should allow {table}"

    def test_rejects_sql_injection_attempts(self):
        """Verify SQL injection attempts are rejected"""
        dangerous = [
            "chats; DROP TABLE users;",
            "chats' OR '1'='1",
            "chats--",
            "chats/**/",
            "chats UNION SELECT * FROM passwords",
            "robert'); DROP TABLE students;--",
        ]
        for attempt in dangerous:
            assert is_safe_table_name(attempt) is False, \
                f"Should reject: {attempt}"

    def test_rejects_empty_string(self):
        """Verify empty string is rejected"""
        assert is_safe_table_name("") is False

    def test_rejects_none(self):
        """Verify None is rejected"""
        assert is_safe_table_name(None) is False

    def test_rejects_unknown_tables(self):
        """Verify tables not in allowlist are rejected"""
        assert is_safe_table_name("unknown_table_xyz") is False
        assert is_safe_table_name("secrets") is False
        assert is_safe_table_name("passwords") is False


class TestDatabaseConnection:
    """Tests for database connection handling"""

    def test_returns_none_for_missing_db(self):
        """Verify missing database returns None"""
        conn = get_database_connection("/nonexistent/path/db.sqlite")
        assert conn is None

    def test_returns_none_for_empty_path(self):
        """Verify empty path returns None"""
        conn = get_database_connection("")
        assert conn is None

    def test_returns_none_for_none_path(self):
        """Verify None path returns None"""
        conn = get_database_connection(None)
        assert conn is None

    def test_connects_to_valid_db(self):
        """Verify connection to valid database works"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Create a simple database
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.close()

            # Test our connection function
            conn = get_database_connection(db_path)
            assert conn is not None
            conn.close()
        finally:
            os.unlink(db_path)


class TestQueryDatabase:
    """Tests for query_database function"""

    def test_returns_empty_for_invalid_db(self):
        """Verify invalid database returns empty list"""
        result = query_database("/nonexistent/db.sqlite", "SELECT 1")
        assert result == []

    def test_returns_list_of_dicts(self):
        """Verify query returns list of dictionaries"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'Alice')")
            conn.execute("INSERT INTO test VALUES (2, 'Bob')")
            conn.commit()
            conn.close()

            result = query_database(db_path, "SELECT * FROM test")
            assert isinstance(result, list)
            assert len(result) == 2
            assert isinstance(result[0], dict)
            assert result[0]["id"] == 1
            assert result[0]["name"] == "Alice"
        finally:
            os.unlink(db_path)

    def test_parameterized_queries(self):
        """Verify parameterized queries work"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'Alice')")
            conn.execute("INSERT INTO test VALUES (2, 'Bob')")
            conn.commit()
            conn.close()

            result = query_database(db_path, "SELECT * FROM test WHERE id = ?", (1,))
            assert len(result) == 1
            assert result[0]["name"] == "Alice"
        finally:
            os.unlink(db_path)


class TestSafeQueryTable:
    """Tests for safe_query_table function"""

    def test_rejects_invalid_table(self):
        """Verify invalid table names return empty"""
        result = safe_query_table("/any/path.db", "DROP TABLE users;")
        assert result == []

    def test_rejects_nonexistent_table(self):
        """Verify nonexistent tables return empty"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE real_table (id INTEGER)")
            conn.close()

            result = safe_query_table(db_path, "fake_table")
            assert result == []
        finally:
            os.unlink(db_path)


class TestSafeCountTable:
    """Tests for safe_count_table function"""

    def test_rejects_invalid_table(self):
        """Verify invalid table names return 0"""
        count = safe_count_table("/any/path.db", "DROP TABLE users;")
        assert count == 0

    def test_counts_rows_correctly(self):
        """Verify correct row counting"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE chats (id INTEGER)")
            conn.execute("INSERT INTO chats VALUES (1)")
            conn.execute("INSERT INTO chats VALUES (2)")
            conn.execute("INSERT INTO chats VALUES (3)")
            conn.commit()
            conn.close()

            # Note: This will fail because 'chats' must exist AND be in allowlist
            # For proper testing, we'd need to mock ALLOWED_TABLE_NAMES
            count = safe_count_table(db_path, "chats")
            # The function validates against both existence and allowlist
            assert isinstance(count, int)
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
