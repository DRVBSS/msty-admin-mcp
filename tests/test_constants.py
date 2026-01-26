#!/usr/bin/env python3
"""
Tests for Msty Admin MCP - Constants Module

Run with: pytest tests/test_constants.py -v
"""

import os
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import (
    SERVER_VERSION,
    SIDECAR_HOST,
    SIDECAR_PROXY_PORT,
    LOCAL_AI_SERVICE_PORT,
    SIDECAR_TIMEOUT,
    MLX_SERVICE_PORT,
    LLAMACPP_SERVICE_PORT,
    VIBE_PROXY_PORT,
    ALLOWED_TABLE_NAMES,
    MODEL_SIZE_PATTERNS,
    MODEL_CAPABILITY_PATTERNS,
    MODEL_SPEED_PATTERNS
)


class TestServerVersion:
    """Tests for server version constant"""

    def test_version_is_string(self):
        """Verify version is a string"""
        assert isinstance(SERVER_VERSION, str)

    def test_version_follows_semver(self):
        """Verify version follows semantic versioning (x.y.z)"""
        parts = SERVER_VERSION.split(".")
        assert len(parts) == 3, f"Version should have 3 parts: {SERVER_VERSION}"
        for part in parts:
            assert part.isdigit(), f"Version parts should be numeric: {SERVER_VERSION}"


class TestNetworkConstants:
    """Tests for network configuration constants"""

    def test_sidecar_host_default(self):
        """Verify default sidecar host is localhost"""
        assert SIDECAR_HOST in ["127.0.0.1", "localhost"]

    def test_ports_are_integers(self):
        """Verify all ports are integers"""
        assert isinstance(SIDECAR_PROXY_PORT, int)
        assert isinstance(LOCAL_AI_SERVICE_PORT, int)
        assert isinstance(MLX_SERVICE_PORT, int)
        assert isinstance(LLAMACPP_SERVICE_PORT, int)
        assert isinstance(VIBE_PROXY_PORT, int)

    def test_ports_in_valid_range(self):
        """Verify ports are in valid range (1-65535)"""
        ports = [SIDECAR_PROXY_PORT, LOCAL_AI_SERVICE_PORT, MLX_SERVICE_PORT,
                 LLAMACPP_SERVICE_PORT, VIBE_PROXY_PORT]
        for port in ports:
            assert 1 <= port <= 65535, f"Port {port} out of valid range"

    def test_timeout_is_positive(self):
        """Verify timeout is positive"""
        assert SIDECAR_TIMEOUT > 0


class TestAllowedTableNames:
    """Tests for SQL injection protection table allowlist"""

    def test_is_frozenset(self):
        """Verify allowed tables is a frozenset (immutable)"""
        assert isinstance(ALLOWED_TABLE_NAMES, frozenset)

    def test_contains_core_tables(self):
        """Verify core Msty tables are included"""
        core_tables = ["chats", "messages", "personas", "prompts"]
        for table in core_tables:
            assert table in ALLOWED_TABLE_NAMES, f"Missing core table: {table}"

    def test_no_sql_injection_characters(self):
        """Verify table names don't contain SQL injection characters"""
        dangerous_chars = [";", "'", '"', "--", "/*", "*/", "DROP", "DELETE"]
        for table in ALLOWED_TABLE_NAMES:
            for char in dangerous_chars:
                assert char.lower() not in table.lower(), \
                    f"Table {table} contains dangerous character: {char}"


class TestModelPatterns:
    """Tests for model tagging patterns"""

    def test_size_patterns_has_categories(self):
        """Verify size patterns has expected categories"""
        expected = ["large", "medium", "small"]
        for cat in expected:
            assert cat in MODEL_SIZE_PATTERNS

    def test_capability_patterns_has_categories(self):
        """Verify capability patterns has expected categories"""
        expected = ["coding", "reasoning", "creative", "vision", "embedding"]
        for cat in expected:
            assert cat in MODEL_CAPABILITY_PATTERNS

    def test_speed_patterns_has_fast(self):
        """Verify speed patterns has fast category"""
        assert "fast" in MODEL_SPEED_PATTERNS

    def test_pattern_values_are_lists(self):
        """Verify all pattern values are lists of strings"""
        for patterns in [MODEL_SIZE_PATTERNS, MODEL_CAPABILITY_PATTERNS, MODEL_SPEED_PATTERNS]:
            for key, value in patterns.items():
                assert isinstance(value, list), f"Pattern {key} should be a list"
                for item in value:
                    assert isinstance(item, str), f"Pattern items should be strings"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
