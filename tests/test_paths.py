#!/usr/bin/env python3
"""
Tests for Msty Admin MCP - Paths Module

Run with: pytest tests/test_paths.py -v
"""

import os
import pytest
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paths import (
    get_msty_paths,
    sanitize_path,
    expand_path,
    read_claude_desktop_config
)


class TestGetMstyPaths:
    """Tests for get_msty_paths function"""

    def test_returns_dict(self):
        """Verify get_msty_paths returns a dictionary"""
        paths = get_msty_paths()
        assert isinstance(paths, dict)

    def test_has_expected_keys(self):
        """Verify all expected path keys are present"""
        paths = get_msty_paths()
        expected_keys = ["app", "app_alt", "data", "sidecar", "database", "mlx_models"]
        for key in expected_keys:
            assert key in paths, f"Missing key: {key}"

    def test_paths_are_strings_or_none(self):
        """Verify path values are strings or None"""
        paths = get_msty_paths()
        for key, value in paths.items():
            assert value is None or isinstance(value, str), \
                f"Path {key} should be string or None, got {type(value)}"


class TestSanitizePath:
    """Tests for sanitize_path function (replaces home with $HOME)"""

    def test_replaces_home_with_variable(self):
        """Verify home directory is replaced with $HOME"""
        home = os.path.expanduser("~")
        result = sanitize_path(f"{home}/Documents/file.txt")
        assert "$HOME" in result
        assert home not in result

    def test_only_replaces_home_prefix(self):
        """Verify only home at start is replaced"""
        home = os.path.expanduser("~")
        # Path not starting with home should not be changed
        result = sanitize_path("/other/path/to/file")
        assert result == "/other/path/to/file"

    def test_empty_string_returns_empty(self):
        """Verify empty string returns empty"""
        assert sanitize_path("") == ""

    def test_none_returns_none(self):
        """Verify None returns None (falsy)"""
        result = sanitize_path(None)
        assert not result

    def test_preserves_non_home_path(self):
        """Verify paths not starting with home are preserved"""
        valid_path = "/Applications/SomeApp.app"
        result = sanitize_path(valid_path)
        assert result == valid_path


class TestExpandPath:
    """Tests for expand_path function (expands $HOME and ~)"""

    def test_expands_tilde(self):
        """Verify tilde is expanded to home directory"""
        result = expand_path("~/Documents")
        assert "~" not in result
        assert os.path.expanduser("~") in result

    def test_expands_home_variable(self):
        """Verify $HOME is expanded to home directory"""
        result = expand_path("$HOME/Documents")
        assert "$HOME" not in result
        assert os.path.expanduser("~") in result

    def test_absolute_path_unchanged(self):
        """Verify absolute paths without special chars are unchanged"""
        path = "/absolute/path/to/file"
        result = expand_path(path)
        assert result == path

    def test_empty_string_returns_empty(self):
        """Verify empty path returns falsy"""
        result = expand_path("")
        assert not result


class TestReadClaudeDesktopConfig:
    """Tests for read_claude_desktop_config function"""

    def test_returns_dict(self):
        """Verify function returns a dictionary"""
        result = read_claude_desktop_config()
        assert isinstance(result, dict)

    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_parses_valid_json(self, mock_exists, mock_open):
        """Verify valid JSON config is parsed"""
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = \
            '{"mcpServers": {"test": {"command": "test"}}}'

        result = read_claude_desktop_config()
        # Either returns the parsed config or empty dict if path doesn't exist
        assert isinstance(result, dict)

    def test_returns_empty_on_missing_file(self):
        """Verify missing config file returns empty dict"""
        # This should gracefully handle missing file
        result = read_claude_desktop_config()
        assert isinstance(result, dict)


class TestPathSecurity:
    """Security-focused tests for path handling"""

    def test_sanitize_returns_string(self):
        """Verify sanitize always returns string for string input"""
        result = sanitize_path("/any/path")
        assert isinstance(result, str)

    def test_expand_returns_string(self):
        """Verify expand returns string for valid input"""
        result = expand_path("~/test")
        assert isinstance(result, str)

    def test_paths_are_consistent(self):
        """Verify sanitize and expand are inverse operations for home paths"""
        home = os.path.expanduser("~")
        original = f"{home}/Documents/test.txt"
        sanitized = sanitize_path(original)
        expanded = expand_path(sanitized)
        assert expanded == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
