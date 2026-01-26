#!/usr/bin/env python3
"""
Tests for Msty Admin MCP Server - Phase 1

Run with: pytest tests/test_server.py -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the server module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from modular structure (v6.5.0+)
from src.paths import get_msty_paths
from src.network import is_process_running
from src.database import get_table_names
from src.models import MstyInstallation, MstyHealthReport, DatabaseStats
from src.constants import SERVER_VERSION


class TestPathResolution:
    """Tests for path resolution utilities"""
    
    def test_get_msty_paths_returns_dict(self):
        """Verify get_msty_paths returns a dictionary"""
        paths = get_msty_paths()
        assert isinstance(paths, dict)
        
    def test_get_msty_paths_has_expected_keys(self):
        """Verify all expected path keys are present"""
        paths = get_msty_paths()
        expected_keys = ["app", "app_alt", "data", "sidecar", "database", "mlx_models"]
        for key in expected_keys:
            assert key in paths, f"Missing key: {key}"


class TestProcessDetection:
    """Tests for process detection"""
    
    def test_is_process_running_returns_bool(self):
        """Verify is_process_running returns boolean"""
        result = is_process_running("nonexistent_process_12345")
        assert isinstance(result, bool)
        assert result is False
    
    @patch('psutil.process_iter')
    def test_is_process_running_finds_process(self, mock_iter):
        """Test that running processes are detected"""
        mock_proc = MagicMock()
        mock_proc.info = {'name': 'MstyStudio'}
        mock_iter.return_value = [mock_proc]
        
        result = is_process_running("MstyStudio")
        assert result is True


class TestDataClasses:
    """Tests for data class structures"""
    
    def test_msty_installation_defaults(self):
        """Verify MstyInstallation default values"""
        install = MstyInstallation(installed=False)
        assert install.installed is False
        assert install.version is None
        assert install.is_running is False
        assert install.platform_info == {}
    
    def test_msty_health_report_defaults(self):
        """Verify MstyHealthReport default values"""
        report = MstyHealthReport(overall_status="unknown")
        assert report.overall_status == "unknown"
        assert report.database_status == {}
        assert report.recommendations == []
    
    def test_database_stats_defaults(self):
        """Verify DatabaseStats default values"""
        stats = DatabaseStats()
        assert stats.total_conversations == 0
        assert stats.total_messages == 0
        assert stats.database_size_mb == 0.0


class TestMCPTools:
    """Integration tests for MCP tools"""
    
    def test_detect_msty_installation_returns_json(self):
        """Verify detect_msty_installation returns valid JSON"""
        from src.server import detect_msty_installation
        result = detect_msty_installation()
        
        # Should be valid JSON
        data = json.loads(result)
        assert isinstance(data, dict)
        assert "installed" in data
        assert "platform_info" in data
    
    def test_read_msty_database_stats_returns_json(self):
        """Verify read_msty_database returns valid JSON"""
        from src.server import read_msty_database
        result = read_msty_database(query_type="stats")
        
        data = json.loads(result)
        assert isinstance(data, dict)
        assert "query_type" in data
    
    def test_read_msty_database_tables_returns_json(self):
        """Verify read_msty_database tables query works"""
        from src.server import read_msty_database
        result = read_msty_database(query_type="tables")
        
        data = json.loads(result)
        assert isinstance(data, dict)
    
    def test_list_configured_tools_returns_json(self):
        """Verify list_configured_tools returns valid JSON"""
        from src.server import list_configured_tools
        result = list_configured_tools()
        
        data = json.loads(result)
        assert isinstance(data, dict)
        assert "tools" in data
        assert "tool_count" in data
    
    def test_get_model_providers_returns_json(self):
        """Verify get_model_providers returns valid JSON"""
        from src.server import get_model_providers
        result = get_model_providers()
        
        data = json.loads(result)
        assert isinstance(data, dict)
        assert "local_models" in data
        assert "remote_providers" in data
    
    def test_analyse_msty_health_returns_json(self):
        """Verify analyse_msty_health returns valid JSON"""
        from src.server import analyse_msty_health
        result = analyse_msty_health()
        
        data = json.loads(result)
        assert isinstance(data, dict)
        assert "overall_status" in data
        assert "recommendations" in data
    
    def test_get_server_status_returns_json(self):
        """Verify get_server_status returns valid JSON"""
        from src.server import get_server_status
        result = get_server_status()
        
        data = json.loads(result)
        assert isinstance(data, dict)
        assert "server" in data
        assert "available_tools" in data
        assert data["server"]["version"] == SERVER_VERSION


class TestSecurityFeatures:
    """Tests for security-related functionality"""
    
    def test_api_keys_redacted(self):
        """Verify API keys are redacted in provider queries"""
        from src.server import get_model_providers
        result = get_model_providers()
        
        # If there are any providers, keys should be redacted
        data = json.loads(result)
        for provider in data.get("database_providers", []):
            for key, value in provider.items():
                if any(x in key.lower() for x in ["key", "secret", "token"]):
                    assert value == "[REDACTED]", f"Key {key} not redacted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
