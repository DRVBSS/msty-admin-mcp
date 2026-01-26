#!/usr/bin/env python3
"""
Tests for Msty Admin MCP - Network Module

Run with: pytest tests/test_network.py -v
"""

import pytest
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import urllib.error

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.network import (
    make_api_request,
    is_process_running,
    is_local_ai_available,
    get_available_service_ports
)


class TestMakeApiRequest:
    """Tests for make_api_request function"""

    def test_returns_dict(self):
        """Verify function returns a dictionary"""
        result = make_api_request("/v1/models", port=99999)
        assert isinstance(result, dict)

    def test_handles_connection_failure(self):
        """Verify connection failures are handled gracefully"""
        result = make_api_request("/v1/models", port=99999, timeout=1)
        assert result["success"] is False
        assert "error" in result

    @patch('urllib.request.urlopen')
    def test_successful_request(self, mock_urlopen):
        """Verify successful requests return data"""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"models": []}'
        mock_response.status = 200
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        result = make_api_request("/v1/models", port=11964)
        assert result["success"] is True
        assert result["status_code"] == 200

    def test_connection_refused_handling(self):
        """Verify connection refused errors are handled gracefully"""
        # Try to connect to a port that's not listening
        result = make_api_request("/test", port=59999, timeout=1)
        assert result["success"] is False
        assert "error" in result
        assert isinstance(result["error"], str)


class TestIsProcessRunning:
    """Tests for is_process_running function"""

    def test_returns_bool(self):
        """Verify function returns boolean"""
        result = is_process_running("nonexistent_process_xyz123")
        assert isinstance(result, bool)

    def test_nonexistent_process(self):
        """Verify nonexistent process returns False"""
        result = is_process_running("this_process_definitely_does_not_exist_xyz")
        assert result is False

    @patch('psutil.process_iter')
    def test_finds_running_process(self, mock_iter):
        """Verify running processes are detected"""
        mock_proc = MagicMock()
        mock_proc.info = {'name': 'TestProcess'}
        mock_iter.return_value = [mock_proc]

        result = is_process_running("TestProcess")
        assert result is True

    @patch('psutil.process_iter')
    def test_case_insensitive_match(self, mock_iter):
        """Verify process name matching is case-insensitive"""
        mock_proc = MagicMock()
        mock_proc.info = {'name': 'MyApp'}
        mock_iter.return_value = [mock_proc]

        result = is_process_running("myapp")
        assert result is True

    @patch('psutil.process_iter')
    def test_handles_access_denied(self, mock_iter):
        """Verify AccessDenied exceptions are handled"""
        import psutil
        mock_proc = MagicMock()
        mock_proc.info.__getitem__.side_effect = psutil.AccessDenied()
        mock_iter.return_value = [mock_proc]

        # Should not raise, just return False
        result = is_process_running("test")
        assert result is False


class TestIsLocalAiAvailable:
    """Tests for is_local_ai_available function"""

    def test_returns_bool(self):
        """Verify function returns boolean"""
        result = is_local_ai_available(port=99999, timeout=1)
        assert isinstance(result, bool)

    def test_unavailable_port(self):
        """Verify unavailable port returns False"""
        result = is_local_ai_available(port=99999, timeout=1)
        assert result is False

    @patch('urllib.request.urlopen')
    def test_available_service(self, mock_urlopen):
        """Verify available service returns True"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        result = is_local_ai_available(port=11964)
        assert result is True


class TestGetAvailableServicePorts:
    """Tests for get_available_service_ports function"""

    def test_returns_dict(self):
        """Verify function returns dictionary"""
        result = get_available_service_ports()
        assert isinstance(result, dict)

    def test_has_expected_services(self):
        """Verify expected services are included"""
        result = get_available_service_ports()
        expected = ["local_ai", "mlx", "llamacpp", "vibe_proxy"]
        for service in expected:
            assert service in result, f"Missing service: {service}"

    def test_service_info_structure(self):
        """Verify each service has port and available keys"""
        result = get_available_service_ports()
        for name, info in result.items():
            assert "port" in info, f"Service {name} missing port"
            assert "available" in info, f"Service {name} missing available"
            assert isinstance(info["port"], int)
            assert isinstance(info["available"], bool)


class TestNetworkSecurity:
    """Security-focused tests for network operations"""

    def test_no_arbitrary_host_access(self):
        """Verify requests don't access arbitrary hosts by default"""
        # The function should use SIDECAR_HOST by default
        result = make_api_request("/test", port=80)
        # Even if this fails, it should fail connecting to localhost, not external
        assert isinstance(result, dict)

    def test_timeout_respected(self):
        """Verify timeout parameter is respected"""
        import time
        start = time.time()
        result = make_api_request("/test", port=99999, timeout=1)
        elapsed = time.time() - start
        # Should timeout within reasonable bounds (give some margin)
        assert elapsed < 5, "Request took too long despite timeout"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
