#!/usr/bin/env python3
"""
Tests for Msty Admin MCP - Cache Module

Run with: pytest tests/test_cache.py -v
"""

import time
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cache import (
    ResponseCache,
    get_cached_models,
    cache_models,
    get_cache
)


class TestResponseCache:
    """Tests for ResponseCache class"""

    def test_get_returns_none_for_missing_key(self):
        """Verify missing keys return None"""
        cache = ResponseCache()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self):
        """Verify basic set and get operations"""
        cache = ResponseCache()
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_ttl_expiry(self):
        """Verify entries expire after TTL"""
        cache = ResponseCache(default_ttl=1)
        cache.set("key", "value", ttl=1)
        assert cache.get("key") == "value"
        time.sleep(1.5)
        assert cache.get("key") is None

    def test_invalidate_specific_key(self):
        """Verify specific key can be invalidated"""
        cache = ResponseCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.invalidate("key1")
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_invalidate_all(self):
        """Verify all entries can be cleared"""
        cache = ResponseCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.invalidate()  # Clear all
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_stats_returns_dict(self):
        """Verify stats returns dictionary"""
        cache = ResponseCache()
        stats = cache.stats()
        assert isinstance(stats, dict)
        assert "total_entries" in stats
        assert "valid_entries" in stats

    def test_stats_counts_correctly(self):
        """Verify stats count entries correctly"""
        cache = ResponseCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        stats = cache.stats()
        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 2

    def test_stores_complex_values(self):
        """Verify cache stores complex Python objects"""
        cache = ResponseCache()
        complex_data = {
            "models": [{"id": "model1"}, {"id": "model2"}],
            "count": 2,
            "nested": {"deep": {"value": True}}
        }
        cache.set("data", complex_data)
        retrieved = cache.get("data")
        assert retrieved == complex_data
        assert retrieved["nested"]["deep"]["value"] is True


class TestCacheModels:
    """Tests for model caching functions"""

    def test_cache_and_retrieve_models(self):
        """Verify models can be cached and retrieved"""
        # Clear any existing cache first
        get_cache().invalidate()

        models_data = {"models": [{"id": "test-model"}]}
        cache_models(models_data, ttl=60)
        retrieved = get_cached_models()
        assert retrieved == models_data

    def test_get_cached_models_returns_none_when_empty(self):
        """Verify empty cache returns None"""
        get_cache().invalidate()  # Clear cache
        result = get_cached_models()
        assert result is None


class TestGetCache:
    """Tests for get_cache function"""

    def test_returns_response_cache_instance(self):
        """Verify get_cache returns ResponseCache instance"""
        cache = get_cache()
        assert isinstance(cache, ResponseCache)

    def test_returns_same_instance(self):
        """Verify get_cache returns singleton instance"""
        cache1 = get_cache()
        cache2 = get_cache()
        assert cache1 is cache2


class TestCacheConcurrency:
    """Tests for cache behavior under concurrent access"""

    def test_overwrite_updates_value(self):
        """Verify setting same key overwrites value"""
        cache = ResponseCache()
        cache.set("key", "value1")
        cache.set("key", "value2")
        assert cache.get("key") == "value2"

    def test_overwrite_updates_ttl(self):
        """Verify overwriting key updates TTL"""
        cache = ResponseCache()
        cache.set("key", "value", ttl=1)
        time.sleep(0.5)
        cache.set("key", "new_value", ttl=5)
        time.sleep(1)
        # Should still be valid with new TTL
        assert cache.get("key") == "new_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
