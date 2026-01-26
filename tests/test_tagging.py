#!/usr/bin/env python3
"""
Tests for Msty Admin MCP - Tagging Module

Run with: pytest tests/test_tagging.py -v
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tagging import (
    MODEL_TAGS,
    get_model_tags,
    find_models_by_tag
)


class TestModelTags:
    """Tests for MODEL_TAGS constant"""

    def test_has_patterns(self):
        """Verify MODEL_TAGS has patterns section"""
        assert "patterns" in MODEL_TAGS

    def test_has_overrides(self):
        """Verify MODEL_TAGS has overrides section"""
        assert "overrides" in MODEL_TAGS

    def test_patterns_has_expected_categories(self):
        """Verify expected tag categories exist"""
        expected = ["fast", "quality", "coding", "creative", "reasoning",
                    "embedding", "vision", "long_context"]
        for cat in expected:
            assert cat in MODEL_TAGS["patterns"], f"Missing pattern: {cat}"

    def test_overrides_are_lists(self):
        """Verify all overrides are lists of strings"""
        for model_id, tags in MODEL_TAGS["overrides"].items():
            assert isinstance(tags, list), f"{model_id} override should be list"
            for tag in tags:
                assert isinstance(tag, str), f"Tags should be strings"


class TestGetModelTags:
    """Tests for get_model_tags function"""

    def test_returns_list(self):
        """Verify function returns a list"""
        tags = get_model_tags("some-model")
        assert isinstance(tags, list)

    def test_override_takes_precedence(self):
        """Verify explicit overrides take precedence over patterns"""
        # Claude models have explicit overrides
        tags = get_model_tags("claude-opus-4.5")
        assert "quality" in tags
        assert "reasoning" in tags
        assert "coding" in tags

    def test_pattern_matching_coding(self):
        """Verify coding pattern matches"""
        tags = get_model_tags("deepseek-coder-v2")
        assert "coding" in tags

    def test_pattern_matching_fast(self):
        """Verify fast pattern matches"""
        tags = get_model_tags("phi-mini-3")
        assert "fast" in tags

    def test_pattern_matching_quality(self):
        """Verify quality pattern matches large models"""
        tags = get_model_tags("llama-70b-instruct")
        assert "quality" in tags

    def test_size_tagging_large(self):
        """Verify large models get 'large' tag"""
        tags = get_model_tags("some-model-70b")
        assert "large" in tags

    def test_size_tagging_medium(self):
        """Verify medium models get 'medium' tag"""
        tags = get_model_tags("some-model-32b")
        assert "medium" in tags

    def test_size_tagging_small(self):
        """Verify small models get 'small' tag"""
        tags = get_model_tags("some-model-7b")
        assert "small" in tags

    def test_default_to_general(self):
        """Verify unknown models get 'general' tag"""
        tags = get_model_tags("completely-unknown-model-name")
        assert "general" in tags

    def test_case_insensitive_patterns(self):
        """Verify pattern matching is case-insensitive"""
        tags1 = get_model_tags("DeepSeek-Coder-V2")
        tags2 = get_model_tags("deepseek-coder-v2")
        assert "coding" in tags1
        assert "coding" in tags2

    def test_embedding_model_detection(self):
        """Verify embedding models are detected"""
        tags = get_model_tags("nomic-embed-text")
        assert "embedding" in tags

    def test_vision_model_detection(self):
        """Verify vision models are detected"""
        tags = get_model_tags("llava-v1.5-7b")
        assert "vision" in tags

    def test_reasoning_model_detection(self):
        """Verify reasoning models are detected"""
        tags = get_model_tags("deepseek-r1-distill")
        assert "reasoning" in tags


class TestFindModelsByTag:
    """Tests for find_models_by_tag function"""

    def test_returns_list(self):
        """Verify function returns a list"""
        models = find_models_by_tag("coding", models=[])
        assert isinstance(models, list)

    def test_filters_by_tag(self):
        """Verify models are filtered by tag"""
        test_models = [
            {"id": "deepseek-coder"},
            {"id": "gpt-general"},
            {"id": "starcoder-base"}
        ]
        result = find_models_by_tag("coding", models=test_models)
        # Should find models matching coding pattern
        model_ids = [m["id"] for m in result]
        assert "deepseek-coder" in model_ids or "starcoder-base" in model_ids

    def test_adds_tags_to_results(self):
        """Verify matched models have _tags added"""
        test_models = [{"id": "deepseek-coder-v2"}]
        result = find_models_by_tag("coding", models=test_models)
        if result:
            assert "_tags" in result[0]
            assert isinstance(result[0]["_tags"], list)

    def test_empty_models_list(self):
        """Verify empty models list returns empty result"""
        result = find_models_by_tag("fast", models=[])
        assert result == []


class TestSpecificModelOverrides:
    """Tests for specific model overrides (v6.1.0 additions)"""

    def test_claude_opus_tags(self):
        """Verify Claude Opus has correct tags"""
        tags = get_model_tags("claude-opus-4-5-20251101")
        expected = ["quality", "reasoning", "coding", "creative", "large"]
        for tag in expected:
            assert tag in tags, f"Claude Opus missing {tag}"

    def test_claude_sonnet_tags(self):
        """Verify Claude Sonnet has correct tags"""
        tags = get_model_tags("claude-sonnet-4-5-20250929")
        assert "quality" in tags
        assert "reasoning" in tags
        assert "coding" in tags

    def test_claude_haiku_tags(self):
        """Verify Claude Haiku has correct tags"""
        tags = get_model_tags("claude-haiku-4-5-20251001")
        assert "fast" in tags
        assert "coding" in tags

    def test_gpt5_tags(self):
        """Verify GPT-5 has correct tags"""
        tags = get_model_tags("gpt-5")
        assert "quality" in tags
        assert "reasoning" in tags
        assert "large" in tags

    def test_gemini_pro_tags(self):
        """Verify Gemini Pro has correct tags"""
        tags = get_model_tags("gemini-2.5-pro")
        assert "quality" in tags
        assert "reasoning" in tags

    def test_gemini_flash_tags(self):
        """Verify Gemini Flash has correct tags"""
        tags = get_model_tags("gemini-2.5-flash")
        assert "fast" in tags

    def test_mistral_nemo_long_context(self):
        """Verify Mistral Nemo has long_context tag"""
        tags = get_model_tags("mistral-nemo-instruct")
        assert "long_context" in tags


class TestSizeDetectionOrder:
    """Tests for size detection order (v6.1.0 fix)"""

    def test_253b_not_small(self):
        """Verify 253B models are not tagged as small"""
        tags = get_model_tags("nemotron-ultra-253b")
        assert "small" not in tags
        assert "large" in tags

    def test_235b_not_small(self):
        """Verify 235B models are not tagged as small"""
        tags = get_model_tags("qwen3-235b-a22b")
        assert "small" not in tags
        assert "large" in tags

    def test_70b_is_large(self):
        """Verify 70B models are tagged as large"""
        tags = get_model_tags("llama-70b")
        assert "large" in tags
        assert "small" not in tags


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
