#!/usr/bin/env python3
"""
Tests for Msty Admin MCP - Tagging Module v2.0

Run with: pytest tests/test_tagging.py -v
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tagging import (
    MODEL_TAGS,
    TAG_DEFINITIONS,
    get_model_tags,
    find_models_by_tag,
    get_all_tags,
    detect_model_size,
    detect_context_length,
)


class TestTagDefinitions:
    """Tests for TAG_DEFINITIONS constant"""

    def test_has_capability_tags(self):
        """Verify capability tags exist"""
        expected = ["fast", "quality", "coding", "creative", "reasoning",
                    "embedding", "vision", "long_context"]
        for tag in expected:
            assert tag in TAG_DEFINITIONS, f"Missing tag: {tag}"

    def test_has_size_tags(self):
        """Verify size tags exist"""
        expected = ["small", "medium", "large", "massive"]
        for tag in expected:
            assert tag in TAG_DEFINITIONS, f"Missing size tag: {tag}"

    def test_has_context_tags(self):
        """Verify context length tags exist"""
        expected = ["long_context", "very_long_context", "massive_context"]
        for tag in expected:
            assert tag in TAG_DEFINITIONS, f"Missing context tag: {tag}"

    def test_has_quantization_tags(self):
        """Verify quantization tags exist"""
        expected = ["fp16", "8bit", "6bit", "5bit", "4bit", "3bit"]
        for tag in expected:
            assert tag in TAG_DEFINITIONS, f"Missing quant tag: {tag}"

    def test_has_architecture_tags(self):
        """Verify architecture tags exist"""
        expected = ["moe", "mlx", "gguf"]
        for tag in expected:
            assert tag in TAG_DEFINITIONS, f"Missing arch tag: {tag}"

    def test_all_tags_have_descriptions(self):
        """Verify all tags have non-empty descriptions"""
        for tag, desc in TAG_DEFINITIONS.items():
            assert isinstance(desc, str), f"{tag} description should be string"
            assert len(desc) > 5, f"{tag} description too short"


class TestModelTags:
    """Tests for MODEL_TAGS constant (legacy compatibility)"""

    def test_has_patterns(self):
        """Verify MODEL_TAGS has patterns section"""
        assert "patterns" in MODEL_TAGS

    def test_has_overrides(self):
        """Verify MODEL_TAGS has overrides section"""
        assert "overrides" in MODEL_TAGS

    def test_overrides_are_lists(self):
        """Verify all overrides are lists of strings"""
        for model_id, tags in MODEL_TAGS["overrides"].items():
            assert isinstance(tags, list), f"{model_id} override should be list"
            for tag in tags:
                assert isinstance(tag, str), f"Tags should be strings"


class TestDetectModelSize:
    """Tests for detect_model_size function"""

    def test_massive_model(self):
        """Verify massive models (200B+) are detected"""
        size, params = detect_model_size("qwen3-235b-model")
        assert size == "massive"
        assert params == 235

    def test_large_model(self):
        """Verify large models (50B+) are detected"""
        size, params = detect_model_size("llama-70b-instruct")
        assert size == "large"
        assert params == 70

    def test_medium_model(self):
        """Verify medium models (10B-50B) are detected"""
        size, params = detect_model_size("qwen-32b-chat")
        assert size == "medium"
        assert params == 32

    def test_small_model(self):
        """Verify small models (<10B) are detected"""
        size, params = detect_model_size("phi-3-mini-7b")
        assert size == "small"
        assert params == 7

    def test_million_params(self):
        """Verify million-param models are detected"""
        size, params = detect_model_size("bge-m3:567m-fp16")
        assert size == "small"
        assert params == pytest.approx(0.567, 0.01)

    def test_unknown_size(self):
        """Verify unknown sizes return None"""
        size, params = detect_model_size("mystery-model")
        assert size is None
        assert params is None

    def test_moe_size_detection(self):
        """Verify MoE models with A22B pattern detected correctly"""
        size, params = detect_model_size("qwen3-235b-a22b-8bit")
        assert size == "massive"
        assert params == 235


class TestDetectContextLength:
    """Tests for detect_context_length function"""

    def test_known_model_context(self):
        """Verify known models return correct context"""
        ctx = detect_context_length("longcat-flash-thinking-test")
        assert ctx == 983040

    def test_model_info_priority(self):
        """Verify model_info takes priority"""
        ctx = detect_context_length("some-model", {"context_length": 65536})
        assert ctx == 65536

    def test_pattern_based_context(self):
        """Verify ctx pattern detection"""
        ctx = detect_context_length("model-128k-ctx-version")
        assert ctx == 131072

    def test_unknown_context(self):
        """Verify unknown models return None"""
        ctx = detect_context_length("mystery-model-xyz")
        assert ctx is None


class TestGetModelTags:
    """Tests for get_model_tags function"""

    def test_returns_list(self):
        """Verify function returns a list"""
        tags = get_model_tags("some-model")
        assert isinstance(tags, list)

    def test_override_takes_precedence(self):
        """Verify explicit overrides take precedence over patterns"""
        tags = get_model_tags("claude-opus-4.5")
        assert "quality" in tags
        assert "reasoning" in tags
        assert "coding" in tags
        assert "massive" in tags

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

    def test_quantization_detection_4bit(self):
        """Verify 4-bit quantization detected"""
        tags = get_model_tags("some-model-q4_k_m.gguf")
        assert "4bit" in tags

    def test_quantization_detection_8bit(self):
        """Verify 8-bit quantization detected"""
        tags = get_model_tags("model-8bit-mlx")
        assert "8bit" in tags

    def test_mlx_detection(self):
        """Verify MLX models detected"""
        tags = get_model_tags("mlx-community/some-model")
        assert "mlx" in tags

    def test_gguf_detection(self):
        """Verify GGUF models detected"""
        tags = get_model_tags("model.gguf")
        assert "gguf" in tags

    def test_moe_detection(self):
        """Verify MoE models detected"""
        tags = get_model_tags("qwen3-235b-a22b-8bit")
        assert "moe" in tags

    def test_thinking_detection(self):
        """Verify thinking models detected"""
        tags = get_model_tags("longcat-flash-thinking-model")
        assert "thinking" in tags


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


class TestGetAllTags:
    """Tests for get_all_tags function"""

    def test_returns_dict(self):
        """Verify function returns a dictionary"""
        tags = get_all_tags()
        assert isinstance(tags, dict)

    def test_returns_copy(self):
        """Verify function returns a copy, not the original"""
        tags1 = get_all_tags()
        tags2 = get_all_tags()
        tags1["test"] = "value"
        assert "test" not in tags2


class TestSpecificModelOverrides:
    """Tests for specific model overrides"""

    def test_claude_opus_tags(self):
        """Verify Claude Opus has correct tags"""
        tags = get_model_tags("claude-opus-4-5-20251101")
        expected = ["quality", "reasoning", "coding", "creative", "massive", "long_context"]
        for tag in expected:
            assert tag in tags, f"Claude Opus missing {tag}"

    def test_claude_sonnet_tags(self):
        """Verify Claude Sonnet has correct tags"""
        tags = get_model_tags("claude-sonnet-4-5-20250929")
        assert "quality" in tags
        assert "reasoning" in tags
        assert "coding" in tags
        assert "long_context" in tags

    def test_claude_haiku_tags(self):
        """Verify Claude Haiku has correct tags"""
        tags = get_model_tags("claude-haiku-4-5-20251001")
        assert "fast" in tags
        assert "coding" in tags
        assert "long_context" in tags

    def test_gpt5_tags(self):
        """Verify GPT-5 has correct tags"""
        tags = get_model_tags("gpt-5")
        assert "quality" in tags
        assert "reasoning" in tags
        assert "massive" in tags
        assert "long_context" in tags

    def test_gemini_pro_tags(self):
        """Verify Gemini Pro has correct tags"""
        tags = get_model_tags("gemini-2.5-pro")
        assert "quality" in tags
        assert "reasoning" in tags
        assert "massive_context" in tags

    def test_gemini_flash_tags(self):
        """Verify Gemini Flash has correct tags"""
        tags = get_model_tags("gemini-2.5-flash")
        assert "fast" in tags
        assert "very_long_context" in tags

    def test_longcat_massive_context(self):
        """Verify LongCat has massive_context tag"""
        tags = get_model_tags("inferencerlabs/LongCat-Flash-Thinking-2601-MLX-5.5bit")
        assert "massive_context" in tags
        assert "thinking" in tags

    def test_qwen_moe_tags(self):
        """Verify Qwen MoE model has moe tag"""
        tags = get_model_tags("mlx-community/Qwen3-235B-A22B-8bit")
        assert "moe" in tags
        assert "massive" in tags

    def test_deepseek_coder_long_context(self):
        """Verify DeepSeek Coder has long_context tag"""
        tags = get_model_tags("GGorman/DeepSeek-Coder-V2-Instruct-Q4-mlx")
        assert "coding" in tags
        assert "long_context" in tags


class TestSizeDetectionOrder:
    """Tests for size detection order"""

    def test_253b_not_small(self):
        """Verify 253B models are not tagged as small"""
        tags = get_model_tags("nemotron-ultra-253b")
        assert "small" not in tags
        assert "massive" in tags

    def test_235b_not_small(self):
        """Verify 235B models are not tagged as small"""
        tags = get_model_tags("qwen3-235b-a22b")
        assert "small" not in tags
        assert "massive" in tags

    def test_70b_is_large(self):
        """Verify 70B models are tagged as large"""
        tags = get_model_tags("llama-70b")
        assert "large" in tags
        assert "small" not in tags

    def test_405b_is_massive(self):
        """Verify 405B models are tagged as massive"""
        tags = get_model_tags("hermes-405b-model")
        assert "massive" in tags
        assert "large" not in tags


class TestContextLengthTags:
    """Tests for context length tagging"""

    def test_massive_context_threshold(self):
        """Verify 500K+ gets massive_context"""
        # LongCat has 983K context
        tags = get_model_tags("inferencerlabs/LongCat-Flash-Thinking-2601-MLX-5.5bit")
        assert "massive_context" in tags
        assert "very_long_context" in tags
        assert "long_context" in tags

    def test_very_long_context_threshold(self):
        """Verify 250K+ gets very_long_context"""
        tags = get_model_tags("mlx-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit")
        assert "very_long_context" in tags
        assert "long_context" in tags

    def test_long_context_threshold(self):
        """Verify 100K+ gets long_context"""
        tags = get_model_tags("mlx-community/Hermes-4-405B-MLX-6bit")
        assert "long_context" in tags


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
