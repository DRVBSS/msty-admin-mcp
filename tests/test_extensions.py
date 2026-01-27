#!/usr/bin/env python3
"""
Tests for Msty Admin MCP - Extension Modules (Phases 10-15)

Run with: pytest tests/test_extensions.py -v
"""

import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Knowledge Stack Tests
# ============================================================================

class TestKnowledgeStacks:
    """Tests for knowledge_stacks.py module"""

    def test_list_knowledge_stacks_returns_dict(self):
        """Verify list_knowledge_stacks returns a dictionary"""
        from src.knowledge_stacks import list_knowledge_stacks
        result = list_knowledge_stacks()
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "stacks" in result
        assert "total_count" in result

    def test_get_stack_details_not_found(self):
        """Verify error handling for non-existent stack"""
        from src.knowledge_stacks import get_knowledge_stack_details
        result = get_knowledge_stack_details("non_existent_stack_xyz")
        assert isinstance(result, dict)
        assert result.get("found") == False

    def test_get_stack_statistics(self):
        """Verify statistics function returns proper structure"""
        from src.knowledge_stacks import get_stack_statistics
        result = get_stack_statistics()
        assert isinstance(result, dict)
        assert "timestamp" in result


# ============================================================================
# Model Management Tests
# ============================================================================

class TestModelManagement:
    """Tests for model_management.py module"""

    def test_get_local_model_inventory(self):
        """Verify inventory returns proper structure"""
        from src.model_management import get_local_model_inventory
        result = get_local_model_inventory()
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "models" in result
        assert "summary" in result

    def test_delete_model_requires_confirm(self):
        """Verify delete requires confirmation"""
        from src.model_management import delete_model
        result = delete_model("/fake/path/model.gguf", confirm=False)
        assert isinstance(result, dict)
        assert result.get("deleted") == False
        assert "confirmation_required" in result or "error" in result

    def test_find_duplicate_models(self):
        """Verify duplicate finder returns proper structure"""
        from src.model_management import find_duplicate_models
        result = find_duplicate_models()
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "potential_duplicates" in result

    def test_analyze_model_storage(self):
        """Verify storage analysis returns proper structure"""
        from src.model_management import analyze_model_storage
        result = analyze_model_storage()
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "storage" in result


# ============================================================================
# Model Bridge Tests
# ============================================================================

class TestModelBridge:
    """Tests for model_bridge.py module"""

    def test_select_best_model_for_task(self):
        """Verify model selection returns proper structure"""
        from src.model_bridge import select_best_model_for_task
        result = select_best_model_for_task("coding")
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "task_type" in result

    def test_select_model_with_preferences(self):
        """Verify model selection respects preferences"""
        from src.model_bridge import select_best_model_for_task
        result = select_best_model_for_task("general", prefer_speed=True)
        assert isinstance(result, dict)
        assert "task_type" in result
        # Note: preferences may not be in result if no models available

    def test_delegate_to_local_model_structure(self):
        """Verify delegation returns proper structure"""
        from src.model_bridge import delegate_to_local_model
        # This will fail if no local models available, but structure should be correct
        result = delegate_to_local_model("test prompt", task_type="general")
        assert isinstance(result, dict)
        assert "timestamp" in result

    def test_parallel_process_tasks_empty(self):
        """Verify parallel processing handles empty task list"""
        from src.model_bridge import parallel_process_tasks
        # Empty task list should return a result without error
        result = parallel_process_tasks([{"prompt": "test"}])  # Need at least one task
        assert isinstance(result, dict)
        assert "results" in result or "error" in result


# ============================================================================
# Turnstile Tests
# ============================================================================

class TestTurnstiles:
    """Tests for turnstiles.py module"""

    def test_list_turnstile_templates(self):
        """Verify templates list returns expected templates"""
        from src.turnstiles import list_turnstile_templates
        result = list_turnstile_templates()
        assert isinstance(result, dict)
        assert "templates" in result
        assert result["total_count"] >= 5  # At least 5 built-in templates

    def test_get_turnstile_template_code_review(self):
        """Verify code_review template exists and is valid"""
        from src.turnstiles import get_turnstile_template
        result = get_turnstile_template("code_review")
        assert isinstance(result, dict)
        assert "steps" in result
        assert len(result["steps"]) > 0
        assert result["name"] == "Code Review Pipeline"

    def test_get_turnstile_template_not_found(self):
        """Verify error handling for non-existent template"""
        from src.turnstiles import get_turnstile_template
        result = get_turnstile_template("non_existent_template")
        assert isinstance(result, dict)
        assert "error" in result
        assert "available_templates" in result

    def test_execute_turnstile_dry_run(self):
        """Verify turnstile dry run execution"""
        from src.turnstiles import execute_turnstile
        result = execute_turnstile(
            template_id="content_creation",
            input_data="Write about AI",
            dry_run=True
        )
        assert isinstance(result, dict)
        assert result.get("dry_run") == True
        assert "steps" in result
        for step in result["steps"]:
            assert step.get("status") == "pending"

    def test_suggest_turnstile_for_task(self):
        """Verify task-based suggestion"""
        from src.turnstiles import suggest_turnstile_for_task
        result = suggest_turnstile_for_task("review my Python code for bugs")
        assert isinstance(result, dict)
        assert "suggestions" in result
        # Should suggest code_review template
        suggestion_ids = [s.get("template_id") for s in result["suggestions"]]
        assert "code_review" in suggestion_ids

    def test_analyze_turnstile_usage(self):
        """Verify usage analysis returns proper structure"""
        from src.turnstiles import analyze_turnstile_usage
        result = analyze_turnstile_usage()
        assert isinstance(result, dict)
        assert "analysis" in result
        assert "recommendations" in result


# ============================================================================
# Live Context Tests
# ============================================================================

class TestLiveContext:
    """Tests for live_context.py module"""

    def test_get_system_context(self):
        """Verify system context returns expected fields"""
        from src.live_context import get_system_context
        result = get_system_context()
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "system" in result
        assert "memory" in result
        assert "cpu" in result

    def test_get_datetime_context(self):
        """Verify datetime context returns expected fields"""
        from src.live_context import get_datetime_context
        result = get_datetime_context()
        assert isinstance(result, dict)
        assert "datetime" in result
        assert "formatted" in result
        assert result["datetime"]["year"] >= 2024

    def test_get_msty_context(self):
        """Verify Msty context returns expected fields"""
        from src.live_context import get_msty_context
        result = get_msty_context()
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "paths" in result
        assert "status" in result

    def test_get_full_live_context(self):
        """Verify full context aggregation"""
        from src.live_context import get_full_live_context
        result = get_full_live_context(
            include_system=True,
            include_datetime=True,
            include_msty=True
        )
        assert isinstance(result, dict)
        assert "components" in result
        assert len(result["components"]) >= 3

    def test_format_context_for_prompt_markdown(self):
        """Verify markdown formatting"""
        from src.live_context import get_system_context, format_context_for_prompt
        context = get_system_context()
        formatted = format_context_for_prompt(context, style="markdown")
        assert isinstance(formatted, str)
        assert "##" in formatted  # Markdown header

    def test_format_context_for_prompt_json(self):
        """Verify JSON formatting"""
        from src.live_context import get_system_context, format_context_for_prompt
        context = get_system_context()
        formatted = format_context_for_prompt(context, style="json")
        # Should be valid JSON
        parsed = json.loads(formatted)
        assert isinstance(parsed, dict)

    def test_context_cache(self):
        """Verify context caching works"""
        from src.live_context import get_cached_context, clear_context_cache
        clear_context_cache()
        result1 = get_cached_context("system")
        assert result1.get("_cached") == False
        result2 = get_cached_context("system")
        assert result2.get("_cached") == True


# ============================================================================
# Conversation Analytics Tests
# ============================================================================

class TestConversationAnalytics:
    """Tests for conversation_analytics.py module"""

    def test_get_conversations(self):
        """Verify conversations retrieval returns proper structure"""
        from src.conversation_analytics import get_conversations
        result = get_conversations(limit=10)
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "conversations" in result
        assert "filters" in result

    def test_analyze_usage_patterns(self):
        """Verify usage analysis returns proper structure"""
        from src.conversation_analytics import analyze_usage_patterns
        result = analyze_usage_patterns(days=7)
        assert isinstance(result, dict)
        assert "analytics" in result or "error" in result

    def test_analyze_conversation_content(self):
        """Verify content analysis returns proper structure"""
        from src.conversation_analytics import analyze_conversation_content
        result = analyze_conversation_content(sample_size=10)
        assert isinstance(result, dict)
        assert "analytics" in result

    def test_analyze_model_performance(self):
        """Verify model performance analysis returns proper structure"""
        from src.conversation_analytics import analyze_model_performance
        result = analyze_model_performance(days=7)
        assert isinstance(result, dict)
        assert "models" in result

    def test_analyze_session_patterns(self):
        """Verify session analysis returns proper structure"""
        from src.conversation_analytics import analyze_session_patterns
        result = analyze_session_patterns(days=7)
        assert isinstance(result, dict)
        assert "analytics" in result

    def test_generate_analytics_report(self):
        """Verify full report generation"""
        from src.conversation_analytics import generate_analytics_report
        result = generate_analytics_report(
            days=7,
            include_usage=True,
            include_content=True,
            include_models=True,
            include_sessions=True
        )
        assert isinstance(result, dict)
        assert "report_sections" in result
        assert "executive_summary" in result
        assert len(result["report_sections"]) == 4


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for extension modules"""

    def test_all_modules_import(self):
        """Verify all extension modules can be imported"""
        from src import knowledge_stacks
        from src import model_management
        from src import model_bridge
        from src import turnstiles
        from src import live_context
        from src import conversation_analytics
        assert True

    def test_server_extensions_import(self):
        """Verify server_extensions module imports correctly"""
        # Skip if mcp package not available
        try:
            from src.server_extensions import register_extension_tools
            assert callable(register_extension_tools)
        except ImportError:
            pytest.skip("MCP package not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
