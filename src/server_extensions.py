"""
Msty Admin MCP Server Extensions v7.0.0

New tools for Phase 10+:
- Knowledge Stack Management
- Model Management (Download/Delete)
- Claude ↔ Local Model Bridge
- Turnstile Workflows
- Live Context
- Conversation Analytics

These extensions add 25+ new tools to the base server.
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP

# Import new modules
from .knowledge_stacks import (
    list_knowledge_stacks,
    get_knowledge_stack_details,
    search_knowledge_stack,
    analyze_knowledge_stack,
    get_stack_statistics,
)
from .model_management import (
    get_local_model_inventory,
    delete_model,
    find_duplicate_models,
    check_huggingface_model,
    get_download_instructions,
    analyze_model_storage,
)
from .model_bridge import (
    select_best_model_for_task,
    delegate_to_local_model,
    multi_model_consensus,
    draft_and_refine,
    parallel_process_tasks,
)
from .turnstiles import (
    list_turnstiles,
    get_turnstile_details,
    list_turnstile_templates,
    get_turnstile_template,
    execute_turnstile,
    analyze_turnstile_usage,
    suggest_turnstile_for_task,
)
from .live_context import (
    get_system_context,
    get_datetime_context,
    get_environment_context,
    get_process_context,
    get_msty_context,
    get_full_live_context,
    format_context_for_prompt,
    get_cached_context,
    clear_context_cache,
    get_context_cache_stats,
)
from .conversation_analytics import (
    get_conversations,
    get_messages,
    analyze_usage_patterns,
    analyze_conversation_content,
    analyze_model_performance,
    analyze_session_patterns,
    generate_analytics_report,
)

logger = logging.getLogger("msty-admin-mcp")


def register_extension_tools(mcp: FastMCP):
    """Register all extension tools with the MCP server."""

    # =========================================================================
    # Phase 10: Knowledge Stack Management
    # =========================================================================

    @mcp.tool()
    def ks_list_stacks() -> str:
        """
        List all Knowledge Stacks with metadata.

        Returns:
            JSON with stacks list, document counts, and size information
        """
        result = list_knowledge_stacks()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def ks_get_details(stack_id: str) -> str:
        """
        Get detailed information about a specific Knowledge Stack.

        Args:
            stack_id: The stack identifier

        Returns:
            JSON with stack details including documents list
        """
        result = get_knowledge_stack_details(stack_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def ks_search(stack_id: str, query: str, limit: int = 10) -> str:
        """
        Search within a Knowledge Stack.

        Args:
            stack_id: The stack to search
            query: Search query
            limit: Maximum results (default: 10)

        Returns:
            JSON with search results and snippets
        """
        result = search_knowledge_stack(stack_id, query, limit)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def ks_analyze(stack_id: str) -> str:
        """
        Analyze a Knowledge Stack for quality and optimization.

        Args:
            stack_id: The stack to analyze

        Returns:
            JSON with analysis results and recommendations
        """
        result = analyze_knowledge_stack(stack_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def ks_statistics() -> str:
        """
        Get aggregate statistics across all Knowledge Stacks.

        Returns:
            JSON with overall statistics and comparisons
        """
        result = get_stack_statistics()
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 11: Model Management
    # =========================================================================

    @mcp.tool()
    def model_inventory() -> str:
        """
        Get complete inventory of local AI models.

        Returns:
            JSON with all models, sizes, and locations
        """
        result = get_local_model_inventory()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def model_delete(model_path: str, confirm: bool = False) -> str:
        """
        Delete a local model file.

        Args:
            model_path: Full path to the model file
            confirm: Must be True to actually delete

        Returns:
            JSON with deletion result or confirmation request
        """
        result = delete_model(model_path, confirm)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def model_find_duplicates() -> str:
        """
        Find duplicate model files.

        Returns:
            JSON with duplicate groups and space savings
        """
        result = find_duplicate_models()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def model_check_hf(repo_id: str) -> str:
        """
        Check a HuggingFace model repository.

        Args:
            repo_id: HuggingFace repo ID (e.g., "meta-llama/Llama-2-7b")

        Returns:
            JSON with model info and availability
        """
        result = check_huggingface_model(repo_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def model_download_guide(repo_id: str, model_type: str = "auto") -> str:
        """
        Get download instructions for a model.

        Args:
            repo_id: HuggingFace repo ID
            model_type: Model type (auto, mlx, gguf)

        Returns:
            JSON with step-by-step download instructions
        """
        result = get_download_instructions(repo_id, model_type)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def model_storage_analysis() -> str:
        """
        Analyze model storage usage and recommend cleanup.

        Returns:
            JSON with storage analysis and recommendations
        """
        result = analyze_model_storage()
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 12: Claude ↔ Local Model Bridge
    # =========================================================================

    @mcp.tool()
    def bridge_select_model(
        task_type: str,
        prefer_speed: bool = False,
        prefer_quality: bool = False,
        max_size: Optional[str] = None
    ) -> str:
        """
        Select the best local model for a task.

        Args:
            task_type: Type of task (coding, reasoning, creative, general)
            prefer_speed: Prioritize faster models
            prefer_quality: Prioritize higher quality
            max_size: Maximum model size (small, medium, large)

        Returns:
            JSON with recommended model and reasoning
        """
        result = select_best_model_for_task(task_type, prefer_speed, prefer_quality, max_size)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def bridge_delegate(
        prompt: str,
        task_type: str = "general",
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Delegate a task to a local model.

        Args:
            prompt: The prompt to send
            task_type: Task type for model selection
            model_id: Specific model (optional)
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            JSON with model response and metadata
        """
        result = delegate_to_local_model(
            prompt, task_type, model_id, system_prompt, temperature, max_tokens
        )
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def bridge_consensus(
        prompt: str,
        models: Optional[List[str]] = None,
        num_models: int = 3,
        voting_method: str = "similarity"
    ) -> str:
        """
        Get consensus from multiple local models.

        Args:
            prompt: The prompt to send to all models
            models: Specific models to use (optional)
            num_models: Number of models if not specified
            voting_method: How to determine consensus

        Returns:
            JSON with all responses and consensus analysis
        """
        result = multi_model_consensus(prompt, models, num_models, voting_method)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def bridge_draft_refine(
        prompt: str,
        draft_model: Optional[str] = None,
        refine_instructions: Optional[str] = None,
        task_type: str = "writing"
    ) -> str:
        """
        Draft with local model, ready for Claude refinement.

        Args:
            prompt: The content prompt
            draft_model: Model for drafting (optional)
            refine_instructions: Instructions for refinement
            task_type: Type of content

        Returns:
            JSON with draft and refinement instructions
        """
        result = draft_and_refine(prompt, draft_model, refine_instructions, task_type)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def bridge_parallel_tasks(
        tasks: List[Dict[str, Any]],
        max_concurrent: int = 3
    ) -> str:
        """
        Process multiple tasks in parallel across models.

        Args:
            tasks: List of task dicts with 'prompt' and optional 'task_type', 'model'
            max_concurrent: Maximum concurrent tasks

        Returns:
            JSON with all results
        """
        result = parallel_process_tasks(tasks, max_concurrent)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 13: Turnstile Workflows
    # =========================================================================

    @mcp.tool()
    def turnstile_list() -> str:
        """
        List all Turnstile workflows.

        Returns:
            JSON with turnstiles and metadata
        """
        result = list_turnstiles()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def turnstile_details(turnstile_id: str) -> str:
        """
        Get details of a specific turnstile.

        Args:
            turnstile_id: The turnstile identifier

        Returns:
            JSON with turnstile configuration and steps
        """
        result = get_turnstile_details(turnstile_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def turnstile_templates() -> str:
        """
        List available turnstile templates.

        Returns:
            JSON with template names and descriptions
        """
        result = list_turnstile_templates()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def turnstile_get_template(template_id: str) -> str:
        """
        Get a specific turnstile template.

        Args:
            template_id: Template identifier

        Returns:
            JSON with full template configuration
        """
        result = get_turnstile_template(template_id)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def turnstile_execute(
        template_id: str,
        input_data: str,
        variables: Optional[Dict[str, str]] = None,
        dry_run: bool = True
    ) -> str:
        """
        Execute or preview a turnstile workflow.

        Args:
            template_id: Template to execute
            input_data: Input for the workflow
            variables: Additional variables
            dry_run: Preview only (default: True)

        Returns:
            JSON with execution plan or results
        """
        result = execute_turnstile(template_id, input_data, variables, dry_run)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def turnstile_analyze() -> str:
        """
        Analyze turnstile usage patterns.

        Returns:
            JSON with usage statistics and recommendations
        """
        result = analyze_turnstile_usage()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def turnstile_suggest(task_description: str) -> str:
        """
        Suggest turnstile templates for a task.

        Args:
            task_description: Description of the task

        Returns:
            JSON with suggested templates and reasoning
        """
        result = suggest_turnstile_for_task(task_description)
        return json.dumps(result, indent=2, default=str)

    # =========================================================================
    # Phase 14: Live Context
    # =========================================================================

    @mcp.tool()
    def context_system() -> str:
        """
        Get system context (CPU, memory, disk).

        Returns:
            JSON with system information
        """
        result = get_system_context()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def context_datetime() -> str:
        """
        Get current datetime context.

        Returns:
            JSON with detailed datetime information
        """
        result = get_datetime_context()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def context_msty() -> str:
        """
        Get Msty-specific context.

        Returns:
            JSON with Msty paths and status
        """
        result = get_msty_context()
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def context_full(
        include_system: bool = True,
        include_datetime: bool = True,
        include_environment: bool = False,
        include_processes: bool = False,
        include_msty: bool = True
    ) -> str:
        """
        Get full aggregated live context.

        Args:
            include_system: Include system info
            include_datetime: Include datetime
            include_environment: Include environment
            include_processes: Include processes
            include_msty: Include Msty info

        Returns:
            JSON with all requested context
        """
        result = get_full_live_context(
            include_system, include_datetime,
            include_environment, include_processes, include_msty
        )
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def context_for_prompt(context_type: str = "full", style: str = "markdown") -> str:
        """
        Get context formatted for AI prompts.

        Args:
            context_type: Type of context (full, system, datetime, msty)
            style: Output style (markdown, text, json)

        Returns:
            Formatted context string
        """
        context = get_cached_context(context_type)
        formatted = format_context_for_prompt(context, style)
        return formatted

    # =========================================================================
    # Phase 15: Conversation Analytics
    # =========================================================================

    @mcp.tool()
    def analytics_usage(days: int = 30) -> str:
        """
        Analyze conversation usage patterns.

        Args:
            days: Number of days to analyze

        Returns:
            JSON with usage analytics and insights
        """
        result = analyze_usage_patterns(days)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def analytics_content(
        conversation_id: Optional[str] = None,
        sample_size: int = 100
    ) -> str:
        """
        Analyze conversation content patterns (privacy-preserving).

        Args:
            conversation_id: Specific conversation (optional)
            sample_size: Number of messages to sample

        Returns:
            JSON with content analytics
        """
        result = analyze_conversation_content(conversation_id, sample_size)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def analytics_models(days: int = 30) -> str:
        """
        Analyze model performance across conversations.

        Args:
            days: Number of days to analyze

        Returns:
            JSON with model performance metrics
        """
        result = analyze_model_performance(days)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def analytics_sessions(days: int = 30) -> str:
        """
        Analyze session patterns and behaviors.

        Args:
            days: Number of days to analyze

        Returns:
            JSON with session analytics
        """
        result = analyze_session_patterns(days)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def analytics_report(
        days: int = 30,
        include_usage: bool = True,
        include_content: bool = True,
        include_models: bool = True,
        include_sessions: bool = True
    ) -> str:
        """
        Generate comprehensive analytics report.

        Args:
            days: Number of days to analyze
            include_usage: Include usage analytics
            include_content: Include content analytics
            include_models: Include model analytics
            include_sessions: Include session analytics

        Returns:
            JSON with full analytics report
        """
        result = generate_analytics_report(
            days, include_usage, include_content,
            include_models, include_sessions
        )
        return json.dumps(result, indent=2, default=str)

    logger.info("Registered 35 extension tools (Phases 10-15)")


__all__ = ["register_extension_tools"]
