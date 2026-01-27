#!/usr/bin/env python3
"""
Msty Admin MCP Server v7.0.0

AI-administered Msty Studio Desktop management system with database insights,
configuration management, hardware optimization, and Claude Desktop sync.

Phase 1: Foundational Tools (Read-Only)
- detect_msty_installation, read_msty_database, list_configured_tools
- get_model_providers, analyse_msty_health, get_server_status
- scan_database_locations

Phase 2: Configuration Management
- export_tool_config, sync_claude_preferences
- generate_persona, import_tool_config

Phase 3: Automation Bridge
- query_local_ai_service, list_available_models, get_sidecar_status
- chat_with_local_model, recommend_model
- list_model_tags, find_model_by_tag
- get_cache_stats, clear_cache

Phase 4: Intelligence Layer
- get_model_performance_metrics, analyse_conversation_patterns
- compare_model_responses, optimise_knowledge_stacks
- suggest_persona_improvements

Phase 5: Tiered AI Workflow
- run_calibration_test, evaluate_response_quality
- identify_handoff_triggers, get_calibration_history

Phase 6: Advanced Model Management
- get_model_details, benchmark_model
- list_local_model_files, estimate_model_requirements

Phase 7: Conversation Management
- export_conversations, search_conversations
- get_conversation_stats

Phase 8: Prompt Templates & Automation
- create_prompt_template, list_prompt_templates
- run_prompt_template, smart_model_router

Phase 9: Backup & System Management
- backup_configuration, restore_configuration
- get_system_resources

Phase 10: Knowledge Stack Management
- ks_list_stacks, ks_get_details, ks_search
- ks_analyze, ks_statistics

Phase 11: Model Download/Delete
- model_inventory, model_delete, model_find_duplicates
- model_check_hf, model_download_guide, model_storage_analysis

Phase 12: Claude ↔ Local Model Bridge
- bridge_select_model, bridge_delegate, bridge_consensus
- bridge_draft_refine, bridge_parallel_tasks

Phase 13: Turnstile Workflows
- turnstile_list, turnstile_details, turnstile_templates
- turnstile_get_template, turnstile_execute
- turnstile_analyze, turnstile_suggest

Phase 14: Live Context
- context_system, context_datetime, context_msty
- context_full, context_for_prompt

Phase 15: Conversation Analytics
- analytics_usage, analytics_content, analytics_models
- analytics_sessions, analytics_report

Total: 77 tools

Created by Pineapple 🍍 AI Administration System
"""

import json
import logging
import os
import platform
import plistlib
import sqlite3
import urllib.request
import urllib.error
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Dict, List

import psutil
from mcp.server.fastmcp import FastMCP

# Import from local modules
from .constants import (
    SERVER_VERSION,
    SIDECAR_HOST,
    SIDECAR_PROXY_PORT,
    LOCAL_AI_SERVICE_PORT,
    SIDECAR_TIMEOUT,
    MLX_SERVICE_PORT,
    LLAMACPP_SERVICE_PORT,
    VIBE_PROXY_PORT,
    ALLOWED_TABLE_NAMES
)
from .models import (
    MstyInstallation,
    MstyHealthReport,
    DatabaseStats,
    PersonaConfig
)
from .errors import (
    ErrorCode,
    error_response,
    success_response,
    make_error_response,
    make_success_response
)
from .paths import (
    get_msty_paths,
    sanitize_path,
    expand_path,
    read_claude_desktop_config
)
from .database import (
    get_database_connection,
    query_database,
    get_table_names,
    is_safe_table_name,
    validate_table_exists,
    safe_query_table,
    safe_count_table,
    get_table_row_count
)
from .network import (
    make_api_request,
    is_process_running,
    is_local_ai_available,
    get_available_service_ports
)
from .cache import (
    ResponseCache,
    get_cached_models,
    cache_models,
    get_cache
)
from .tagging import (
    MODEL_TAGS,
    get_model_tags,
    find_models_by_tag
)

# Import Phase 4 & 5 utilities
from .phase4_5_tools import (
    init_metrics_db,
    get_metrics_db_path,
    record_model_metric,
    get_model_metrics_summary,
    save_calibration_result,
    get_calibration_results,
    record_handoff_trigger,
    get_handoff_triggers,
    evaluate_response_heuristic,
    CALIBRATION_PROMPTS,
    QUALITY_RUBRIC
)

# Import extension tools (Phases 10-15)
from .server_extensions import register_extension_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("msty-admin-mcp")

# Initialize FastMCP server
mcp = FastMCP("msty-admin")

# Get global cache instance for tools that need direct access
_response_cache = get_cache()


# =============================================================================
# Phase 1: Foundational Tools (Read-Only)
# =============================================================================

@mcp.tool()
def detect_msty_installation() -> str:
    """
    Detect and analyse Msty Studio Desktop installation.
    
    Returns comprehensive information about:
    - Installation status and paths
    - Application version (if detectable)
    - Running status of Msty Studio and Sidecar
    - Platform information
    - Data directory locations
    
    This is the first tool to run when working with Msty Admin MCP.
    """
    paths = get_msty_paths()
    app_path = paths.get("app") or paths.get("app_alt")
    installed = app_path is not None
    
    version = None
    if app_path:
        plist_path = Path(app_path) / "Contents/Info.plist"
        if plist_path.exists():
            try:
                import plistlib
                with open(plist_path, 'rb') as f:
                    plist = plistlib.load(f)
                    version = plist.get('CFBundleShortVersionString', plist.get('CFBundleVersion'))
            except (OSError, plistlib.InvalidFileException, KeyError):
                pass
    
    platform_info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "is_apple_silicon": platform.machine() in ["arm64", "aarch64"],
    }
    
    installation = MstyInstallation(
        installed=installed,
        version=version,
        app_path=app_path,
        data_path=paths.get("data"),
        sidecar_path=paths.get("sidecar"),
        database_path=paths.get("database"),
        mlx_models_path=paths.get("mlx_models"),
        is_running=is_process_running("MstyStudio"),
        sidecar_running=is_local_ai_available(),  # Msty 2.4.0+ has services built-in
        platform_info=platform_info
    )
    
    return json.dumps(asdict(installation), indent=2)


@mcp.tool()
def read_msty_database(
    query_type: str = "stats",
    table_name: Optional[str] = None,
    limit: int = 50
) -> str:
    """
    Query the Msty Studio database for insights.
    
    Args:
        query_type: Type of query to run:
            - "stats": Get overall database statistics
            - "tables": List all tables in the database
            - "conversations": Get recent conversations
            - "personas": List configured personas
            - "prompts": List saved prompts
            - "tools": List configured MCP tools
            - "custom": Query a specific table (requires table_name)
        table_name: Table name for custom queries (only used with query_type="custom")
        limit: Maximum number of results to return (default: 50)
    
    Returns:
        JSON string with query results
    """
    paths = get_msty_paths()
    db_path = paths.get("database")
    
    if not db_path:
        return error_response(
            ErrorCode.DATABASE_NOT_FOUND,
            "Msty database not found",
            suggestion="Run scan_database_locations to find your database, or set MSTY_DATABASE_PATH environment variable"
        )
    
    result = {"query_type": query_type, "database_path": db_path}
    
    try:
        if query_type == "tables":
            tables = get_table_names(db_path)
            result["tables"] = [{"name": t, "row_count": get_table_row_count(db_path, t)} for t in tables]
            
        elif query_type == "stats":
            tables = get_table_names(db_path)
            stats = DatabaseStats()
            table_mapping = {
                "chat_sessions": "total_conversations",
                "conversations": "total_conversations",
                "messages": "total_messages",
                "chat_messages": "total_messages",
                "personas": "total_personas",
                "prompts": "total_prompts",
                "knowledge_stacks": "total_knowledge_stacks",
                "tools": "total_tools",
            }
            for table in tables:
                for pattern, attr in table_mapping.items():
                    if pattern in table.lower():
                        count = get_table_row_count(db_path, table)
                        setattr(stats, attr, getattr(stats, attr, 0) + count)
                        break
            db_file = Path(db_path)
            stats.database_size_mb = round(db_file.stat().st_size / (1024 * 1024), 2)
            result["stats"] = asdict(stats)
            result["available_tables"] = tables
            
        elif query_type == "custom" and table_name:
            data = safe_query_table(db_path, table_name, limit=limit)
            if data:
                result["data"] = data
            else:
                result["error"] = f"Table '{table_name}' not found or empty"
        else:
            table_map = {
                "conversations": ["chat_sessions", "conversations", "chats"],
                "personas": ["personas"],
                "prompts": ["prompts", "prompt_library"],
                "tools": ["tools", "mcp_tools"],
            }
            if query_type in table_map:
                for t in table_map[query_type]:
                    data = safe_query_table(db_path, t, limit=limit)
                    if data:
                        result[query_type] = data
                        break
    except Exception as e:
        result["error"] = str(e)
    
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def list_configured_tools() -> str:
    """
    List all MCP tools configured in Msty Studio's Toolbox.
    
    Returns detailed information about each tool including:
    - Tool ID and name
    - Configuration (command, args, env vars)
    - Status and notes
    """
    paths = get_msty_paths()
    db_path = paths.get("database")
    
    if not db_path:
        return error_response(
            ErrorCode.DATABASE_NOT_FOUND,
            "Msty database not found",
            suggestion="Run scan_database_locations to find your database"
        )
    
    result = {"database_path": db_path, "tools": [], "tool_count": 0}
    tables = get_table_names(db_path)
    tool_tables = [t for t in tables if "tool" in t.lower() or "mcp" in t.lower()]
    
    for table in tool_tables:
        tools = safe_query_table(db_path, table, limit=1000)
        if tools:
            result["tools"].extend(tools)
            break
    
    result["tool_count"] = len(result["tools"])
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_model_providers() -> str:
    """
    Get configured AI model providers in Msty Studio.
    
    Returns information about:
    - Local models (MLX, Ollama connections)
    - Remote providers (OpenAI, Anthropic, etc.)
    - Model configurations and parameters
    
    Note: API keys are NOT returned for security reasons.
    """
    paths = get_msty_paths()
    mlx_path = paths.get("mlx_models")
    
    result = {
        "local_models": {"mlx_available": mlx_path is not None, "mlx_models": []},
        "remote_providers": []
    }
    
    if mlx_path and Path(mlx_path).exists():
        for model_dir in Path(mlx_path).iterdir():
            if model_dir.is_dir():
                result["local_models"]["mlx_models"].append({
                    "name": model_dir.name,
                    "size_mb": sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024 * 1024)
                })
    
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def analyse_msty_health() -> str:
    """
    Perform comprehensive health analysis of Msty Studio installation.
    
    Checks:
    - Database integrity and size
    - Storage usage and available space
    - Model cache status
    - Application and Sidecar status
    - Configuration completeness
    
    Returns a health report with status and recommendations.
    """
    paths = get_msty_paths()
    health = MstyHealthReport(overall_status="unknown", timestamp=datetime.now().isoformat())
    issues, warnings = [], []
    
    if not paths.get("app") and not paths.get("app_alt"):
        health.overall_status = "critical"
        health.recommendations.append("Install Msty Studio Desktop from https://msty.ai")
        return json.dumps(asdict(health), indent=2)
    
    db_path = paths.get("database")
    if db_path and Path(db_path).exists():
        db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)
        health.database_status = {"exists": True, "size_mb": round(db_size_mb, 2)}
        if db_size_mb > 500:
            warnings.append(f"Database is large ({db_size_mb:.0f}MB)")
    
    # Check all Msty 2.4.0+ services
    services = get_available_service_ports()
    health.recommendations.append(f"Msty Studio: {'Running ✅' if is_process_running('MstyStudio') or is_process_running('Msty') else 'Not running'}")
    health.recommendations.append(f"Local AI Service (port {LOCAL_AI_SERVICE_PORT}): {'Running ✅' if services['local_ai']['available'] else 'Not running'}")
    health.recommendations.append(f"MLX Service (port {MLX_SERVICE_PORT}): {'Running ✅' if services['mlx']['available'] else 'Not running'}")
    health.recommendations.append(f"LLaMA.cpp Service (port {LLAMACPP_SERVICE_PORT}): {'Running ✅' if services['llamacpp']['available'] else 'Not running'}")
    
    health.overall_status = "critical" if issues else ("warning" if warnings else "healthy")
    return json.dumps(asdict(health), indent=2)


@mcp.tool()
def get_server_status() -> str:
    """
    Get the current status of the Msty Admin MCP server.
    
    Returns server information including:
    - Server name and version
    - Available tools
    - Msty installation status summary
    - Current capabilities
    """
    paths = get_msty_paths()
    
    return json.dumps({
        "server": {
            "name": "msty-admin-mcp",
            "version": SERVER_VERSION,
            "phase": "Phase 5+ - Enhanced",
            "author": "Pineapple 🍍 + Claude"
        },
        "available_tools": {
            "phase_1_foundational": ["detect_msty_installation", "read_msty_database", "list_configured_tools", "get_model_providers", "analyse_msty_health", "get_server_status", "scan_database_locations"],
            "phase_2_configuration": ["export_tool_config", "sync_claude_preferences", "generate_persona", "import_tool_config"],
            "phase_3_automation": ["get_sidecar_status", "list_available_models", "query_local_ai_service", "chat_with_local_model", "recommend_model", "list_model_tags", "find_model_by_tag"],
            "phase_3_cache": ["get_cache_stats", "clear_cache"],
            "phase_4_intelligence": ["get_model_performance_metrics", "analyse_conversation_patterns", "compare_model_responses", "optimise_knowledge_stacks", "suggest_persona_improvements"],
            "phase_5_calibration": ["run_calibration_test", "evaluate_response_quality", "identify_handoff_triggers", "get_calibration_history"],
            "phase_6_model_management": ["get_model_details", "benchmark_model", "list_local_model_files", "estimate_model_requirements"],
            "phase_7_conversations": ["export_conversations", "search_conversations", "get_conversation_stats"],
            "phase_8_automation": ["create_prompt_template", "list_prompt_templates", "run_prompt_template", "smart_model_router"],
            "phase_9_backup": ["backup_configuration", "restore_configuration", "get_system_resources"]
        },
        "tool_count": 42,
        "msty_status": {
            "installed": paths.get("app") is not None or paths.get("app_alt") is not None,
            "database_available": paths.get("database") is not None,
            "database_type": paths.get("database_type"),
            "local_ai_available": is_local_ai_available()
        },
        "cache_stats": _response_cache.stats()
    }, indent=2)


# =============================================================================
# Phase 2: Configuration Management Tools
# =============================================================================

@mcp.tool()
def export_tool_config(
    tool_name: Optional[str] = None,
    source: str = "claude",
    output_format: str = "msty",
    include_env: bool = False
) -> str:
    """
    Export MCP tool configurations for backup or sync.
    
    Args:
        tool_name: Specific tool to export (None = all tools)
        source: Where to read config from ("claude" or "msty")
        output_format: Output format ("msty", "claude", or "raw")
        include_env: Include environment variables (may contain secrets)
    
    Returns:
        JSON with tool configurations ready for import
    """
    result = {"source": source, "output_format": output_format, "timestamp": datetime.now().isoformat(), "tools": []}
    
    if source == "claude":
        config = read_claude_desktop_config()
        if "error" in config:
            return json.dumps(config, indent=2)
        
        for name, server_config in config.get("mcpServers", {}).items():
            if tool_name and name != tool_name:
                continue
            tool = {
                "name": name,
                "command": server_config.get("command", ""),
                "args": server_config.get("args", []),
            }
            if include_env:
                tool["env"] = server_config.get("env", {})
            result["tools"].append(tool)
    
    result["tool_count"] = len(result["tools"])
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def sync_claude_preferences(
    output_path: Optional[str] = None,
    include_memory_protocol: bool = True,
    include_tool_priorities: bool = True
) -> str:
    """
    Convert Claude Desktop preferences to Msty persona format.
    
    Args:
        output_path: Optional path to save the persona JSON file
        include_memory_protocol: Include memory system integration instructions
        include_tool_priorities: Include MCP tool priority order
    
    Returns:
        JSON with Msty persona configuration
    """
    sections = ["# AI Assistant Persona - Opus Style\n\nBritish English, conversational tone, quality over quantity."]
    
    if include_memory_protocol:
        sections.append("\n## Memory Protocol\nCheck memory MCP at conversation start. Store important info proactively.")
    
    if include_tool_priorities:
        sections.append("\n## Tool Priorities\n1. Memory MCP\n2. Filesystem MCP\n3. Other specialised MCPs")
    
    persona = PersonaConfig(
        name="Opus Style Assistant",
        description="Claude Opus behaviour patterns",
        system_prompt="\n".join(sections),
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    result = {"persona": asdict(persona), "system_prompt_length": len(persona.system_prompt)}
    
    if output_path:
        try:
            with open(expand_path(output_path), 'w') as f:
                json.dump(result["persona"], f, indent=2)
            result["saved_to"] = output_path
        except Exception as e:
            result["save_error"] = str(e)
    
    return json.dumps(result, indent=2)


@mcp.tool()
def generate_persona(
    name: str,
    description: str = "",
    base_template: str = "opus",
    custom_instructions: Optional[str] = None,
    temperature: float = 0.7,
    model_preference: Optional[str] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a complete Msty persona configuration.
    
    Args:
        name: Name for the persona
        description: Brief description of the persona's purpose
        base_template: Starting template ("opus", "minimal", "coder", "writer")
        custom_instructions: Additional instructions to append
        temperature: Model temperature (0.0-1.0)
        model_preference: Preferred model identifier
        output_path: Optional path to save the persona JSON
    
    Returns:
        JSON with complete persona configuration ready for Msty import
    """
    templates = {
        "opus": "AI assistant with British English, quality focus, executive mindset.",
        "minimal": "Helpful AI assistant.",
        "coder": "Development assistant with code review focus.",
        "writer": "Writing assistant with British English standards."
    }
    
    if base_template not in templates:
        return json.dumps({"error": f"Unknown template: {base_template}", "available": list(templates.keys())})
    
    system_prompt = templates[base_template]
    if custom_instructions:
        system_prompt += f"\n\n{custom_instructions}"
    
    persona = PersonaConfig(
        name=name,
        description=description or f"{base_template} persona",
        system_prompt=system_prompt,
        temperature=temperature,
        model_preference=model_preference,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    result = {"persona": asdict(persona), "base_template": base_template}
    
    if output_path:
        try:
            Path(expand_path(output_path)).parent.mkdir(parents=True, exist_ok=True)
            with open(expand_path(output_path), 'w') as f:
                json.dump(result["persona"], f, indent=2)
            result["saved_to"] = output_path
        except Exception as e:
            result["save_error"] = str(e)
    
    return json.dumps(result, indent=2)


@mcp.tool()
def import_tool_config(
    config_json: Optional[str] = None,
    config_file: Optional[str] = None,
    source: str = "claude",
    dry_run: bool = True
) -> str:
    """
    Import MCP tool configurations into Msty Studio.
    
    Args:
        config_json: JSON string with tool configuration(s)
        config_file: Path to JSON file with tool configuration(s)
        source: Source format ("claude", "msty", "auto")
        dry_run: If True, validate only without importing (default: True)
    
    Returns:
        JSON with validation results and import instructions
    """
    result = {"dry_run": dry_run, "validation": {"valid": False, "errors": []}, "tools_to_import": []}
    
    config = None
    if config_json:
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as e:
            result["validation"]["errors"].append(f"Invalid JSON: {e}")
            return json.dumps(result, indent=2)
    elif config_file:
        try:
            with open(expand_path(config_file), 'r') as f:
                config = json.load(f)
        except Exception as e:
            result["validation"]["errors"].append(str(e))
            return json.dumps(result, indent=2)
    else:
        config = read_claude_desktop_config()
        if "error" in config:
            result["validation"]["errors"].append(config["error"])
            return json.dumps(result, indent=2)
    
    tools = []
    if "mcpServers" in config:
        for name, sc in config["mcpServers"].items():
            tools.append({"name": name, "config": {"command": sc.get("command", ""), "args": sc.get("args", [])}})
    
    result["tools_to_import"] = tools
    result["validation"]["valid"] = len(tools) > 0
    result["validation"]["tool_count"] = len(tools)
    
    return json.dumps(result, indent=2, default=str)


# =============================================================================
# Phase 3: Automation Bridge - Sidecar API Integration
# =============================================================================

@mcp.tool()
def get_sidecar_status() -> str:
    """
    Get comprehensive status of Msty Local AI Services.

    Returns:
        - Msty Studio process status
        - Local AI Service availability (port 11964)
        - MLX Service availability (port 11973)
        - LLaMA.cpp Service availability (port 11454)
        - Vibe CLI Proxy availability (port 8317)
        - Available models
        - Port information

    Note: Msty 2.4.0+ has services built into the main app, no separate Sidecar needed.
    """
    # Check all services (Msty 2.4.0+ architecture)
    services = get_available_service_ports()

    result = {
        "timestamp": datetime.now().isoformat(),
        "msty_studio": {"process_running": is_process_running("MstyStudio") or is_process_running("Msty")},
        "services": services,
        "local_ai_service": {"port": LOCAL_AI_SERVICE_PORT, "reachable": False, "models_available": 0},
        "recommendations": []
    }

    # Check if any service is available
    any_service_available = any(s["available"] for s in services.values())

    if not any_service_available:
        result["recommendations"].append("Start Msty Studio and enable Local AI services in Settings")
        return json.dumps(result, indent=2)

    # Try to get models from Local AI Service first, then fallback to other services
    for service_name, service_info in services.items():
        if service_info["available"]:
            models_response = make_api_request("/v1/models", port=service_info["port"], timeout=5)
            if models_response.get("success"):
                result["local_ai_service"]["reachable"] = True
                result["local_ai_service"]["port"] = service_info["port"]
                result["local_ai_service"]["service"] = service_name
                data = models_response.get("data", {})
                if isinstance(data, dict) and "data" in data:
                    result["local_ai_service"]["models_available"] = len(data["data"])
                    result["local_ai_service"]["model_list"] = [m.get("id") for m in data["data"]]
                break

    return json.dumps(result, indent=2)


@mcp.tool()
def list_available_models() -> str:
    """
    List all AI models available through Msty's Local AI Services.

    Checks all available services (Local AI, MLX, LLaMA.cpp, Vibe Proxy)
    and returns models from the first available service.

    Returns detailed information about each model.
    """
    result = {"timestamp": datetime.now().isoformat(), "models": [], "model_count": 0}

    # Check all services
    services = get_available_service_ports()
    any_service_available = any(s["available"] for s in services.values())

    if not any_service_available:
        result["error"] = "No Local AI services are running. Start Msty Studio and enable services in Settings."
        return json.dumps(result, indent=2)

    # Collect models from ALL available services
    all_models = []
    services_with_models = {}

    for service_name, service_info in services.items():
        if service_info["available"]:
            response = make_api_request("/v1/models", port=service_info["port"])
            if response.get("success"):
                data = response.get("data", {})
                if isinstance(data, dict) and "data" in data:
                    service_models = data["data"]
                    # Tag each model with its service
                    for model in service_models:
                        model["_service"] = service_name
                        model["_port"] = service_info["port"]
                    all_models.extend(service_models)
                    services_with_models[service_name] = {
                        "port": service_info["port"],
                        "model_count": len(service_models),
                        "models": [m.get("id") for m in service_models]
                    }

    result["models"] = all_models
    result["model_count"] = len(all_models)
    result["by_service"] = services_with_models

    if not all_models:
        result["error"] = "No models found on any service"

    return json.dumps(result, indent=2)


@mcp.tool()
def query_local_ai_service(
    endpoint: str = "/v1/models",
    method: str = "GET",
    request_body: Optional[str] = None
) -> str:
    """
    Query the Sidecar Local AI Service API directly.
    
    Args:
        endpoint: API endpoint (e.g., "/v1/models", "/v1/chat/completions")
        method: HTTP method (GET, POST)
        request_body: JSON string for POST requests
    
    Returns:
        Raw API response with status information
    """
    if not is_local_ai_available():
        return error_response(
            ErrorCode.SERVICE_UNAVAILABLE,
            "No Local AI service is running",
            suggestion="Start Msty Studio and enable Local AI services"
        )

    data = json.loads(request_body) if request_body else None
    response = make_api_request(endpoint, port=LOCAL_AI_SERVICE_PORT, method=method, data=data, timeout=30)
    
    return json.dumps({"endpoint": endpoint, "method": method, "response": response}, indent=2, default=str)


def get_chat_port_for_model(model_id: str) -> int:
    """
    Determine which port to use for a given model.
    Checks all services to find where the model is available.
    """
    services = get_available_service_ports()

    for service_name, service_info in services.items():
        if service_info["available"]:
            response = make_api_request("/v1/models", port=service_info["port"], timeout=5)
            if response.get("success"):
                data = response.get("data", {})
                if isinstance(data, dict) and "data" in data:
                    model_ids = [m.get("id") for m in data["data"]]
                    if model_id in model_ids:
                        return service_info["port"]

    # Default to MLX port if available, then LLaMA.cpp, then local_ai
    if services["mlx"]["available"]:
        return MLX_SERVICE_PORT
    if services["llamacpp"]["available"]:
        return LLAMACPP_SERVICE_PORT
    return LOCAL_AI_SERVICE_PORT


def get_first_chat_model() -> tuple:
    """
    Find the first available chat model (not embedding model).
    Returns (model_id, port) or (None, None) if none found.
    """
    services = get_available_service_ports()

    # Prefer MLX and LLaMA.cpp services for chat models
    for service_name in ["mlx", "llamacpp", "vibe_proxy", "local_ai"]:
        service_info = services.get(service_name, {})
        if service_info.get("available"):
            response = make_api_request("/v1/models", port=service_info["port"], timeout=5)
            if response.get("success"):
                data = response.get("data", {})
                if isinstance(data, dict) and "data" in data:
                    for model in data["data"]:
                        model_id = model.get("id", "")
                        # Skip embedding models
                        if "embed" in model_id.lower() or "bge" in model_id.lower() or "nomic" in model_id.lower():
                            continue
                        return (model_id, service_info["port"])

    return (None, None)


@mcp.tool()
def chat_with_local_model(
    message: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    track_metrics: bool = True
) -> str:
    """
    Send a chat message to a local model via Sidecar.

    Args:
        message: The user message to send
        model: Model ID to use (if None, uses first available chat model)
        system_prompt: Optional system prompt for context
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens in response
        track_metrics: Record performance metrics (default: True)

    Returns:
        Model response with timing and token information
    """
    result = {"timestamp": datetime.now().isoformat(), "request": {"message": message[:100] + "..." if len(message) > 100 else message}}

    # Check if any service is available
    services = get_available_service_ports()
    any_available = any(s["available"] for s in services.values())

    if not any_available:
        result["error"] = "No Local AI service is running. Start Msty Studio and enable services."
        return json.dumps(result, indent=2)

    # Find model and appropriate port
    if model:
        port = get_chat_port_for_model(model)
    else:
        model, port = get_first_chat_model()
        if not model:
            result["error"] = "No chat models available (only embedding models found)"
            return json.dumps(result, indent=2)

    result["request"]["model"] = model
    result["request"]["port"] = port

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})

    request_data = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": False}

    start_time = time.time()
    response = make_api_request("/v1/chat/completions", port=port, method="POST", data=request_data, timeout=120)
    elapsed_time = time.time() - start_time
    
    result["timing"] = {"elapsed_seconds": round(elapsed_time, 2)}
    
    if response.get("success"):
        data = response.get("data", {})
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            content = msg.get("content", "") or msg.get("reasoning", "")
            result["response"] = {"content": content, "finish_reason": data["choices"][0].get("finish_reason")}
        
        if "usage" in data:
            result["usage"] = data["usage"]
            completion_tokens = data["usage"].get("completion_tokens", 0)
            result["timing"]["tokens_per_second"] = round(completion_tokens / max(elapsed_time, 0.1), 1)
            
            if track_metrics:
                try:
                    init_metrics_db()
                    record_model_metric(
                        model_id=model,
                        prompt_tokens=data["usage"].get("prompt_tokens", 0),
                        completion_tokens=completion_tokens,
                        latency_seconds=elapsed_time,
                        success=True,
                        use_case="chat"
                    )
                except (sqlite3.Error, OSError) as e:
                    logger.warning(f"Failed to record metrics: {e}")
    else:
        result["error"] = response.get("error")
        if track_metrics:
            try:
                init_metrics_db()
                record_model_metric(model_id=model, latency_seconds=elapsed_time, success=False, error_message=response.get("error"))
            except (sqlite3.Error, OSError) as e:
                logger.warning(f"Failed to record metrics: {e}")
    
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def recommend_model(use_case: str = "general", max_size_gb: Optional[float] = None) -> str:
    """
    Get model recommendations based on use case and hardware.
    
    Args:
        use_case: Type of work ("general", "coding", "writing", "analysis", "fast")
        max_size_gb: Maximum model size in GB (optional)
    
    Returns:
        Recommended models with installation instructions
    """
    model_db = {
        "general": [{"id": "qwen2.5:32b", "size_gb": 19, "quality": "very good"}, {"id": "qwen2.5:7b", "size_gb": 4.5, "quality": "good"}],
        "coding": [{"id": "qwen2.5-coder:32b", "size_gb": 19, "quality": "excellent"}, {"id": "qwen2.5-coder:7b", "size_gb": 4.5, "quality": "good"}],
        "writing": [{"id": "qwen2.5:32b", "size_gb": 19, "quality": "very good"}],
        "analysis": [{"id": "qwen2.5:32b", "size_gb": 19, "quality": "very good"}],
        "fast": [{"id": "qwen3:0.6b", "size_gb": 0.5, "quality": "basic"}, {"id": "gemma3:4b", "size_gb": 3, "quality": "acceptable"}]
    }
    
    if use_case not in model_db:
        return error_response(
            ErrorCode.INVALID_PARAMETER,
            f"Unknown use case: {use_case}",
            suggestion=f"Valid use cases: {', '.join(model_db.keys())}"
        )
    
    recommendations = model_db[use_case]
    if max_size_gb:
        recommendations = [m for m in recommendations if m["size_gb"] <= max_size_gb]
    
    return json.dumps({"use_case": use_case, "recommendations": recommendations}, indent=2)


@mcp.tool()
def list_model_tags(model_id: Optional[str] = None) -> str:
    """
    Get tags for models to help with smart selection.

    Tags include: fast, quality, coding, creative, reasoning, embedding, vision, long_context, small, medium, large, general

    Args:
        model_id: Specific model to get tags for (None = all models with their tags)

    Returns:
        Model tags and available tag categories
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "available_tags": list(MODEL_TAGS["patterns"].keys()) + ["small", "medium", "large", "general"],
    }

    if model_id:
        tags = get_model_tags(model_id)
        result["model_id"] = model_id
        result["tags"] = tags
    else:
        # Get all models and their tags
        services = get_available_service_ports()
        models_with_tags = []

        for service_name, service_info in services.items():
            if service_info["available"]:
                response = make_api_request("/v1/models", port=service_info["port"])
                if response.get("success"):
                    data = response.get("data", {})
                    if isinstance(data, dict) and "data" in data:
                        for m in data["data"]:
                            mid = m.get("id", "")
                            tags = get_model_tags(mid)
                            models_with_tags.append({
                                "model_id": mid,
                                "tags": tags,
                                "service": service_name,
                                "port": service_info["port"]
                            })

        result["models"] = models_with_tags
        result["model_count"] = len(models_with_tags)

        # Summarize by tag
        tag_counts = {}
        for m in models_with_tags:
            for tag in m["tags"]:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        result["tag_summary"] = tag_counts

    return json.dumps(result, indent=2)


@mcp.tool()
def find_model_by_tag(
    tag: str,
    prefer_fast: bool = False,
    exclude_embedding: bool = True
) -> str:
    """
    Find models matching a specific tag.

    Args:
        tag: Tag to search for (fast, quality, coding, creative, reasoning, etc.)
        prefer_fast: If True, sort smaller/faster models first
        exclude_embedding: If True, exclude embedding models from results

    Returns:
        List of matching models with their details
    """
    result = {"timestamp": datetime.now().isoformat(), "tag": tag, "models": []}

    # Get all models
    services = get_available_service_ports()
    all_models = []

    for service_name, service_info in services.items():
        if service_info["available"]:
            response = make_api_request("/v1/models", port=service_info["port"])
            if response.get("success"):
                data = response.get("data", {})
                if isinstance(data, dict) and "data" in data:
                    for m in data["data"]:
                        m["_service"] = service_name
                        m["_port"] = service_info["port"]
                        all_models.append(m)

    # Filter by tag
    matching = find_models_by_tag(tag, all_models)

    # Optionally exclude embedding models
    if exclude_embedding:
        matching = [m for m in matching if "embedding" not in m.get("_tags", [])]

    # Sort if prefer_fast
    if prefer_fast:
        # Sort by size indicators in name (smaller first)
        def size_key(m):
            mid = m.get("id", "").lower()
            if any(s in mid for s in ["0.6b", "1b", "2b", "3b"]):
                return 0
            if any(s in mid for s in ["4b", "7b", "8b"]):
                return 1
            if any(s in mid for s in ["13b", "14b", "27b", "32b"]):
                return 2
            return 3
        matching.sort(key=size_key)

    result["models"] = matching
    result["match_count"] = len(matching)

    if not matching:
        result["note"] = f"No models found with tag '{tag}'. Available tags: fast, quality, coding, creative, reasoning, embedding, vision, small, medium, large, general"

    return json.dumps(result, indent=2)


@mcp.tool()
def get_cache_stats() -> str:
    """
    Get statistics about the response cache.

    Returns cache hit/miss information and current entries.
    Useful for debugging and performance monitoring.
    """
    stats = _response_cache.stats()
    stats["timestamp"] = datetime.now().isoformat()
    stats["cache_ttl_seconds"] = _response_cache._default_ttl
    return json.dumps(stats, indent=2)


@mcp.tool()
def clear_cache() -> str:
    """
    Clear the response cache.

    Forces fresh data to be fetched on next request.
    Useful after model changes or configuration updates.
    """
    _response_cache.invalidate()
    return json.dumps({
        "timestamp": datetime.now().isoformat(),
        "status": "Cache cleared successfully",
        "message": "Next requests will fetch fresh data"
    }, indent=2)


@mcp.tool()
def scan_database_locations() -> str:
    """
    Scan for Msty database files in common locations.

    Useful for debugging when database is not found automatically.
    Shows all potential database files and their sizes.

    Returns:
        List of found database files with details
    """
    home = Path.home()
    result = {
        "timestamp": datetime.now().isoformat(),
        "scan_locations": [],
        "found_databases": [],
        "found_data_files": [],
        "current_config": get_msty_paths()
    }

    # Locations to scan - expanded for Msty 2.4.0+
    scan_paths = [
        home / "Library/Application Support/MstyStudio",
        home / "Library/Application Support/MstySidecar",
        home / "Library/Containers/ai.msty.MstyStudio",
        home / "Library/Caches/MstyStudio",
        home / "Library/Preferences",  # May have plist files
        home / ".msty",
        home / ".config/msty",
        home / ".local/share/msty",
    ]

    # File patterns to search for (Msty 2.4.0+ may use various formats)
    db_patterns = ["**/*.db", "**/*.sqlite", "**/*.sqlite3"]
    data_patterns = ["**/*.json", "**/*.leveldb", "**/*.ldb", "**/MANIFEST*", "**/LOG", "**/CURRENT"]

    for scan_path in scan_paths:
        location_info = {
            "path": str(scan_path),
            "exists": scan_path.exists(),
            "databases": [],
            "data_files": [],
            "all_files": []
        }

        if scan_path.exists():
            try:
                # List all files in directory (limited depth for performance)
                all_files = []
                for item in scan_path.iterdir():
                    try:
                        if item.is_file():
                            stat = item.stat()
                            all_files.append({
                                "name": item.name,
                                "size_kb": round(stat.st_size / 1024, 1),
                                "type": item.suffix or "no_ext"
                            })
                        elif item.is_dir():
                            all_files.append({
                                "name": item.name + "/",
                                "type": "directory"
                            })
                    except (OSError, PermissionError):
                        pass
                location_info["all_files"] = all_files[:30]  # Limit for readability

                # Find database files
                for pattern in db_patterns:
                    for db_file in scan_path.glob(pattern):
                        try:
                            if db_file.is_file():
                                stat = db_file.stat()
                                location_info["databases"].append({
                                    "path": str(db_file),
                                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                                })
                                result["found_databases"].append(str(db_file))
                        except (OSError, PermissionError):
                            pass

                # Find other data files (JSON configs, LevelDB, etc.)
                for pattern in data_patterns:
                    for data_file in scan_path.glob(pattern):
                        try:
                            if data_file.is_file() and data_file.stat().st_size > 100:  # Skip tiny files
                                stat = data_file.stat()
                                file_info = {
                                    "path": str(data_file),
                                    "size_kb": round(stat.st_size / 1024, 1),
                                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                                }
                                # Check if it might contain conversation data
                                if data_file.suffix == ".json" and stat.st_size > 1000:
                                    file_info["note"] = "Potential config/data file"
                                location_info["data_files"].append(file_info)
                                result["found_data_files"].append(str(data_file))
                        except (OSError, PermissionError):
                            pass

            except PermissionError:
                location_info["error"] = "Permission denied"

        result["scan_locations"].append(location_info)

    result["total_databases_found"] = len(result["found_databases"])
    result["total_data_files_found"] = len(result["found_data_files"])

    # Provide recommendations
    if result["found_databases"]:
        result["recommendation"] = "Set MSTY_DATABASE_PATH environment variable to one of the found databases"
    elif result["found_data_files"]:
        result["recommendation"] = "No SQLite databases found, but data files exist. Msty 2.4.0+ may use JSON or other storage formats."
    else:
        result["recommendation"] = "No data files found. Check if Msty has been used to create conversations, or check Msty settings for data location."

    return json.dumps(result, indent=2)


# =============================================================================
# Phase 6: Advanced Model Management
# =============================================================================

@mcp.tool()
def get_model_details(model_id: str) -> str:
    """
    Get detailed information about a specific model.

    Args:
        model_id: The model ID to get details for

    Returns:
        Comprehensive model details including:
        - Context length and parameters
        - Service and port information
        - Tags and capabilities
        - File size (for local models)
        - Recommended use cases
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "found": False,
    }

    # Search for model in all services
    services = get_available_service_ports()

    for service_name, service_info in services.items():
        if service_info["available"]:
            response = make_api_request("/v1/models", port=service_info["port"])
            if response.get("success"):
                data = response.get("data", {})
                if isinstance(data, dict) and "data" in data:
                    for model in data["data"]:
                        if model.get("id") == model_id:
                            result["found"] = True
                            result["service"] = service_name
                            result["port"] = service_info["port"]
                            result["raw_info"] = model

                            # Extract key details
                            result["details"] = {
                                "id": model.get("id"),
                                "owned_by": model.get("owned_by"),
                                "object": model.get("object"),
                                "context_length": model.get("context_length"),
                                "created": model.get("created"),
                            }

                            # Get tags
                            result["tags"] = get_model_tags(model_id)

                            # Determine capabilities based on tags
                            tags = result["tags"]
                            capabilities = []
                            if "coding" in tags:
                                capabilities.append("Code generation and review")
                            if "reasoning" in tags:
                                capabilities.append("Complex reasoning and analysis")
                            if "creative" in tags:
                                capabilities.append("Creative writing and storytelling")
                            if "fast" in tags:
                                capabilities.append("Quick responses, good for simple tasks")
                            if "quality" in tags:
                                capabilities.append("High-quality outputs")
                            if "embedding" in tags:
                                capabilities.append("Text embeddings for semantic search")
                            if "vision" in tags:
                                capabilities.append("Image understanding")
                            if "long_context" in tags:
                                capabilities.append("Long document processing")
                            result["capabilities"] = capabilities

                            # Recommend use cases
                            use_cases = []
                            if "coding" in tags:
                                use_cases.extend(["Code review", "Bug fixing", "Code generation"])
                            if "reasoning" in tags:
                                use_cases.extend(["Problem solving", "Analysis", "Research"])
                            if "creative" in tags:
                                use_cases.extend(["Writing", "Brainstorming", "Content creation"])
                            if "fast" in tags:
                                use_cases.extend(["Quick Q&A", "Summarization", "Simple tasks"])
                            if not use_cases:
                                use_cases = ["General conversation", "Question answering"]
                            result["recommended_use_cases"] = list(set(use_cases))

                            break
                if result["found"]:
                    break

    if not result["found"]:
        result["error"] = f"Model '{model_id}' not found in any service"
        result["suggestion"] = "Use list_available_models to see all available models"

    return json.dumps(result, indent=2)


@mcp.tool()
def benchmark_model(
    model_id: str,
    num_runs: int = 3,
    prompt_lengths: Optional[List[int]] = None
) -> str:
    """
    Run performance benchmarks on a local model.

    Args:
        model_id: Model to benchmark
        num_runs: Number of test runs per prompt length (default: 3)
        prompt_lengths: List of prompt lengths to test (default: [50, 200, 500])

    Returns:
        Benchmark results with tokens/second at different context sizes
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "num_runs": num_runs,
        "benchmarks": [],
        "summary": {}
    }

    if prompt_lengths is None:
        prompt_lengths = [50, 200, 500]

    # Find the model's port
    port = get_chat_port_for_model(model_id)
    result["port"] = port

    # Skip embedding models
    if any(x in model_id.lower() for x in ["embed", "bge", "nomic"]):
        result["error"] = "Cannot benchmark embedding models - they don't support chat completions"
        return json.dumps(result, indent=2)

    # Generate test prompts of different lengths
    base_prompt = "Explain the concept of "
    topics = ["machine learning", "quantum computing", "renewable energy", "artificial intelligence", "blockchain technology"]

    all_results = []

    for target_length in prompt_lengths:
        # Create a prompt of approximately target_length characters
        prompt = base_prompt + topics[target_length % len(topics)]
        while len(prompt) < target_length:
            prompt += " and its applications in modern technology, including various use cases"
        prompt = prompt[:target_length] + "? Be concise."

        run_results = []
        for run in range(num_runs):
            start_time = time.time()
            response = make_api_request(
                "/v1/chat/completions",
                port=port,
                method="POST",
                data={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 256,
                    "stream": False
                },
                timeout=60
            )
            elapsed = time.time() - start_time

            if response.get("success"):
                data = response.get("data", {})
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)

                tps = completion_tokens / max(elapsed, 0.01)
                run_results.append({
                    "run": run + 1,
                    "latency_seconds": round(elapsed, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tokens_per_second": round(tps, 1)
                })
            else:
                run_results.append({
                    "run": run + 1,
                    "error": response.get("error")
                })

        # Calculate averages for this prompt length
        successful_runs = [r for r in run_results if "error" not in r]
        if successful_runs:
            avg_tps = sum(r["tokens_per_second"] for r in successful_runs) / len(successful_runs)
            avg_latency = sum(r["latency_seconds"] for r in successful_runs) / len(successful_runs)
        else:
            avg_tps = 0
            avg_latency = 0

        all_results.append({
            "prompt_length": target_length,
            "runs": run_results,
            "avg_tokens_per_second": round(avg_tps, 1),
            "avg_latency_seconds": round(avg_latency, 3)
        })

    result["benchmarks"] = all_results

    # Overall summary
    all_successful = [r for bench in all_results for r in bench["runs"] if "error" not in r]
    if all_successful:
        result["summary"] = {
            "total_runs": len(all_successful),
            "overall_avg_tps": round(sum(r["tokens_per_second"] for r in all_successful) / len(all_successful), 1),
            "min_tps": round(min(r["tokens_per_second"] for r in all_successful), 1),
            "max_tps": round(max(r["tokens_per_second"] for r in all_successful), 1),
            "overall_avg_latency": round(sum(r["latency_seconds"] for r in all_successful) / len(all_successful), 3),
        }

        # Performance rating
        avg_tps = result["summary"]["overall_avg_tps"]
        if avg_tps >= 50:
            result["summary"]["rating"] = "Excellent"
        elif avg_tps >= 30:
            result["summary"]["rating"] = "Very Good"
        elif avg_tps >= 15:
            result["summary"]["rating"] = "Good"
        elif avg_tps >= 5:
            result["summary"]["rating"] = "Acceptable"
        else:
            result["summary"]["rating"] = "Slow"

    # Record metrics
    try:
        init_metrics_db()
        for bench in all_results:
            for run in bench["runs"]:
                if "error" not in run:
                    record_model_metric(
                        model_id=model_id,
                        prompt_tokens=run.get("prompt_tokens", 0),
                        completion_tokens=run.get("completion_tokens", 0),
                        latency_seconds=run["latency_seconds"],
                        success=True,
                        use_case="benchmark"
                    )
    except (sqlite3.Error, OSError) as e:
        logger.debug(f"Could not record benchmark metrics: {e}")

    return json.dumps(result, indent=2)


@mcp.tool()
def list_local_model_files() -> str:
    """
    List all local model files on disk (MLX and GGUF models).

    Returns:
        List of model files with sizes and locations
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "mlx_models": [],
        "gguf_models": [],
        "total_size_gb": 0
    }

    paths = get_msty_paths()
    total_size = 0

    # Scan MLX models
    mlx_path = paths.get("mlx_models")
    if mlx_path and Path(mlx_path).exists():
        mlx_dir = Path(mlx_path)
        for model_dir in mlx_dir.iterdir():
            if model_dir.is_dir():
                # Calculate total size of model directory
                size_bytes = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                size_gb = size_bytes / (1024 ** 3)
                total_size += size_bytes

                # Count files
                file_count = sum(1 for f in model_dir.rglob('*') if f.is_file())

                result["mlx_models"].append({
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "size_gb": round(size_gb, 2),
                    "file_count": file_count
                })

    # Look for GGUF models in common locations
    home = Path.home()
    gguf_locations = [
        home / ".cache/lm-studio/models",
        home / ".ollama/models",
        home / "Library/Application Support/MstyStudio/models",
        home / "models",
    ]

    for gguf_dir in gguf_locations:
        if gguf_dir.exists():
            for gguf_file in gguf_dir.rglob("*.gguf"):
                try:
                    size_bytes = gguf_file.stat().st_size
                    size_gb = size_bytes / (1024 ** 3)
                    total_size += size_bytes

                    result["gguf_models"].append({
                        "name": gguf_file.stem,
                        "path": str(gguf_file),
                        "size_gb": round(size_gb, 2)
                    })
                except (OSError, PermissionError):
                    pass

    result["total_size_gb"] = round(total_size / (1024 ** 3), 2)
    result["mlx_count"] = len(result["mlx_models"])
    result["gguf_count"] = len(result["gguf_models"])

    return json.dumps(result, indent=2)


@mcp.tool()
def estimate_model_requirements(model_id: str) -> str:
    """
    Estimate hardware requirements for running a model.

    Args:
        model_id: Model ID to analyze

    Returns:
        Estimated memory requirements and recommendations
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "estimates": {}
    }

    # Get model details first
    details_json = get_model_details(model_id)
    details = json.loads(details_json)

    if not details.get("found"):
        result["error"] = f"Model '{model_id}' not found"
        return json.dumps(result, indent=2)

    result["service"] = details.get("service")
    result["tags"] = details.get("tags", [])

    # Extract size from model name
    model_lower = model_id.lower()
    param_size = None

    # Try to extract parameter count
    import re
    size_patterns = [
        (r'(\d+\.?\d*)b', 1),  # 7b, 13b, 70b
        (r'(\d+)m', 0.001),     # 567m -> 0.567b
    ]

    for pattern, multiplier in size_patterns:
        match = re.search(pattern, model_lower)
        if match:
            param_size = float(match.group(1)) * multiplier
            break

    if param_size:
        result["estimated_parameters"] = f"{param_size}B"

        # Estimate memory based on quantization
        if "fp16" in model_lower or "f16" in model_lower:
            bytes_per_param = 2
            quant = "FP16"
        elif "q8" in model_lower or "8bit" in model_lower:
            bytes_per_param = 1
            quant = "Q8/8-bit"
        elif "q6" in model_lower or "6bit" in model_lower:
            bytes_per_param = 0.75
            quant = "Q6/6-bit"
        elif "q5" in model_lower or "5bit" in model_lower:
            bytes_per_param = 0.625
            quant = "Q5/5-bit"
        elif "q4" in model_lower or "4bit" in model_lower:
            bytes_per_param = 0.5
            quant = "Q4/4-bit"
        elif "q3" in model_lower or "3bit" in model_lower:
            bytes_per_param = 0.375
            quant = "Q3/3-bit"
        elif "q2" in model_lower or "2bit" in model_lower:
            bytes_per_param = 0.25
            quant = "Q2/2-bit"
        else:
            bytes_per_param = 0.5  # Assume Q4 as default
            quant = "Unknown (assuming Q4)"

        # Calculate memory
        model_memory_gb = (param_size * 1e9 * bytes_per_param) / (1024 ** 3)
        # Add overhead for KV cache and runtime (roughly 20-30%)
        total_memory_gb = model_memory_gb * 1.25

        result["estimates"] = {
            "quantization": quant,
            "model_memory_gb": round(model_memory_gb, 1),
            "recommended_vram_gb": round(total_memory_gb, 1),
            "recommended_ram_gb": round(total_memory_gb * 1.5, 1),  # CPU inference needs more
        }

        # Recommendations
        recommendations = []
        if total_memory_gb <= 8:
            recommendations.append("✅ Should run on most modern Macs (8GB+ RAM)")
        elif total_memory_gb <= 16:
            recommendations.append("⚠️ Requires 16GB+ RAM Mac")
        elif total_memory_gb <= 32:
            recommendations.append("⚠️ Requires 32GB+ RAM Mac")
        elif total_memory_gb <= 64:
            recommendations.append("⚠️ Requires 64GB+ RAM Mac (M1/M2/M3 Max/Ultra)")
        else:
            recommendations.append("❌ Requires high-end workstation (96GB+ RAM)")

        if details.get("service") == "mlx":
            recommendations.append("💡 MLX optimized for Apple Silicon - uses unified memory efficiently")
        elif details.get("service") == "llamacpp":
            recommendations.append("💡 LLaMA.cpp can offload layers to CPU if needed")

        result["recommendations"] = recommendations
    else:
        result["estimates"]["note"] = "Could not determine model size from name"
        result["estimates"]["suggestion"] = "Check model card or documentation for requirements"

    return json.dumps(result, indent=2)


# =============================================================================
# Phase 7: Conversation & Memory
# =============================================================================

@mcp.tool()
def export_conversations(
    output_format: str = "json",
    days: Optional[int] = None,
    model_filter: Optional[str] = None,
    limit: int = 100
) -> str:
    """
    Export chat history from Msty database.

    Args:
        output_format: Output format ("json", "markdown", "csv")
        days: Only export conversations from last N days (None = all)
        model_filter: Only export conversations using this model
        limit: Maximum conversations to export (default: 100)

    Returns:
        Exported conversations in requested format
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "format": output_format,
        "filters": {"days": days, "model": model_filter, "limit": limit},
    }

    paths = get_msty_paths()
    db_path = paths.get("database")

    if not db_path:
        return error_response(
            ErrorCode.DATABASE_NOT_FOUND,
            "Msty database not found",
            suggestion="Run scan_database_locations to find it, or set MSTY_DATABASE_PATH environment variable"
        )

    try:
        # Query conversations
        tables = get_table_names(db_path)
        conv_table = None
        for t in ["chat_sessions", "conversations", "sessions"]:
            if t in tables:
                conv_table = t
                break

        if not conv_table:
            result["error"] = "No conversation table found in database"
            result["available_tables"] = tables
            return json.dumps(result, indent=2)

        # Validate table exists before querying (SQL injection protection)
        if not validate_table_exists(db_path, conv_table):
            result["error"] = f"Table validation failed for {conv_table}"
            return json.dumps(result, indent=2)

        # Build query with parameterized limit
        query = f"SELECT * FROM {conv_table}"
        params = []

        if days:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            query += " WHERE created_at > ? OR updated_at > ?"
            params.extend([cutoff, cutoff])

        query += " ORDER BY rowid DESC LIMIT ?"
        params.append(limit)

        conversations = query_database(db_path, query, tuple(params))

        if model_filter:
            conversations = [c for c in conversations if model_filter.lower() in str(c.get("model", "")).lower()]

        result["conversation_count"] = len(conversations)

        if output_format == "json":
            result["conversations"] = conversations
        elif output_format == "markdown":
            md_lines = ["# Msty Conversations Export\n"]
            for conv in conversations:
                md_lines.append(f"## {conv.get('title', 'Untitled')}")
                md_lines.append(f"- Model: {conv.get('model', 'Unknown')}")
                md_lines.append(f"- Created: {conv.get('created_at', 'Unknown')}")
                md_lines.append("")
            result["markdown"] = "\n".join(md_lines)
        elif output_format == "csv":
            if conversations:
                headers = list(conversations[0].keys())
                csv_lines = [",".join(headers)]
                for conv in conversations:
                    csv_lines.append(",".join(str(conv.get(h, "")).replace(",", ";") for h in headers))
                result["csv"] = "\n".join(csv_lines)

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def search_conversations(
    query: str,
    search_type: str = "keyword",
    limit: int = 20
) -> str:
    """
    Search through past conversations.

    Args:
        query: Search query (keyword or phrase)
        search_type: "keyword" for text search, "title" for title-only search
        limit: Maximum results to return

    Returns:
        Matching conversations with context
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "search_type": search_type,
        "matches": []
    }

    paths = get_msty_paths()
    db_path = paths.get("database")

    if not db_path:
        result["error"] = "Msty database not found. Use scan_database_locations to find it."
        return json.dumps(result, indent=2)

    try:
        tables = get_table_names(db_path)

        # Search in conversations/sessions table
        conv_table = None
        for t in ["chat_sessions", "conversations", "sessions"]:
            if t in tables:
                conv_table = t
                break

        if conv_table and validate_table_exists(db_path, conv_table):
            search_pattern = f"%{query}%"

            if search_type == "title":
                sql = f"SELECT * FROM {conv_table} WHERE title LIKE ? LIMIT ?"
                matches = query_database(db_path, sql, (search_pattern, limit))
            else:
                sql = f"SELECT * FROM {conv_table} WHERE title LIKE ? OR summary LIKE ? LIMIT ?"
                matches = query_database(db_path, sql, (search_pattern, search_pattern, limit))

            result["matches"] = matches

        # Also search messages if available
        msg_table = None
        for t in ["messages", "chat_messages"]:
            if t in tables:
                msg_table = t
                break

        if msg_table and validate_table_exists(db_path, msg_table):
            msg_sql = f"SELECT * FROM {msg_table} WHERE content LIKE ? LIMIT ?"
            msg_matches = query_database(db_path, msg_sql, (f"%{query}%", limit))
            result["message_matches"] = len(msg_matches)
            if msg_matches:
                result["message_samples"] = msg_matches[:5]  # Return first 5 as samples

        result["total_matches"] = len(result.get("matches", []))

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_conversation_stats(days: int = 30) -> str:
    """
    Get usage analytics for conversations.

    Args:
        days: Number of days to analyze (default: 30)

    Returns:
        Statistics including messages per day, model usage, session lengths
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "period_days": days,
        "stats": {}
    }

    paths = get_msty_paths()
    db_path = paths.get("database")

    if not db_path:
        result["error"] = "Msty database not found. Use scan_database_locations to find it."
        return json.dumps(result, indent=2)

    try:
        tables = get_table_names(db_path)

        # Get conversation counts
        conv_table = None
        for t in ["chat_sessions", "conversations", "sessions"]:
            if t in tables:
                conv_table = t
                break

        if conv_table and validate_table_exists(db_path, conv_table):
            result["stats"]["total_conversations"] = safe_count_table(db_path, conv_table)

            # Recent conversations
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            result["stats"]["recent_conversations"] = safe_count_table(
                db_path, conv_table, "created_at > ?", (cutoff,)
            )

            # Model usage - safe query with validated table
            model_usage = query_database(
                db_path,
                f"SELECT model, COUNT(*) as count FROM {conv_table} GROUP BY model ORDER BY count DESC LIMIT 10"
            )
            result["stats"]["model_usage"] = model_usage

        # Get message counts
        msg_table = None
        for t in ["messages", "chat_messages"]:
            if t in tables:
                msg_table = t
                break

        if msg_table and validate_table_exists(db_path, msg_table):
            result["stats"]["total_messages"] = safe_count_table(db_path, msg_table)

            # Messages by role - safe query with validated table
            role_counts = query_database(
                db_path,
                f"SELECT role, COUNT(*) as count FROM {msg_table} GROUP BY role"
            )
            result["stats"]["messages_by_role"] = {r["role"]: r["count"] for r in role_counts}

        # Calculate averages
        if result["stats"].get("total_conversations", 0) > 0 and result["stats"].get("total_messages", 0) > 0:
            result["stats"]["avg_messages_per_conversation"] = round(
                result["stats"]["total_messages"] / result["stats"]["total_conversations"], 1
            )

        # Personas if available
        if "personas" in tables and validate_table_exists(db_path, "personas"):
            result["stats"]["total_personas"] = safe_count_table(db_path, "personas")

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result, indent=2, default=str)


# =============================================================================
# Phase 8: Automation & Task Templates
# =============================================================================

@mcp.tool()
def create_prompt_template(
    name: str,
    template: str,
    description: str = "",
    variables: Optional[List[str]] = None,
    preferred_model: Optional[str] = None,
    category: str = "general"
) -> str:
    """
    Create a reusable prompt template with variables.

    Args:
        name: Template name (unique identifier)
        template: Prompt template with {{variable}} placeholders
        description: What this template is for
        variables: List of variable names used in template
        preferred_model: Recommended model for this template
        category: Category (general, coding, writing, analysis, creative)

    Returns:
        Created template configuration
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "action": "create",
        "template": {}
    }

    # Extract variables from template if not provided
    import re
    if variables is None:
        variables = re.findall(r'\{\{(\w+)\}\}', template)

    template_config = {
        "name": name,
        "description": description,
        "template": template,
        "variables": variables,
        "preferred_model": preferred_model,
        "category": category,
        "created_at": datetime.now().isoformat()
    }

    # Store in metrics database
    try:
        init_metrics_db()
        db_path = get_metrics_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create templates table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                template TEXT NOT NULL,
                variables TEXT,
                preferred_model TEXT,
                category TEXT DEFAULT 'general',
                created_at TEXT,
                use_count INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            INSERT OR REPLACE INTO prompt_templates
            (name, description, template, variables, preferred_model, category, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            description,
            template,
            json.dumps(variables),
            preferred_model,
            category,
            template_config["created_at"]
        ))
        conn.commit()
        conn.close()

        result["template"] = template_config
        result["status"] = "created"

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result, indent=2)


@mcp.tool()
def list_prompt_templates(category: Optional[str] = None) -> str:
    """
    List all saved prompt templates.

    Args:
        category: Filter by category (optional)

    Returns:
        List of templates with usage stats
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "templates": []
    }

    try:
        init_metrics_db()
        db_path = get_metrics_db_path()
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if category:
            cursor.execute("SELECT * FROM prompt_templates WHERE category = ? ORDER BY use_count DESC", (category,))
        else:
            cursor.execute("SELECT * FROM prompt_templates ORDER BY use_count DESC")

        templates = [dict(row) for row in cursor.fetchall()]
        conn.close()

        # Parse variables back to list
        for t in templates:
            if t.get("variables"):
                t["variables"] = json.loads(t["variables"])

        result["templates"] = templates
        result["count"] = len(templates)

    except sqlite3.OperationalError:
        result["templates"] = []
        result["note"] = "No templates created yet"
    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result, indent=2)


@mcp.tool()
def run_prompt_template(
    template_name: str,
    variables: Dict[str, str],
    model: Optional[str] = None
) -> str:
    """
    Execute a saved prompt template with provided variables.

    Args:
        template_name: Name of the template to run
        variables: Dictionary of variable values to substitute
        model: Override model (uses template's preferred_model if not specified)

    Returns:
        Model response from executing the template
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "template_name": template_name,
    }

    try:
        init_metrics_db()
        db_path = get_metrics_db_path()
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM prompt_templates WHERE name = ?", (template_name,))
        row = cursor.fetchone()

        if not row:
            result["error"] = f"Template '{template_name}' not found"
            conn.close()
            return json.dumps(result, indent=2)

        template = dict(row)
        template_text = template["template"]

        # Substitute variables
        for var_name, var_value in variables.items():
            template_text = template_text.replace(f"{{{{{var_name}}}}}", str(var_value))

        # Check for missing variables
        import re
        missing = re.findall(r'\{\{(\w+)\}\}', template_text)
        if missing:
            result["error"] = f"Missing variables: {missing}"
            conn.close()
            return json.dumps(result, indent=2)

        result["rendered_prompt"] = template_text[:500] + "..." if len(template_text) > 500 else template_text

        # Use specified model or template's preferred model
        use_model = model or template.get("preferred_model")

        # Update use count
        cursor.execute("UPDATE prompt_templates SET use_count = use_count + 1 WHERE name = ?", (template_name,))
        conn.commit()
        conn.close()

        # Execute with chat_with_local_model
        chat_result = chat_with_local_model(
            message=template_text,
            model=use_model
        )
        chat_data = json.loads(chat_result)

        result["model_used"] = chat_data.get("request", {}).get("model")
        result["response"] = chat_data.get("response", {}).get("content")
        result["timing"] = chat_data.get("timing")

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def smart_model_router(
    task_description: str,
    prefer_speed: bool = False,
    prefer_quality: bool = False
) -> str:
    """
    Automatically select the best model for a given task.

    Args:
        task_description: Description of what you want to accomplish
        prefer_speed: Prioritize faster models
        prefer_quality: Prioritize higher quality models

    Returns:
        Recommended model with reasoning
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "task": task_description[:200],
        "preferences": {"speed": prefer_speed, "quality": prefer_quality}
    }

    # Analyze task to determine category
    task_lower = task_description.lower()

    detected_category = "general"
    if any(word in task_lower for word in ["code", "program", "function", "debug", "script", "api"]):
        detected_category = "coding"
    elif any(word in task_lower for word in ["write", "essay", "story", "email", "blog", "article"]):
        detected_category = "writing"
    elif any(word in task_lower for word in ["analyze", "compare", "evaluate", "research", "study"]):
        detected_category = "analysis"
    elif any(word in task_lower for word in ["think", "reason", "solve", "math", "logic", "puzzle"]):
        detected_category = "reasoning"
    elif any(word in task_lower for word in ["creative", "brainstorm", "idea", "imagine"]):
        detected_category = "creative"

    result["detected_category"] = detected_category

    # Map category to preferred tags
    tag_priority = {
        "coding": ["coding", "quality"],
        "writing": ["creative", "quality"],
        "analysis": ["reasoning", "quality"],
        "reasoning": ["reasoning", "quality"],
        "creative": ["creative"],
        "general": ["general"]
    }

    tags_to_find = tag_priority.get(detected_category, ["general"])

    if prefer_speed:
        tags_to_find = ["fast"] + tags_to_find
    if prefer_quality:
        tags_to_find = ["quality"] + tags_to_find

    result["search_tags"] = tags_to_find

    # Find matching models
    best_model = None
    best_score = 0

    services = get_available_service_ports()
    candidates = []

    for service_name, service_info in services.items():
        if service_info["available"]:
            response = make_api_request("/v1/models", port=service_info["port"])
            if response.get("success"):
                data = response.get("data", {})
                if isinstance(data, dict) and "data" in data:
                    for m in data["data"]:
                        model_id = m.get("id", "")
                        # Skip embedding models
                        if any(x in model_id.lower() for x in ["embed", "bge", "nomic"]):
                            continue

                        tags = get_model_tags(model_id)
                        score = sum(1 for t in tags_to_find if t in tags)

                        if score > 0:
                            candidates.append({
                                "model_id": model_id,
                                "service": service_name,
                                "port": service_info["port"],
                                "tags": tags,
                                "score": score
                            })

    # Sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)

    if candidates:
        result["recommended"] = candidates[0]
        result["alternatives"] = candidates[1:4]  # Top 3 alternatives
        result["reasoning"] = f"Selected based on tags matching: {tags_to_find}"
    else:
        # Fallback to first available chat model
        model, port = get_first_chat_model()
        if model:
            result["recommended"] = {"model_id": model, "port": port, "tags": get_model_tags(model)}
            result["reasoning"] = "Fallback to first available chat model"
        else:
            result["error"] = "No suitable models found"

    return json.dumps(result, indent=2)


# =============================================================================
# Phase 9: Backup & Integration
# =============================================================================

@mcp.tool()
def backup_configuration(
    output_path: Optional[str] = None,
    include_personas: bool = True,
    include_prompts: bool = True,
    include_templates: bool = True,
    include_tools: bool = True
) -> str:
    """
    Create a comprehensive backup of Msty configuration.

    Args:
        output_path: Path to save backup file (optional, returns JSON if not specified)
        include_personas: Include persona configurations
        include_prompts: Include saved prompts
        include_templates: Include prompt templates
        include_tools: Include MCP tool configurations

    Returns:
        Backup data or confirmation of saved file
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "backup": {
            "version": SERVER_VERSION,
            "created_at": datetime.now().isoformat(),
        }
    }

    backup_data = {
        "metadata": {
            "version": SERVER_VERSION,
            "created_at": datetime.now().isoformat(),
            "source": "msty-admin-mcp"
        }
    }

    # Get Msty database data
    paths = get_msty_paths()
    db_path = paths.get("database")

    if db_path:
        tables = get_table_names(db_path)

        if include_personas and validate_table_exists(db_path, "personas"):
            backup_data["personas"] = safe_query_table(db_path, "personas", limit=10000)

        if include_prompts:
            for t in ["prompts", "prompt_library"]:
                if validate_table_exists(db_path, t):
                    backup_data["prompts"] = safe_query_table(db_path, t, limit=10000)
                    break

        if include_tools:
            for t in ["tools", "mcp_tools"]:
                if validate_table_exists(db_path, t):
                    backup_data["tools"] = safe_query_table(db_path, t, limit=10000)
                    break

    # Get prompt templates from our metrics DB
    if include_templates:
        try:
            init_metrics_db()
            metrics_db = get_metrics_db_path()
            conn = sqlite3.connect(str(metrics_db))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM prompt_templates")
            backup_data["prompt_templates"] = [dict(row) for row in cursor.fetchall()]
            conn.close()
        except (sqlite3.Error, OSError) as e:
            logger.debug(f"Could not backup prompt templates: {e}")
            backup_data["prompt_templates"] = []

    # Get Claude Desktop config
    if include_tools:
        claude_config = read_claude_desktop_config()
        if "error" not in claude_config:
            backup_data["claude_desktop_mcp"] = claude_config.get("mcpServers", {})

    result["backup"]["sections"] = list(backup_data.keys())
    result["backup"]["item_counts"] = {k: len(v) if isinstance(v, list) else 1 for k, v in backup_data.items()}

    if output_path:
        try:
            expanded_path = expand_path(output_path)
            Path(expanded_path).parent.mkdir(parents=True, exist_ok=True)
            with open(expanded_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            result["saved_to"] = output_path
            result["status"] = "Backup saved successfully"
        except Exception as e:
            result["error"] = f"Failed to save backup: {e}"
            result["backup_data"] = backup_data
    else:
        result["backup_data"] = backup_data

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def restore_configuration(
    backup_path: Optional[str] = None,
    backup_json: Optional[str] = None,
    restore_templates: bool = True,
    dry_run: bool = True
) -> str:
    """
    Restore configuration from a backup.

    Args:
        backup_path: Path to backup file
        backup_json: Backup data as JSON string (alternative to file)
        restore_templates: Restore prompt templates
        dry_run: If True, only validate without restoring (default: True)

    Returns:
        Restore status and what would be/was restored
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "restore_plan": []
    }

    # Load backup data
    backup_data = None

    if backup_path:
        try:
            with open(expand_path(backup_path), 'r') as f:
                backup_data = json.load(f)
        except Exception as e:
            result["error"] = f"Failed to load backup: {e}"
            return json.dumps(result, indent=2)
    elif backup_json:
        try:
            backup_data = json.loads(backup_json)
        except json.JSONDecodeError as e:
            result["error"] = f"Invalid JSON: {e}"
            return json.dumps(result, indent=2)
    else:
        result["error"] = "Provide either backup_path or backup_json"
        return json.dumps(result, indent=2)

    result["backup_metadata"] = backup_data.get("metadata", {})

    # Plan restoration
    if restore_templates and "prompt_templates" in backup_data:
        templates = backup_data["prompt_templates"]
        result["restore_plan"].append({
            "type": "prompt_templates",
            "count": len(templates),
            "items": [t.get("name") for t in templates]
        })

        if not dry_run:
            try:
                init_metrics_db()
                db_path = get_metrics_db_path()
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()

                for t in templates:
                    cursor.execute("""
                        INSERT OR REPLACE INTO prompt_templates
                        (name, description, template, variables, preferred_model, category, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        t.get("name"),
                        t.get("description"),
                        t.get("template"),
                        t.get("variables") if isinstance(t.get("variables"), str) else json.dumps(t.get("variables", [])),
                        t.get("preferred_model"),
                        t.get("category", "general"),
                        t.get("created_at", datetime.now().isoformat())
                    ))
                conn.commit()
                conn.close()
                result["restored"] = {"prompt_templates": len(templates)}
            except Exception as e:
                result["error"] = f"Failed to restore templates: {e}"

    if dry_run:
        result["status"] = "Dry run complete - no changes made"
        result["note"] = "Set dry_run=False to actually restore"
    else:
        result["status"] = "Restore complete"

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_system_resources() -> str:
    """
    Get current system resource usage.

    Returns:
        CPU, memory, and disk usage information relevant to AI model inference
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "system": {},
        "memory": {},
        "disk": {},
        "recommendations": []
    }

    # CPU info
    result["system"]["cpu_count"] = psutil.cpu_count()
    result["system"]["cpu_percent"] = psutil.cpu_percent(interval=0.5)

    # Memory info
    mem = psutil.virtual_memory()
    result["memory"]["total_gb"] = round(mem.total / (1024 ** 3), 1)
    result["memory"]["available_gb"] = round(mem.available / (1024 ** 3), 1)
    result["memory"]["used_gb"] = round(mem.used / (1024 ** 3), 1)
    result["memory"]["percent_used"] = mem.percent

    # Disk info (for model storage)
    paths = get_msty_paths()
    data_path = paths.get("data") or str(Path.home())

    try:
        disk = psutil.disk_usage(data_path)
        result["disk"]["total_gb"] = round(disk.total / (1024 ** 3), 1)
        result["disk"]["free_gb"] = round(disk.free / (1024 ** 3), 1)
        result["disk"]["used_gb"] = round(disk.used / (1024 ** 3), 1)
        result["disk"]["percent_used"] = round(disk.percent, 1)
    except (OSError, FileNotFoundError):
        pass

    # Recommendations based on resources
    if mem.percent > 90:
        result["recommendations"].append("⚠️ Memory usage is very high - consider closing other applications")
    elif mem.percent > 75:
        result["recommendations"].append("💡 Memory usage is elevated - larger models may run slowly")

    if result["memory"]["available_gb"] < 8:
        result["recommendations"].append("⚠️ Less than 8GB RAM available - stick to smaller models (7B or less)")
    elif result["memory"]["available_gb"] < 16:
        result["recommendations"].append("💡 16-32B models should work well")
    else:
        result["recommendations"].append("✅ Plenty of RAM available for large models")

    if result.get("disk", {}).get("free_gb", 0) < 50:
        result["recommendations"].append("⚠️ Low disk space - may affect model downloads")

    # Check if Msty is using significant resources
    for proc in psutil.process_iter(['name', 'memory_percent', 'cpu_percent']):
        try:
            if 'msty' in proc.info['name'].lower():
                result["msty_process"] = {
                    "name": proc.info['name'],
                    "memory_percent": round(proc.info['memory_percent'], 1),
                    "cpu_percent": round(proc.info['cpu_percent'], 1)
                }
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, KeyError, TypeError):
            pass

    return json.dumps(result, indent=2)


# =============================================================================
# Phase 4: Intelligence Layer
# =============================================================================

@mcp.tool()
def get_model_performance_metrics(model_id: Optional[str] = None, days: int = 30) -> str:
    """
    Get performance metrics for local models over time.
    
    Args:
        model_id: Specific model to query (None = all models)
        days: Number of days to include in analysis (default: 30)
    
    Returns:
        Aggregated performance metrics with trends
    """
    result = {"timestamp": datetime.now().isoformat(), "period_days": days}
    
    try:
        init_metrics_db()
        metrics = get_model_metrics_summary(model_id=model_id, days=days)
        result["metrics"] = metrics
        
        if metrics.get("models"):
            insights = []
            for m in metrics["models"]:
                tps = m.get("avg_tokens_per_second", 0) or 0
                if tps > 50:
                    insights.append(f"✅ {m['model_id']}: Excellent speed ({tps:.1f} tok/s)")
                elif tps > 20:
                    insights.append(f"👍 {m['model_id']}: Good speed ({tps:.1f} tok/s)")
                elif tps > 0:
                    insights.append(f"⚠️ {m['model_id']}: Slow ({tps:.1f} tok/s)")
            result["insights"] = insights
    except Exception as e:
        result["error"] = str(e)
    
    return json.dumps(result, indent=2)


@mcp.tool()
def analyse_conversation_patterns(days: int = 30) -> str:
    """
    Analyse conversation patterns from Msty database.
    
    Privacy-respecting analysis that tracks session counts, message volumes,
    and model usage distribution without exposing conversation content.
    
    Args:
        days: Number of days to analyse (default: 30)
    
    Returns:
        Aggregated usage patterns
    """
    result = {"timestamp": datetime.now().isoformat(), "period_days": days, "patterns": {}}
    
    paths = get_msty_paths()
    db_path = paths.get("database")
    
    if not db_path:
        result["error"] = "Msty database not found"
        return json.dumps(result, indent=2)
    
    try:
        tables = get_table_names(db_path)
        patterns = {"session_analysis": {}, "model_usage": {}}
        
        for t in ["chat_sessions", "conversations", "chats"]:
            if validate_table_exists(db_path, t):
                patterns["session_analysis"]["total_sessions"] = safe_count_table(db_path, t)

                recent = safe_query_table(db_path, t, limit=100)
                if recent:
                    model_counts = {}
                    for s in recent:
                        model = s.get("model") or s.get("model_id") or s.get("llm_model") or "unknown"
                        model_counts[model] = model_counts.get(model, 0) + 1
                    patterns["model_usage"] = model_counts
                break
        
        result["patterns"] = patterns
    except Exception as e:
        result["error"] = str(e)
    
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def compare_model_responses(
    prompt: str,
    models: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    evaluation_criteria: str = "balanced"
) -> str:
    """
    Send the same prompt to multiple models and compare responses.

    Args:
        prompt: The prompt to send to all models
        models: List of model IDs to compare (None = use all available chat models, max 5)
        system_prompt: Optional system prompt for context
        evaluation_criteria: What to optimise for ("quality", "speed", "balanced")

    Returns:
        Comparison of responses with timing and quality scores
    """
    result = {"timestamp": datetime.now().isoformat(), "prompt": prompt[:200] + "...", "responses": [], "comparison": {}}

    # Check if any service is available
    services = get_available_service_ports()
    any_available = any(s["available"] for s in services.values())

    if not any_available:
        result["error"] = "No Local AI service is running. Start Msty Studio and enable services."
        return json.dumps(result, indent=2)

    # Collect all chat models from all services if not specified
    if not models:
        all_models = []
        for service_name, service_info in services.items():
            if service_info["available"]:
                response = make_api_request("/v1/models", port=service_info["port"])
                if response.get("success"):
                    data = response.get("data", {})
                    if isinstance(data, dict) and "data" in data:
                        for m in data["data"]:
                            model_id = m.get("id", "")
                            # Skip embedding models
                            if "embed" in model_id.lower() or "bge" in model_id.lower() or "nomic" in model_id.lower():
                                continue
                            all_models.append({"id": model_id, "port": service_info["port"]})
        models = all_models[:5]  # Limit to 5 models
        if not models:
            result["error"] = "No chat models available"
            return json.dumps(result, indent=2)

    init_metrics_db()

    for model_entry in models:
        # Handle both dict format (from auto-discovery) and string format (user-provided)
        if isinstance(model_entry, dict):
            model_id = model_entry["id"]
            port = model_entry["port"]
        else:
            model_id = model_entry
            port = get_chat_port_for_model(model_id)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        response = make_api_request("/v1/chat/completions", port=port, method="POST",
            data={"model": model_id, "messages": messages, "temperature": 0.7, "max_tokens": 1024, "stream": False}, timeout=120)
        elapsed = time.time() - start_time
        
        model_result = {"model_id": model_id, "success": response.get("success", False), "latency_seconds": round(elapsed, 2)}
        
        if response.get("success"):
            data = response.get("data", {})
            if "choices" in data and data["choices"]:
                content = data["choices"][0].get("message", {}).get("content", "") or data["choices"][0].get("message", {}).get("reasoning", "")
                model_result["response"] = content[:500] + "..." if len(content) > 500 else content
                model_result["response_length"] = len(content)
                
                if "usage" in data:
                    model_result["tokens_per_second"] = round(data["usage"].get("completion_tokens", 0) / max(elapsed, 0.1), 1)
                
                eval_result = evaluate_response_heuristic(prompt, content, "general")
                model_result["quality_score"] = round(eval_result["score"], 2)
                
                record_model_metric(model_id=model_id, completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
                    latency_seconds=elapsed, success=True, use_case="comparison")
        
        result["responses"].append(model_result)
    
    successful = [r for r in result["responses"] if r["success"]]
    if successful:
        if evaluation_criteria == "speed":
            best = min(successful, key=lambda x: x["latency_seconds"])
        elif evaluation_criteria == "quality":
            best = max(successful, key=lambda x: x.get("quality_score", 0))
        else:
            best = max(successful, key=lambda x: x.get("quality_score", 0.5) * 0.6 + (1.0 / max(x["latency_seconds"], 0.1)) * 0.4)
        result["comparison"]["winner"] = best["model_id"]
    
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def optimise_knowledge_stacks() -> str:
    """
    Analyse and recommend optimisations for knowledge stacks.
    
    Returns:
        Recommendations for knowledge stack improvements
    """
    result = {"timestamp": datetime.now().isoformat(), "analysis": {}, "recommendations": []}
    
    paths = get_msty_paths()
    db_path = paths.get("database")
    
    if not db_path:
        result["error"] = "Msty database not found"
        return json.dumps(result, indent=2)
    
    try:
        tables = get_table_names(db_path)
        ks_table = None
        for t in ["knowledge_stacks", "knowledge_stack"]:
            if t in tables:
                ks_table = t
                break
        
        if not ks_table or not validate_table_exists(db_path, ks_table):
            result["note"] = "No knowledge stack table found"
            return json.dumps(result, indent=2)

        stacks = safe_query_table(db_path, ks_table, limit=1000)
        result["analysis"]["total_stacks"] = len(stacks)
        
        if len(stacks) == 0:
            result["recommendations"].append("No knowledge stacks found. Consider creating domain-specific stacks.")
        elif len(stacks) > 10:
            result["recommendations"].append(f"Many stacks ({len(stacks)}). Consider consolidating.")
    except Exception as e:
        result["error"] = str(e)
    
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def suggest_persona_improvements(persona_name: Optional[str] = None) -> str:
    """
    Analyse personas and suggest improvements.
    
    Args:
        persona_name: Specific persona to analyse (None = all)
    
    Returns:
        Suggestions for persona optimisation
    """
    result = {"timestamp": datetime.now().isoformat(), "analysis": {}, "suggestions": []}
    
    paths = get_msty_paths()
    db_path = paths.get("database")
    
    if not db_path:
        result["error"] = "Msty database not found"
        return json.dumps(result, indent=2)
    
    try:
        tables = get_table_names(db_path)
        persona_table = "personas" if "personas" in tables else None
        
        if not persona_table or not validate_table_exists(db_path, persona_table):
            result["note"] = "No persona table found"
            return json.dumps(result, indent=2)

        if persona_name:
            personas = safe_query_table(db_path, persona_table, limit=1000, where_clause="name LIKE ?", params=(f"%{persona_name}%",))
        else:
            personas = safe_query_table(db_path, persona_table, limit=1000)
        
        result["analysis"]["total_personas"] = len(personas)
        
        for p in personas:
            name = p.get("name", "Unknown")
            prompt_len = len(p.get("system_prompt", "") or p.get("prompt", "") or "")
            temp = p.get("temperature")
            
            if prompt_len < 100:
                result["suggestions"].append(f"'{name}': System prompt too short.")
            elif prompt_len > 4000:
                result["suggestions"].append(f"'{name}': System prompt very long.")
            
            if temp is not None and temp > 0.9:
                result["suggestions"].append(f"'{name}': High temperature ({temp}) may cause inconsistency.")
        
        if len(personas) == 0:
            result["suggestions"].append("No personas found. Create task-specific personas.")
    except Exception as e:
        result["error"] = str(e)
    
    return json.dumps(result, indent=2, default=str)


# =============================================================================
# Phase 5: Tiered AI Workflow / Calibration
# =============================================================================

@mcp.tool()
def run_calibration_test(
    model_id: Optional[str] = None,
    category: str = "general",
    custom_prompt: Optional[str] = None,
    passing_threshold: float = 0.6
) -> str:
    """
    Run a calibration test on a local model.

    Args:
        model_id: Model to test (None = auto-select first available chat model)
        category: Test category ("general", "reasoning", "coding", "writing", "analysis", "creative")
        custom_prompt: Use a custom prompt instead of built-in tests
        passing_threshold: Minimum score to pass (0.0-1.0, default 0.6)

    Returns:
        Test results with quality scores and recommendations
    """
    result = {"timestamp": datetime.now().isoformat(), "category": category, "tests": [], "summary": {}}

    # Check if any service is available
    services = get_available_service_ports()
    any_available = any(s["available"] for s in services.values())

    if not any_available:
        result["error"] = "No Local AI service is running. Start Msty Studio and enable services."
        return json.dumps(result, indent=2)

    # Find model and port
    if model_id:
        port = get_chat_port_for_model(model_id)
    else:
        model_id, port = get_first_chat_model()
        if not model_id:
            result["error"] = "No chat models available (only embedding models found)"
            return json.dumps(result, indent=2)

    result["model_id"] = model_id
    result["port"] = port

    prompts_to_test = []
    if custom_prompt:
        prompts_to_test = [(category, custom_prompt)]
    elif category == "general":
        for cat, prompts in CALIBRATION_PROMPTS.items():
            if prompts:
                prompts_to_test.append((cat, prompts[0]))
    elif category in CALIBRATION_PROMPTS:
        for p in CALIBRATION_PROMPTS[category]:
            prompts_to_test.append((category, p))
    else:
        result["error"] = f"Unknown category: {category}"
        return json.dumps(result, indent=2)

    init_metrics_db()
    passed_count, total_score = 0, 0.0

    for test_cat, prompt in prompts_to_test:
        import hashlib
        test_id = hashlib.md5(f"{model_id}:{prompt}:{datetime.now().isoformat()}".encode()).hexdigest()[:12]

        start_time = time.time()
        response = make_api_request("/v1/chat/completions", port=port, method="POST",
            data={"model": model_id, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 1024, "stream": False}, timeout=120)
        elapsed = time.time() - start_time
        
        test_result = {"test_id": test_id, "category": test_cat, "prompt": prompt[:100] + "...", "latency_seconds": round(elapsed, 2)}
        
        if response.get("success"):
            data = response.get("data", {})
            if "choices" in data and data["choices"]:
                content = data["choices"][0].get("message", {}).get("content", "") or data["choices"][0].get("message", {}).get("reasoning", "")
                
                evaluation = evaluate_response_heuristic(prompt, content, test_cat)
                test_result["quality_score"] = round(evaluation["score"], 2)
                test_result["passed"] = evaluation["score"] >= passing_threshold
                test_result["notes"] = evaluation["notes"]
                
                total_score += evaluation["score"]
                if test_result["passed"]:
                    passed_count += 1
                
                save_calibration_result(test_id, model_id, test_cat, prompt, content, evaluation["score"], 
                    json.dumps(evaluation["notes"]), elapsed, test_result["passed"])
        else:
            test_result["error"] = response.get("error")
            test_result["passed"] = False
        
        result["tests"].append(test_result)
    
    total_tests = len(prompts_to_test)
    result["summary"] = {
        "total_tests": total_tests,
        "passed": passed_count,
        "pass_rate": round(passed_count / max(total_tests, 1) * 100, 1),
        "average_score": round(total_score / max(total_tests, 1), 2)
    }
    
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def evaluate_response_quality(prompt: str, response: str, category: str = "general") -> str:
    """
    Evaluate the quality of a model response.
    
    Args:
        prompt: The original prompt
        response: The model's response
        category: Response category for specific evaluation criteria
    
    Returns:
        Quality score (0.0-1.0) with detailed breakdown
    """
    evaluation = evaluate_response_heuristic(prompt, response, category)
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "quality_score": round(evaluation["score"], 3),
        "passed": evaluation["passed"],
        "criteria_scores": {k: round(v, 2) for k, v in evaluation.get("criteria_scores", {}).items()},
        "notes": evaluation["notes"],
        "rubric": QUALITY_RUBRIC
    }
    
    score = evaluation["score"]
    if score >= 0.8:
        result["interpretation"] = "Excellent"
    elif score >= 0.6:
        result["interpretation"] = "Good"
    elif score >= 0.4:
        result["interpretation"] = "Fair"
    else:
        result["interpretation"] = "Poor"
    
    return json.dumps(result, indent=2)


@mcp.tool()
def identify_handoff_triggers(
    analyse_recent: bool = True,
    add_pattern: Optional[str] = None,
    pattern_type: Optional[str] = None
) -> str:
    """
    Identify and manage patterns that should trigger escalation to Claude.
    
    Args:
        analyse_recent: Analyse recent calibration tests for triggers
        add_pattern: Manually add a trigger pattern description
        pattern_type: Type of pattern ("complexity", "domain", "quality", "safety", "creativity")
    
    Returns:
        List of identified handoff triggers with confidence scores
    """
    result = {"timestamp": datetime.now().isoformat(), "triggers": [], "analysis": {}}
    
    init_metrics_db()
    
    if add_pattern and pattern_type:
        record_handoff_trigger(pattern_type, add_pattern, 0.7)
        result["added_pattern"] = {"type": pattern_type, "description": add_pattern}
    
    if analyse_recent:
        calibration_results = get_calibration_results(limit=100)
        failed = [r for r in calibration_results if not r.get("passed")]
        
        category_failures = {}
        for test in failed:
            cat = test.get("prompt_category", "unknown")
            category_failures[cat] = category_failures.get(cat, 0) + 1
        
        result["analysis"]["failed_tests_count"] = len(failed)
        result["analysis"]["failure_by_category"] = category_failures
        
        for cat, count in category_failures.items():
            if count >= 3:
                record_handoff_trigger("category_failure", f"Local model fails {cat} tasks", min(count / 10, 1.0))
    
    result["triggers"] = get_handoff_triggers(active_only=True)
    result["trigger_count"] = len(result["triggers"])
    
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_calibration_history(
    model_id: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50
) -> str:
    """
    Get historical calibration test results.
    
    Args:
        model_id: Filter by specific model (None = all)
        category: Filter by test category (None = all)
        limit: Maximum results to return (default: 50)
    
    Returns:
        Historical test results with trends and statistics
    """
    result = {"timestamp": datetime.now().isoformat(), "filters": {"model_id": model_id, "category": category}, "history": [], "statistics": {}}
    
    init_metrics_db()
    
    all_results = get_calibration_results(model_id=model_id, limit=limit)
    
    if category:
        all_results = [r for r in all_results if r.get("prompt_category") == category]
    
    result["history"] = all_results
    result["total_tests"] = len(all_results)
    
    if all_results:
        scores = [r.get("quality_score", 0) for r in all_results if r.get("quality_score")]
        passed = sum(1 for r in all_results if r.get("passed"))
        
        result["statistics"] = {
            "average_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "pass_count": passed,
            "pass_rate": round(passed / len(all_results) * 100, 1)
        }
    else:
        result["note"] = "No calibration tests found. Run run_calibration_test to generate data."
    
    return json.dumps(result, indent=2, default=str)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the Msty Admin MCP server"""
    logger.info(f"Starting Msty Admin MCP Server v{SERVER_VERSION}")
    logger.info("Phase 1: Foundational Tools (Read-Only)")
    logger.info("Phase 2: Configuration Management")
    logger.info("Phase 3: Automation Bridge")
    logger.info("Phase 4: Intelligence Layer")
    logger.info("Phase 5: Tiered AI Workflow")
    logger.info("Phase 6: Advanced Model Management")
    logger.info("Phase 7: Conversation Management")
    logger.info("Phase 8: Prompt Templates & Automation")
    logger.info("Phase 9: Backup & System Management")
    logger.info("Phase 10: Knowledge Stack Management")
    logger.info("Phase 11: Model Download/Delete")
    logger.info("Phase 12: Claude ↔ Local Model Bridge")
    logger.info("Phase 13: Turnstile Workflows")
    logger.info("Phase 14: Live Context")
    logger.info("Phase 15: Conversation Analytics")

    # Register extension tools
    register_extension_tools(mcp)

    logger.info("Total tools: 77")
    mcp.run()


if __name__ == "__main__":
    main()
