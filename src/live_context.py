"""
Msty Admin MCP - Live Context Tools

Live Context is Msty's feature for providing real-time system information
to AI models. This module provides tools for managing and utilizing live context.
"""

import json
import logging
import platform
import subprocess
import psutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from .paths import get_msty_paths

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# SYSTEM CONTEXT PROVIDERS
# ============================================================================

def get_system_context() -> Dict[str, Any]:
    """
    Get comprehensive system context information.

    Returns:
        Dict with system information useful for AI context
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "context_type": "system",
    }

    # Basic system info
    result["system"] = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
    }

    # Memory info
    try:
        memory = psutil.virtual_memory()
        result["memory"] = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent_used": memory.percent,
        }
    except Exception as e:
        result["memory"] = {"error": str(e)}

    # CPU info
    try:
        result["cpu"] = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "percent_used": psutil.cpu_percent(interval=0.1),
            "frequency_mhz": getattr(psutil.cpu_freq(), 'current', None) if psutil.cpu_freq() else None,
        }
    except Exception as e:
        result["cpu"] = {"error": str(e)}

    # Disk info
    try:
        disk = psutil.disk_usage('/')
        result["disk"] = {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "percent_used": round(disk.percent, 1),
        }
    except Exception as e:
        result["disk"] = {"error": str(e)}

    return result


def get_datetime_context() -> Dict[str, Any]:
    """
    Get current date/time context.

    Returns:
        Dict with detailed datetime information
    """
    now = datetime.now()

    return {
        "timestamp": now.isoformat(),
        "context_type": "datetime",
        "datetime": {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "month": now.strftime("%B"),
            "year": now.year,
            "week_number": now.isocalendar()[1],
            "day_of_year": now.timetuple().tm_yday,
            "timezone": str(now.astimezone().tzinfo),
            "utc_offset": now.astimezone().strftime("%z"),
            "is_dst": bool(now.dst()) if now.dst() is not None else None,
        },
        "formatted": {
            "short": now.strftime("%m/%d/%Y %I:%M %p"),
            "long": now.strftime("%A, %B %d, %Y at %I:%M %p"),
            "iso": now.isoformat(),
        }
    }


def get_environment_context() -> Dict[str, Any]:
    """
    Get environment context (safe environment variables).

    Returns:
        Dict with safe environment information
    """
    import os

    # Only include safe, non-sensitive environment variables
    safe_vars = [
        "HOME", "USER", "SHELL", "LANG", "LC_ALL", "TERM",
        "PATH", "PWD", "EDITOR", "VISUAL", "PAGER",
        "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"
    ]

    env_info = {}
    for var in safe_vars:
        value = os.environ.get(var)
        if value:
            # Truncate PATH to avoid verbosity
            if var == "PATH":
                paths = value.split(os.pathsep)
                env_info[var] = f"{len(paths)} directories"
            else:
                env_info[var] = value

    return {
        "timestamp": datetime.now().isoformat(),
        "context_type": "environment",
        "environment": env_info,
        "working_directory": os.getcwd(),
    }


def get_process_context() -> Dict[str, Any]:
    """
    Get context about running processes.

    Returns:
        Dict with process information
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "context_type": "processes",
    }

    try:
        # Get interesting processes
        interesting = ["msty", "python", "node", "ollama", "mlx", "llama"]
        found_processes = []

        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                pinfo = proc.info
                name_lower = pinfo['name'].lower()
                if any(interest in name_lower for interest in interesting):
                    found_processes.append({
                        "pid": pinfo['pid'],
                        "name": pinfo['name'],
                        "cpu_percent": pinfo.get('cpu_percent', 0),
                        "memory_percent": round(pinfo.get('memory_percent', 0), 2),
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        result["relevant_processes"] = found_processes[:20]  # Limit
        result["total_processes"] = len(list(psutil.process_iter()))

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================================
# MSTY-SPECIFIC CONTEXT
# ============================================================================

def get_msty_context() -> Dict[str, Any]:
    """
    Get Msty-specific context information.

    Returns:
        Dict with Msty installation and status context
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "context_type": "msty",
    }

    paths = get_msty_paths()
    result["paths"] = paths

    # Check Msty processes
    msty_running = False
    sidecar_running = False

    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name'].lower()
            if 'msty' in name:
                msty_running = True
            if 'sidecar' in name:
                sidecar_running = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    result["status"] = {
        "msty_running": msty_running,
        "sidecar_running": sidecar_running,
    }

    # Check data directory stats
    data_dir = paths.get("data_dir")
    if data_dir:
        data_path = Path(data_dir)
        if data_path.exists():
            result["data_stats"] = {
                "exists": True,
                "subdirectories": [d.name for d in data_path.iterdir() if d.is_dir()][:10],
            }
        else:
            result["data_stats"] = {"exists": False}

    return result


# ============================================================================
# CONTEXT AGGREGATION
# ============================================================================

def get_full_live_context(
    include_system: bool = True,
    include_datetime: bool = True,
    include_environment: bool = False,
    include_processes: bool = False,
    include_msty: bool = True
) -> Dict[str, Any]:
    """
    Get aggregated live context from multiple providers.

    Args:
        include_system: Include system info
        include_datetime: Include datetime info
        include_environment: Include environment info
        include_processes: Include process info
        include_msty: Include Msty-specific info

    Returns:
        Dict with aggregated context
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "context_type": "full",
        "components": [],
    }

    if include_system:
        result["system"] = get_system_context()
        result["components"].append("system")

    if include_datetime:
        result["datetime"] = get_datetime_context()
        result["components"].append("datetime")

    if include_environment:
        result["environment"] = get_environment_context()
        result["components"].append("environment")

    if include_processes:
        result["processes"] = get_process_context()
        result["components"].append("processes")

    if include_msty:
        result["msty"] = get_msty_context()
        result["components"].append("msty")

    return result


def format_context_for_prompt(
    context: Dict[str, Any],
    style: str = "markdown"
) -> str:
    """
    Format context information for inclusion in AI prompts.

    Args:
        context: Context dictionary from any context provider
        style: Output style ("markdown", "text", "json")

    Returns:
        Formatted string suitable for prompt injection
    """
    if style == "json":
        return json.dumps(context, indent=2)

    if style == "text":
        lines = []
        lines.append(f"Context as of {context.get('timestamp', 'unknown')}")
        lines.append("-" * 40)

        def flatten(d, prefix=""):
            for key, value in d.items():
                if key in ["timestamp", "context_type", "components"]:
                    continue
                if isinstance(value, dict):
                    lines.append(f"{prefix}{key}:")
                    flatten(value, prefix + "  ")
                else:
                    lines.append(f"{prefix}{key}: {value}")

        flatten(context)
        return "\n".join(lines)

    # Default: markdown
    lines = []
    lines.append(f"## Live Context ({context.get('timestamp', 'unknown')})")
    lines.append("")

    def format_section(name, data, level=3):
        if not isinstance(data, dict):
            return
        header = "#" * level
        lines.append(f"{header} {name.replace('_', ' ').title()}")
        lines.append("")
        for key, value in data.items():
            if key in ["timestamp", "context_type", "error"]:
                continue
            if isinstance(value, dict):
                format_section(key, value, level + 1)
            else:
                lines.append(f"- **{key}**: {value}")
        lines.append("")

    for key, value in context.items():
        if key in ["timestamp", "context_type", "components"]:
            continue
        if isinstance(value, dict):
            format_section(key, value)

    return "\n".join(lines)


# ============================================================================
# CONTEXT CACHING & MANAGEMENT
# ============================================================================

class ContextCache:
    """Simple in-memory context cache with TTL."""

    def __init__(self, ttl_seconds: int = 60):
        self.ttl = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, datetime] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached context if not expired."""
        if key not in self._cache:
            return None

        cached_time = self._timestamps.get(key)
        if cached_time:
            age = (datetime.now() - cached_time).total_seconds()
            if age < self.ttl:
                return self._cache[key]

        # Expired
        del self._cache[key]
        del self._timestamps[key]
        return None

    def set(self, key: str, value: Dict[str, Any]):
        """Cache a context value."""
        self._cache[key] = value
        self._timestamps[key] = datetime.now()

    def clear(self):
        """Clear all cached contexts."""
        self._cache.clear()
        self._timestamps.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "ttl_seconds": self.ttl,
            "keys": list(self._cache.keys())
        }


# Global cache instance
_context_cache = ContextCache(ttl_seconds=30)


def get_cached_context(context_type: str = "full") -> Dict[str, Any]:
    """
    Get context with caching for performance.

    Args:
        context_type: Type of context ("full", "system", "datetime", "msty")

    Returns:
        Context dictionary (cached if available)
    """
    cached = _context_cache.get(context_type)
    if cached:
        cached["_cached"] = True
        return cached

    # Generate fresh context
    if context_type == "system":
        context = get_system_context()
    elif context_type == "datetime":
        context = get_datetime_context()
    elif context_type == "msty":
        context = get_msty_context()
    elif context_type == "environment":
        context = get_environment_context()
    elif context_type == "processes":
        context = get_process_context()
    else:
        context = get_full_live_context()

    _context_cache.set(context_type, context)
    context["_cached"] = False
    return context


def clear_context_cache() -> Dict[str, Any]:
    """
    Clear the context cache.

    Returns:
        Dict with confirmation
    """
    stats = _context_cache.stats()
    _context_cache.clear()
    return {
        "timestamp": datetime.now().isoformat(),
        "cleared": True,
        "previous_entries": stats["entries"]
    }


def get_context_cache_stats() -> Dict[str, Any]:
    """
    Get context cache statistics.

    Returns:
        Dict with cache info
    """
    return {
        "timestamp": datetime.now().isoformat(),
        **_context_cache.stats()
    }


__all__ = [
    "get_system_context",
    "get_datetime_context",
    "get_environment_context",
    "get_process_context",
    "get_msty_context",
    "get_full_live_context",
    "format_context_for_prompt",
    "get_cached_context",
    "clear_context_cache",
    "get_context_cache_stats",
    "ContextCache",
]
