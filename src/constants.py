"""
Msty Admin MCP - Constants and Configuration

Centralized configuration for the Msty Admin MCP server.
All values can be overridden via environment variables.
"""

import os

# Server version
SERVER_VERSION = "6.6.0"

# Network configuration (configurable via environment)
SIDECAR_HOST = os.environ.get("MSTY_SIDECAR_HOST", "127.0.0.1")
SIDECAR_PROXY_PORT = int(os.environ.get("MSTY_PROXY_PORT", 11932))
LOCAL_AI_SERVICE_PORT = int(os.environ.get("MSTY_AI_PORT", 11964))
SIDECAR_TIMEOUT = int(os.environ.get("MSTY_TIMEOUT", 10))

# Msty 2.4.0+ ports (services built into main app)
MLX_SERVICE_PORT = int(os.environ.get("MSTY_MLX_PORT", 11973))
LLAMACPP_SERVICE_PORT = int(os.environ.get("MSTY_LLAMACPP_PORT", 11454))
VIBE_PROXY_PORT = int(os.environ.get("MSTY_VIBE_PORT", 8317))

# Safe table names allowlist for SQL queries (prevents SQL injection)
ALLOWED_TABLE_NAMES = frozenset([
    "chats", "messages", "personas", "prompts", "mcp_tools", "tools",
    "knowledge_stacks", "models", "settings", "conversations", "users",
    "attachments", "embeddings", "tags", "folders", "providers",
    "chat_sessions", "chat_messages", "prompt_library"
])

# Model tagging patterns
MODEL_SIZE_PATTERNS = {
    "large": ["70b", "72b", "65b", "180b", "405b", "235b", "253b", "110b", "123b"],
    "medium": ["32b", "34b", "27b", "22b", "13b", "14b", "20b"],
    "small": ["7b", "8b", "3b", "4b", "1b", "0.5b", "0.6b", "1.5b", "2b", "6b", "9b", "11b", "12b"]
}

MODEL_CAPABILITY_PATTERNS = {
    "coding": ["code", "coder", "codestral", "starcoder", "deepseek-coder", "codellama", "kimi-dev", "oswe", "grok-code"],
    "reasoning": ["think", "reason", "r1", "qwq", "deepseek-r"],
    "creative": ["creative", "writer", "story", "claude", "gpt"],
    "vision": ["vision", "llava", "bakllava", "moondream", "cogvlm", "pro-image"],
    "embedding": ["embed", "bge", "nomic-embed", "jina-embed"],
    "long_context": ["1m", "128k", "200k", "long", "mistral-nemo"]
}

MODEL_SPEED_PATTERNS = {
    "fast": ["flash", "lite", "haiku", "mini", "tiny", "nano", "small", "fast", "turbo", "instant"]
}

# Export all constants
__all__ = [
    "SERVER_VERSION",
    "SIDECAR_HOST",
    "SIDECAR_PROXY_PORT",
    "LOCAL_AI_SERVICE_PORT",
    "SIDECAR_TIMEOUT",
    "MLX_SERVICE_PORT",
    "LLAMACPP_SERVICE_PORT",
    "VIBE_PROXY_PORT",
    "ALLOWED_TABLE_NAMES",
    "MODEL_SIZE_PATTERNS",
    "MODEL_CAPABILITY_PATTERNS",
    "MODEL_SPEED_PATTERNS"
]
