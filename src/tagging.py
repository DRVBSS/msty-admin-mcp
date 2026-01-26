"""
Msty Admin MCP - Model Tagging System v2.0

Smart model categorization for intelligent selection.
Improved with:
- Context length awareness
- Quantization detection
- MoE (Mixture of Experts) detection
- Better pattern matching
- Architecture-aware tagging
"""

import re
from typing import List, Dict, Optional, Tuple

from .cache import get_cached_models
from .network import get_available_service_ports, make_api_request


# ============================================================================
# TAG DEFINITIONS
# ============================================================================

# All available tags with descriptions
TAG_DEFINITIONS = {
    # Capability tags
    "fast": "Optimized for speed, lower latency",
    "quality": "High-quality outputs, best reasoning",
    "coding": "Specialized for code generation/review",
    "creative": "Good for creative writing, storytelling",
    "reasoning": "Strong logical/analytical capabilities",
    "embedding": "Vector embeddings for RAG/search",
    "vision": "Can process images",
    "long_context": "Large context window (100K+)",
    "very_long_context": "Very large context window (250K+)",
    "massive_context": "Massive context window (500K+)",

    # Size tags
    "small": "Under 10B parameters",
    "medium": "10B-50B parameters",
    "large": "50B+ parameters",
    "massive": "200B+ parameters",

    # Architecture tags
    "moe": "Mixture of Experts architecture",
    "mlx": "Optimized for Apple Silicon (MLX)",
    "gguf": "GGUF quantized format",

    # Quantization tags
    "fp16": "Full FP16 precision",
    "8bit": "8-bit quantization",
    "6bit": "6-bit quantization",
    "5bit": "5-bit quantization",
    "4bit": "4-bit quantization",
    "3bit": "3-bit quantization",

    # Use case tags
    "general": "Good all-around model",
    "chat": "Optimized for conversation",
    "instruct": "Instruction-following",
    "thinking": "Chain-of-thought reasoning",
    "math": "Strong at mathematics",
    "science": "Strong at scientific topics",
}


# ============================================================================
# PATTERN MATCHING
# ============================================================================

# Patterns for detecting capabilities (regex patterns, case-insensitive)
CAPABILITY_PATTERNS = {
    "fast": [
        r"granite.*[23]b",
        r"phi[.-]?[234]?[.-]?mini",
        r"gemma.*[23]b",
        r"qwen.*[01]\.?[56]b",
        r"tiny",
        r"small",
        r"mini(?!stral)",  # mini but not ministral
        r"flash",
        r"lite",
        r"haiku",
        r"turbo",
    ],
    "quality": [
        r"[^a-z]70b",
        r"[^a-z]72b",
        r"[^a-z]80b",
        r"[^a-z]120b",
        r"[^a-z]235b",
        r"[^a-z]253b",
        r"[^a-z]405b",
        r"[^a-z]671b",
        r"opus",
        r"sonnet",
        r"pro(?!xy)",  # pro but not proxy
        r"ultra",
        r"nemotron",
        r"deepseek-v3",
    ],
    "coding": [
        r"coder",
        r"codex",
        r"starcoder",
        r"codellama",
        r"wizardcoder",
        r"phind",
        r"code-",
        r"-code",
        r"dev(?!ice)",  # dev but not device
        r"kimi-dev",
        r"oswe",
        r"grok-code",
        r"qwen.*coder",
    ],
    "creative": [
        r"creative",
        r"writer",
        r"story",
        r"hermes",
        r"nous",
        r"mythomax",
        r"goliath",
    ],
    "reasoning": [
        r"[-_]r1",
        r"thinking",
        r"reason",
        r"[-_]o1",
        r"qwen3",
        r"glm",
        r"nemotron",
        r"deepseek.*v3",
        r"longcat.*thinking",
    ],
    "embedding": [
        r"embed",
        r"bge[-_]",
        r"nomic[-_]embed",
        r"e5[-_]",
        r"gte[-_]",
        r"jina[-_]embed",
        r"text-embedding",
    ],
    "vision": [
        r"vision",
        r"llava",
        r"visual",
        r"image",
        r"cogvlm",
        r"qwen.*vl",
        r"internvl",
    ],
    "long_context": [
        r"longcat",
        r"yarn",
        r"longrope",
        r"nemo",
        r"mistral-nemo",
        r"qwen.*next",  # Qwen3-Next has 262K context
    ],
    "thinking": [
        r"thinking",
        r"[-_]r1",
        r"cot",  # chain of thought
        r"reason",
    ],
    "math": [
        r"math",
        r"wizard.*math",
        r"deepseek.*math",
        r"qwen.*math",
        r"metamath",
    ],
    "moe": [
        r"moe",
        r"mixture",
        r"a\d+b",  # Pattern like A22B (active params in MoE)
        r"mixtral",
        r"dbrx",
        r"grok-1",
    ],
    "instruct": [
        r"instruct",
        r"[-_]it(?:[^a-z]|$)",  # -it or _it at end
        r"chat",
    ],
}

# Quantization patterns (for MLX and GGUF)
QUANT_PATTERNS = {
    "fp16": [r"fp16", r"f16", r"full"],
    "8bit": [r"8bit", r"q8", r"8-bit"],
    "6bit": [r"6bit", r"6\.5bit", r"q6"],
    "5bit": [r"5bit", r"5\.5bit", r"q5"],
    "4bit": [r"4bit", r"q4", r"4-bit", r"q4_k"],
    "3bit": [r"3bit", r"q3", r"3-bit"],
}


# ============================================================================
# SIZE DETECTION
# ============================================================================

def detect_model_size(model_id: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Detect model size from ID.
    Returns (size_tag, estimated_params_billions)
    """
    model_lower = model_id.lower()

    # Extract parameter count - check larger numbers first!
    size_patterns = [
        (r"(\d{3})b", 1),      # 405b, 253b, 235b, 120b, 671b
        (r"(\d{2})b", 1),      # 70b, 72b, 80b, 32b, etc.
        (r"(\d)b", 1),         # 7b, 8b, 3b, etc.
        (r"(\d+)m", 0.001),    # 567m, 137m, etc. (millions)
    ]

    params_b = None
    for pattern, multiplier in size_patterns:
        match = re.search(pattern, model_lower)
        if match:
            params_b = int(match.group(1)) * multiplier
            break

    # Also check for MoE active parameters (e.g., A22B = 22B active)
    moe_match = re.search(r"a(\d+)b", model_lower)
    if moe_match:
        # For MoE, use active params for speed estimation
        active_params = int(moe_match.group(1))
        # But total params still matter for quality
        if params_b and params_b > 100:
            return ("massive", params_b)

    # Determine size category
    if params_b is None:
        return (None, None)
    elif params_b >= 200:
        return ("massive", params_b)
    elif params_b >= 50:
        return ("large", params_b)
    elif params_b >= 10:
        return ("medium", params_b)
    else:
        return ("small", params_b)


# ============================================================================
# CONTEXT LENGTH DETECTION
# ============================================================================

# Known context lengths for specific models
KNOWN_CONTEXT_LENGTHS = {
    # MLX models
    "hermes-4-405b": 131072,
    "qwen3-235b": 40960,
    "qwen3-next-80b": 262144,
    "deepseek-coder-v2": 163840,
    "longcat-flash-thinking": 983040,
    "glm-4.7": 202752,
    "mistral-nemo": 128000,

    # Vibe proxy models
    "claude-opus": 200000,
    "claude-sonnet": 200000,
    "claude-haiku": 200000,
    "gpt-5": 128000,
    "gpt-4": 128000,
    "gemini-2.5-pro": 2000000,
    "gemini-2.5-flash": 1000000,
    "gemini-3": 1000000,
}


def detect_context_length(model_id: str, model_info: dict = None) -> Optional[int]:
    """
    Detect context length from model ID or info.
    Returns context length in tokens or None if unknown.
    """
    # First check if model_info has it
    if model_info and "context_length" in model_info:
        return model_info["context_length"]

    model_lower = model_id.lower()

    # Check known models
    for pattern, length in KNOWN_CONTEXT_LENGTHS.items():
        if pattern in model_lower:
            return length

    # Try to detect from name patterns
    ctx_match = re.search(r"(\d+)k[-_]?ctx", model_lower)
    if ctx_match:
        return int(ctx_match.group(1)) * 1024

    return None


def get_context_tags(context_length: Optional[int]) -> List[str]:
    """Get context-related tags based on length."""
    if context_length is None:
        return []

    tags = []
    if context_length >= 500000:
        tags.append("massive_context")
        tags.append("very_long_context")
        tags.append("long_context")
    elif context_length >= 250000:
        tags.append("very_long_context")
        tags.append("long_context")
    elif context_length >= 100000:
        tags.append("long_context")

    return tags


# ============================================================================
# MANUAL OVERRIDES
# ============================================================================

# Explicit overrides for specific models (takes precedence)
MODEL_OVERRIDES = {
    # MLX Models - Apple Silicon optimized
    "mlx-community/granite-3.3-2b-instruct-4bit": ["fast", "general", "small", "mlx", "4bit"],
    "mlx-community/Qwen3-32B-MLX-4bit": ["quality", "general", "reasoning", "medium", "mlx", "4bit"],
    "mlx-community/Qwen3-235B-A22B-8bit": ["quality", "reasoning", "massive", "moe", "mlx", "8bit"],
    "mlx-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit": ["quality", "reasoning", "large", "moe", "very_long_context", "long_context", "mlx", "4bit"],
    "mlx-community/Hermes-4-405B-MLX-6bit": ["quality", "creative", "reasoning", "massive", "mlx", "6bit", "long_context"],
    "mlx-community/Hermes-4-70B-MLX-4bit": ["quality", "creative", "reasoning", "large", "mlx", "4bit"],
    "mlx-community/Kimi-Dev-72B-4bit-DWQ": ["quality", "coding", "reasoning", "large", "mlx", "4bit"],
    "mlx-community/Mistral-Nemo-Instruct-2407-4bit": ["quality", "long_context", "general", "medium", "mlx", "4bit"],
    "GGorman/DeepSeek-Coder-V2-Instruct-Q4-mlx": ["coding", "quality", "reasoning", "long_context", "mlx", "4bit"],
    "inferencelabs/GLM-4.7-MLX-6.5bit": ["quality", "reasoning", "general", "very_long_context", "mlx", "6bit"],
    "inferencerlabs/LongCat-Flash-Thinking-2601-MLX-5.5bit": ["reasoning", "thinking", "massive_context", "very_long_context", "long_context", "fast", "mlx", "5bit"],

    # GGUF Models
    "DeepSeek/DeepSeek-R1-Distill-Llama-70B-GGUF/DeepSeek-R1-Distill-Llama-70B-Q8_0.gguf": ["quality", "reasoning", "thinking", "large", "gguf", "8bit"],
    "DeepSeek/DeepSeek-V3-0324-UD-Q4_K_XL/DeepSeek-V3-0324-UD-Q4_K_XL-merged.gguf": ["quality", "reasoning", "coding", "massive", "moe", "gguf", "4bit"],
    "DevQuasar/nvidia.Llama-3_1-Nemotron-Ultra-253B-v1-GGUF/nvidia.Llama-3_1-Nemotron-Ultra-253B-v1.Q4_K_M_Dima.gguf": ["quality", "reasoning", "massive", "gguf", "4bit"],
    "Gemma/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf": ["quality", "general", "medium", "gguf", "4bit"],
    "Llama/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-Q8_0.gguf": ["quality", "general", "reasoning", "large", "gguf", "8bit"],
    "OpenAi/gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4_Dima.gguf": ["quality", "reasoning", "coding", "large", "gguf", "4bit"],
    "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/qwen2.5-coder-32b-instruct-fp16.gguf": ["coding", "quality", "medium", "gguf", "fp16"],
    "mradermacher/Qwen3-72B-Instruct-2-i1-GGUF/Qwen3-72B-Instruct-2.i1-Q5_K_M.gguf": ["quality", "reasoning", "large", "gguf", "5bit"],
    "yinghy2018/CLIENS_PHI4_14B_BF16-Q8_0-GGUF/cliens_phi4_14b_bf16-q8_0.gguf": ["fast", "general", "medium", "gguf", "8bit"],
    "jinaai/jina-embeddings-v4-text-matching-GGUF/jina-embeddings-v4-text-matching-F16.gguf": ["embedding", "gguf", "fp16"],

    # Embedding models
    "bge-m3:567m-fp16": ["embedding", "small", "fp16"],
    "nomic-embed-text:137m-v1.5-fp16": ["embedding", "small", "fp16", "fast"],

    # Claude Models (Anthropic)
    "claude-opus-4-5-20251101": ["quality", "reasoning", "coding", "creative", "massive", "long_context"],
    "claude-opus-4.5": ["quality", "reasoning", "coding", "creative", "massive", "long_context"],
    "claude-opus-4-20250514": ["quality", "reasoning", "coding", "creative", "massive", "long_context"],
    "claude-opus-4.1": ["quality", "reasoning", "coding", "creative", "massive", "long_context"],
    "claude-opus-4-1-20250805": ["quality", "reasoning", "coding", "creative", "massive", "long_context"],
    "claude-sonnet-4.5": ["quality", "reasoning", "coding", "creative", "long_context"],
    "claude-sonnet-4-5-20250929": ["quality", "reasoning", "coding", "creative", "long_context"],
    "claude-sonnet-4": ["quality", "reasoning", "coding", "long_context"],
    "claude-sonnet-4-20250514": ["quality", "reasoning", "coding", "long_context"],
    "claude-3-7-sonnet-20250219": ["quality", "reasoning", "coding", "long_context"],
    "claude-haiku-4.5": ["fast", "general", "coding", "long_context"],
    "claude-haiku-4-5-20251001": ["fast", "general", "coding", "long_context"],
    "claude-3-5-haiku-20241022": ["fast", "general", "long_context"],

    # GPT Models (OpenAI)
    "gpt-5": ["quality", "reasoning", "general", "massive", "long_context"],
    "gpt-5.1": ["quality", "reasoning", "general", "massive", "long_context"],
    "gpt-5.2": ["quality", "reasoning", "general", "massive", "long_context"],
    "gpt-5-mini": ["fast", "general", "long_context"],
    "gpt-5-codex": ["coding", "quality", "long_context"],
    "gpt-5.1-codex": ["coding", "quality", "long_context"],
    "gpt-5.1-codex-mini": ["coding", "fast", "long_context"],
    "gpt-5.1-codex-max": ["coding", "quality", "massive", "long_context"],
    "gpt-5.2-codex": ["coding", "quality", "long_context"],
    "gpt-4.1": ["quality", "reasoning", "general", "long_context"],
    "gpt-oss-120b-medium": ["quality", "reasoning", "large"],

    # Gemini Models (Google)
    "gemini-2.5-pro": ["quality", "reasoning", "massive_context", "long_context"],
    "gemini-2.5-flash": ["fast", "general", "very_long_context"],
    "gemini-2.5-flash-lite": ["fast", "general", "long_context"],
    "gemini-3-pro-preview": ["quality", "reasoning", "very_long_context"],
    "gemini-3-flash-preview": ["fast", "general", "very_long_context"],
    "gemini-3-pro-image-preview": ["quality", "vision", "very_long_context"],

    # Hybrid/Other Models
    "gemini-claude-sonnet-4-5": ["quality", "reasoning", "coding"],
    "gemini-claude-sonnet-4-5-thinking": ["quality", "reasoning", "coding", "thinking"],
    "gemini-claude-opus-4-5-thinking": ["quality", "reasoning", "coding", "massive", "thinking"],
    "grok-code-fast-1": ["coding", "fast"],
    "oswe-vscode-prime": ["coding", "quality"],
    "tab_flash_lite_preview": ["fast", "coding"],
}


# ============================================================================
# MAIN TAGGING FUNCTION
# ============================================================================

def get_model_tags(model_id: str, model_info: dict = None) -> List[str]:
    """
    Get tags for a model based on its ID and optional info.

    Args:
        model_id: The model identifier
        model_info: Optional dict with model metadata (context_length, etc.)

    Returns:
        List of tags describing the model's capabilities
    """
    tags = set()
    model_lower = model_id.lower()

    # 1. Check manual overrides first (highest priority)
    if model_id in MODEL_OVERRIDES:
        return MODEL_OVERRIDES[model_id]

    # 2. Pattern-based capability detection
    for tag, patterns in CAPABILITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, model_lower):
                tags.add(tag)
                break

    # 3. Size detection
    size_tag, params = detect_model_size(model_id)
    if size_tag:
        tags.add(size_tag)

    # 4. Quantization detection
    for quant_tag, patterns in QUANT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, model_lower):
                tags.add(quant_tag)
                break

    # 5. Context length detection
    context_length = detect_context_length(model_id, model_info)
    context_tags = get_context_tags(context_length)
    tags.update(context_tags)

    # 6. Format/architecture detection
    if "mlx" in model_lower or "mlx-community" in model_lower:
        tags.add("mlx")
    if ".gguf" in model_lower:
        tags.add("gguf")

    # 7. Default to general if no capability tags
    capability_tags = {"fast", "quality", "coding", "creative", "reasoning",
                       "embedding", "vision", "math", "science", "thinking"}
    if not tags.intersection(capability_tags):
        tags.add("general")

    return list(tags)


def find_models_by_tag(tag: str, models: List[dict] = None) -> List[dict]:
    """
    Find models matching a specific tag.
    If models not provided, fetches from cache or API.
    """
    if models is None:
        cached = get_cached_models()
        if cached:
            models = cached.get("models", [])
        else:
            # Fetch fresh
            services = get_available_service_ports()
            models = []
            for service_name, service_info in services.items():
                if service_info["available"]:
                    response = make_api_request("/v1/models", port=service_info["port"])
                    if response.get("success"):
                        data = response.get("data", {})
                        if isinstance(data, dict) and "data" in data:
                            for m in data["data"]:
                                m["_service"] = service_name
                                m["_port"] = service_info["port"]
                            models.extend(data["data"])

    matching = []
    for model in models:
        model_id = model.get("id", "")
        model_tags = get_model_tags(model_id, model)
        if tag in model_tags:
            model["_tags"] = model_tags
            matching.append(model)

    return matching


def get_all_tags() -> Dict[str, str]:
    """Return all available tags with descriptions."""
    return TAG_DEFINITIONS.copy()


# Legacy compatibility
MODEL_TAGS = {
    "patterns": CAPABILITY_PATTERNS,
    "overrides": MODEL_OVERRIDES,
}


__all__ = [
    "MODEL_TAGS",
    "TAG_DEFINITIONS",
    "get_model_tags",
    "find_models_by_tag",
    "get_all_tags",
    "detect_model_size",
    "detect_context_length",
]
