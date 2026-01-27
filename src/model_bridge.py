"""
Msty Admin MCP - Claude ↔ Local Model Bridge

Enables Claude to orchestrate local models through Msty for:
- Cost-effective task delegation
- Multi-model consensus
- Draft-refine workflows
- Parallel processing
"""

import json
import logging
import time
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .network import make_api_request, get_available_service_ports
from .tagging import get_model_tags, find_models_by_tag
from .cache import get_cache

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# MODEL SELECTION
# ============================================================================

def select_best_model_for_task(
    task_type: str,
    prefer_speed: bool = False,
    prefer_quality: bool = False,
    max_size: str = None,
) -> Dict[str, Any]:
    """
    Intelligently select the best local model for a task.

    Args:
        task_type: Type of task (coding, writing, reasoning, general, fast)
        prefer_speed: Prioritize faster models
        prefer_quality: Prioritize quality models
        max_size: Maximum model size (small, medium, large, massive)

    Returns:
        Dict with recommended model and reasoning
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "task_type": task_type,
        "recommendation": None,
        "reasoning": [],
    }

    # Get available models
    services = get_available_service_ports()
    available_models = []

    for service_name, service_info in services.items():
        if service_info.get("available"):
            response = make_api_request("/v1/models", port=service_info["port"])
            if response.get("success"):
                data = response.get("data", {})
                if isinstance(data, dict) and "data" in data:
                    for model in data["data"]:
                        model["_service"] = service_name
                        model["_port"] = service_info["port"]
                        model["_tags"] = get_model_tags(model.get("id", ""), model)
                        available_models.append(model)

    if not available_models:
        result["error"] = "No models available"
        return result

    # Filter by task type
    task_tag_map = {
        "coding": ["coding"],
        "writing": ["creative", "quality"],
        "reasoning": ["reasoning", "thinking"],
        "math": ["math", "reasoning"],
        "general": ["general", "quality"],
        "fast": ["fast"],
    }

    target_tags = task_tag_map.get(task_type, ["general"])

    # Score models
    scored_models = []
    for model in available_models:
        tags = model.get("_tags", [])
        score = 0

        # Task match score
        for tag in target_tags:
            if tag in tags:
                score += 10

        # Speed preference
        if prefer_speed:
            if "fast" in tags:
                score += 5
            if "small" in tags:
                score += 3
            if "massive" in tags or "large" in tags:
                score -= 2

        # Quality preference
        if prefer_quality:
            if "quality" in tags:
                score += 5
            if "massive" in tags:
                score += 3
            if "large" in tags:
                score += 2
            if "small" in tags:
                score -= 2

        # Size filter
        if max_size:
            size_order = ["small", "medium", "large", "massive"]
            max_idx = size_order.index(max_size) if max_size in size_order else 3
            model_sizes = [s for s in size_order if s in tags]
            if model_sizes:
                model_idx = size_order.index(model_sizes[0])
                if model_idx > max_idx:
                    score -= 100  # Effectively exclude

        # MLX bonus for Apple Silicon
        if "mlx" in tags:
            score += 2

        scored_models.append({
            "model": model,
            "score": score,
        })

    # Sort by score
    scored_models.sort(key=lambda x: x["score"], reverse=True)

    if scored_models and scored_models[0]["score"] > -50:
        best = scored_models[0]["model"]
        result["recommendation"] = {
            "model_id": best.get("id"),
            "service": best.get("_service"),
            "port": best.get("_port"),
            "tags": best.get("_tags"),
            "context_length": best.get("context_length"),
        }
        result["reasoning"] = [
            f"Selected for {task_type} task",
            f"Tags: {', '.join(best.get('_tags', []))}",
        ]
        if prefer_speed:
            result["reasoning"].append("Optimized for speed")
        if prefer_quality:
            result["reasoning"].append("Optimized for quality")

        # Also return alternatives
        result["alternatives"] = [
            {
                "model_id": m["model"]["id"],
                "score": m["score"],
                "tags": m["model"].get("_tags", []),
            }
            for m in scored_models[1:4]
        ]

    else:
        result["error"] = "No suitable model found for this task"

    return result


# ============================================================================
# TASK DELEGATION
# ============================================================================

def delegate_to_local_model(
    prompt: str,
    task_type: str = "general",
    model_id: str = None,
    system_prompt: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """
    Delegate a task to a local model.

    Args:
        prompt: The prompt to send
        task_type: Type of task for auto-selection
        model_id: Specific model to use (optional)
        system_prompt: System prompt (optional)
        temperature: Sampling temperature
        max_tokens: Maximum response tokens

    Returns:
        Dict with model response and metadata
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "task_type": task_type,
        "response": None,
        "model_used": None,
        "timing": {},
    }

    start_time = time.time()

    # Select model if not specified
    if not model_id:
        selection = select_best_model_for_task(task_type)
        if "error" in selection:
            result["error"] = selection["error"]
            return result
        rec = selection.get("recommendation", {})
        model_id = rec.get("model_id")
        port = rec.get("port")
        result["model_selection"] = selection
    else:
        # Find the port for the specified model
        services = get_available_service_ports()
        port = None
        for service_name, service_info in services.items():
            if service_info.get("available"):
                response = make_api_request("/v1/models", port=service_info["port"])
                if response.get("success"):
                    data = response.get("data", {})
                    if isinstance(data, dict) and "data" in data:
                        for model in data["data"]:
                            if model.get("id") == model_id:
                                port = service_info["port"]
                                break
                if port:
                    break

    if not port:
        result["error"] = f"Model '{model_id}' not found in any service"
        return result

    result["model_used"] = model_id
    result["port"] = port

    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Make request
    selection_time = time.time()
    result["timing"]["selection_seconds"] = round(selection_time - start_time, 3)

    response = make_api_request(
        "/v1/chat/completions",
        port=port,
        method="POST",
        data={
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )

    inference_time = time.time()
    result["timing"]["inference_seconds"] = round(inference_time - selection_time, 3)
    result["timing"]["total_seconds"] = round(inference_time - start_time, 3)

    if response.get("success"):
        data = response.get("data", {})
        if "choices" in data and data["choices"]:
            result["response"] = data["choices"][0]["message"]["content"]
            result["usage"] = data.get("usage", {})
        else:
            result["error"] = "Empty response from model"
    else:
        result["error"] = response.get("error", "Request failed")

    return result


# ============================================================================
# MULTI-MODEL CONSENSUS
# ============================================================================

def multi_model_consensus(
    prompt: str,
    models: List[str] = None,
    num_models: int = 3,
    task_type: str = "reasoning",
    system_prompt: str = None,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Query multiple models and synthesize a consensus response.

    Args:
        prompt: The prompt to send to all models
        models: Specific model IDs (optional)
        num_models: Number of models if auto-selecting
        task_type: Task type for model selection
        system_prompt: System prompt for all models
        temperature: Sampling temperature

    Returns:
        Dict with individual responses and consensus analysis
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
        "responses": [],
        "consensus": None,
        "agreement_analysis": {},
    }

    start_time = time.time()

    # Get models to use
    if not models:
        # Auto-select diverse models
        services = get_available_service_ports()
        available = []
        for service_name, service_info in services.items():
            if service_info.get("available"):
                response = make_api_request("/v1/models", port=service_info["port"])
                if response.get("success"):
                    data = response.get("data", {})
                    if isinstance(data, dict) and "data" in data:
                        for model in data["data"]:
                            tags = get_model_tags(model.get("id", ""))
                            # Skip embedding models
                            if "embedding" not in tags:
                                available.append({
                                    "id": model.get("id"),
                                    "port": service_info["port"],
                                    "tags": tags,
                                })

        # Select diverse models
        models = []
        selected_families = set()
        for m in available:
            # Get model family (first part of name)
            family = m["id"].split("/")[-1].split("-")[0].lower()
            if family not in selected_families and len(models) < num_models:
                models.append(m)
                selected_families.add(family)

    if len(models) < 2:
        result["error"] = "Need at least 2 models for consensus"
        return result

    result["models_used"] = [m["id"] if isinstance(m, dict) else m for m in models]

    # Query models in parallel
    def query_model(model_info):
        if isinstance(model_info, dict):
            model_id = model_info["id"]
            port = model_info["port"]
        else:
            # Find port for model ID
            model_id = model_info
            port = None
            services = get_available_service_ports()
            for svc in services.values():
                if svc.get("available"):
                    port = svc["port"]
                    break

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = make_api_request(
            "/v1/chat/completions",
            port=port,
            method="POST",
            data={
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2048,
            },
            timeout=120,
        )

        if response.get("success"):
            data = response.get("data", {})
            if "choices" in data and data["choices"]:
                return {
                    "model_id": model_id,
                    "response": data["choices"][0]["message"]["content"],
                    "success": True,
                }

        return {
            "model_id": model_id,
            "response": None,
            "success": False,
            "error": response.get("error", "Failed"),
        }

    with ThreadPoolExecutor(max_workers=min(len(models), 5)) as executor:
        futures = [executor.submit(query_model, m) for m in models]
        for future in as_completed(futures):
            result["responses"].append(future.result())

    # Analyze agreement
    successful_responses = [r for r in result["responses"] if r.get("success")]

    if len(successful_responses) >= 2:
        # Simple agreement analysis based on key terms
        all_words = []
        for resp in successful_responses:
            words = set(resp["response"].lower().split())
            all_words.append(words)

        # Find common words (excluding stop words)
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "must", "shall",
                      "can", "to", "of", "in", "for", "on", "with", "at", "by",
                      "from", "as", "into", "through", "during", "before", "after",
                      "above", "below", "between", "under", "again", "further",
                      "then", "once", "here", "there", "when", "where", "why",
                      "how", "all", "each", "few", "more", "most", "other", "some",
                      "such", "no", "nor", "not", "only", "own", "same", "so",
                      "than", "too", "very", "just", "and", "but", "if", "or",
                      "because", "until", "while", "this", "that", "these", "those",
                      "i", "you", "he", "she", "it", "we", "they", "what", "which"}

        if all_words:
            common = all_words[0]
            for words in all_words[1:]:
                common = common.intersection(words)
            common = common - stop_words

            result["agreement_analysis"] = {
                "models_responded": len(successful_responses),
                "common_key_terms": list(common)[:20],
                "response_lengths": [len(r["response"]) for r in successful_responses],
            }

        # Create synthesis prompt
        result["consensus"] = {
            "note": "Multiple model responses collected",
            "synthesis_suggestion": "Review the responses above and synthesize the key points of agreement",
            "response_count": len(successful_responses),
        }

    result["timing"] = {
        "total_seconds": round(time.time() - start_time, 3),
    }

    return result


# ============================================================================
# DRAFT-REFINE WORKFLOW
# ============================================================================

def draft_and_refine(
    prompt: str,
    draft_model: str = None,
    refine_instructions: str = None,
    task_type: str = "writing",
) -> Dict[str, Any]:
    """
    Use a local model for drafting, preparing for Claude to refine.

    This creates a cost-effective workflow where:
    1. Local model creates initial draft (free/cheap)
    2. Claude refines and polishes (paid but efficient)

    Args:
        prompt: The original task prompt
        draft_model: Model to use for drafting (auto-selects if None)
        refine_instructions: Instructions for the refinement step
        task_type: Type of task

    Returns:
        Dict with draft and refinement prompt
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "workflow": "draft_and_refine",
        "draft": None,
        "refinement_prompt": None,
    }

    # Get draft from local model
    draft_result = delegate_to_local_model(
        prompt=prompt,
        task_type=task_type,
        model_id=draft_model,
        temperature=0.7,
        max_tokens=4096,
    )

    if "error" in draft_result:
        result["error"] = f"Draft failed: {draft_result['error']}"
        return result

    result["draft"] = {
        "content": draft_result["response"],
        "model_used": draft_result["model_used"],
        "timing": draft_result["timing"],
    }

    # Create refinement prompt for Claude
    default_refine = """Please review and improve this draft:
- Fix any errors or inconsistencies
- Improve clarity and flow
- Enhance the quality while preserving the core content
- Make it more polished and professional"""

    refine_instructions = refine_instructions or default_refine

    result["refinement_prompt"] = f"""{refine_instructions}

---
ORIGINAL TASK: {prompt}

---
DRAFT FROM LOCAL MODEL ({draft_result['model_used']}):

{draft_result['response']}

---
Please provide your refined version:"""

    result["instructions"] = [
        "1. Review the draft above",
        "2. Use the refinement_prompt to ask Claude to improve it",
        "3. This saves cost by using local model for initial work",
    ]

    return result


# ============================================================================
# PARALLEL TASK PROCESSING
# ============================================================================

def parallel_process_tasks(
    tasks: List[Dict[str, Any]],
    max_concurrent: int = 3,
) -> Dict[str, Any]:
    """
    Process multiple tasks in parallel using local models.

    Args:
        tasks: List of task dicts with 'prompt' and optional 'task_type', 'model_id'
        max_concurrent: Maximum concurrent requests

    Returns:
        Dict with all task results
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "total_tasks": len(tasks),
        "completed": 0,
        "failed": 0,
        "results": [],
    }

    start_time = time.time()

    def process_task(task_info):
        task_idx = task_info.get("_index", 0)
        return {
            "index": task_idx,
            "result": delegate_to_local_model(
                prompt=task_info.get("prompt", ""),
                task_type=task_info.get("task_type", "general"),
                model_id=task_info.get("model_id"),
                system_prompt=task_info.get("system_prompt"),
                temperature=task_info.get("temperature", 0.7),
                max_tokens=task_info.get("max_tokens", 2048),
            )
        }

    # Handle empty task list
    if not tasks:
        result["timing"] = {"total_seconds": 0, "average_per_task": 0}
        return result

    # Add index to tasks
    for i, task in enumerate(tasks):
        task["_index"] = i

    with ThreadPoolExecutor(max_workers=min(max_concurrent, len(tasks))) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        for future in as_completed(futures):
            task_result = future.result()
            result["results"].append(task_result)
            if "error" in task_result.get("result", {}):
                result["failed"] += 1
            else:
                result["completed"] += 1

    # Sort by original index
    result["results"].sort(key=lambda x: x["index"])

    result["timing"] = {
        "total_seconds": round(time.time() - start_time, 3),
        "average_per_task": round((time.time() - start_time) / len(tasks), 3) if tasks else 0,
    }

    return result


__all__ = [
    "select_best_model_for_task",
    "delegate_to_local_model",
    "multi_model_consensus",
    "draft_and_refine",
    "parallel_process_tasks",
]
