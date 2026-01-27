"""
Msty Admin MCP - Model Management

Tools for downloading, deleting, and managing local AI models.
"""

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import urllib.request
import urllib.error

from .paths import get_msty_paths
from .network import make_api_request, get_available_service_ports
from .tagging import get_model_tags

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# MODEL STORAGE PATHS
# ============================================================================

def get_mlx_models_path() -> Optional[Path]:
    """Get the path to MLX models storage."""
    paths = get_msty_paths()
    data_dir = paths.get("data_dir")
    if data_dir:
        mlx_path = Path(data_dir) / "models-mlx" / "hub"
        if mlx_path.exists():
            return mlx_path
        # Also check without /hub
        mlx_path = Path(data_dir) / "models-mlx"
        if mlx_path.exists():
            return mlx_path
    return None


def get_gguf_models_path() -> Optional[Path]:
    """Get the path to GGUF models storage."""
    paths = get_msty_paths()
    data_dir = paths.get("data_dir")
    if data_dir:
        # Common GGUF paths
        for subdir in ["models-gguf", "models", "gguf"]:
            gguf_path = Path(data_dir) / subdir
            if gguf_path.exists():
                return gguf_path
    return None


def get_ollama_models_path() -> Optional[Path]:
    """Get the path to Ollama models (Local AI service)."""
    # Standard Ollama path on macOS
    ollama_path = Path.home() / ".ollama" / "models"
    if ollama_path.exists():
        return ollama_path
    return None


# ============================================================================
# MODEL INVENTORY
# ============================================================================

def get_local_model_inventory() -> Dict[str, Any]:
    """
    Get complete inventory of all local models with file details.

    Returns:
        Dict with models categorized by type (MLX, GGUF, Ollama)
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "models": {
            "mlx": [],
            "gguf": [],
            "ollama": [],
        },
        "summary": {
            "total_models": 0,
            "total_size_gb": 0,
            "by_type": {},
        }
    }

    # Scan MLX models
    mlx_path = get_mlx_models_path()
    if mlx_path:
        mlx_models = _scan_mlx_models(mlx_path)
        result["models"]["mlx"] = mlx_models
        result["summary"]["by_type"]["mlx"] = {
            "count": len(mlx_models),
            "size_gb": sum(m.get("size_gb", 0) for m in mlx_models),
        }

    # Scan GGUF models
    gguf_path = get_gguf_models_path()
    if gguf_path:
        gguf_models = _scan_gguf_models(gguf_path)
        result["models"]["gguf"] = gguf_models
        result["summary"]["by_type"]["gguf"] = {
            "count": len(gguf_models),
            "size_gb": sum(m.get("size_gb", 0) for m in gguf_models),
        }

    # Scan Ollama models
    ollama_path = get_ollama_models_path()
    if ollama_path:
        ollama_models = _scan_ollama_models(ollama_path)
        result["models"]["ollama"] = ollama_models
        result["summary"]["by_type"]["ollama"] = {
            "count": len(ollama_models),
            "size_gb": sum(m.get("size_gb", 0) for m in ollama_models),
        }

    # Calculate totals
    for model_type in result["models"].values():
        result["summary"]["total_models"] += len(model_type)
        result["summary"]["total_size_gb"] += sum(m.get("size_gb", 0) for m in model_type)

    result["summary"]["total_size_gb"] = round(result["summary"]["total_size_gb"], 2)

    return result


def _scan_mlx_models(mlx_path: Path) -> List[Dict[str, Any]]:
    """Scan MLX models directory."""
    models = []

    for model_dir in mlx_path.iterdir():
        if model_dir.is_dir():
            model_info = {
                "name": model_dir.name,
                "path": str(model_dir),
                "type": "mlx",
                "size_gb": 0,
                "file_count": 0,
                "modified": None,
                "tags": [],
            }

            # Calculate size
            total_size = 0
            file_count = 0
            latest_mtime = 0

            for f in model_dir.rglob("*"):
                if f.is_file():
                    stat = f.stat()
                    total_size += stat.st_size
                    file_count += 1
                    if stat.st_mtime > latest_mtime:
                        latest_mtime = stat.st_mtime

            model_info["size_gb"] = round(total_size / (1024**3), 2)
            model_info["file_count"] = file_count
            if latest_mtime > 0:
                model_info["modified"] = datetime.fromtimestamp(latest_mtime).isoformat()

            # Add tags
            model_info["tags"] = get_model_tags(model_dir.name)

            models.append(model_info)

    return sorted(models, key=lambda x: x["size_gb"], reverse=True)


def _scan_gguf_models(gguf_path: Path) -> List[Dict[str, Any]]:
    """Scan GGUF models directory."""
    models = []

    for f in gguf_path.rglob("*.gguf"):
        stat = f.stat()
        model_info = {
            "name": f.stem,
            "filename": f.name,
            "path": str(f),
            "type": "gguf",
            "size_gb": round(stat.st_size / (1024**3), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "tags": get_model_tags(f.name),
        }
        models.append(model_info)

    return sorted(models, key=lambda x: x["size_gb"], reverse=True)


def _scan_ollama_models(ollama_path: Path) -> List[Dict[str, Any]]:
    """Scan Ollama models directory."""
    models = []

    manifests_path = ollama_path / "manifests" / "registry.ollama.ai" / "library"
    if manifests_path.exists():
        for model_dir in manifests_path.iterdir():
            if model_dir.is_dir():
                for tag_file in model_dir.iterdir():
                    if tag_file.is_file():
                        model_info = {
                            "name": f"{model_dir.name}:{tag_file.name}",
                            "path": str(model_dir),
                            "type": "ollama",
                            "size_gb": 0,  # Ollama stores blobs separately
                            "tags": get_model_tags(model_dir.name),
                        }
                        models.append(model_info)

    return models


# ============================================================================
# MODEL DELETION
# ============================================================================

def delete_model(
    model_path: str,
    confirm: bool = False
) -> Dict[str, Any]:
    """
    Delete a local model file or directory.

    SAFETY: Requires explicit confirmation and validates path.

    Args:
        model_path: Path to the model to delete
        confirm: Must be True to actually delete

    Returns:
        Dict with deletion result
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "deleted": False,
    }

    path = Path(model_path)

    # Safety checks
    if not path.exists():
        result["error"] = "Path does not exist"
        return result

    # Validate it's actually a model path
    valid_model_dirs = []
    for p in [get_mlx_models_path(), get_gguf_models_path(), get_ollama_models_path()]:
        if p:
            valid_model_dirs.append(str(p))

    is_valid_model_path = False
    for valid_dir in valid_model_dirs:
        if str(path).startswith(valid_dir):
            is_valid_model_path = True
            break

    if not is_valid_model_path:
        result["error"] = "Path is not within a recognized model directory"
        result["valid_directories"] = valid_model_dirs
        return result

    # Calculate size before deletion
    if path.is_file():
        size_gb = path.stat().st_size / (1024**3)
    else:
        size_gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**3)

    result["size_gb"] = round(size_gb, 2)

    if not confirm:
        result["warning"] = "Set confirm=True to actually delete this model"
        result["preview"] = f"Would delete: {path.name} ({result['size_gb']} GB)"
        return result

    # Perform deletion
    try:
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)

        result["deleted"] = True
        result["freed_gb"] = result["size_gb"]
        result["message"] = f"Successfully deleted {path.name}"

    except Exception as e:
        result["error"] = f"Deletion failed: {str(e)}"

    return result


def find_duplicate_models() -> Dict[str, Any]:
    """
    Find potentially duplicate models based on name similarity.

    Returns:
        Dict with potential duplicates for review
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "potential_duplicates": [],
        "recommendations": [],
    }

    inventory = get_local_model_inventory()

    # Collect all models
    all_models = []
    for model_type, models in inventory["models"].items():
        for model in models:
            all_models.append({
                **model,
                "model_type": model_type,
            })

    # Find models with similar names (different quants of same model)
    model_families = {}
    for model in all_models:
        # Extract base name (remove quant suffixes)
        name = model["name"].lower()
        for suffix in ["-q4", "-q5", "-q6", "-q8", "-4bit", "-5bit", "-6bit", "-8bit",
                       "-fp16", "-f16", "_k_m", "_k_s", "_k_l", "-gguf", ".gguf"]:
            name = name.replace(suffix, "")

        if name not in model_families:
            model_families[name] = []
        model_families[name].append(model)

    # Find families with multiple versions
    for family_name, models in model_families.items():
        if len(models) > 1:
            total_size = sum(m.get("size_gb", 0) for m in models)
            result["potential_duplicates"].append({
                "family": family_name,
                "versions": [
                    {
                        "name": m["name"],
                        "type": m.get("model_type"),
                        "size_gb": m.get("size_gb", 0),
                        "path": m.get("path"),
                    }
                    for m in models
                ],
                "total_size_gb": round(total_size, 2),
                "potential_savings_gb": round(total_size - min(m.get("size_gb", 0) for m in models), 2),
            })

    # Sort by potential savings
    result["potential_duplicates"].sort(
        key=lambda x: x["potential_savings_gb"],
        reverse=True
    )

    # Generate recommendations
    if result["potential_duplicates"]:
        total_savings = sum(d["potential_savings_gb"] for d in result["potential_duplicates"])
        result["recommendations"].append({
            "type": "cleanup",
            "message": f"Found {len(result['potential_duplicates'])} model families with multiple versions",
            "potential_savings_gb": round(total_savings, 2),
        })

    return result


# ============================================================================
# MODEL DOWNLOAD (Hugging Face)
# ============================================================================

def check_huggingface_model(repo_id: str) -> Dict[str, Any]:
    """
    Check if a model exists on Hugging Face and get its info.

    Args:
        repo_id: HuggingFace repo ID (e.g., "mlx-community/Llama-3-8B-Instruct-4bit")

    Returns:
        Dict with model information from HuggingFace
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "repo_id": repo_id,
        "found": False,
    }

    try:
        # Use HuggingFace API
        api_url = f"https://huggingface.co/api/models/{repo_id}"

        req = urllib.request.Request(api_url)
        req.add_header("User-Agent", "msty-admin-mcp/6.6.0")

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

            result["found"] = True
            result["model_info"] = {
                "id": data.get("id"),
                "author": data.get("author"),
                "downloads": data.get("downloads"),
                "likes": data.get("likes"),
                "tags": data.get("tags", []),
                "pipeline_tag": data.get("pipeline_tag"),
                "library_name": data.get("library_name"),
                "created_at": data.get("createdAt"),
                "last_modified": data.get("lastModified"),
            }

            # Check if it's an MLX model
            if "mlx" in str(data.get("tags", [])).lower():
                result["model_type"] = "mlx"
            elif any(f.endswith(".gguf") for f in data.get("siblings", [])):
                result["model_type"] = "gguf"
            else:
                result["model_type"] = "unknown"

    except urllib.error.HTTPError as e:
        if e.code == 404:
            result["error"] = f"Model '{repo_id}' not found on HuggingFace"
        else:
            result["error"] = f"HuggingFace API error: {e.code}"
    except Exception as e:
        result["error"] = f"Failed to check model: {str(e)}"

    return result


def get_download_instructions(
    repo_id: str,
    model_type: str = "auto"
) -> Dict[str, Any]:
    """
    Get instructions for downloading a model from HuggingFace.

    Args:
        repo_id: HuggingFace repo ID
        model_type: "mlx", "gguf", or "auto" to detect

    Returns:
        Dict with download instructions and commands
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "repo_id": repo_id,
        "instructions": [],
    }

    # Check model exists
    model_info = check_huggingface_model(repo_id)
    if not model_info.get("found"):
        result["error"] = model_info.get("error", "Model not found")
        return result

    detected_type = model_info.get("model_type", "unknown")
    if model_type == "auto":
        model_type = detected_type

    result["model_type"] = model_type
    result["model_info"] = model_info.get("model_info")

    if model_type == "mlx":
        result["instructions"] = [
            {
                "step": 1,
                "description": "Install/Update MLX LM library",
                "command": "pip install -U mlx-lm",
            },
            {
                "step": 2,
                "description": "Download the model",
                "command": f"mlx_lm.download --model {repo_id}",
            },
            {
                "step": 3,
                "description": "Or download via Python",
                "code": f"""
from mlx_lm import load
model, tokenizer = load("{repo_id}")
""",
            },
        ]
        result["note"] = "Model will be cached in ~/.cache/huggingface/hub"

    elif model_type == "gguf":
        result["instructions"] = [
            {
                "step": 1,
                "description": "Install huggingface_hub CLI",
                "command": "pip install -U huggingface_hub",
            },
            {
                "step": 2,
                "description": "Download GGUF file",
                "command": f"huggingface-cli download {repo_id} --include '*.gguf' --local-dir ./models",
            },
        ]
        result["note"] = "Move the .gguf file to Msty's models directory after download"

    else:
        result["instructions"] = [
            {
                "step": 1,
                "description": "Clone the model repository",
                "command": f"git lfs install && git clone https://huggingface.co/{repo_id}",
            },
        ]
        result["warning"] = "Model type could not be auto-detected. Manual setup may be required."

    return result


# ============================================================================
# DISK SPACE ANALYSIS
# ============================================================================

def analyze_model_storage() -> Dict[str, Any]:
    """
    Analyze disk space usage by local models.

    Returns:
        Dict with storage analysis and recommendations
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "storage": {},
        "recommendations": [],
    }

    inventory = get_local_model_inventory()

    result["storage"] = {
        "total_models": inventory["summary"]["total_models"],
        "total_size_gb": inventory["summary"]["total_size_gb"],
        "by_type": inventory["summary"]["by_type"],
    }

    # Find largest models
    all_models = []
    for model_type, models in inventory["models"].items():
        for model in models:
            all_models.append({**model, "model_type": model_type})

    all_models.sort(key=lambda x: x.get("size_gb", 0), reverse=True)

    result["storage"]["largest_models"] = [
        {
            "name": m["name"],
            "type": m.get("model_type"),
            "size_gb": m.get("size_gb", 0),
        }
        for m in all_models[:10]
    ]

    # Get disk info
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        result["storage"]["disk"] = {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "models_percent_of_used": round(
                (result["storage"]["total_size_gb"] / (used / (1024**3))) * 100, 1
            ) if used > 0 else 0,
        }
    except:
        pass

    # Recommendations
    if result["storage"]["total_size_gb"] > 500:
        result["recommendations"].append({
            "type": "cleanup",
            "priority": "medium",
            "message": f"Models are using {result['storage']['total_size_gb']:.0f}GB. Consider removing unused models.",
        })

    # Check for duplicates
    duplicates = find_duplicate_models()
    if duplicates.get("potential_duplicates"):
        savings = sum(d["potential_savings_gb"] for d in duplicates["potential_duplicates"])
        result["recommendations"].append({
            "type": "duplicates",
            "priority": "high",
            "message": f"Found {len(duplicates['potential_duplicates'])} model families with duplicates. Potential savings: {savings:.1f}GB",
        })

    return result


__all__ = [
    "get_mlx_models_path",
    "get_gguf_models_path",
    "get_ollama_models_path",
    "get_local_model_inventory",
    "delete_model",
    "find_duplicate_models",
    "check_huggingface_model",
    "get_download_instructions",
    "analyze_model_storage",
]
