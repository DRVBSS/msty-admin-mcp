"""
Msty Admin MCP - Knowledge Stack Management

Tools for creating, managing, and querying Msty Knowledge Stacks (RAG).
"""

import json
import logging
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from .paths import get_msty_paths
from .database import query_database, get_database_connection

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# KNOWLEDGE STACK DATA STRUCTURES
# ============================================================================

def get_knowledge_stacks_path() -> Optional[Path]:
    """Get the path to Knowledge Stacks storage."""
    paths = get_msty_paths()
    data_dir = paths.get("data_dir")
    if data_dir:
        ks_path = Path(data_dir) / "KnowledgeStacks"
        if ks_path.exists():
            return ks_path
    return None


def get_embeddings_path() -> Optional[Path]:
    """Get the path to embeddings storage."""
    paths = get_msty_paths()
    data_dir = paths.get("data_dir")
    if data_dir:
        embed_path = Path(data_dir) / "Embeddings"
        if embed_path.exists():
            return embed_path
    return None


# ============================================================================
# KNOWLEDGE STACK OPERATIONS
# ============================================================================

def list_knowledge_stacks() -> Dict[str, Any]:
    """
    List all Knowledge Stacks with metadata.

    Returns:
        Dict with stacks list and summary statistics
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "stacks": [],
        "total_count": 0,
        "total_documents": 0,
        "total_chunks": 0,
    }

    ks_path = get_knowledge_stacks_path()
    if not ks_path:
        result["error"] = "Knowledge Stacks directory not found"
        return result

    # Scan for knowledge stack folders
    for stack_dir in ks_path.iterdir():
        if stack_dir.is_dir():
            stack_info = _get_stack_info(stack_dir)
            if stack_info:
                result["stacks"].append(stack_info)
                result["total_documents"] += stack_info.get("document_count", 0)
                result["total_chunks"] += stack_info.get("chunk_count", 0)

    result["total_count"] = len(result["stacks"])
    return result


def _get_stack_info(stack_dir: Path) -> Optional[Dict[str, Any]]:
    """Get metadata for a single knowledge stack."""
    try:
        info = {
            "id": stack_dir.name,
            "name": stack_dir.name,
            "path": str(stack_dir),
            "document_count": 0,
            "chunk_count": 0,
            "size_mb": 0,
            "created": None,
            "modified": None,
            "sources": [],
        }

        # Get directory stats
        stat = stack_dir.stat()
        info["created"] = datetime.fromtimestamp(stat.st_ctime).isoformat()
        info["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

        # Count files and calculate size
        total_size = 0
        for f in stack_dir.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
                info["document_count"] += 1

                # Track source types
                ext = f.suffix.lower()
                if ext not in info["sources"]:
                    info["sources"].append(ext)

        info["size_mb"] = round(total_size / (1024 * 1024), 2)

        # Look for metadata file
        meta_file = stack_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                info["name"] = meta.get("name", stack_dir.name)
                info["description"] = meta.get("description", "")
                info["chunk_count"] = meta.get("chunk_count", 0)

        return info
    except Exception as e:
        logger.error(f"Error reading stack {stack_dir}: {e}")
        return None


def get_knowledge_stack_details(stack_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific Knowledge Stack.

    Args:
        stack_id: The stack identifier

    Returns:
        Dict with detailed stack information including documents
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "stack_id": stack_id,
        "found": False,
    }

    ks_path = get_knowledge_stacks_path()
    if not ks_path:
        result["error"] = "Knowledge Stacks directory not found"
        return result

    stack_dir = ks_path / stack_id
    if not stack_dir.exists():
        result["error"] = f"Stack '{stack_id}' not found"
        return result

    info = _get_stack_info(stack_dir)
    if info:
        result.update(info)
        result["found"] = True

        # List documents in stack
        result["documents"] = []
        for f in stack_dir.iterdir():
            if f.is_file() and f.name != "metadata.json":
                doc_info = {
                    "name": f.name,
                    "type": f.suffix.lower(),
                    "size_kb": round(f.stat().st_size / 1024, 2),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                }
                result["documents"].append(doc_info)

    return result


def search_knowledge_stack(
    stack_id: str,
    query: str,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search within a Knowledge Stack.

    Note: This is a simple text search. For semantic search,
    use the embedding-based search through Msty's API.

    Args:
        stack_id: The stack to search
        query: Search query
        limit: Maximum results

    Returns:
        Dict with search results
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "stack_id": stack_id,
        "query": query,
        "results": [],
        "total_matches": 0,
    }

    ks_path = get_knowledge_stacks_path()
    if not ks_path:
        result["error"] = "Knowledge Stacks directory not found"
        return result

    stack_dir = ks_path / stack_id
    if not stack_dir.exists():
        result["error"] = f"Stack '{stack_id}' not found"
        return result

    query_lower = query.lower()
    matches = []

    # Simple text search across documents
    for f in stack_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in ['.txt', '.md', '.json', '.csv']:
            try:
                content = f.read_text(encoding='utf-8', errors='ignore')
                if query_lower in content.lower():
                    # Find context around match
                    idx = content.lower().find(query_lower)
                    start = max(0, idx - 100)
                    end = min(len(content), idx + len(query) + 100)
                    snippet = content[start:end]

                    matches.append({
                        "document": f.name,
                        "path": str(f.relative_to(stack_dir)),
                        "snippet": f"...{snippet}...",
                        "match_position": idx,
                    })
            except Exception as e:
                logger.warning(f"Error searching {f}: {e}")

    result["results"] = matches[:limit]
    result["total_matches"] = len(matches)

    return result


def analyze_knowledge_stack(stack_id: str) -> Dict[str, Any]:
    """
    Analyze a Knowledge Stack for quality and optimization opportunities.

    Args:
        stack_id: The stack to analyze

    Returns:
        Dict with analysis results and recommendations
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "stack_id": stack_id,
        "analysis": {},
        "recommendations": [],
    }

    ks_path = get_knowledge_stacks_path()
    if not ks_path:
        result["error"] = "Knowledge Stacks directory not found"
        return result

    stack_dir = ks_path / stack_id
    if not stack_dir.exists():
        result["error"] = f"Stack '{stack_id}' not found"
        return result

    # Analyze content
    analysis = {
        "document_count": 0,
        "total_size_mb": 0,
        "file_types": {},
        "largest_files": [],
        "empty_files": [],
        "duplicate_content": [],
    }

    file_hashes = {}
    files_data = []

    for f in stack_dir.rglob("*"):
        if f.is_file() and f.name != "metadata.json":
            size = f.stat().st_size
            analysis["document_count"] += 1
            analysis["total_size_mb"] += size / (1024 * 1024)

            # Track file types
            ext = f.suffix.lower() or "no_extension"
            analysis["file_types"][ext] = analysis["file_types"].get(ext, 0) + 1

            files_data.append((f.name, size))

            # Check for empty files
            if size == 0:
                analysis["empty_files"].append(f.name)

            # Hash for duplicate detection
            if size < 10 * 1024 * 1024:  # Only hash files < 10MB
                try:
                    content = f.read_bytes()
                    file_hash = hashlib.md5(content).hexdigest()
                    if file_hash in file_hashes:
                        analysis["duplicate_content"].append({
                            "file1": file_hashes[file_hash],
                            "file2": f.name,
                        })
                    else:
                        file_hashes[file_hash] = f.name
                except:
                    pass

    # Find largest files
    files_data.sort(key=lambda x: x[1], reverse=True)
    analysis["largest_files"] = [
        {"name": name, "size_mb": round(size / (1024 * 1024), 2)}
        for name, size in files_data[:5]
    ]

    analysis["total_size_mb"] = round(analysis["total_size_mb"], 2)
    result["analysis"] = analysis

    # Generate recommendations
    recommendations = []

    if analysis["empty_files"]:
        recommendations.append({
            "type": "cleanup",
            "priority": "medium",
            "message": f"Remove {len(analysis['empty_files'])} empty files",
            "files": analysis["empty_files"][:5],
        })

    if analysis["duplicate_content"]:
        recommendations.append({
            "type": "cleanup",
            "priority": "high",
            "message": f"Found {len(analysis['duplicate_content'])} duplicate files",
            "duplicates": analysis["duplicate_content"][:5],
        })

    if analysis["total_size_mb"] > 500:
        recommendations.append({
            "type": "optimization",
            "priority": "medium",
            "message": "Large stack may slow down queries. Consider splitting into focused sub-stacks.",
        })

    if len(analysis["file_types"]) > 5:
        recommendations.append({
            "type": "organization",
            "priority": "low",
            "message": f"Stack contains {len(analysis['file_types'])} different file types. Consider organizing by type.",
        })

    result["recommendations"] = recommendations

    return result


def get_stack_statistics() -> Dict[str, Any]:
    """
    Get aggregate statistics across all Knowledge Stacks.

    Returns:
        Dict with overall statistics and usage patterns
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "statistics": {},
    }

    stacks = list_knowledge_stacks()
    if "error" in stacks:
        return stacks

    stats = {
        "total_stacks": stacks["total_count"],
        "total_documents": stacks["total_documents"],
        "total_size_mb": 0,
        "average_stack_size_mb": 0,
        "largest_stack": None,
        "smallest_stack": None,
        "file_type_distribution": {},
        "stacks_by_size": [],
    }

    if not stacks["stacks"]:
        result["statistics"] = stats
        return result

    # Calculate totals
    for stack in stacks["stacks"]:
        size = stack.get("size_mb", 0)
        stats["total_size_mb"] += size
        stats["stacks_by_size"].append({
            "name": stack["name"],
            "size_mb": size,
            "documents": stack.get("document_count", 0),
        })

        # Aggregate file types
        for src in stack.get("sources", []):
            stats["file_type_distribution"][src] = \
                stats["file_type_distribution"].get(src, 0) + 1

    # Sort by size
    stats["stacks_by_size"].sort(key=lambda x: x["size_mb"], reverse=True)

    if stats["stacks_by_size"]:
        stats["largest_stack"] = stats["stacks_by_size"][0]
        stats["smallest_stack"] = stats["stacks_by_size"][-1]

    stats["average_stack_size_mb"] = round(
        stats["total_size_mb"] / stats["total_stacks"], 2
    ) if stats["total_stacks"] > 0 else 0

    stats["total_size_mb"] = round(stats["total_size_mb"], 2)

    result["statistics"] = stats

    return result


__all__ = [
    "get_knowledge_stacks_path",
    "get_embeddings_path",
    "list_knowledge_stacks",
    "get_knowledge_stack_details",
    "search_knowledge_stack",
    "analyze_knowledge_stack",
    "get_stack_statistics",
]
