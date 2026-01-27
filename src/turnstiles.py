"""
Msty Admin MCP - Turnstile Workflow Tools

Turnstiles are Msty's automation system for creating multi-step AI workflows.
This module provides tools for managing, analyzing, and optimizing turnstiles.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from .paths import get_msty_paths
from .database import query_database, get_database_connection

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# TURNSTILE DATA ACCESS
# ============================================================================

def get_turnstiles_path() -> Optional[Path]:
    """Get the path to Turnstiles storage."""
    paths = get_msty_paths()
    data_dir = paths.get("data_dir")
    if data_dir:
        turnstiles_path = Path(data_dir) / "Turnstiles"
        if turnstiles_path.exists():
            return turnstiles_path
    return None


def list_turnstiles() -> Dict[str, Any]:
    """
    List all Turnstile workflows.

    Returns:
        Dict with turnstiles list and metadata
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "turnstiles": [],
        "total_count": 0,
    }

    # Try database first
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            # Check for turnstiles/workflows tables
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name LIKE '%turnstile%'
                OR name LIKE '%workflow%' OR name LIKE '%automation%'
            """)
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 100")
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

                for row in rows:
                    turnstile = dict(zip(columns, row))
                    result["turnstiles"].append({
                        "source": "database",
                        "table": table_name,
                        **turnstile
                    })
            conn.close()
    except Exception as e:
        logger.debug(f"Database turnstile query: {e}")

    # Try file system
    turnstiles_path = get_turnstiles_path()
    if turnstiles_path:
        for f in turnstiles_path.iterdir():
            if f.suffix.lower() == '.json':
                try:
                    with open(f, 'r') as file:
                        data = json.load(file)
                        result["turnstiles"].append({
                            "source": "file",
                            "path": str(f),
                            "name": f.stem,
                            **data
                        })
                except Exception as e:
                    logger.warning(f"Error reading turnstile {f}: {e}")

    result["total_count"] = len(result["turnstiles"])

    if not result["turnstiles"]:
        result["note"] = "No turnstiles found. Turnstiles may be stored in a different location or the feature may not be enabled."

    return result


def get_turnstile_details(turnstile_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific turnstile.

    Args:
        turnstile_id: The turnstile identifier

    Returns:
        Dict with turnstile details including steps and configuration
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "turnstile_id": turnstile_id,
        "found": False,
    }

    # Search in files
    turnstiles_path = get_turnstiles_path()
    if turnstiles_path:
        for f in turnstiles_path.iterdir():
            if f.stem == turnstile_id or f.name == turnstile_id:
                try:
                    with open(f, 'r') as file:
                        data = json.load(file)
                        result.update(data)
                        result["found"] = True
                        result["path"] = str(f)
                        return result
                except Exception as e:
                    result["error"] = f"Error reading turnstile: {e}"
                    return result

    # Search in database
    try:
        conn = get_database_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND (name LIKE '%turnstile%' OR name LIKE '%workflow%')
            """)
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]
                cursor.execute(f"""
                    SELECT * FROM {table_name}
                    WHERE id = ? OR name = ? OR title = ?
                    LIMIT 1
                """, (turnstile_id, turnstile_id, turnstile_id))
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    result.update(dict(zip(columns, row)))
                    result["found"] = True
                    result["source"] = "database"
                    break
            conn.close()
    except Exception as e:
        logger.debug(f"Database turnstile search: {e}")

    if not result["found"]:
        result["error"] = f"Turnstile '{turnstile_id}' not found"

    return result


# ============================================================================
# TURNSTILE TEMPLATES
# ============================================================================

TURNSTILE_TEMPLATES = {
    "code_review": {
        "name": "Code Review Pipeline",
        "description": "Multi-model code review with different perspectives",
        "steps": [
            {
                "name": "Initial Analysis",
                "model_preference": "coding",
                "prompt_template": "Analyze this code for bugs and logic errors:\n\n{input}",
                "output_key": "bug_analysis"
            },
            {
                "name": "Security Review",
                "model_preference": "coding",
                "prompt_template": "Review this code for security vulnerabilities:\n\n{input}\n\nPrevious analysis:\n{bug_analysis}",
                "output_key": "security_analysis"
            },
            {
                "name": "Performance Review",
                "model_preference": "reasoning",
                "prompt_template": "Analyze this code for performance issues:\n\n{input}",
                "output_key": "performance_analysis"
            },
            {
                "name": "Final Summary",
                "model_preference": "quality",
                "prompt_template": "Create a comprehensive code review summary:\n\nCode:\n{input}\n\nBug Analysis:\n{bug_analysis}\n\nSecurity Analysis:\n{security_analysis}\n\nPerformance Analysis:\n{performance_analysis}",
                "output_key": "final_review"
            }
        ]
    },
    "research_synthesis": {
        "name": "Research Synthesis",
        "description": "Multi-step research and synthesis workflow",
        "steps": [
            {
                "name": "Topic Exploration",
                "model_preference": "reasoning",
                "prompt_template": "Explore the key aspects of this topic:\n\n{input}",
                "output_key": "exploration"
            },
            {
                "name": "Deep Analysis",
                "model_preference": "quality",
                "prompt_template": "Provide deep analysis on:\n\n{input}\n\nKey aspects identified:\n{exploration}",
                "output_key": "analysis"
            },
            {
                "name": "Synthesis",
                "model_preference": "creative",
                "prompt_template": "Synthesize insights into a cohesive narrative:\n\nTopic: {input}\n\nExploration: {exploration}\n\nAnalysis: {analysis}",
                "output_key": "synthesis"
            }
        ]
    },
    "content_creation": {
        "name": "Content Creation Pipeline",
        "description": "Draft, review, and polish content",
        "steps": [
            {
                "name": "Draft",
                "model_preference": "fast",
                "prompt_template": "Create an initial draft for:\n\n{input}",
                "output_key": "draft"
            },
            {
                "name": "Critique",
                "model_preference": "reasoning",
                "prompt_template": "Critique this draft for improvements:\n\n{draft}",
                "output_key": "critique"
            },
            {
                "name": "Polish",
                "model_preference": "creative",
                "prompt_template": "Polish and improve this content based on feedback:\n\nOriginal: {draft}\n\nCritique: {critique}",
                "output_key": "final_content"
            }
        ]
    },
    "data_analysis": {
        "name": "Data Analysis Pipeline",
        "description": "Multi-step data exploration and insight generation",
        "steps": [
            {
                "name": "Data Exploration",
                "model_preference": "coding",
                "prompt_template": "Analyze this data and identify patterns:\n\n{input}",
                "output_key": "patterns"
            },
            {
                "name": "Statistical Analysis",
                "model_preference": "reasoning",
                "prompt_template": "Provide statistical insights on:\n\n{input}\n\nPatterns found:\n{patterns}",
                "output_key": "statistics"
            },
            {
                "name": "Insight Summary",
                "model_preference": "quality",
                "prompt_template": "Create an executive summary of insights:\n\nData: {input}\n\nPatterns: {patterns}\n\nStatistics: {statistics}",
                "output_key": "summary"
            }
        ]
    },
    "translation_pipeline": {
        "name": "Translation with Review",
        "description": "Translate and verify with back-translation",
        "steps": [
            {
                "name": "Initial Translation",
                "model_preference": "general",
                "prompt_template": "Translate to {target_language}:\n\n{input}",
                "output_key": "translation"
            },
            {
                "name": "Back Translation",
                "model_preference": "general",
                "prompt_template": "Translate this back to the original language:\n\n{translation}",
                "output_key": "back_translation"
            },
            {
                "name": "Quality Check",
                "model_preference": "reasoning",
                "prompt_template": "Compare and identify translation issues:\n\nOriginal: {input}\n\nTranslation: {translation}\n\nBack-translation: {back_translation}",
                "output_key": "quality_report"
            },
            {
                "name": "Final Translation",
                "model_preference": "quality",
                "prompt_template": "Produce final translation addressing issues:\n\nOriginal: {input}\n\nInitial translation: {translation}\n\nIssues found: {quality_report}",
                "output_key": "final_translation"
            }
        ]
    }
}


def list_turnstile_templates() -> Dict[str, Any]:
    """
    List available turnstile templates.

    Returns:
        Dict with template names and descriptions
    """
    templates = []
    for key, template in TURNSTILE_TEMPLATES.items():
        templates.append({
            "id": key,
            "name": template["name"],
            "description": template["description"],
            "step_count": len(template["steps"]),
            "steps": [s["name"] for s in template["steps"]]
        })

    return {
        "timestamp": datetime.now().isoformat(),
        "templates": templates,
        "total_count": len(templates)
    }


def get_turnstile_template(template_id: str) -> Dict[str, Any]:
    """
    Get a specific turnstile template.

    Args:
        template_id: Template identifier

    Returns:
        Dict with full template configuration
    """
    if template_id not in TURNSTILE_TEMPLATES:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": f"Template '{template_id}' not found",
            "available_templates": list(TURNSTILE_TEMPLATES.keys())
        }

    return {
        "timestamp": datetime.now().isoformat(),
        "template_id": template_id,
        **TURNSTILE_TEMPLATES[template_id]
    }


# ============================================================================
# TURNSTILE EXECUTION
# ============================================================================

def execute_turnstile(
    template_id: str,
    input_data: str,
    variables: Optional[Dict[str, str]] = None,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Execute a turnstile workflow (or preview in dry_run mode).

    Args:
        template_id: Template to execute
        input_data: Input data for the workflow
        variables: Additional variables for prompt templates
        dry_run: If True, only show what would be executed

    Returns:
        Dict with execution plan or results
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "template_id": template_id,
        "dry_run": dry_run,
        "steps": [],
    }

    if template_id not in TURNSTILE_TEMPLATES:
        result["error"] = f"Template '{template_id}' not found"
        return result

    template = TURNSTILE_TEMPLATES[template_id]
    result["template_name"] = template["name"]

    # Build context with input and variables
    context = {"input": input_data}
    if variables:
        context.update(variables)

    # Process each step
    for i, step in enumerate(template["steps"]):
        step_info = {
            "step_number": i + 1,
            "name": step["name"],
            "model_preference": step["model_preference"],
            "output_key": step["output_key"],
        }

        # Format the prompt with available context
        try:
            prompt = step["prompt_template"].format(**context)
            step_info["prompt_preview"] = prompt[:500] + "..." if len(prompt) > 500 else prompt
        except KeyError as e:
            step_info["prompt_preview"] = f"[Missing variable: {e}]"

        if dry_run:
            step_info["status"] = "pending"
            step_info["note"] = "Dry run - not executed"
        else:
            # TODO: Integrate with model_bridge for actual execution
            step_info["status"] = "skipped"
            step_info["note"] = "Live execution not yet implemented. Use model_bridge functions."

        result["steps"].append(step_info)

        # In actual execution, would add output to context
        # context[step["output_key"]] = response

    result["total_steps"] = len(result["steps"])

    return result


# ============================================================================
# TURNSTILE ANALYTICS
# ============================================================================

def analyze_turnstile_usage() -> Dict[str, Any]:
    """
    Analyze turnstile usage patterns.

    Returns:
        Dict with usage statistics and recommendations
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "analysis": {},
        "recommendations": [],
    }

    turnstiles = list_turnstiles()

    if not turnstiles.get("turnstiles"):
        result["analysis"] = {
            "total_turnstiles": 0,
            "note": "No turnstiles found to analyze"
        }
        result["recommendations"].append({
            "type": "getting_started",
            "priority": "high",
            "message": "Create your first turnstile using one of the built-in templates",
            "suggested_templates": list(TURNSTILE_TEMPLATES.keys())[:3]
        })
        return result

    # Analyze existing turnstiles
    analysis = {
        "total_turnstiles": len(turnstiles["turnstiles"]),
        "sources": {},
        "complexity": {
            "simple": 0,  # 1-2 steps
            "medium": 0,  # 3-4 steps
            "complex": 0  # 5+ steps
        }
    }

    for t in turnstiles["turnstiles"]:
        # Track sources
        source = t.get("source", "unknown")
        analysis["sources"][source] = analysis["sources"].get(source, 0) + 1

        # Analyze complexity
        steps = t.get("steps", [])
        if isinstance(steps, list):
            step_count = len(steps)
            if step_count <= 2:
                analysis["complexity"]["simple"] += 1
            elif step_count <= 4:
                analysis["complexity"]["medium"] += 1
            else:
                analysis["complexity"]["complex"] += 1

    result["analysis"] = analysis

    # Generate recommendations
    if analysis["complexity"]["complex"] > analysis["complexity"]["simple"]:
        result["recommendations"].append({
            "type": "optimization",
            "priority": "medium",
            "message": "Consider breaking complex turnstiles into smaller, reusable components"
        })

    if analysis["total_turnstiles"] < 3:
        result["recommendations"].append({
            "type": "exploration",
            "priority": "low",
            "message": "Explore more turnstile templates to automate common workflows"
        })

    return result


def suggest_turnstile_for_task(task_description: str) -> Dict[str, Any]:
    """
    Suggest appropriate turnstile templates for a given task.

    Args:
        task_description: Description of the task to automate

    Returns:
        Dict with suggested templates and reasoning
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "task": task_description,
        "suggestions": [],
    }

    task_lower = task_description.lower()

    # Score each template based on keyword matching
    scores = []
    for template_id, template in TURNSTILE_TEMPLATES.items():
        score = 0
        reasons = []

        # Check template name and description
        template_text = f"{template['name']} {template['description']}".lower()

        # Keyword matching
        keywords = {
            "code_review": ["code", "review", "bug", "security", "programming"],
            "research_synthesis": ["research", "analyze", "study", "explore", "synthesis"],
            "content_creation": ["write", "content", "draft", "article", "blog", "create"],
            "data_analysis": ["data", "analysis", "statistics", "patterns", "insights"],
            "translation_pipeline": ["translate", "language", "translation", "localize"]
        }

        for keyword in keywords.get(template_id, []):
            if keyword in task_lower:
                score += 10
                reasons.append(f"Matches keyword '{keyword}'")

        # Check step relevance
        for step in template["steps"]:
            if any(word in task_lower for word in step["name"].lower().split()):
                score += 5
                reasons.append(f"Step '{step['name']}' may be relevant")

        if score > 0:
            scores.append({
                "template_id": template_id,
                "template_name": template["name"],
                "score": score,
                "reasons": reasons,
                "description": template["description"],
                "step_count": len(template["steps"])
            })

    # Sort by score
    scores.sort(key=lambda x: x["score"], reverse=True)
    result["suggestions"] = scores[:3]  # Top 3

    if not result["suggestions"]:
        result["note"] = "No strong template matches found. Consider creating a custom turnstile."
        result["suggestions"] = [
            {
                "template_id": "content_creation",
                "template_name": "Content Creation Pipeline",
                "reason": "General purpose workflow suitable for many tasks"
            }
        ]

    return result


__all__ = [
    "get_turnstiles_path",
    "list_turnstiles",
    "get_turnstile_details",
    "list_turnstile_templates",
    "get_turnstile_template",
    "execute_turnstile",
    "analyze_turnstile_usage",
    "suggest_turnstile_for_task",
    "TURNSTILE_TEMPLATES",
]
