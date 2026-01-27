"""
Msty Admin MCP - Conversation Analytics

Advanced analytics for conversation history, usage patterns, and insights.
"""

import json
import logging
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .paths import get_msty_paths
from .database import query_database, get_database_connection

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# CONVERSATION DATA ACCESS
# ============================================================================

def get_conversations(
    limit: int = 100,
    days: Optional[int] = None,
    model_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve conversations from Msty database.

    Args:
        limit: Maximum conversations to return
        days: Only include conversations from last N days
        model_filter: Filter by model name

    Returns:
        Dict with conversation list
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "conversations": [],
        "filters": {
            "limit": limit,
            "days": days,
            "model_filter": model_filter
        }
    }

    try:
        conn = get_database_connection()
        if not conn:
            result["error"] = "Could not connect to database"
            return result

        cursor = conn.cursor()

        # Find conversation-related tables
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND (
                name LIKE '%chat%' OR
                name LIKE '%conversation%' OR
                name LIKE '%message%' OR
                name LIKE '%session%'
            )
        """)
        tables = [t[0] for t in cursor.fetchall()]

        # Try common table names
        chat_tables = ["chats", "chat_sessions", "conversations"]
        message_tables = ["messages", "chat_messages"]

        chat_table = None
        for t in chat_tables:
            if t in tables:
                chat_table = t
                break

        if chat_table:
            # Build query with filters
            query = f"SELECT * FROM {chat_table}"
            params = []

            if days:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                query += " WHERE created_at > ? OR updated_at > ?"
                params.extend([cutoff, cutoff])

            query += f" ORDER BY id DESC LIMIT {limit}"

            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            for row in rows:
                conv = dict(zip(columns, row))

                # Apply model filter if specified
                if model_filter:
                    model = conv.get("model", conv.get("model_id", ""))
                    if model_filter.lower() not in str(model).lower():
                        continue

                result["conversations"].append(conv)

        result["total_count"] = len(result["conversations"])
        result["tables_found"] = tables

        conn.close()
    except Exception as e:
        result["error"] = str(e)

    return result


def get_messages(
    conversation_id: Optional[str] = None,
    limit: int = 500
) -> Dict[str, Any]:
    """
    Retrieve messages from Msty database.

    Args:
        conversation_id: Filter by conversation (optional)
        limit: Maximum messages to return

    Returns:
        Dict with message list
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "messages": [],
    }

    try:
        conn = get_database_connection()
        if not conn:
            result["error"] = "Could not connect to database"
            return result

        cursor = conn.cursor()

        # Find message tables
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name LIKE '%message%'
        """)
        tables = [t[0] for t in cursor.fetchall()]

        message_table = None
        for t in ["messages", "chat_messages"]:
            if t in tables:
                message_table = t
                break

        if message_table:
            if conversation_id:
                cursor.execute(f"""
                    SELECT * FROM {message_table}
                    WHERE chat_id = ? OR conversation_id = ? OR session_id = ?
                    ORDER BY id DESC LIMIT ?
                """, (conversation_id, conversation_id, conversation_id, limit))
            else:
                cursor.execute(f"""
                    SELECT * FROM {message_table}
                    ORDER BY id DESC LIMIT ?
                """, (limit,))

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            for row in rows:
                result["messages"].append(dict(zip(columns, row)))

        result["total_count"] = len(result["messages"])
        conn.close()
    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================================
# USAGE ANALYTICS
# ============================================================================

def analyze_usage_patterns(days: int = 30) -> Dict[str, Any]:
    """
    Analyze conversation usage patterns.

    Args:
        days: Number of days to analyze

    Returns:
        Dict with usage analytics
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "period_days": days,
        "analytics": {},
    }

    conversations = get_conversations(limit=1000, days=days)
    messages = get_messages(limit=5000)

    if "error" in conversations:
        result["error"] = conversations["error"]
        return result

    convs = conversations.get("conversations", [])
    msgs = messages.get("messages", [])

    analytics = {
        "overview": {
            "total_conversations": len(convs),
            "total_messages": len(msgs),
            "avg_messages_per_conversation": round(len(msgs) / max(len(convs), 1), 2),
        },
        "model_usage": {},
        "daily_activity": {},
        "hourly_distribution": {},
        "conversation_lengths": {
            "short": 0,    # 1-5 messages
            "medium": 0,   # 6-20 messages
            "long": 0,     # 21-50 messages
            "extended": 0  # 50+ messages
        }
    }

    # Analyze models
    model_counter = Counter()
    for conv in convs:
        model = conv.get("model", conv.get("model_id", "unknown"))
        model_counter[model] += 1

    analytics["model_usage"] = dict(model_counter.most_common(10))

    # Analyze time patterns
    for msg in msgs:
        # Try to parse timestamp
        for field in ["created_at", "timestamp", "date"]:
            if field in msg and msg[field]:
                try:
                    ts = datetime.fromisoformat(str(msg[field]).replace("Z", "+00:00"))
                    date_key = ts.strftime("%Y-%m-%d")
                    hour_key = ts.strftime("%H:00")

                    analytics["daily_activity"][date_key] = \
                        analytics["daily_activity"].get(date_key, 0) + 1
                    analytics["hourly_distribution"][hour_key] = \
                        analytics["hourly_distribution"].get(hour_key, 0) + 1
                    break
                except:
                    pass

    # Sort time distributions
    analytics["daily_activity"] = dict(
        sorted(analytics["daily_activity"].items())[-days:]
    )
    analytics["hourly_distribution"] = dict(
        sorted(analytics["hourly_distribution"].items())
    )

    result["analytics"] = analytics

    # Generate insights
    result["insights"] = generate_usage_insights(analytics)

    return result


def generate_usage_insights(analytics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate insights from analytics data."""
    insights = []

    overview = analytics.get("overview", {})

    # Activity level insight
    total_convs = overview.get("total_conversations", 0)
    if total_convs > 100:
        insights.append({
            "type": "high_usage",
            "message": f"Heavy usage detected: {total_convs} conversations in the analysis period",
            "recommendation": "Consider organizing conversations into folders or cleaning up old ones"
        })
    elif total_convs < 10:
        insights.append({
            "type": "low_usage",
            "message": f"Light usage: only {total_convs} conversations",
            "recommendation": "Explore more features to get the most out of Msty"
        })

    # Model diversity insight
    model_usage = analytics.get("model_usage", {})
    if len(model_usage) == 1:
        model = list(model_usage.keys())[0]
        insights.append({
            "type": "single_model",
            "message": f"Only using one model: {model}",
            "recommendation": "Try different models for different tasks to optimize quality and speed"
        })
    elif len(model_usage) > 5:
        insights.append({
            "type": "diverse_models",
            "message": f"Using {len(model_usage)} different models",
            "recommendation": "Great model diversity! Consider creating task-specific personas"
        })

    # Peak hours insight
    hourly = analytics.get("hourly_distribution", {})
    if hourly:
        peak_hour = max(hourly, key=hourly.get)
        insights.append({
            "type": "peak_activity",
            "message": f"Most active hour: {peak_hour}",
            "detail": f"{hourly[peak_hour]} messages during this hour"
        })

    return insights


# ============================================================================
# CONTENT ANALYTICS
# ============================================================================

def analyze_conversation_content(
    conversation_id: Optional[str] = None,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Analyze conversation content patterns (privacy-preserving).

    Args:
        conversation_id: Specific conversation to analyze (optional)
        sample_size: Number of messages to sample

    Returns:
        Dict with content analytics (no actual content exposed)
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "analytics": {},
    }

    messages = get_messages(conversation_id=conversation_id, limit=sample_size)

    if "error" in messages:
        result["error"] = messages["error"]
        return result

    msgs = messages.get("messages", [])

    analytics = {
        "message_lengths": {
            "user": {"min": 0, "max": 0, "avg": 0, "total": 0},
            "assistant": {"min": 0, "max": 0, "avg": 0, "total": 0},
        },
        "message_counts": {
            "user": 0,
            "assistant": 0,
            "system": 0,
            "other": 0,
        },
        "topics_detected": [],
        "code_presence": False,
        "questions_asked": 0,
    }

    user_lengths = []
    assistant_lengths = []

    for msg in msgs:
        content = msg.get("content", msg.get("text", ""))
        role = msg.get("role", msg.get("type", "other")).lower()

        if not content:
            continue

        length = len(content)

        # Count by role
        if "user" in role or "human" in role:
            analytics["message_counts"]["user"] += 1
            user_lengths.append(length)
        elif "assistant" in role or "ai" in role or "bot" in role:
            analytics["message_counts"]["assistant"] += 1
            assistant_lengths.append(length)
        elif "system" in role:
            analytics["message_counts"]["system"] += 1
        else:
            analytics["message_counts"]["other"] += 1

        # Detect code
        if "```" in content or "def " in content or "function " in content:
            analytics["code_presence"] = True

        # Count questions
        if "?" in content:
            analytics["questions_asked"] += content.count("?")

    # Calculate length statistics
    if user_lengths:
        analytics["message_lengths"]["user"] = {
            "min": min(user_lengths),
            "max": max(user_lengths),
            "avg": round(sum(user_lengths) / len(user_lengths), 1),
            "total": len(user_lengths)
        }

    if assistant_lengths:
        analytics["message_lengths"]["assistant"] = {
            "min": min(assistant_lengths),
            "max": max(assistant_lengths),
            "avg": round(sum(assistant_lengths) / len(assistant_lengths), 1),
            "total": len(assistant_lengths)
        }

    result["analytics"] = analytics

    return result


# ============================================================================
# MODEL PERFORMANCE ANALYTICS
# ============================================================================

def analyze_model_performance(days: int = 30) -> Dict[str, Any]:
    """
    Analyze model performance across conversations.

    Args:
        days: Number of days to analyze

    Returns:
        Dict with model performance metrics
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "period_days": days,
        "models": {},
    }

    conversations = get_conversations(limit=1000, days=days)

    if "error" in conversations:
        result["error"] = conversations["error"]
        return result

    convs = conversations.get("conversations", [])

    model_stats = {}

    for conv in convs:
        model = conv.get("model", conv.get("model_id", "unknown"))

        if model not in model_stats:
            model_stats[model] = {
                "conversation_count": 0,
                "total_messages": 0,
                "avg_response_length": 0,
                "response_lengths": [],
            }

        model_stats[model]["conversation_count"] += 1

        # Get message count if available
        msg_count = conv.get("message_count", conv.get("messages", 0))
        if isinstance(msg_count, int):
            model_stats[model]["total_messages"] += msg_count

    # Calculate averages
    for model, stats in model_stats.items():
        if stats["conversation_count"] > 0:
            stats["avg_messages_per_conversation"] = round(
                stats["total_messages"] / stats["conversation_count"], 1
            )

        # Remove internal tracking list
        del stats["response_lengths"]

    result["models"] = model_stats

    # Rank models
    result["rankings"] = {
        "most_used": sorted(
            model_stats.items(),
            key=lambda x: x[1]["conversation_count"],
            reverse=True
        )[:5],
        "highest_engagement": sorted(
            model_stats.items(),
            key=lambda x: x[1].get("avg_messages_per_conversation", 0),
            reverse=True
        )[:5]
    }

    return result


# ============================================================================
# SESSION ANALYTICS
# ============================================================================

def analyze_session_patterns(days: int = 30) -> Dict[str, Any]:
    """
    Analyze session patterns and behaviors.

    Args:
        days: Number of days to analyze

    Returns:
        Dict with session analytics
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "period_days": days,
        "analytics": {},
    }

    conversations = get_conversations(limit=500, days=days)

    if "error" in conversations:
        result["error"] = conversations["error"]
        return result

    convs = conversations.get("conversations", [])

    # Analyze session durations
    session_durations = []
    sessions_by_day = {}

    for conv in convs:
        # Try to get duration
        created = conv.get("created_at")
        updated = conv.get("updated_at")

        if created and updated:
            try:
                created_dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                updated_dt = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
                duration_minutes = (updated_dt - created_dt).total_seconds() / 60
                if 0 < duration_minutes < 1440:  # Reasonable session (< 24h)
                    session_durations.append(duration_minutes)

                # Count by day
                day_key = created_dt.strftime("%A")
                sessions_by_day[day_key] = sessions_by_day.get(day_key, 0) + 1
            except:
                pass

    analytics = {
        "session_count": len(convs),
        "sessions_by_weekday": sessions_by_day,
    }

    if session_durations:
        analytics["session_duration_minutes"] = {
            "min": round(min(session_durations), 1),
            "max": round(max(session_durations), 1),
            "avg": round(sum(session_durations) / len(session_durations), 1),
            "median": round(sorted(session_durations)[len(session_durations) // 2], 1)
        }

    result["analytics"] = analytics

    return result


# ============================================================================
# EXPORT & REPORTING
# ============================================================================

def generate_analytics_report(
    days: int = 30,
    include_usage: bool = True,
    include_content: bool = True,
    include_models: bool = True,
    include_sessions: bool = True
) -> Dict[str, Any]:
    """
    Generate a comprehensive analytics report.

    Args:
        days: Number of days to analyze
        include_usage: Include usage analytics
        include_content: Include content analytics
        include_models: Include model analytics
        include_sessions: Include session analytics

    Returns:
        Dict with full analytics report
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "period_days": days,
        "report_sections": [],
    }

    if include_usage:
        result["usage"] = analyze_usage_patterns(days)
        result["report_sections"].append("usage")

    if include_content:
        result["content"] = analyze_conversation_content(sample_size=200)
        result["report_sections"].append("content")

    if include_models:
        result["models"] = analyze_model_performance(days)
        result["report_sections"].append("models")

    if include_sessions:
        result["sessions"] = analyze_session_patterns(days)
        result["report_sections"].append("sessions")

    # Generate executive summary
    result["executive_summary"] = _generate_executive_summary(result)

    return result


def _generate_executive_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Generate executive summary from report data."""
    summary = {
        "key_metrics": {},
        "highlights": [],
        "recommendations": []
    }

    # Extract key metrics
    if "usage" in report:
        usage = report["usage"].get("analytics", {}).get("overview", {})
        summary["key_metrics"]["total_conversations"] = usage.get("total_conversations", 0)
        summary["key_metrics"]["total_messages"] = usage.get("total_messages", 0)

    if "models" in report:
        models = report["models"].get("models", {})
        summary["key_metrics"]["models_used"] = len(models)

    # Add highlights
    if "usage" in report:
        insights = report["usage"].get("insights", [])
        for insight in insights[:3]:
            summary["highlights"].append(insight.get("message", ""))

    # Add recommendations
    if summary["key_metrics"].get("models_used", 0) == 1:
        summary["recommendations"].append(
            "Diversify model usage to optimize for different task types"
        )

    if summary["key_metrics"].get("total_conversations", 0) > 50:
        summary["recommendations"].append(
            "Consider organizing conversations with tags or folders"
        )

    return summary


__all__ = [
    "get_conversations",
    "get_messages",
    "analyze_usage_patterns",
    "analyze_conversation_content",
    "analyze_model_performance",
    "analyze_session_patterns",
    "generate_analytics_report",
]
