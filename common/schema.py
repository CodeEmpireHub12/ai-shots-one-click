"""Shared response schema utilities for Project Raven agents."""

from typing import Any, Dict, Optional


VALID_STATUSES = {"success", "partial", "failed"}


def build_agent_response(
    agent: str,
    status: str,
    data: Optional[Dict[str, Any]] = None,
    confidence: float = 0.0,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the standard Project Raven agent response envelope.

    Every agent should return this structure:
    {
        "agent": "<name>",
        "status": "success | partial | failed",
        "data": {},
        "confidence": 0.0,
        "error": null
    }
    """
    if status not in VALID_STATUSES:
        status = "failed"
        error = error or "Invalid agent status provided."

    return {
        "agent": agent,
        "status": status,
        "data": data or {},
        "confidence": float(confidence),
        "error": error,
    }
