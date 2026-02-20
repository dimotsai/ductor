"""Stream event models and NDJSON parser for --output-format stream-json.
This module provides a unified protocol layer for multiple CLI providers.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StreamEvent(BaseModel):
    """Base event for CLI stream-json output."""

    type: str
    subtype: str | None = None


class AssistantTextDelta(StreamEvent):
    """Text chunk from an assistant response."""

    text: str = ""


class SystemInitEvent(StreamEvent):
    """Initialization event containing session metadata."""

    session_id: str | None = None


class ResultEvent(StreamEvent):
    """Final summary event with metrics and execution status."""

    session_id: str | None = None
    result: str = ""
    is_error: bool = False
    returncode: int | None = None
    duration_ms: float | None = None
    duration_api_ms: float | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    model_usage: dict[str, Any] = Field(default_factory=dict)
    num_turns: int | None = None


class ToolUseEvent(StreamEvent):
    """Request to execute a tool."""

    tool_name: str = ""
    tool_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolResultEvent(StreamEvent):
    """Output from a tool execution."""

    tool_id: str | None = None
    status: str = ""
    output: str = ""


class ThinkingEvent(StreamEvent):
    """Internal reasoning or chain-of-thought content."""

    text: str = ""


class SystemStatusEvent(StreamEvent):
    """Transient system state updates (e.g. 'compacting')."""

    status: str | None = None


class CompactBoundaryEvent(StreamEvent):
    """Marker for context compaction events."""

    trigger: str = ""
    pre_tokens: int = 0


def parse_stream_line(line: str) -> list[StreamEvent]:
    """Parse a single NDJSON line into normalized stream events.
    Supports both nested (block-based) and flat (role-based) formats.
    """
    stripped = line.strip()
    if not stripped:
        return []

    try:
        data: dict[str, Any] = json.loads(stripped)
    except json.JSONDecodeError:
        logger.debug("Unparseable stream line: %.200s", stripped)
        return []

    etype = data.get("type", "")

    # 1. Result/Summary Events
    if etype == "result":
        return [_parse_result_event(data)]

    # 2. Assistant/Message Events (The heart of the stream)
    if etype in ("assistant", "message"):
        return _parse_message_event(data)

    # 3. Tool Events (Direct or Nested)
    if etype == "tool_use":
        return [
            ToolUseEvent(
                type="assistant",
                tool_name=data.get("tool_name", ""),
                tool_id=data.get("tool_id"),
                parameters=data.get("parameters", {}),
            )
        ]
    if etype == "tool_result":
        return [
            ToolResultEvent(
                type="tool_result",
                tool_id=data.get("tool_id"),
                status=data.get("status", ""),
                output=data.get("output", ""),
            )
        ]

    # 4. System/Meta Events
    if etype in ("system", "init"):
        return _parse_system_meta_event(data)

    return []


def _parse_result_event(data: dict[str, Any]) -> ResultEvent:
    """Extract metrics and results from a summary event."""
    stats = data.get("stats", {})
    usage = data.get("usage") or {
        "input_tokens": stats.get("input_tokens", 0),
        "output_tokens": stats.get("output_tokens", 0),
        "cached_tokens": stats.get("cached", 0),  # Capture precise cache stats
    }

    # Extract result content using fallback chain
    res = data.get("result") or data.get("response") or data.get("output")
    is_error = data.get("is_error", False) or data.get("status") == "error"

    if not res and is_error:
        err = data.get("error")
        res = err.get("message") if isinstance(err, dict) else str(err)

    return ResultEvent(
        type="result",
        subtype=data.get("subtype"),
        session_id=data.get("session_id"),
        result=res or "",
        is_error=is_error,
        duration_ms=data.get("duration_ms") or stats.get("duration_ms"),
        duration_api_ms=data.get("duration_api_ms"),
        total_cost_usd=data.get("total_cost_usd"),
        usage=usage,
        model_usage=data.get("modelUsage", {}),
        returncode=data.get("returncode"),
        num_turns=data.get("num_turns"),
    )


def _parse_message_event(data: dict[str, Any]) -> list[StreamEvent]:
    """Unify nested block-based and flat string-based messages."""
    events: list[StreamEvent] = []

    # Case A: Nested structure (Claude style)
    if "message" in data:
        content = data["message"].get("content", [])
        for block in content:
            if not isinstance(block, dict):
                continue
            events.extend(_parse_content_block(block))
        return events

    # Case B: Flat structure (Gemini style)
    role = data.get("role")
    content = data.get("content")
    if role == "assistant" and content:
        if isinstance(content, str):
            events.append(AssistantTextDelta(type="assistant", text=content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    events.extend(_parse_content_block(block))

    return events


def _parse_content_block(block: dict[str, Any]) -> list[StreamEvent]:
    """Parse a single content block into a specific event."""
    b_type = block.get("type", "")
    if b_type == "text":
        text = block.get("text", "")
        if not text:
            return []
        return [AssistantTextDelta(type="assistant", text=text)]
    if b_type == "tool_use":
        return [
            ToolUseEvent(
                type="assistant",
                tool_name=block.get("name", ""),
                tool_id=block.get("id"),
                parameters=block.get("input", {}),
            )
        ]
    if b_type == "thinking":
        return [ThinkingEvent(type="assistant", text=block.get("text", ""))]
    return []


def _parse_system_meta_event(data: dict[str, Any]) -> list[StreamEvent]:
    """Route system metadata events (init, status, compaction)."""
    etype = data.get("type")
    subtype = data.get("subtype") or (etype if etype == "init" else "")

    if subtype == "init":
        return [
            SystemInitEvent(
                type="system",
                subtype="init",
                session_id=data.get("session_id"),
            )
        ]

    if subtype == "status":
        return [
            SystemStatusEvent(
                type="system",
                subtype="status",
                status=data.get("status"),
            )
        ]

    if subtype == "compact_boundary":
        meta = data.get("compact_metadata", {})
        return [
            CompactBoundaryEvent(
                type="system",
                subtype="compact_boundary",
                trigger=meta.get("trigger", ""),
                pre_tokens=meta.get("pre_tokens", 0),
            )
        ]

    return []
