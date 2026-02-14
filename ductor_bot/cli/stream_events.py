"""Stream event models and NDJSON parser for --output-format stream-json."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StreamEvent(BaseModel):
    """Base event from the Claude CLI stream-json output."""

    type: str
    subtype: str | None = None
    delta: bool = False


class AssistantTextDelta(StreamEvent):
    """Text from an assistant turn."""

    text: str = ""


class SystemInitEvent(StreamEvent):
    """First event of a stream -- contains session_id and tool list."""

    session_id: str | None = None


class ResultEvent(StreamEvent):
    """Final event with usage, cost, and session_id."""

    session_id: str | None = None
    result: str = ""
    is_error: bool = False
    duration_ms: float | None = None
    duration_api_ms: float | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    model_usage: dict[str, Any] = Field(default_factory=dict)
    num_turns: int | None = None


class ToolUseEvent(StreamEvent):
    """Tool invocation detected during streaming."""

    tool_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class ToolResultEvent(StreamEvent):
    """Tool execution result detected during streaming."""

    tool_id: str | None = None
    status: str = ""
    output: str = ""


class ThinkingEvent(StreamEvent):
    """Extended thinking/reasoning block."""

    text: str = ""


class SystemStatusEvent(StreamEvent):
    """System status update (e.g. ``compacting``)."""

    status: str | None = None


class CompactBoundaryEvent(StreamEvent):
    """Marks a context compaction boundary."""

    trigger: str = ""
    pre_tokens: int = 0


def parse_stream_line(line: str, last_session_id: str | None = None) -> list[StreamEvent]:
    """Parse a single NDJSON line into normalized stream events."""
    stripped = line.strip()
    if not stripped:
        return []

    try:
        data: dict[str, Any] = json.loads(stripped)
    except json.JSONDecodeError:
        logger.debug("Unparseable stream line: %.200s", stripped)
        return []

    event_type = data.get("type", "")

    if event_type == "result":
        logger.debug("Stream event parsed type=%s", event_type)
        stats = data.get("stats", {})
        usage = data.get("usage") or {
            "input_tokens": stats.get("input_tokens", 0),
            "output_tokens": stats.get("output_tokens", 0),
        }
        return [
            ResultEvent(
                type=event_type,
                subtype=data.get("subtype"),
                session_id=data.get("session_id") or last_session_id,
                result=data.get("result") or (data.get("error", {}).get("message") if isinstance(data.get("error"), dict) else str(data.get("error"))) or "",
                is_error=data.get("is_error", False) or data.get("status") == "error",
                duration_ms=data.get("duration_ms") or stats.get("duration_ms"),
                duration_api_ms=data.get("duration_api_ms"),
                total_cost_usd=data.get("total_cost_usd"),
                usage=usage,
                model_usage=data.get("model_usage") or data.get("modelUsage") or {},
                num_turns=data.get("num_turns") or stats.get("tool_calls"),
            ),
        ]

    if event_type == "assistant":
        return _parse_assistant_content(data)

    if event_type == "tool_use":
        name = data.get("tool_name", "")
        if name:
            return [
                ToolUseEvent(
                    type="assistant",
                    tool_name=name,
                    arguments=data.get("parameters", {}),
                    call_id=data.get("tool_id"),
                )
            ]
        return []

    if event_type == "tool_result":
        return [
            ToolResultEvent(
                type="tool_result",
                tool_id=data.get("tool_id"),
                status=data.get("status", ""),
                output=data.get("output", ""),
            )
        ]

    if event_type == "message":
        # Gemini specific message event
        # {"type":"message", "role":"assistant", "content":"...","delta":true}
        # OR tool use in content blocks
        role = data.get("role")
        if role == "assistant":
            events = _parse_gemini_message_content(data)
            is_delta = data.get("delta", False)
            for e in events:
                e.delta = is_delta
            return events
        return []

    if event_type == "system":
        logger.debug("Stream event parsed type=%s subtype=%s", event_type, data.get("subtype"))
        return _parse_system_event(data)
        
    if event_type == "init":
        # Gemini init: {"type":"init", "session_id":"...", "model":"..."}
        return [
            SystemInitEvent(
                type="system",
                subtype="init",
                session_id=data.get("session_id"),
            ),
        ]

    return []


def _parse_system_event(data: dict[str, Any]) -> list[StreamEvent]:
    """Route system events by subtype."""
    subtype = data.get("subtype", "")

    if subtype == "init":
        return [
            SystemInitEvent(
                type="system",
                subtype="init",
                session_id=data.get("session_id"),
            ),
        ]

    if subtype == "status":
        return [
            SystemStatusEvent(
                type="system",
                subtype="status",
                status=data.get("status"),
            ),
        ]

    if subtype == "compact_boundary":
        meta = data.get("compact_metadata", {})
        return [
            CompactBoundaryEvent(
                type="system",
                subtype="compact_boundary",
                trigger=meta.get("trigger", ""),
                pre_tokens=meta.get("pre_tokens", 0),
            ),
        ]

    return []


def _parse_assistant_content(data: dict[str, Any]) -> list[StreamEvent]:
    """Extract all content blocks from an assistant message."""
    message = data.get("message", {})
    content = message.get("content", [])
    events: list[StreamEvent] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")

        if block_type == "text":
            text = block.get("text", "")
            if text:
                events.append(AssistantTextDelta(type="assistant", text=text))

        elif block_type == "tool_use":
            name = block.get("name", "")
            if name:
                events.append(
                    ToolUseEvent(
                        type="assistant",
                        tool_name=name,
                        arguments=block.get("arguments", {}),
                        call_id=block.get("call_id"),
                    )
                )

        elif block_type == "thinking":
            events.append(
                ThinkingEvent(type="assistant", text=block.get("text", "")),
            )

    return events


def _parse_gemini_message_content(data: dict[str, Any]) -> list[StreamEvent]:
    """Parse Gemini message content which can be a string or blocks."""
    content = data.get("content")
    if not content:
        return []

    if isinstance(content, str):
        return [AssistantTextDelta(type="assistant", text=content)]

    if isinstance(content, list):
        events: list[StreamEvent] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            b_type = block.get("type")
            if b_type == "text":
                events.append(AssistantTextDelta(type="assistant", text=block.get("text", "")))
            elif b_type == "tool_use":
                events.append(
                    ToolUseEvent(
                        type="assistant",
                        tool_name=block.get("name", ""),
                        arguments=block.get("arguments", {}),
                        call_id=block.get("call_id"),
                    )
                )
        return events

    return []


