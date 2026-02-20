"""Tests for Gemini-specific stream event parsing."""

from __future__ import annotations

import json

from ductor_bot.cli.gemini_events import parse_gemini_stream_line
from ductor_bot.cli.stream_events import (
    AssistantTextDelta,
    ResultEvent,
    ToolUseEvent,
    SystemInitEvent,
    ToolResultEvent,
)


def test_parse_gemini_init() -> None:
    data = {"type": "init", "session_id": "gem-123"}
    events = parse_gemini_stream_line(json.dumps(data))
    assert len(events) == 1
    assert isinstance(events[0], SystemInitEvent)
    assert events[0].session_id == "gem-123"


def test_parse_flat_gemini_message() -> None:
    # Flat format (Gemini)
    data = {"type": "message", "role": "assistant", "content": "Hello Gemini"}
    events = parse_gemini_stream_line(json.dumps(data))
    assert len(events) == 1
    assert isinstance(events[0], AssistantTextDelta)
    assert events[0].text == "Hello Gemini"


def test_parse_flat_gemini_tool_use() -> None:
    # Top-level tool_use (Gemini)
    data = {
        "type": "tool_use",
        "tool_name": "bash",
        "tool_id": "bash_1",
        "parameters": {"cmd": "ls"},
    }
    events = parse_gemini_stream_line(json.dumps(data))
    assert len(events) == 1
    assert isinstance(events[0], ToolUseEvent)
    assert events[0].tool_name == "bash"
    assert events[0].tool_id == "bash_1"
    assert events[0].parameters == {"cmd": "ls"}


def test_parse_gemini_tool_result() -> None:
    data = {
        "type": "tool_result",
        "tool_id": "bash_1",
        "status": "success",
        "output": "file.txt",
    }
    events = parse_gemini_stream_line(json.dumps(data))
    assert len(events) == 1
    assert isinstance(events[0], ToolResultEvent)
    assert events[0].tool_id == "bash_1"
    assert events[0].status == "success"
    assert events[0].output == "file.txt"


def test_parse_gemini_stats() -> None:
    # Result with Gemini-specific stats structure
    data = {
        "type": "result",
        "status": "success",
        "stats": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cached": 20,
            "duration_ms": 1234,
        },
    }
    events = parse_gemini_stream_line(json.dumps(data))
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ResultEvent)
    assert event.usage["cached_tokens"] == 20
    assert event.duration_ms == 1234


def test_parse_gemini_error() -> None:
    data = {"type": "error", "message": "API Key Invalid"}
    events = parse_gemini_stream_line(json.dumps(data))
    assert len(events) == 1
    assert isinstance(events[0], ResultEvent)
    assert events[0].is_error is True
    assert events[0].result == "API Key Invalid"
