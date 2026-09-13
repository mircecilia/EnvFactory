"""Typed adapter for FastMCP 3.1.0 / MCP 1.30.0 CallToolResult values."""

from __future__ import annotations

import json
from typing import Any, Mapping, Tuple

from .adapters import UNKNOWN


def _attr(result: Any, camel: str, snake: str) -> Any:
    if isinstance(result, Mapping):
        if camel in result:
            return result[camel]
        return result.get(snake, UNKNOWN)
    if hasattr(result, camel):
        return getattr(result, camel)
    if hasattr(result, snake):
        return getattr(result, snake)
    return UNKNOWN


def _json_object_text_fallback(result: Any) -> Any:
    content = (
        result.get("content", UNKNOWN)
        if isinstance(result, Mapping)
        else getattr(result, "content", UNKNOWN)
    )
    if not isinstance(content, list) or len(content) != 1:
        return UNKNOWN
    block = content[0]
    block_type = (
        block.get("type")
        if isinstance(block, Mapping)
        else getattr(block, "type", None)
    )
    text = (
        block.get("text")
        if isinstance(block, Mapping)
        else getattr(block, "text", None)
    )
    if block_type != "text" or not isinstance(text, str):
        return UNKNOWN
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return UNKNOWN
    return parsed if isinstance(parsed, (dict, list)) else UNKNOWN


def fastmcp_result_fields(result: Any) -> Any:
    """Recover typed fields, preferring the protocol structured content."""

    structured = _attr(result, "structuredContent", "structured_content")
    if isinstance(structured, (dict, list)):
        return structured
    return _json_object_text_fallback(result)


def fastmcp_execution_success(result: Any) -> Any:
    """Return the typed MCP error bit; never classify natural-language text."""

    is_error = _attr(result, "isError", "is_error")
    return not is_error if isinstance(is_error, bool) else UNKNOWN


def adapt_fastmcp_result(result: Any) -> Tuple[Any, Any]:
    return fastmcp_result_fields(result), fastmcp_execution_success(result)


class FastMCPTraceAdapter:
    """Connect an existing async FastMCP client call to TypedRolloutRecorder."""

    def __init__(self, recorder: Any, call_tool: Any) -> None:
        self.recorder = recorder
        self.call_tool = call_tool

    async def call(
        self,
        step_index: int,
        tool_name: str,
        arguments: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        return await self.recorder.record_async_call(
            step_index,
            tool_name,
            dict(arguments),
            lambda name, args: self.call_tool(name, args, **kwargs),
            success_from_response=fastmcp_execution_success,
            fields_from_response=fastmcp_result_fields,
        )


def trace_quality_metrics(trace: Mapping[str, Any]) -> Mapping[str, float | int]:
    events = list(trace.get("events") or [])
    count = len(events)
    recovered = sum(
        event.get("returned_fields", UNKNOWN) != UNKNOWN for event in events
    )
    typed_status = sum(
        isinstance(event.get("execution_success"), bool) for event in events
    )
    return {
        "events": count,
        "typed_value_recovery_rate": recovered / count if count else 0.0,
        "typed_execution_status_rate": typed_status / count if count else 0.0,
    }
