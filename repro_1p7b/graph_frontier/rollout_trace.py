"""Typed, machine-readable rollout trace recorder and executor wrappers."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from .adapters import UNKNOWN

_FINAL_STATUSES = {"success", "failure", UNKNOWN}


def _known_bool(value: Any) -> Any:
    return value if isinstance(value, bool) else UNKNOWN


def _is_unknown(value: Any) -> bool:
    return isinstance(value, str) and value == UNKNOWN


def _structured_value(value: Any) -> Any:
    if _is_unknown(value):
        return UNKNOWN
    candidate = value
    if hasattr(candidate, "model_dump") and callable(candidate.model_dump):
        candidate = candidate.model_dump()
    elif hasattr(candidate, "to_dict") and callable(candidate.to_dict):
        candidate = candidate.to_dict()
    try:
        json.dumps(candidate, ensure_ascii=False)
    except (TypeError, ValueError):
        return UNKNOWN
    return candidate


def _returned_fields(response: Any) -> Any:
    structured = _structured_value(response)
    return structured if isinstance(structured, (Mapping, list)) else UNKNOWN


def _exception_record(exception: Any) -> Any:
    if exception is None or _is_unknown(exception):
        return UNKNOWN
    if isinstance(exception, BaseException):
        return {"type": type(exception).__name__, "message": str(exception)}
    structured = _structured_value(exception)
    if isinstance(structured, (str, Mapping)):
        return structured
    return UNKNOWN


def _timestamp(auto_timestamp: bool, value: Any) -> Any:
    if isinstance(value, str) and value:
        return value
    if auto_timestamp:
        return datetime.now(timezone.utc).isoformat()
    return UNKNOWN


class TypedRolloutRecorder:
    """Accumulate a typed trace without interpreting human-readable responses."""

    def __init__(
        self,
        task_id: str,
        environment_identifiers: Any = UNKNOWN,
        initial_environment_state: Any = UNKNOWN,
        auto_timestamp: bool = False,
    ) -> None:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a non-empty string")
        self.auto_timestamp = auto_timestamp
        self.trace: Dict[str, Any] = {
            "schema_version": "envfactory_rollout_trace_v1",
            "task_id": task_id,
            "environment_identifiers": _structured_value(environment_identifiers),
            "initial_environment_state": _structured_value(initial_environment_state),
            "events": [],
            "final_status": UNKNOWN,
            "terminal_success": UNKNOWN,
            "final_environment_state": UNKNOWN,
            "verifier_result": UNKNOWN,
            "verifier_details": UNKNOWN,
            "trace_notes": [],
            "trace_quality_metrics": {
                "events": 0,
                "typed_value_recovery_rate": 0.0,
                "typed_execution_status_rate": 0.0,
            },
        }

    def record_tool_event(
        self,
        step_index: int,
        tool_name: str,
        tool_arguments: Any,
        tool_response: Any = UNKNOWN,
        returned_fields: Any = UNKNOWN,
        execution_success: Any = UNKNOWN,
        exception: Any = UNKNOWN,
        timestamp: Any = UNKNOWN,
        environment_id: Any = UNKNOWN,
        server_id: Any = UNKNOWN,
    ) -> Dict[str, Any]:
        if not isinstance(step_index, int) or step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("tool_name must be a non-empty string")
        known_success = _known_bool(execution_success)
        exception_record = _exception_record(exception)
        if known_success is True and exception_record != UNKNOWN:
            raise ValueError("a successful event cannot also contain an exception")

        response = _structured_value(tool_response)
        arguments = _structured_value(tool_arguments)
        fields = (
            _returned_fields(tool_response)
            if _is_unknown(returned_fields)
            else _structured_value(returned_fields)
        )
        if _is_unknown(arguments) and not _is_unknown(tool_arguments):
            self.trace["trace_notes"].append(f"step {step_index}: tool arguments were not JSON serializable")
        if _is_unknown(response) and not _is_unknown(tool_response):
            self.trace["trace_notes"].append(f"step {step_index}: tool response was not JSON serializable")

        event = {
            "step_index": step_index,
            "tool_name": tool_name,
            "tool_arguments": arguments,
            "tool_response": response,
            "returned_fields": fields,
            "execution_success": known_success,
            "exception": exception_record,
            "timestamp": _timestamp(self.auto_timestamp, timestamp),
            "environment_id": environment_id if isinstance(environment_id, str) and environment_id else UNKNOWN,
            "server_id": server_id if isinstance(server_id, str) and server_id else UNKNOWN,
        }
        self.trace["events"].append(event)
        events = self.trace["events"]
        count = len(events)
        self.trace["trace_quality_metrics"] = {
            "events": count,
            "typed_value_recovery_rate": (
                sum(item["returned_fields"] != UNKNOWN for item in events) / count
            ),
            "typed_execution_status_rate": (
                sum(isinstance(item["execution_success"], bool) for item in events) / count
            ),
        }
        return event

    def finalize(
        self,
        final_status: str = UNKNOWN,
        final_environment_state: Any = UNKNOWN,
        verifier_result: Any = UNKNOWN,
        verifier_details: Any = UNKNOWN,
    ) -> Dict[str, Any]:
        if final_status not in _FINAL_STATUSES:
            raise ValueError(f"final_status must be one of {sorted(_FINAL_STATUSES)}")
        self.trace["final_status"] = final_status
        self.trace["terminal_success"] = (
            True if final_status == "success" else False if final_status == "failure" else UNKNOWN
        )
        self.trace["final_environment_state"] = _structured_value(final_environment_state)
        self.trace["verifier_result"] = _known_bool(verifier_result)
        self.trace["verifier_details"] = _structured_value(verifier_details)
        return self.as_dict()

    def as_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self.trace, ensure_ascii=False))

    def write(self, output_path: str | Path) -> Path:
        return write_rollout_trace(self.trace, output_path)

    def record_sync_call(
        self,
        step_index: int,
        tool_name: str,
        tool_arguments: Any,
        executor: Callable[[str, Any], Any],
        success_from_response: Optional[Callable[[Any], Any]] = None,
        fields_from_response: Optional[Callable[[Any], Any]] = None,
        environment_id: Any = UNKNOWN,
        server_id: Any = UNKNOWN,
    ) -> Any:
        """Wrap an executor; without a typed classifier, non-exception success stays unknown."""
        try:
            response = executor(tool_name, tool_arguments)
        except Exception as exc:
            self.record_tool_event(
                step_index,
                tool_name,
                tool_arguments,
                execution_success=False,
                exception=exc,
                environment_id=environment_id,
                server_id=server_id,
            )
            raise
        success = success_from_response(response) if success_from_response is not None else UNKNOWN
        self.record_tool_event(
            step_index,
            tool_name,
            tool_arguments,
            tool_response=response,
            returned_fields=(fields_from_response(response) if fields_from_response is not None else UNKNOWN),
            execution_success=success,
            environment_id=environment_id,
            server_id=server_id,
        )
        return response

    async def record_async_call(
        self,
        step_index: int,
        tool_name: str,
        tool_arguments: Any,
        executor: Callable[[str, Any], Awaitable[Any]],
        success_from_response: Optional[Callable[[Any], Any]] = None,
        fields_from_response: Optional[Callable[[Any], Any]] = None,
        environment_id: Any = UNKNOWN,
        server_id: Any = UNKNOWN,
    ) -> Any:
        """Async equivalent intended for the client.call_tool boundary."""
        try:
            response = executor(tool_name, tool_arguments)
            if inspect.isawaitable(response):
                response = await response
        except Exception as exc:
            self.record_tool_event(
                step_index,
                tool_name,
                tool_arguments,
                execution_success=False,
                exception=exc,
                environment_id=environment_id,
                server_id=server_id,
            )
            raise
        success = success_from_response(response) if success_from_response is not None else UNKNOWN
        self.record_tool_event(
            step_index,
            tool_name,
            tool_arguments,
            tool_response=response,
            returned_fields=(fields_from_response(response) if fields_from_response is not None else UNKNOWN),
            execution_success=success,
            environment_id=environment_id,
            server_id=server_id,
        )
        return response


def write_rollout_trace(trace: Mapping[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(trace), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    return output


def load_rollout_trace(source: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    with Path(source).open("r", encoding="utf-8") as handle:
        return json.load(handle)
