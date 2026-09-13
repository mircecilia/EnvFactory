"""Conservative canonical final-state comparison."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .adapters import UNKNOWN


def canonical_state(value: Any) -> Any:
    """Canonicalize JSON state: map key order is irrelevant, list order is not."""

    if isinstance(value, Mapping):
        return {
            str(key): canonical_state(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, list):
        return [canonical_state(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return UNKNOWN


def compare_final_states(actual: Any, expected: Any) -> Any:
    if actual == UNKNOWN or expected == UNKNOWN:
        return UNKNOWN
    canonical_actual = canonical_state(actual)
    canonical_expected = canonical_state(expected)
    if canonical_actual == UNKNOWN or canonical_expected == UNKNOWN:
        return UNKNOWN
    try:
        return json.dumps(
            canonical_actual, sort_keys=True, separators=(",", ":")
        ) == json.dumps(
            canonical_expected, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return UNKNOWN
