# Real Probe Readiness

## Decision

- **CPU SEMANTICS: READY**
- **REAL PROBE: NOT READY**

The v1 sidecar, eligibility gates, typed adapters, and profiler semantics are
covered by CPU tests. Runtime generation/executor wiring and one real executable
task smoke remain pending. This stage does not authorize those workloads.

## CPU semantic readiness

| Gate | Status | Evidence |
|---|---|---|
| Selected producer trace | PASS | `DependencyTraceRecorder` delegates every random decision once and records the selected producer. |
| Every selected dependency resolves | PASS | `dependency_resolution` counts selected/resolved dependencies and records ambiguous or missing graph edges explicitly. |
| OR alternatives are not treated as AND | PASS | Alternatives remain an `or` group; only the uniquely resolved selected edge is structural gold. |
| Required user-provided semantics | PASS | `missing_producer` requires both `required is True` and `internal_parameter is True`. |
| FastMCP typed adapter | PASS | Verified with FastMCP 3.1.0 / MCP 1.30.0 fields `structuredContent`, `isError`, `content`, and `TextContent.text`. |
| Conservative typed matching | PASS | Scalar exact, collection exact or schema-labeled multiset, object structural; ambiguous conversions remain unknown. |
| Authoritative depth | PASS | Sidecar node/edge depths override fallback; cycles and incomplete dependency resolution produce unknown. |
| v1 cross-turn rejection | PASS | A selected dependency whose producer and consumer are not in the same turn gets `cross_turn_dependency_not_supported_v1`. |
| Path adherence versus task success | PASS | A different tool path reaching the same reference final state remains task-successful. |
| Eligibility checker | PASS | `evaluate_probe_eligibility` and `is_structurally_diagnosable` implement a single acceptance rule. |
| CPU tests | PASS | See the commit handoff for the exact total. |

A task is structurally diagnosable only when:

```text
dependency_semantics == selected_reference
selected_dependency_trace_present == true
selected_dependency_count == resolved_dependency_count
unresolved_dependency_count == 0
cross_turn_dependency_count == 0
raw_tool_call is non-empty
initial state is available
```

A missing expected final state does not fail structural eligibility. It only makes
state diagnosis unavailable and leaves `state_success` unknown unless the
runtime supplies a typed verifier result.

## Graph-Frontier v1 scope

Graph-Frontier v1 supports **single-turn structural diagnosis only**. EnvFactory
multi-turn tasks are not declared invalid; only sidecars with selected
dependencies crossing a turn boundary are rejected for v1 structural diagnosis.
Multi-turn dependency provenance is deferred to v2.

Ambiguous selected dependencies are never silently dropped. They are retained
under `dependency_resolution.unresolved_dependencies`, the sidecar becomes
`selected_reference_incomplete`, dependency depth becomes `unknown`, and
eligibility is false.

## Final-state reference assumption

For this internship/project implementation, the selected QueryGen reference
trajectory's `final_scenario` may be used as `expected_final_state`, but it
must be labeled with provenance
`selected_querygen_reference_trajectory`. It is a reference state under
EnvFactory's own engineering assumption, not an independent oracle. The canonical sidecar field is `expected_final_state`; `expected_final_scenario` is retained only as a backward-compatible alias.

The profiler therefore keeps:

```text
path_adherence != task_success
```

A non-reference path that reaches the same canonical reference state may have
`path_adherence=false` and `task_success=true`.

## Runtime readiness

| Gate | Status | Required action |
|---|---|---|
| Production QueryGen hook wiring | PENDING | Use `sample_with_dependency_trace` and export before the lossy save boundary. |
| Production executor hook wiring | PENDING | Observe `Client.call_tool` before MCPManager flattens the result to text. |
| One executable task smoke | PENDING | Run only after explicit authorization. |
| 200-500 task pool | BLOCKED | Do not launch until all runtime gates pass. |

No core patch is required. A probe driver plus `QueryGenNonConv` and
`MCPManager` subclasses can install both hooks.

## Future one-task smoke acceptance criteria

Do not execute this smoke in the current stage. When authorized, a dependent
task must satisfy all of:

```text
dependency_semantics = selected_reference

selected_dependency_count
==
resolved_dependency_count

structural_diagnosis_eligible = true
cross_turn_dependency_count = 0
typed_execution_status_rate = 1.0
dependency edge count > 0
```

Additionally:

- every key dependency has a typed source and typed target value;
- `state_success` is boolean, or its unknown reason is explicit;
- there is no unexpected unresolved dependency;
- the profile exposes `path_adherence`,
  `structural_root_cause_failures`, `task_success`, and
  `max_dependency_depth_reached`;
- no runtime log, checkpoint, credential, or full graph dump is committed.

## Minimal lossless hook points

Gold metadata must be exported before
`src/gen/query_gen/query_gen_non_conv.py:609` calls
`context.tool_chain.save(save_path)`.

**generation-time graph metadata must be preserved before ToolQueryNode.save()
discards raw_tool_call information**

Typed rollout evidence must be captured immediately after
`Client.call_tool(short_name, args)` at
`src/manager/mcp_client_manager.py:242/245`, before line 247 joins text
content.
