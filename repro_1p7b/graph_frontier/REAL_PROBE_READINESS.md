# Real Probe Readiness

## Decision

- **CPU SEMANTICS: READY**
- **REAL SINGLE-TASK PROBE: READY**
- **12-CASE PIPELINE STABILITY PROBE: READY**
- **200-500 TASK PROBE POOL: NOT STARTED**

The v1 sidecar, eligibility gates, typed adapters, profiler semantics, production
wrapper hooks, and both single-task and 12-case executable smokes are verified.
This does not authorize the planned probe pool or turn these fixture-driven
results into a model-quality claim.

## CPU semantic readiness

| Gate | Status | Evidence |
|---|---|---|
| Selected producer trace | PASS | `DependencyTraceRecorder` delegates every random decision once and records the selected producer. |
| Every selected dependency resolves | PASS | `dependency_resolution` counts selected/resolved dependencies and records ambiguous or missing graph edges explicitly. |
| OR alternatives are not treated as AND | PASS | Alternatives remain an `or` group; only the uniquely resolved selected edge is structural gold. |
| Required user-provided semantics | PASS | `missing_producer` requires both `required is True` and `internal_parameter is True`. |
| FastMCP typed adapter | PASS | Verified against actual `CallToolResult` fields `structuredContent`, `isError`, `content`, and `TextContent.text`. |
| Conservative typed matching | PASS | Scalar exact, collection exact or schema-labeled multiset, object structural; ambiguous conversions remain unknown. |
| Authoritative depth | PASS | Sidecar node/edge depths override fallback; cycles and incomplete dependency resolution produce unknown. |
| v1 cross-turn rejection | PASS | A selected dependency whose producer and consumer are not in the same turn gets `cross_turn_dependency_not_supported_v1`. |
| Path adherence versus task success | PASS | A different tool path reaching the same reference final state remains task-successful. |
| Eligibility checker | PASS | `evaluate_probe_eligibility` and `is_structurally_diagnosable` implement a single acceptance rule. |
| CPU tests | PASS | 52 tests passed in the project environment before the real smoke. |

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

A missing expected final state does not fail structural eligibility. It only
makes state diagnosis unavailable and leaves `state_success` unknown unless the
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
EnvFactory's own engineering assumption, not an independent oracle. The
canonical sidecar field is `expected_final_state`;
`expected_final_scenario` is retained only as a backward-compatible alias.

The profiler therefore keeps:

```text
path_adherence != task_success
```

A non-reference path that reaches the same canonical reference state may have
`path_adherence=false` and `task_success=true`.

## Runtime readiness

| Gate | Status | Evidence |
|---|---|---|
| QueryGen hook wiring | PASS | `ProbeQueryGenNonConv.terminate()` exports through `GenerationSidecarCallback` before delegating to the lossy core save. |
| Executor hook wiring | PASS | The probe-only wrapper observes `MCPManager._call_tool_async` at the typed `client.call_tool` boundary and restores the original method afterward. |
| One executable task smoke | PASS | Parameter-Aware checkpoint, one CampusCard dependency, two real MCP calls, schemas and profiler all passed on GPU 1. |
| Expanded pipeline smoke | PASS | 12/12 deterministic cases, 24 typed MCP calls, all 12 dependency edges satisfied. |
| 200-500 task pool | PENDING | Do not launch without a separate authorization and a frozen manifest. |

No core patch is required. The reusable drivers are `real_probe_smoke.py` and
`real_probe_batch.py`. Runtime JSON and logs remain outside Git.

## Real smoke evidence (2026-09-14)

Task `graph-frontier-real-smoke-001` used the gold dependency:

```text
CampusCard-query_balance.userId
  -> CampusCard-recharge.userId
```

Observed typed execution:

```text
step 0 query_balance(userId="student-001")
       -> userId="student-001", balance=20.0, currency="CNY"

step 1 recharge(userId="student-001", amount=50, paymentMethod="bank_card")
       -> success=true, balanceAfter=70.0
```

Acceptance results:

```text
dependency_semantics = selected_reference
selected_dependency_count = resolved_dependency_count = 1
structural_diagnosis_eligible = true
cross_turn_dependency_count = 0
typed_value_recovery_rate = 1.0
typed_execution_status_rate = 1.0
dependency edge status = satisfied
path_adherence = true
task_success = true
max_dependency_depth_reached = 1
root_cause_failures = []
```

The smoke uses an explicit execution-fixture instruction requiring both target
calls and exact literal inputs. Therefore it verifies the export/runtime/profile
pipeline, not autonomous model capability. Earlier unassisted attempts were
correctly rejected: one hallucinated `user123`; another stopped after
`query_balance`. Those failures were not relabeled as passes.

## Expanded pipeline smoke evidence (2026-09-14)

The opt-in batch driver completed 12/12 cases in 61.01 seconds on GPU 1. It
covered 12 distinct user IDs, all three supported payment methods, initial
balances from 0.0 through 1000.0 CNY, and recharge amounts from 0.01 through
333.0 CNY.

Across 12 sidecars, 12 typed rollouts, and 12 profiles:

```text
real MCP events = 24
typed_value_recovery_rate = 1.0
typed_execution_status_rate = 1.0
dependency edges with both typed values = 12/12
dependency edges with exact value match = 12/12
state_success = 12/12
task_success = 12/12
path_adherence = 12/12
root_cause_failures = 0
```

This batch uses the same explicit two-call execution fixture as the one-task
smoke. Its purpose is to test repeated scenario reset, identifier and numeric
value preservation, payment-method coverage, artifact isolation, and profiler
stability. It remains explicitly excluded from model-capability claims.

## Reproduction outline

Use an idle GPU and a port distinct from ongoing evaluation. Start a local
OpenAI-compatible SGLang endpoint for the checkpoint, then run:

```bash
export SGLANG_BASE_URL=http://127.0.0.1:1054/v1
export SGLANG_API_KEY=graph-frontier-smoke
export SGLANG_MODEL=repro_1p7b/checkpoints/parameter_aware_sft_8k_1p7b

/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python \
  -m repro_1p7b.graph_frontier.real_probe_smoke \
  --output-dir /tmp/envfactory_graph_frontier_real_smoke

/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python \
  -m repro_1p7b.graph_frontier.real_probe_batch \
  --limit 12 \
  --output-dir /tmp/envfactory_graph_frontier_mini_probe
```

For a long-lived launch, put the server and driver in separate tmux sessions.
Stop only the smoke server after the driver completes; never attach to or stop
the production evaluation server.

## Minimal lossless hook points

Gold metadata is exported before
`src/gen/query_gen/query_gen_non_conv.py:609` calls
`context.tool_chain.save(save_path)`.

**generation-time graph metadata must be preserved before ToolQueryNode.save()
discards raw_tool_call information**

Typed rollout evidence is captured immediately around
`Client.call_tool(short_name, args)` in
`src/manager/mcp_client_manager.py`, before the result is flattened to text.
