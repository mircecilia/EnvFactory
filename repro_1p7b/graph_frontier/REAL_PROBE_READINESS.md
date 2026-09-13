# Real Probe Readiness

## Decision

**NOT READY for the 200-500 task real probe launch.**

The CPU-side semantics are implemented and tested, but the production generation
and execution entry points have not yet been wired to the adapters, and no
single-task live environment smoke has been authorized. Starting the pool now
would risk producing traces without selected-dependency or typed-result evidence.

## Readiness checklist

| Gate | Status | Evidence / remaining action |
|---|---|---|
| Selected producer is distinguished from all possible producers | PASS | `DependencyTraceRecorder` observes the existing sampler and records the chosen producer plus an explicit OR alternative group without adding RNG calls. |
| Sidecar contains only the selected/reference dependency | PASS | Ambiguous untraced graphs are marked `possible_graph_unselected` and emit no asserted dependency edge. |
| Required user-provided input is not a missing producer | PASS | `missing_producer` requires both `required is True` and `internal_parameter is True`; unknown stays unknown. |
| FastMCP result fields are verified against the installed environment | PASS | Remote environment: FastMCP 3.1.0 and MCP 1.30.0. `CallToolResult` has `structuredContent`, `isError`, and `content`; `TextContent` has `text`. |
| Typed response recovery is conservative | PASS | Structured content first; strict JSON-only single TextContent fallback; non-JSON text unknown; exceptions false. |
| Trace quality is measurable | PASS | Raw trace and profile expose `typed_value_recovery_rate` and `typed_execution_status_rate`. |
| Final state has a named expected source | PASS | QueryGen node `final_scenario` is accepted only under the explicit engineering assumption `selected_querygen_reference_trajectory`. |
| Path adherence and task success are separate | PASS | State success suppresses reference-path deviations as task-level roots while retaining structural diagnostics. |
| Sidecar depth is authoritative | PASS | Node/edge sidecar depths override fallback; cycle fallback is unknown; optional unknown incoming edges do not block reached depth. |
| Typed parameter values use exact semantics | PASS | Scalar exact, collection exact, object structural; collection-to-scalar needs explicit `member_selection`; ambiguous type is unknown. |
| CPU test suite | PASS | Full result is recorded in the commit handoff; no GPU workload is used. |
| Production QueryGen hook wired | PENDING | Use the wrapper/subclass hook below before the first real task. |
| Production MCP executor hook wired | PENDING | Wrap the `Client.call_tool` result before MCPManager flattens `result.content`. |
| One-task live executable smoke | PENDING | Requires later authorization because it invokes a real environment/model path. |
| 200-500 task pool | BLOCKED BY ABOVE | Do not launch until both hooks and the one-task smoke pass. |

## Minimal hook points

### A. Gold graph and selected dependency

At the caller that currently invokes `ToolGraph.sample`, replace only the call
boundary:

```python
chain = sample_with_dependency_trace(
    tool_graph,
    topology_sampler,
    max_nodes=max_nodes,
    start_node=start_node,
    seed=seed,
)
```

Then use a wrapper/subclass of `QueryGenNonConv` whose `terminate` calls
`GenerationSidecarCallback.before_save(context)` immediately before delegating
to the existing terminate implementation. The current core save is
`src/gen/query_gen/query_gen_non_conv.py:609`.

**generation-time graph metadata must be preserved before ToolQueryNode.save()
discards raw_tool_call information**

No core patch is required: both operations can be performed by the probe driver
and a `QueryGenNonConv` subclass. If the future driver cannot inject a subclass,
the minimal core proposal is one optional `before_save_callback` invocation
immediately before line 609, defaulting to `None`.

### B. Typed rollout

The last lossless boundary is `Client.call_tool(short_name, args)` at
`src/manager/mcp_client_manager.py:242/245`. Line 247 flattens content to text.
Wrap the client call with `FastMCPTraceAdapter` and only then preserve the
existing outward text conversion. The wrapper returns the original
`CallToolResult`, so existing behavior can remain unchanged.

If the probe runner must use `MCPManager` directly, subclass it and override
`_call_tool_async`; do not patch the training checkout. A core callback proposal
would accept an optional result observer immediately after lines 242/245 and
before line 247.

## Required one-task acceptance gate

Before a real pool, one executable task must show all of:

- dependency semantics is `selected_reference`;
- selected edge count is nonzero for a dependent task;
- OR alternatives are retained but not treated as AND requirements;
- typed value recovery and typed execution status rates are reported;
- final state or verifier result is typed;
- no runtime log, checkpoint, credential, or full graph dump is committed.

External BFCL remains an external evaluation only and is not used for
capability diagnosis.
