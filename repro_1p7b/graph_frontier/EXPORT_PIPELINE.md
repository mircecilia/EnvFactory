# EnvFactory Generation-to-Profiler Export Pipeline

## Scope and invariant

This pipeline preserves structured task metadata and execution evidence without changing `src/`, persisting a full `graph.pkl`, or recovering a graph from flattened SFT text.

> generation-time graph metadata must be preserved before ToolQueryNode.save() discards raw_tool_call information

Current `ToolQueryNode.save()` writes `raw_tool_call: []`, and `ToolQueryChain.save()` does not serialize `init_tool_chain`. Old flattened 26K SFT artifacts therefore cannot be upgraded into reliable gold dependency graphs with text heuristics.

## Data flow

```text
live ToolGraph + live ToolQueryChain/ToolQueryNode
  -> <task>.gold.json (task-local relevant subgraph)

actual structured executor boundary
  -> <task>.rollout.json (typed calls, responses, exceptions, verifier evidence)

gold sidecar + typed rollout
  -> sidecar_and_trace_to_bundle()
  -> profile_rollout()
  -> Graph-Frontier profile
```

A full NetworkX pickle may still be saved separately for debugging, but it is not an input requirement for profiling.

## Gold sidecar

`gold_sidecar.py` exports one sidecar per query turn while `raw_tool_call` and the official graph are still live. It records the sampled sequence, task-local required tool nodes, forward dependency edges among those tools, input/output parameter records, seed, servers, query/turn identity, and available scenarios.

Stable identifiers are content-derived:

```text
tool::<tool_name>
<tool_name>::input::<parameter_name>
<tool_name>::output::<parameter_name>
```

They never use `id(obj)`, object addresses, or NetworkX node identity. For this internship/project implementation, the selected QueryGen reference trajectory's `final_scenario` may be used as `expected_final_state` only with provenance `selected_querygen_reference_trajectory`. It is an EnvFactory engineering reference state, not an independent oracle. If absent, structural diagnosis remains available while state success is `unknown`.

### Hook A: generation-time gold export

The narrowest point is `src/gen/query_gen/query_gen_non_conv.py:603-609`: `terminate()` still receives a `QueryGenContext` holding both `context.tool_graph` and `context.tool_chain`, immediately before `context.tool_chain.save(save_path)` serializes a lossy representation.

No core patch is required. A probe-only subclass can invoke the supplied callback and then delegate:

```python
from repro_1p7b.graph_frontier.gold_sidecar import GenerationSidecarCallback
from src.gen.query_gen.query_gen_non_conv import QueryGenNonConv

callback = GenerationSidecarCallback("probe_artifacts/gold")

class ProbeQueryGenNonConv(QueryGenNonConv):
    async def terminate(self, context):
        callback.before_save(context)  # must precede super().terminate(context)
        await super().terminate(context)
```

The callback can accept a caller-owned task-id factory. Without one, the exporter assigns a deterministic content hash from seed, turn index, query, and sampled tool sequence. Probe manifests should still provide explicit task IDs.

## Typed rollout trace

`TypedRolloutRecorder` writes machine-readable `envfactory_rollout_trace_v1` JSON. Every event retains structured arguments, structured response/returned fields where the executor exposes them, a typed success flag or `unknown`, a typed exception record, optional timestamp, and environment/server identifiers. Final status and verifier success are independent fields.

The recorder deliberately does not parse strings for words such as `error`, `failed`, or `success`. Non-exception returns stay `unknown` unless a runtime-owned typed classifier is supplied.

### Hook B: actual tool execution

The exact structured boundary is `src/manager/mcp_client_manager.py:232-247`, specifically the two `await client.call_tool(short_name, args)` calls at lines 242 and 245:

- `args` is structured JSON;
- the MCP result object still owns structured content;
- exceptions have not yet been converted to strings;
- line 247 has not yet flattened content to joined text.

Wrapping `Gen.execute()` (`src/gen/__init__.py:374-402`) is too late for reliable status: `MCPClientManager.call_tool()` converts timeout, `ToolError`, and other exceptions to ordinary strings at lines 223-230. The repository's RL converter (`src/utils/data_process.py:290-351`) serializes selected calls plus initial/final configs; it is not an execution runtime and provides no typed executor result or verifier callback. No separate reward/rollout runtime implementation is present in this checkout.

At this inner boundary, use the typed FastMCP adapter:

```python
adapter = FastMCPTraceAdapter(recorder, client.call_tool)
result = await adapter.call(step_index, short_name, args)
```

The adapter reads typed `structuredContent` / `isError` before flattening. Exceptions are recorded as failures, strict JSON TextContent is the only fallback, and natural-language response text is never classified.

### Optional minimal core callback proposal (not applied)

For first-party integration without a client proxy, add an optional observer to `MCPClientManager` and call it around the two inner `client.call_tool` awaits:

```text
before call: observer.on_tool_start(client_id, short_name, args)
on return:   observer.on_tool_result(client_id, short_name, args, result)
on except:   observer.on_tool_error(client_id, short_name, args, exception); re-raise
```

The observer should be disabled by default and supplied only by probe execution. This is the only proposed core change; none was made in this stage.

## Joining and profiling

```python
from repro_1p7b.graph_frontier.export_pipeline import profile_sidecar_and_trace

profile = profile_sidecar_and_trace(
    "probe-001.gold.json",
    "probe-001.rollout.json",
)
```

The adapter rejects mismatched task IDs. Unknown response fields, final state, verifier status, or edge parameters remain exactly `unknown` and do not become failed attempts.

## CPU verification

```bash
python -m py_compile repro_1p7b/graph_frontier/*.py
python -m unittest discover -s repro_1p7b/graph_frontier/tests -v
```

`test_export_pipeline.py` constructs the required `A -> user_id -> B -> order_id -> C` graph and verifies the `456 -> 999` first broken edge, missing producers, extra/repeated calls, final-state-only failure, unknown final state, atomic JSON files, and both schemas. It does not load a model, environment server, SGLang, or GPU runtime.

## Semantic hardening (2026-09-13)

The probe path now has three CPU-only adapters:

- `sample_with_dependency_trace(...)` delegates to the existing sampler while
  recording the selected producer and explicit OR alternatives.
- `FastMCPTraceAdapter` consumes the real FastMCP `CallToolResult` before
  text flattening, preferring `structuredContent` and typed `isError`.
- `compare_final_states` provides canonical JSON comparison when a runtime
  verifier result is absent.

The gold sidecar marks dependency provenance as `selected_reference`,
`selected_reference_incomplete`, `unique_possible_equals_selected`, or
`possible_graph_unselected`. Every selected dependency is counted in
`dependency_resolution`. Ambiguous source parameters and missing graph edges
remain explicit unresolved records; they are never silently converted to a
zero-edge/depth-0 task.

The profile reports reference `path_adherence` separately from `task_success`,
and reports structural roots separately from task-level roots. It also carries
`typed_value_recovery_rate` and `typed_execution_status_rate`.

For current readiness and exact hook points, see
`REAL_PROBE_READINESS.md`.

## v1 structural eligibility

Graph-Frontier v1 supports single-turn structural diagnosis only. For each turn,
the exporter inspects every trace record whose consumer is in that turn. If its
selected producer is outside the same `raw_tool_call`, the sidecar records a
cross-turn dependency and `evaluate_probe_eligibility` returns
`cross_turn_dependency_not_supported_v1`.

Structural acceptance requires `selected_reference`, a present sampler trace,
equal selected/resolved dependency counts, no unresolved or cross-turn
dependency, non-empty `raw_tool_call`, and an available initial state. Expected
final state availability is reported separately and is not a structural gate.
The standalone report schema is `PROBE_ELIGIBILITY_SCHEMA.json`.

