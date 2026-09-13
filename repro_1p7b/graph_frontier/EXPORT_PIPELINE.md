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

They never use `id(obj)`, object addresses, or NetworkX node identity. `expected_final_scenario` is `unknown` by default because the current node's `final_scenario` is the selected solver's observed state, not an independent gold verifier result. A caller may supply an expected final scenario only when its provenance is reliable.

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

For a runtime where exceptions propagate, use the supplied wrapper:

```python
response = await recorder.record_async_call(
    step_index=step_index,
    tool_name=short_name,
    tool_arguments=args,
    executor=client.call_tool,
    success_from_response=lambda _result: True,
    server_id=server_name,
)
```

`success_from_response=lambda _: True` is valid only at this inner boundary, where a return means `client.call_tool` completed without raising. It must not be used around the current outer `MCPManager.call_tool()`.

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
