# Graph-Frontier Schema Audit

Audit date: 2026-09-13. Repository branch: `repro/envfactory-1p7b`.

## Step 0: read-only training snapshot

At `2026-09-13T21:48:47+08:00`:

- `tmux has-session -t parameter_aware_sft_8k_1p7b`: success.
- Progress observed in the formal log: `87/414`.
- GPU 0: 34,635 MiB, 100% utilization.
- GPU 1: 34,635 MiB, 98% utilization.
- No attach, signal, restart, process mutation, or GPU workload was performed by this audit.

## Direct answers

### 1. Is the original tool chain saved with a synthesized task?

It exists in memory but is intentionally discarded by the current serializer.

`ToolQueryChain.init_tool_chain` and each live `ToolQueryNode.raw_tool_call` contain the sampled target tools. Query generation also uses these objects when constructing target-tool prompts. However, `ToolQueryNode.save()` serializes `"raw_tool_call": []`, and `ToolQueryNode.load()` restores an empty list. The two query-generation smoke artifacts in this checkout confirm the persisted field is empty.

Therefore a live generation object can export the gold chain, but the current persisted task JSON cannot recover it.

### 2. Can every tool dependency edge be recovered?

Only from a live or separately saved `ToolGraph`, not from the persisted task/SFT artifacts currently present.

The official graph contains four edge types:

- `Tool_Input`: parameter to consumer tool, with `required`.
- `Tool_Output`: producer tool to output parameter.
- `Parameter_Relate`: producer output parameter to consumer input parameter.
- `Tool_Depend`: tool to tool.

A parameter-flow edge can be grounded by joining `Tool -> output parameter -> related/input parameter -> Tool`. The graph serializer preserves the NetworkX graph and node attributes in `graph.pkl`. No `graph.pkl` or other pickle exists in this checkout, so the present artifacts cannot reconstruct those joins.

### 3. Can `Parameter` distinguish user-provided and internal parameters?

Yes in a built graph. `Parameter.user_provided` returns the stored `_user_provided` classification. `ToolGraph.validate_parameter()` treats a parameter as available when it is user-provided or when a visited upstream tool reaches it through `Tool_Output` or `Parameter_Relate`.

Caveat: `user_provided` is produced during graph construction by an LLM-based batch classification. It is graph metadata, not an execution-time proof. If the property is absent/None in a legacy graph, the profiler must emit `unknown`.

### 4. Does a rollout artifact save tool name, arguments, result, and order?

The QueryGen trajectory does.

`QueryGenNonConv.solve()` appends ordered `tool_call` steps containing tool dictionaries and immediately following `tool_response` steps containing execution results. Selected `steps` and all `pass_k_trace` candidates are fields on `ToolQueryNode`; initial and final scenarios and pass@k scenarios are also retained.

Execution success is not stored as a typed boolean. `MCPClientManager.call_tool()` converts timeout/tool errors/exceptions into response strings. A legacy adapter can preserve those strings but must not infer success from their wording.

No real RL rollout artifact is present in this checkout. The only downloaded formal dataset is the filtered SFT JSON.

### 5. Where is the final-state verifier/reward entry?

There is no programmatic final-state verifier or RL reward implementation in this repository.

QueryGen snapshots environment state with `save_scenario`, but it does not compare final state against a verifier. Its solver-level `pass_k_decision` merely checks whether a trace called at least one tool. `select()` asks an LLM solution selector for decisions and a selected trace; that is not a strong LLM-free environment verifier.

`convert_to_rl_data()` exports:

- ground-truth tool calls under `reward_model.ground_truth`;
- `initial_config` and `final_config` under `extra_info.mcp_factory_kwargs`.

The consumer/reward function is in an external VeRL fork and is absent here, so its actual success semantics cannot be audited or invoked from this checkout.

### 6. What metadata does flattened SFT lose?

`convert_to_sft_data()` emits one Alpaca record per retained input/output pair. The flattened record keeps rendered tool calls/responses and accumulated conversational history, but loses or cannot reliably retain:

- task/conversation identifier and seed;
- original target tool chain;
- ToolGraph nodes, edge types, parameter nodes, `required`, and `user_provided`;
- node/turn identity after global dataset shuffle;
- initial and final scenarios;
- selected vs non-selected pass@k traces and scenarios;
- typed execution-success status;
- accuracy/decision provenance;
- final-state verifier outcome.

Consequently, flattened SFT is unsuitable as the primary input for a graph-grounded profiler. Exact-value text reuse is only a sampling proxy, not graph evidence.

### 7. What should the profiler consume?

Preferred input: a graph-grounded rollout bundle exported before serialization loss, containing:

1. a normalized projection of the actual `ToolGraph` for the task, including expected tools and parameter-flow edges;
2. ordered execution events with typed tool name, arguments, result, and explicit execution-success status from the executor;
3. explicit terminal/final-state verifier outcomes when available;
4. task id and optional initial/final states.

The MVP provides an adapter from a live official `ToolGraph` to that normalized graph schema, plus a conservative adapter for saved QueryGen artifacts. The latter marks expected edges, execution success, and verifier outcomes `unknown` unless a companion graph/verifier record is supplied.

## Artifact inventory

- Official graph implementation: `src/graph/tool_graph.py`, `src/graph/tool_node.py`, `src/graph/sampler.py`.
- Query generation/execution: `src/gen/query_gen/query_gen_non_conv.py`, `query_gen_conv.py`.
- Persisted chain schema: `src/graph/tool_chain.py`.
- SFT/RL exporters: `src/utils/data_process.py`.
- Environment schemas: `envs/metadata/*_metadata.json`.
- Environment generation checkpoints: `envs/intermediate/*_checkpoint.json`; these are environment-construction artifacts, not agent rollouts.
- QueryGen smoke artifacts: two terminated artifacts under `repro_1p7b/results/querygen/`; both have empty `raw_tool_call` and no executed steps.
- `graph.pkl`: absent.
- Generated gold trajectory corpus: absent.
- RL dataset: absent.
- RL rollout logs/reward implementation: absent (external VeRL integration).
