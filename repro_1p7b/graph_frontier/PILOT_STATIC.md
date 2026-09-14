# Static Graph-Frontier capability pilot

This pilot freezes 100 single-turn, autonomous, executable tasks at seed
20260914. The task-local gold graph is exported from live `ToolGraph` and
`ToolQueryChain` objects after a typed FastMCP reference execution and before
`ToolQueryNode.save()` can discard `raw_tool_call`.

The model receives only the normal user query, the normal EnvFactory assistant
system prompt, and every public MCP tool schema for that task's environment.
Gold tools, dependency edges, reference values, reference trajectories, expected
final state, and profiler labels remain evaluator-only.

## Frozen artifacts

The committed manifest index is
`graph_frontier/manifests/pilot_100_seed_20260914.jsonl`. Full sidecars,
canonical states, typed reference traces, rollouts, profiles, and server logs live
under the ignored worktree path
`repro_1p7b/results/graph_frontier/pilot_100/`.

Every accepted candidate passes: real reference execution, a second
load/save reset round trip, exact typed source-to-target bindings, gold-sidecar
schema validation, selected dependency resolution, single-turn eligibility, and
availability of canonical initial/final state. Rejected candidates and reasons
are retained in `frozen/candidate_audit.json`.

## Fixed inference semantics

The versioned protocol is `PILOT_PROTOCOL_20260914.json`. All three models use
temperature 0, top_p 1, 1024 generated tokens, at most 8 executed tool calls,
four tool-call rounds, one user turn, and a 120-second per-task timeout. There is
no outer inference retry. Each probe receives a fresh MCP client and reloads the
same frozen initial state.

A probe is capability-valid only if no timeout, model-server failure, environment
failure, or profiler failure occurred. Task success requires every gold node to
execute successfully, every selected internal edge to be complete, and exact
canonical final-state equality. System failures are excluded rather than scored
as capability failures.

## Commands

Freeze/validate:

    LITELLM_LOCAL_MODEL_COST_MAP=True python -m repro_1p7b.graph_frontier.pilot_static freeze

Run one model against the same full manifest:

    LITELLM_LOCAL_MODEL_COST_MAP=True SGLANG_BASE_URL=http://127.0.0.1:PORT/v1 SGLANG_API_KEY=KEY SGLANG_MODEL=MODEL python -m repro_1p7b.graph_frontier.pilot_static run --model base --output-dir repro_1p7b/results/graph_frontier/pilot_100/base

Aggregate after all three complete:

    python -m repro_1p7b.graph_frontier.pilot_analyze

No training path exists in either pilot module.
