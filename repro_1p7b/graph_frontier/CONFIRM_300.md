# Graph-Frontier 300-Probe Confirmation

This study freezes 300 executable, single-turn tasks before any model rollout.
The diagnosis/held-out split is 240/60 and is task-disjoint, not
environment-disjoint. The target depth distribution is 120/110/70 for depth
1/2/3+, with all tasks drawn from the five already validated executable
environments.

Every accepted task has a real FastMCP reference execution, a reset round trip,
a generation-time gold sidecar, resolved selected producers, typed internal
bindings, canonical initial/final states, and a supported deterministic semantic
verifier. Candidate rejection reasons are retained rather than silently skipped.

The frozen full manifest and runtime artifacts live under the ignored
`repro_1p7b/results/graph_frontier/confirm_300/` tree. The compact manifest,
protocol, summary, deterministic analyzer, and final reports are committed.

## Semantic verification

Mutation tasks use template-specific target-state predicates. Read-only and
mixed tasks additionally require the exact typed reference observation and
deterministic normalized values in persisted final assistant content. No LLM
grader is used. The text matcher is deliberately conservative and can produce
false negatives for semantically correct paraphrases.

Every rollout persists:

- final assistant content and finish reason;
- complete assistant/tool trace;
- typed tool arguments and recovered returned fields;
- execution status and exceptions;
- final environment state and canonical-state verifier result.

## Fixed protocol

The protocol is `CONFIRM_300_PROTOCOL_20260914.json`: temperature 0, top-p 1,
thinking disabled, 1024 generated tokens, at most 8 executed calls and four
tool-call rounds, a 120-second per-task timeout, and no outer retry. All models
consume the same frozen manifest.

## Commands

Freeze:

    LITELLM_LOCAL_MODEL_COST_MAP=True python -m repro_1p7b.graph_frontier.confirm_300 freeze

Run one model after starting its single-GPU SGLang server:

    LITELLM_LOCAL_MODEL_COST_MAP=True SGLANG_BASE_URL=http://127.0.0.1:PORT/v1 SGLANG_API_KEY=KEY SGLANG_MODEL=MODEL python -m repro_1p7b.graph_frontier.confirm_300 run --model LABEL --output-dir repro_1p7b/results/graph_frontier/confirm_300/LABEL

Analyze only after all three runs complete:

    python -m repro_1p7b.graph_frontier.confirm_analyze

The analyzer enforces a minimum of 285 valid probes per model before allowing a
capability interpretation. It uses paired exact McNemar tests at task level;
edge-level results remain descriptive because edges within a task are clustered.
