# Graph-Frontier Probe Dataset Plan

## Purpose

Build a small, fixed diagnostic pool from EnvFactory's own executable environments. This pool is separate from SFT/RL training data and external evaluation data. It diagnoses where tool-graph execution breaks; it is not a new training set.

BFCL remains an external final evaluation only and does not contribute examples, graph metadata, capability labels, thresholds, or selection decisions to this probe pool.

## First pool

Target 300 accepted tasks, within the requested 200-500 range. Generation may over-sample candidates, but acceptance requires an executable environment, a non-empty live `raw_tool_call`, a valid task-local sidecar, and a reproducible initial scenario.

Suggested depth allocation:

| Gold dependency depth | Accepted tasks |
|---|---:|
| 0 (single tool/control) | 40 |
| 1 | 70 |
| 2 | 80 |
| 3 | 70 |
| 4+ | 40 |

Within each bucket, cap any one environment/server family so that one API family cannot dominate the pool. Record actual counts rather than silently filling a missing depth with easier tasks.

## Generation protocol

1. Reserve probe-only seeds and a probe output directory before generation; reject any seed/task ID found in training manifests.
2. Sample from EnvFactory's executable environments with the official graph and sampler, using `sample_with_dependency_trace` at the sampling boundary.
3. During QueryGen, call `GenerationSidecarCallback.before_save(context)` before the lossy `ToolQueryNode.save()` path.
4. Save the original generated query artifact and `<task>.gold.json` under the same manifest task ID.
5. Validate each sidecar against `GOLD_SIDECAR_SCHEMA.json` and `PROBE_ELIGIBILITY_SCHEMA.json`; accept structural tasks only when `is_structurally_diagnosable(sidecar)` is true, and reject empty sequences, unresolved selected dependencies, cross-turn selected dependencies, missing initial scenarios, task-ID collisions, and non-executable servers.
6. Freeze the accepted manifest, sidecars, generation seeds, environment versions, and depth counts. Do not regenerate per model.

For this internship/project implementation, the selected QueryGen reference trajectory's `final_scenario` may be used as `expected_final_state` when it is explicitly labeled with provenance `selected_querygen_reference_trajectory`. It is a reference state under EnvFactory's own engineering assumption, not an independent oracle. If it is unavailable, structural diagnosis may proceed while state success remains `unknown`.

## Evaluation matrix

Run the exact same frozen task IDs, initial scenarios, sidecars, environment versions, decoding settings, and rollout seeds for:

1. Base Qwen3-1.7B;
2. Original EnvFactory SFT;
3. Static Parameter-Aware SFT.

Each run writes one typed `<task>.rollout.json` and one profiler output. Missing/invalid runtime evidence remains `unknown`; do not infer it from response text. Aggregate pass rate and failure localization only after schema validation and task-ID joins succeed.

## Separation and leakage controls

- Store probe artifacts outside training dataset directories.
- Compare task IDs, generation seeds, normalized queries, gold tool sequences, and initial-scenario identifiers against training manifests before freezing.
- Do not add probe tasks back into SFT or curriculum generation.
- Keep failed generation candidates in a separate audit manifest, not in the accepted probe denominator.
- Version a single immutable `probe_manifest.json` containing task ID, seed, server IDs, sidecar path, depth, and acceptance reason.
- Record model/checkpoint IDs and runtime code commit per evaluation, while leaving the probe manifest unchanged.

## Deferred work

Do not generate the pool until the required generation API/model access and executable environments are intentionally scheduled. This stage creates only the schemas, exporters, hook design, and CPU mock. It launches no probe rollout, LLM/API generation, SGLang service, training, or GPU workload.

## Preflight required by semantic hardening

The probe driver must call `sample_with_dependency_trace` instead of calling
`ToolGraph.sample` directly, then export each sidecar before QueryGen saves the
chain. Graph-Frontier v1 accepts only single-turn structural dependencies.
`possible_graph_unselected`, `selected_reference_incomplete`, unresolved
selected dependencies, and cross-turn selected dependencies are invalid for
structural diagnosis and must be rejected or regenerated.

The executor must record the original FastMCP `CallToolResult` before any text
flattening and must report both typed recovery rates. Run one executable task as
a gate before creating the 200-500 task pool. See `REAL_PROBE_READINESS.md`.
