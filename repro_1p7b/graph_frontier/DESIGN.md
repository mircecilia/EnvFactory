# Graph-Frontier Curriculum Design

## Experiment family

| Track | Definition | Current role |
|---|---|---|
| A | Original EnvFactory sampling | Static topology-aware baseline. |
| B | Static Parameter-Aware | Fixed, proxy-reweighted SFT dataset; the current 1.7B training run. |
| C | Dynamic Graph-Frontier curriculum | Aggregate graph-grounded rollout capability and prioritize tasks near the current capability frontier. This MVP prepares its offline infrastructure only. |
| D | Future Graph-Frontier RL rollout allocation | Use the same capability map to allocate executable rollouts, including optional privileged graph hints. Not run in this stage. |

The lightweight design borrows four ideas without reproducing any paper: dependency/root-cause localization, capability-frontier sampling, hint-based regret, and learning-progress curriculum.

## Boundary

This stage is CPU-only. It does not run a model, environment, rollout, grader, hint experiment, or training job. It does not modify `src/`, the running SFT config/launcher/dataset, or runtime artifacts.

## Normalized inputs

### Graph specification

A task-specific graph projection contains:

- expected tool nodes;
- dependency edges;
- producer and consumer tool names;
- source and target parameter paths where the official graph supplies them;
- official edge provenance/type;
- required and internal-parameter flags, each allowing `unknown`;
- optional reliable tool metadata such as `state_changing`, otherwise `unknown`;
- selected dependency resolution counts, unresolved records, and cross-turn flags.

`tool_graph_to_spec()` projects a live EnvFactory `ToolGraph` by joining official `Tool_Output`, `Parameter_Relate`, and `Tool_Input` edges. Explicit `Tool_Depend` edges with no parameter mapping remain order-only dependencies; parameter propagation for them is `unknown`.

### Rollout bundle

A rollout bundle contains ordered typed events:

```text
task_id
terminal_success: true | false | unknown
state_success: true | false | unknown
events[]:
  step_index
  tool_name
  arguments
  result
  execution_success: true | false | unknown
```

Adapters never infer a typed success flag from an error-looking string. Legacy QueryGen JSON can recover call/response ordering, but missing graph and verifier fields remain `unknown`.

## Profiler semantics

For the first consumer invocation of each expected edge:

1. If an input is explicitly both required and internal and its selected producer was never called before the consumer, record `missing_producer` as a root dependency failure. User-provided or unknown provenance never becomes a missing-producer failure.
2. If the producer has an earlier root failure, record the consumer edge/call as downstream propagation.
3. If the producer explicitly failed execution without an earlier dependency failure, record one producer `tool_execution_failure` root; downstream failures point to it.
4. If source and target parameter paths and values are available, exact typed equality satisfies propagation; a mismatch is `wrong_propagated_value`.
5. If evidence is missing, emit `unknown`; never use an LLM or response-text heuristic.
6. If dependencies and tool execution succeed but an explicit final-state verifier is false, record `final_state_failure` as the root.
7. Repeated identical `tool_name + canonical arguments` calls are exact duplicates. Calls outside a known expected set are extras. Semantic unnecessary-call detection remains unavailable without stronger metadata.

Only the earliest causal failure is `first_failure_step`. Later tool failures reachable from it are listed as propagated rather than independent roots.



## v1 diagnosis boundary and state reference

Graph-Frontier v1 accepts structural sidecars only when every selected dependency
is uniquely resolved and producer/consumer are in the same turn. Ambiguous
source parameters, missing graph edges, absent sampler traces, and cross-turn
selected dependencies are explicit rejection reasons. This restriction applies
to v1 structural diagnosis, not to the validity of EnvFactory multi-turn tasks.

For this internship/project implementation, the selected QueryGen reference
trajectory's `final_scenario` may be used as `expected_final_state` with
provenance `selected_querygen_reference_trajectory`. This is an EnvFactory
engineering reference, not an independent oracle. Consequently
`path_adherence` and `task_success` remain separate.

## Capability map

Profiles aggregate into:

- dependency-depth buckets;
- per-edge and per-edge-type success;
- internal-parameter propagation success;
- state-changing vs query-tool execution success only when reliable metadata exists;
- root-cause failure frequencies;
- terminal pass rate.

Unknown observations do not count as attempts. Every reported Bernoulli success rate uses Beta(1,1) smoothing:

```text
p = (successes + 1) / (attempts + 2)
```

Attempts and raw successes are always emitted.

Each capability metric exposes scoring components rather than one mandatory formula:

- `pass_rate = p`
- `frontier_score = 4 * p * (1 - p)`
- `weakness = 1 - p`
- `learning_progress = null` until historical checkpoints exist
- `hint_regret = p_with_hint - p_without_hint`, or `null` without paired hint evidence

Future privileged hints may reveal only graph structure or tool-order strategy, such as “obtain user_id, use it to obtain order_id, then perform the target action.” They must not reveal real parameter values, final answers, or final environment state.

## Selector skeleton

The selector accepts a capability map, candidate task/tool-chain descriptors, deterministic seed, sample count, character/token targets, exploration floor, and repeat cap.

It returns selected task ids plus per-draw reasons and a budget/diversity report. Candidate priority is derived from referenced capability keys and the requested scoring component. A non-zero uniform exploration floor prevents zero probability; the repeat cap bounds duplicates. Multiple deterministic trials choose the plan with the smallest normalized budget error and then the highest priority.

This stage only demonstrates the interface on tiny mock candidates. It does not generate a new 26K dataset.
