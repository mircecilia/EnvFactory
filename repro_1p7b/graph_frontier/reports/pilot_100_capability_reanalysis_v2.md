# Static Graph-Frontier pilot: capability reanalysis v2

Derived only from the frozen manifest, reference traces, typed rollouts, and profiler outputs. No model or environment runtime was invoked.

## Metric audit

- task_success is the legacy name of reference_path_complete_success, not generic semantic completion.
- It requires all gold nodes, all selected gold edges, and exact final-state equality.
- All 213 selected edges are internal dependencies.
- Legacy internal attempts count consumer-reached edges, so the denominator is model-dependent.
- Read-only exact final-state equality is vacuous and is not used as semantic evidence in v2.

## Core metrics

| Metric | Base | Original SFT | Parameter-Aware |
|---|---:|---:|---:|
| Semantic task success | 11/89 = 12.4% (Beta 13.2%) | 37/89 = 41.6% (Beta 41.8%) | 31/89 = 34.8% (Beta 35.2%) |
| Reference-path complete | 0/100 = 0.0% (Beta 1.0%) | 29/100 = 29.0% (Beta 29.4%) | 20/100 = 20.0% (Beta 20.6%) |
| Internal reach | 53/213 = 24.9% (Beta 25.1%) | 159/213 = 74.6% (Beta 74.4%) | 149/213 = 70.0% (Beta 69.8%) |
| Internal end-to-end | 6/213 = 2.8% (Beta 3.3%) | 97/213 = 45.5% (Beta 45.6%) | 38/213 = 17.8% (Beta 18.1%) |
| Conditional propagation | 6/53 = 11.3% (Beta 12.7%) | 97/159 = 61.0% (Beta 60.9%) | 38/149 = 25.5% (Beta 25.8%) |

## Internal dependency funnel

### base

gold=213 -> producer reached=54 -> producer succeeded=44 -> consumer reached=34 -> values inspectable=25 -> correct=6

### original_sft

gold=213 -> producer reached=176 -> producer succeeded=163 -> consumer reached=132 -> values inspectable=129 -> correct=97

### parameter_aware

gold=213 -> producer reached=182 -> producer succeeded=146 -> consumer reached=101 -> values inspectable=88 -> correct=38

## Calls and post-success continuation

### base

- Legacy exact redundant / unexpected: 42 / 18
- Repeated patterns (non-exclusive): {'same_tool_different_args': 25, 'retry_after_tool_error': 51, 'retry_after_wrong_value': 22, 'same_tool_same_args': 42, 'producer_repeated_after_success': 17, 'duplicate_mutation_action': 1}
- Cycle patterns (non-exclusive): {'a_b_a': 17, 'repeated_block_cycle': 20, 'a_b_b': 3}
- Unexpected patterns (non-exclusive): {'tool_outside_gold': 18, 'alternative_producer_candidate': 5, 'exploratory_lookup': 14, 'unnecessary_mutation': 4}
- Post-success: 0 tasks, 0 calls, mean 0.0
- Semantic success but reference-path failure: 11

### original_sft

- Legacy exact redundant / unexpected: 15 / 56
- Repeated patterns (non-exclusive): {'same_tool_different_args': 16, 'retry_after_tool_error': 16, 'retry_after_wrong_value': 14, 'same_tool_same_args': 15, 'producer_repeated_after_success': 7, 'duplicate_mutation_action': 7, 'consumer_repeated_after_success': 4}
- Cycle patterns (non-exclusive): {'a_b_a': 5, 'repeated_block_cycle': 2, 'a_b_b': 3}
- Unexpected patterns (non-exclusive): {'tool_outside_gold': 56, 'alternative_producer_candidate': 22, 'exploratory_lookup': 42, 'malformed_or_failed_tool_call': 12, 'unnecessary_mutation': 12, 'other': 2}
- Post-success: 0 tasks, 0 calls, mean 0.0
- Semantic success but reference-path failure: 20

### parameter_aware

- Legacy exact redundant / unexpected: 202 / 215
- Repeated patterns (non-exclusive): {'same_tool_different_args': 23, 'retry_after_tool_error': 165, 'retry_after_wrong_value': 17, 'same_tool_same_args': 202, 'producer_repeated_after_success': 101, 'duplicate_mutation_action': 19, 'consumer_repeated_after_success': 94}
- Cycle patterns (non-exclusive): {'a_b_a': 11, 'repeated_block_cycle': 298}
- Unexpected patterns (non-exclusive): {'tool_outside_gold': 215, 'exploratory_lookup': 140, 'alternative_producer_candidate': 20, 'malformed_or_failed_tool_call': 128, 'unnecessary_mutation': 67, 'other': 5}
- Post-success: 0 tasks, 0 calls, mean 0.0
- Semantic success but reference-path failure: 17

## Original SFT vs Parameter-Aware

- both_false: 48
- both_true: 27
- original_false_pa_true: 4
- original_true_pa_false: 10
- both_true_with_pa_efficiency_or_propagation_regression: 6
- Regression failure attribution: {'wrong_propagated_value': 7, 'target_observation_not_acquired': 3}
- Regression environments: {'Calendar': 2, 'GoogleTasks': 1, 'UUPaoTui': 4, 'Weather': 3}

## Read-only final-state audit

| Model | Read-only tasks | Exact state match | State match with zero calls | Semantic support |
|---|---:|---:|---:|---:|
| base | 24 | 24 | 23 | 13 |
| original_sft | 24 | 24 | 0 | 13 |
| parameter_aware | 24 | 24 | 0 | 13 |

All 24 read-only tasks have exact final-state equality for every model. Base has 23 such matches with no tool call, proving that state equality alone is not capability evidence.

## Depth breakdown

| Depth | Model | Semantic | Reference path | Internal end-to-end | Reach | Conditional propagation | Redundant/task | Unexpected/task |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | base | 11/42 = 26.2% (Beta 27.3%) | 0/53 = 0.0% (Beta 1.8%) | 6/77 = 7.8% (Beta 8.9%) | 12/77 = 15.6% (Beta 16.5%) | 6/12 = 50.0% (Beta 50.0%) | 0.00 | 0.09 |
| 1 | original_sft | 31/42 = 73.8% (Beta 72.7%) | 29/53 = 54.7% (Beta 54.5%) | 58/77 = 75.3% (Beta 74.7%) | 68/77 = 88.3% (Beta 87.3%) | 58/68 = 85.3% (Beta 84.3%) | 0.08 | 0.09 |
| 1 | parameter_aware | 30/42 = 71.4% (Beta 70.5%) | 20/53 = 37.7% (Beta 38.2%) | 36/77 = 46.8% (Beta 46.8%) | 52/77 = 67.5% (Beta 67.1%) | 36/52 = 69.2% (Beta 68.5%) | 0.00 | 0.23 |
| 2 | base | 0/35 = 0.0% (Beta 2.7%) | 0/35 = 0.0% (Beta 2.7%) | 0/94 = 0.0% (Beta 1.0%) | 11/94 = 11.7% (Beta 12.5%) | 0/11 = 0.0% (Beta 7.7%) | 0.34 | 0.23 |
| 2 | original_sft | 6/35 = 17.1% (Beta 18.9%) | 0/35 = 0.0% (Beta 2.7%) | 21/94 = 22.3% (Beta 22.9%) | 65/94 = 69.1% (Beta 68.8%) | 21/65 = 32.3% (Beta 32.8%) | 0.31 | 1.17 |
| 2 | parameter_aware | 1/35 = 2.9% (Beta 5.4%) | 0/35 = 0.0% (Beta 2.7%) | 2/94 = 2.1% (Beta 3.1%) | 62/94 = 66.0% (Beta 65.6%) | 2/62 = 3.2% (Beta 4.7%) | 3.77 | 2.86 |
| 3+ | base | 0/12 = 0.0% (Beta 7.1%) | 0/12 = 0.0% (Beta 7.1%) | 0/42 = 0.0% (Beta 2.3%) | 30/42 = 71.4% (Beta 70.5%) | 0/30 = 0.0% (Beta 3.1%) | 2.50 | 0.42 |
| 3+ | original_sft | 0/12 = 0.0% (Beta 7.1%) | 0/12 = 0.0% (Beta 7.1%) | 18/42 = 42.9% (Beta 43.2%) | 26/42 = 61.9% (Beta 61.4%) | 18/26 = 69.2% (Beta 67.9%) | 0.00 | 0.83 |
| 3+ | parameter_aware | 0/12 = 0.0% (Beta 7.1%) | 0/12 = 0.0% (Beta 7.1%) | 0/42 = 0.0% (Beta 2.3%) | 35/42 = 83.3% (Beta 81.8%) | 0/35 = 0.0% (Beta 2.7%) | 5.83 | 8.58 |

## Environment breakdown

| Environment | Model | Semantic | Reference path | Internal end-to-end | Redundant/task | Unexpected/task |
|---|---|---:|---:|---:|---:|---:|
| Calendar | base | 5/10 = 50.0% (Beta 50.0%) | 0/10 = 0.0% (Beta 8.3%) | 0/15 = 0.0% (Beta 5.9%) | 0.10 | 0.70 |
| Calendar | original_sft | 3/10 = 30.0% (Beta 33.3%) | 0/10 = 0.0% (Beta 8.3%) | 1/15 = 6.7% (Beta 11.8%) | 0.50 | 1.20 |
| Calendar | parameter_aware | 3/10 = 30.0% (Beta 33.3%) | 0/10 = 0.0% (Beta 8.3%) | 2/15 = 13.3% (Beta 17.6%) | 0.40 | 1.00 |
| GoogleTasks | base | 0/30 = 0.0% (Beta 3.1%) | 0/30 = 0.0% (Beta 3.1%) | 0/66 = 0.0% (Beta 1.5%) | 0.00 | 0.00 |
| GoogleTasks | original_sft | 14/30 = 46.7% (Beta 46.9%) | 6/30 = 20.0% (Beta 21.9%) | 22/66 = 33.3% (Beta 33.8%) | 0.20 | 1.03 |
| GoogleTasks | parameter_aware | 13/30 = 43.3% (Beta 43.8%) | 6/30 = 20.0% (Beta 21.9%) | 6/66 = 9.1% (Beta 10.3%) | 0.20 | 1.20 |
| TradingBot | base | 0/18 = 0.0% (Beta 5.0%) | 0/18 = 0.0% (Beta 5.0%) | 0/48 = 0.0% (Beta 2.0%) | 0.17 | 0.28 |
| TradingBot | original_sft | 5/18 = 27.8% (Beta 30.0%) | 0/18 = 0.0% (Beta 5.0%) | 0/48 = 0.0% (Beta 2.0%) | 0.00 | 0.61 |
| TradingBot | parameter_aware | 5/18 = 27.8% (Beta 30.0%) | 0/18 = 0.0% (Beta 5.0%) | 0/48 = 0.0% (Beta 2.0%) | 10.56 | 8.72 |
| UUPaoTui | base | 6/18 = 33.3% (Beta 35.0%) | 0/18 = 0.0% (Beta 5.0%) | 6/36 = 16.7% (Beta 18.4%) | 2.11 | 0.33 |
| UUPaoTui | original_sft | 4/18 = 22.2% (Beta 25.0%) | 0/18 = 0.0% (Beta 5.0%) | 28/36 = 77.8% (Beta 76.3%) | 0.00 | 0.11 |
| UUPaoTui | parameter_aware | 2/18 = 11.1% (Beta 15.0%) | 0/18 = 0.0% (Beta 5.0%) | 2/36 = 5.6% (Beta 7.9%) | 0.11 | 0.67 |
| Weather | base | 0/13 = 0.0% (Beta 6.7%) | 0/24 = 0.0% (Beta 3.8%) | 0/48 = 0.0% (Beta 2.0%) | 0.00 | 0.00 |
| Weather | original_sft | 11/13 = 84.6% (Beta 80.0%) | 23/24 = 95.8% (Beta 92.3%) | 46/48 = 95.8% (Beta 94.0%) | 0.17 | 0.00 |
| Weather | parameter_aware | 8/13 = 61.5% (Beta 60.0%) | 14/24 = 58.3% (Beta 57.7%) | 28/48 = 58.3% (Beta 58.0%) | 0.00 | 0.00 |

## Q1-Q7 decisions

### Q1: What does legacy task_success measure?

Reference-path completion, not generic semantic completion: all required gold nodes, all selected gold edges, and the exact canonical final state must pass.

### Q2: What is the semantic Original-versus-PA gap?

Original is 37/89 = 41.6% (Beta 41.8%); PA is 31/89 = 34.8% (Beta 35.2%). The paired gap is six tasks (6.74 percentage points); PA wins 4 and loses 10 discordant pairs, exact McNemar p=0.1796.

### Q3: What drives the PA regression?

Primarily propagation correctness, tool execution failures, and redundant pre-completion loops. Reach alone does not explain it, and no model has post-success tool continuation under the deterministic completion boundary. Alternative valid paths explain part of the strict-path gap but not the large internal-edge regression.

### Q4: Why are final-state rates close?

The 40% versus 41% comparison is materially inflated by a permissive read-only state check. Supported deterministic semantic success is 31/89 for PA versus 37/89 for Original.

### Q5: What are PA's dominant extra-call patterns?

Repeated blocks/cycles, exact same-argument repeats, producer and consumer repetition, error retries, and tool-budget exhaustion. They occur before verified completion, not after it.

### Q6: Is the depth-2/3 zero only a strict-path artifact?

No. Strict reference-path scoring hides some alternative success, especially at depth 2, but PA's fixed-denominator and conditional propagation collapse at depth 2 and depth 3+ is a real deeper-dependency failure signal.

### Q7: Can this profiler support dynamic allocation?

Yes, after using semantic support-aware task success, fixed-denominator internal reach/end-to-end metrics, conditional propagation, and separate efficiency/call-pattern metrics. Future probes should persist final assistant content so read-only observation completion can be verified more strongly.

## Recommendation

Option 1: freeze a 300-task confirmation protocol using the corrected v2 metrics. Do not begin dynamic allocation from the uncorrected legacy metrics.

## Limitations

- No final assistant response content was persisted; observation-based semantic success verifies typed acquisition plus terminal assistant status, not the factual content of final prose.
- Empty reference observations are unsupported and excluded from semantic denominators.
- State predicates are template-specific deterministic predicates, not perfect semantic oracles.
- Unexpected tools may be valid alternatives; alternative_producer_candidate is evidence of typed value reuse, not proof of necessity.
- Call-pattern categories are deterministic and non-exclusive.
- Edges within a task are clustered and are not independent statistical samples.
