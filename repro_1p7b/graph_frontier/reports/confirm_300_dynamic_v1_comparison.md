# Dynamic Graph-Frontier v1: Frozen 300-Probe Comparison

## Manifest

- Seed: 20260914
- Candidate/accepted/rejected: 300/300/0
- Split: {'diagnosis': 240, 'heldout': 60}
- Environment: {'Calendar': 32, 'GoogleTasks': 80, 'TradingBot': 68, 'UUPaoTui': 68, 'Weather': 52}
- Depth: {'1': 120, '2': 110, '3+': 70}
- Gold internal edges: 710
- Semantic support: 300/300
- SHA256: 4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c

## Runtime

| Model | Completed | Valid | Runtime seconds | System errors |
|---|---:|---:|---:|---|
| base | 300 | 300 | 1552.3 | {} |
| original_sft | 300 | 300 | 1164.3 | {} |
| parameter_aware | 300 | 300 | 1271.1 | {} |
| dynamic_v1 | 300 | 300 | 1132.0 | {} |

## Core metrics

| Model | BFCL | BFCL miss param | Semantic | Reference path | Internal reach | Conditional propagation | Internal end-to-end | Calls/task | Redundant/task | Unexpected/task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 8.75% | 8.5% | 28/300=9.3% | 0/300=0.0% | 188/710=26.5% | 14/188=7.4% | 14/710=2.0% | 1.89 | 0.47 | 0.22 |
| original_sft | 9.75% | 8.0% | 41/300=13.7% | 7/300=2.3% | 390/710=54.9% | 196/390=50.3% | 196/710=27.6% | 3.07 | 0.06 | 0.64 |
| parameter_aware | 11.00% | 15.0% | 55/300=18.3% | 19/300=6.3% | 407/710=57.3% | 64/407=15.7% | 64/710=9.0% | 3.51 | 0.07 | 0.88 |
| dynamic_v1 | 10.38% | 10.5% | 57/300=19.0% | 23/300=7.7% | 428/710=60.3% | 237/428=55.4% | 237/710=33.4% | 3.32 | 0.06 | 0.71 |

## Depth

### Depth 1

| Model | Semantic | Reach | Conditional | End-to-end | Calls/task | Redundant/task | Unexpected/task |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 28/120=23.3% | 28/172=16.3% | 14/28=50.0% | 14/172=8.1% | 0.71 | 0.00 | 0.12 |
| original_sft | 33/120=27.5% | 53/172=30.8% | 39/53=73.6% | 39/172=22.7% | 1.52 | 0.00 | 0.12 |
| parameter_aware | 34/120=28.3% | 64/172=37.2% | 35/64=54.7% | 35/172=20.3% | 1.86 | 0.02 | 0.23 |
| dynamic_v1 | 39/120=32.5% | 73/172=42.4% | 59/73=80.8% | 59/172=34.3% | 1.68 | 0.02 | 0.14 |

### Depth 2

| Model | Semantic | Reach | Conditional | End-to-end | Calls/task | Redundant/task | Unexpected/task |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0/110=0.0% | 55/293=18.8% | 0/55=0.0% | 0/293=0.0% | 2.57 | 1.15 | 0.18 |
| original_sft | 8/110=7.3% | 198/293=67.6% | 53/198=26.8% | 53/293=18.1% | 4.17 | 0.17 | 1.00 |
| parameter_aware | 21/110=19.1% | 203/293=69.3% | 20/203=9.9% | 20/293=6.8% | 4.70 | 0.16 | 1.23 |
| dynamic_v1 | 18/110=16.4% | 215/293=73.4% | 73/215=34.0% | 73/293=24.9% | 4.66 | 0.15 | 1.15 |

### Depth 3+

| Model | Semantic | Reach | Conditional | End-to-end | Calls/task | Redundant/task | Unexpected/task |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0/70=0.0% | 105/245=42.9% | 0/105=0.0% | 0/245=0.0% | 2.83 | 0.20 | 0.44 |
| original_sft | 0/70=0.0% | 139/245=56.7% | 104/139=74.8% | 104/245=42.4% | 3.99 | 0.00 | 0.99 |
| parameter_aware | 0/70=0.0% | 140/245=57.1% | 9/140=6.4% | 9/245=3.7% | 4.46 | 0.00 | 1.46 |
| dynamic_v1 | 0/70=0.0% | 140/245=57.1% | 105/140=75.0% | 105/245=42.9% | 4.00 | 0.00 | 1.00 |

## Dynamic paired statistics

### original_sft_vs_dynamic_v1

- semantic_task_success: Dynamic-only wins=23, comparator-only wins=7, ties=270, effect=5.33pp, exact p=0.00522288
- reference_path_complete_success: Dynamic-only wins=16, comparator-only wins=0, ties=284, effect=5.33pp, exact p=3.05176e-05
- task_level_internal_edge_complete: Dynamic-only wins=28, comparator-only wins=3, ties=269, effect=8.33pp, exact p=4.64916e-06

### parameter_aware_vs_dynamic_v1

- semantic_task_success: Dynamic-only wins=14, comparator-only wins=12, ties=274, effect=0.67pp, exact p=0.845019
- reference_path_complete_success: Dynamic-only wins=5, comparator-only wins=1, ties=294, effect=1.33pp, exact p=0.21875
- task_level_internal_edge_complete: Dynamic-only wins=68, comparator-only wins=2, ties=230, effect=22.00pp, exact p=4.21145e-18


## Diagnosis vs held-out

- diagnosis original_sft_vs_dynamic_v1 semantic_task_success: Dynamic-only=21, comparator-only=6, effect=6.25pp, exact p=0.00592461
- diagnosis original_sft_vs_dynamic_v1 task_level_internal_edge_complete: Dynamic-only=24, comparator-only=2, effect=9.17pp, exact p=1.04904e-05
- diagnosis parameter_aware_vs_dynamic_v1 semantic_task_success: Dynamic-only=13, comparator-only=11, effect=0.83pp, exact p=0.83882
- diagnosis parameter_aware_vs_dynamic_v1 task_level_internal_edge_complete: Dynamic-only=58, comparator-only=1, effect=23.75pp, exact p=2.08167e-16
- heldout original_sft_vs_dynamic_v1 semantic_task_success: Dynamic-only=2, comparator-only=1, effect=1.67pp, exact p=1
- heldout original_sft_vs_dynamic_v1 task_level_internal_edge_complete: Dynamic-only=4, comparator-only=1, effect=5.00pp, exact p=0.375
- heldout parameter_aware_vs_dynamic_v1 semantic_task_success: Dynamic-only=1, comparator-only=1, effect=0.00pp, exact p=1
- heldout parameter_aware_vs_dynamic_v1 task_level_internal_edge_complete: Dynamic-only=10, comparator-only=1, effect=15.00pp, exact p=0.0117188

## Dynamic failure and efficiency

- Root failure attribution: {'wrong_propagated_value': 66, 'consumer_not_executed': 102, 'observation_failure': 62, 'missing_internal_value': 12, 'producer_not_executed': 1}
- Call-count distribution: mean=3.32, median=3.0, p90=6, p95=6, max=7
- Repeated/retry: same-tool-same-args=19, retry-after-error=94, repeated-block-cycle=1, tool-budget-exhaustion-tasks=0
- Internal dependency funnel: {'gold_internal_opportunities': 710, 'producer_reached': 673, 'producer_succeeded': 614, 'consumer_reached': 428, 'consumer_reached_after_successful_producer': 370, 'consumer_input_inspectable': 428, 'source_and_target_value_inspectable': 340, 'correct_propagated': 237, 'correct_edges_with_final_semantic_success': 51, 'reach_rate': {'successes': 428, 'attempts': 710, 'raw_rate': 0.6028169014084507, 'smoothed_rate': 0.6025280898876404}, 'conditional_propagation_accuracy': {'successes': 237, 'attempts': 428, 'raw_rate': 0.5537383177570093, 'smoothed_rate': 0.5534883720930233}, 'end_to_end_internal_edge_success': {'successes': 237, 'attempts': 710, 'raw_rate': 0.33380281690140845, 'smoothed_rate': 0.3342696629213483}, 'stage_conversion_rates': {'gold_internal_opportunities_to_producer_reached': 0.9478873239436619, 'producer_reached_to_producer_succeeded': 0.912332838038633, 'producer_succeeded_to_consumer_reached_after_successful_producer': 0.6026058631921825, 'consumer_reached_after_successful_producer_to_source_and_target_value_inspectable': 0.918918918918919, 'source_and_target_value_inspectable_to_correct_propagated': 0.6970588235294117, 'correct_propagated_to_correct_edges_with_final_semantic_success': 0.21518987341772153}}

## Conclusions

- Validity gate passed: True
- validity_gate_passed: True
- interpretation_allowed: True
- pilot_semantic_direction_reproduced: False
- pilot_internal_direction_reproduced: True
- pilot_efficiency_direction_reproduced: True
- bfcl_context: BFCL Missing Parameter Original=8.0%, PA=15.0%, Dynamic v1=10.5%; comparison is descriptive only and no causal claim is made.
- dynamic_semantic_preserved_or_improved_vs_original: True
- dynamic_propagation_improved_vs_parameter_aware: True
- dynamic_bfcl_better_than_original: True
- dynamic_efficiency_acceptable_vs_parameter_aware: True
- dynamic_extra_execution_judgment: mostly_useful_depth_with_some_retry_overhead
- dynamic_case: A
- recommended_next_step: enter_dynamic_v2_before_rl

## Limitations

- Task-level McNemar tests are primary; edges within a task are clustered.
- Deterministic text matching can create false negatives for valid paraphrases.
- Unexpected tools can be valid alternative paths and are not necessarily errors.
- The diagnosis split is intended for future allocation and is not an unbiased test set.
- The 60-task held-out split is a directional sanity check and is not over-interpreted.
