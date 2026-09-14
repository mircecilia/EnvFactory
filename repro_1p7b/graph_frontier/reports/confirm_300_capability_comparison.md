# Graph-Frontier 300-Probe Confirmation Study

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

## Core metrics

| Model | Semantic | Reference path | Internal reach | Conditional propagation | Internal end-to-end | Redundant/task | Unexpected/task |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 28/300=9.3% | 0/300=0.0% | 188/710=26.5% | 14/188=7.4% | 14/710=2.0% | 0.47 | 0.22 |
| original_sft | 41/300=13.7% | 7/300=2.3% | 390/710=54.9% | 196/390=50.3% | 196/710=27.6% | 0.06 | 0.64 |
| parameter_aware | 55/300=18.3% | 19/300=6.3% | 407/710=57.3% | 64/407=15.7% | 64/710=9.0% | 0.07 | 0.88 |

## Depth

### Depth 1

| Model | Semantic | Reach | Conditional | End-to-end | Redundant/task | Unexpected/task |
|---|---:|---:|---:|---:|---:|---:|
| base | 28/120=23.3% | 28/172=16.3% | 14/28=50.0% | 14/172=8.1% | 0.00 | 0.12 |
| original_sft | 33/120=27.5% | 53/172=30.8% | 39/53=73.6% | 39/172=22.7% | 0.00 | 0.12 |
| parameter_aware | 34/120=28.3% | 64/172=37.2% | 35/64=54.7% | 35/172=20.3% | 0.02 | 0.23 |

### Depth 2

| Model | Semantic | Reach | Conditional | End-to-end | Redundant/task | Unexpected/task |
|---|---:|---:|---:|---:|---:|---:|
| base | 0/110=0.0% | 55/293=18.8% | 0/55=0.0% | 0/293=0.0% | 1.15 | 0.18 |
| original_sft | 8/110=7.3% | 198/293=67.6% | 53/198=26.8% | 53/293=18.1% | 0.17 | 1.00 |
| parameter_aware | 21/110=19.1% | 203/293=69.3% | 20/203=9.9% | 20/293=6.8% | 0.16 | 1.23 |

### Depth 3+

| Model | Semantic | Reach | Conditional | End-to-end | Redundant/task | Unexpected/task |
|---|---:|---:|---:|---:|---:|---:|
| base | 0/70=0.0% | 105/245=42.9% | 0/105=0.0% | 0/245=0.0% | 0.20 | 0.44 |
| original_sft | 0/70=0.0% | 139/245=56.7% | 104/139=74.8% | 104/245=42.4% | 0.00 | 0.99 |
| parameter_aware | 0/70=0.0% | 140/245=57.1% | 9/140=6.4% | 9/245=3.7% | 0.00 | 1.46 |

## Paired Original vs PA

- semantic_task_success: PA wins=22, losses=8, effect=4.67pp, exact p=0.0161248
- reference_path_complete_success: PA wins=13, losses=1, effect=4.00pp, exact p=0.00183105
- task_level_internal_edge_complete: PA wins=15, losses=56, effect=-13.67pp, exact p=1.0414e-06

## Diagnosis vs held-out

- diagnosis: PA wins=20, losses=7, effect=5.42pp, exact p=0.0191573
- heldout: PA wins=2, losses=1, effect=1.67pp, exact p=1

## Conclusions

- Validity gate passed: True
- validity_gate_passed: True
- interpretation_allowed: True
- pilot_semantic_direction_reproduced: False
- pilot_internal_direction_reproduced: True
- pilot_efficiency_direction_reproduced: True
- bfcl_context: BFCL Missing Parameter Original=8.0%, PA=15.0%; comparison is descriptive only and no causal claim is made.

## Limitations

- Task-level McNemar tests are primary; edges within a task are clustered.
- Deterministic text matching can create false negatives for valid paraphrases.
- Unexpected tools can be valid alternative paths and are not necessarily errors.
- The diagnosis split is intended for future allocation and is not an unbiased test set.
- The 60-task held-out split is a directional sanity check and is not over-interpreted.
