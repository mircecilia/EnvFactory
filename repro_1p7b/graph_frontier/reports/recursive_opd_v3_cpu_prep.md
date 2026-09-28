VERDICT = READY_FOR_GPU_TRAINING

TRAINING BUDGET = UNDECIDED

# Recursive OPD V3 / Complete-C CPU preparation

All counts below are train-split only; heldout rows are excluded from every dataset.

## Transition lineage

| transition | historical | crossed | regressed | uncertain | ambiguous | retention eligible |
|---|---:|---:|---:|---:|---:|---:|
| pi0->pi1 | 734 | 161 | 516 | 57 | 0 | 51 |
| pi1->pi2 | 894 | 128 | 687 | 79 | 0 | 43 |
| pi2->pi3 | 960 | 82 | 830 | 48 | 0 | 25 |
| pi3->pi4 | 1063 | 20 | 1028 | 15 | 0 | 9 |

## Natural Complete-C pools

| cycle | CURRENT_FRONTIER | SUCCESS_TERMINAL | CROSSED_RETENTION | TOTAL | tool | EOS | tool/EOS | 0/1/2 tasks | overflow |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | 752 | 362 | 0 | 1114 | 734 | 380 | 1.931578947368421 | {'0': 612, '1': 1114, '2': 0} | 1 |
| 1 | 1069 | 159 | 51 | 1279 | 778 | 501 | 1.5528942115768463 | {'0': 498, '1': 1177, '2': 51} | 0 |
| 2 | 1118 | 159 | 43 | 1320 | 809 | 511 | 1.5831702544031312 | {'0': 449, '1': 1234, '2': 43} | 1 |
| 3 | 1104 | 286 | 25 | 1415 | 972 | 443 | 2.194130925507901 | {'0': 336, '1': 1365, '2': 25} | 1 |
| 4 | 1246 | 295 | 9 | 1550 | 1249 | 301 | 4.149501661129568 | {'0': 185, '1': 1532, '2': 9} | 1 |

## π2→π3 versus π3→π4

π3 strict retention-eligible crossings: 25.
Of these, π4 still crossed 7, regressed 18, and regressed by premature stop 18.

The latest-relevant-frontier rule admits at most one retention candidate per task/cycle; mechanical accumulation gate: PASS.

## Gates

CPU tests: 47 passed in 5.53s

V3 GPU parameters remain deliberately unset. READY means protocol/data engineering is complete, not that a training budget has been approved.
