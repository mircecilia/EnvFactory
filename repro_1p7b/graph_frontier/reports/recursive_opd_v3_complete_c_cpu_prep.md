VERDICT = READY_FOR_GPU_TRAINING

GPU used = NO

# Recursive OPD V3 Complete-C CPU preparation

All pool counts are train-split, context-compatible rows. No fresh rollout, inference, replay, or training was run.

## Natural Complete-C pools

| D | CURRENT_FRONTIER | SUCCESS_TERMINAL | CURRENT_CONTINUATION | TOTAL | Tool | EOS | Tool/EOS | EOS fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 752 | 362 | 290 | 1404 | 1024 | 380 | 2.6947 | 27.07% |
| 1 | 1069 | 159 | 256 | 1484 | 983 | 501 | 1.9621 | 33.76% |
| 2 | 1118 | 159 | 273 | 1550 | 1039 | 511 | 2.0333 | 32.97% |
| 3 | 1104 | 286 | 331 | 1721 | 1278 | 443 | 2.8849 | 25.74% |
| 4 | 1246 | 295 | 515 | 2056 | 1755 | 301 | 5.8306 | 14.64% |

## Historical-only versus Complete-C

| D | historical-only | Complete-C | matched historical | new non-historical |
|---:|---:|---:|---:|---:|
| 0 | 0 | 290 | 0 | 290 |
| 1 | 51 | 256 | 24 | 232 |
| 2 | 43 | 273 | 25 | 248 |
| 3 | 25 | 331 | 16 | 315 |
| 4 | 9 | 515 | 8 | 507 |

## Continuation distributions

- D0: depth={'1': 14, '2': 1, '3+': 2, 'START-adjacent': 273}; distance={'1': 290}; 0/1/2 rows={'0': 612, '1': 824, '2': 290}; context overflow=1.
- D1: depth={'1': 50, '2': 9, '3+': 5, 'START-adjacent': 192}; distance={'1': 256}; 0/1/2 rows={'0': 498, '1': 972, '2': 256}; context overflow=0.
- D2: depth={'1': 61, '2': 11, '3+': 5, 'START-adjacent': 196}; distance={'1': 273}; 0/1/2 rows={'0': 449, '1': 1004, '2': 273}; context overflow=1.
- D3: depth={'1': 65, '2': 8, '3+': 2, 'START-adjacent': 256}; distance={'1': 331}; 0/1/2 rows={'0': 336, '1': 1059, '2': 331}; context overflow=1.
- D4: depth={'1': 29, '2': 5, 'START-adjacent': 481}; distance={'1': 515}; 0/1/2 rows={'0': 185, '1': 1026, '2': 515}; context overflow=1.

## Boundary coverage

- D0 SUCCESS: {'continuation_positive': 157, 'success_tasks': 363, 'terminal_and_continuation': 157, 'terminal_positive': 362}.
- D1 SUCCESS: {'continuation_positive': 83, 'success_tasks': 159, 'terminal_and_continuation': 83, 'terminal_positive': 159}.
- D2 SUCCESS: {'continuation_positive': 86, 'success_tasks': 159, 'terminal_and_continuation': 86, 'terminal_positive': 159}.
- D3 SUCCESS: {'continuation_positive': 145, 'success_tasks': 286, 'terminal_and_continuation': 145, 'terminal_positive': 286}.
- D4 SUCCESS: {'continuation_positive': 172, 'success_tasks': 296, 'terminal_and_continuation': 172, 'terminal_positive': 295}.

## π3→π4 premature-stop regression

Definition: pi3 deeper than pi4; pi4 FIRST_ERROR START/ADVANCE MODEL_FINAL.
Complete-C coverage: 57/252.
Historical 18-task subset coverage: 14/18.

## Audit

CPU tests: 64 passed in 5.33s

TRAINING STEPS = UNDECIDED
EXPOSURE BUDGET = UNDECIDED
LEARNING RATE = UNDECIDED
