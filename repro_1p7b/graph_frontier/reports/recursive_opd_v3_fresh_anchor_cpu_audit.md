VERDICT = ANCHOR_READY

GPU used = NO

# Recursive OPD V3 Fresh Crossed-Frontier Anchor CPU Audit

Fresh anchors are event-triggered by previous-cycle failure followed by current-cycle verified mastery. They are not fixed-ratio continuation oversampling and expire after the immediately following recursive update.

## Anchor pools

| Pool | Transition | Raw | Compatible | Overlap current | Anchor-only | START/ADVANCE | Depth |
|---|---|---:|---:|---:|---:|---|---|
| A1 | pi0->pi1 | 51 | 51 | 24 | 27 | {'ADVANCE': 48, 'START': 3} | {'0': 3, '1': 44, '2': 3, '3+': 1} |
| A2 | pi1->pi2 | 4 | 4 | 1 | 3 | {'ADVANCE': 4} | {'1': 3, '2': 1} |
| A3 | pi2->pi3 | 3 | 3 | 1 | 2 | {'ADVANCE': 2, 'START': 1} | {'0': 1, '1': 2} |
| A4 | pi3->pi4 | 2 | 2 | 2 | 0 | {'START': 2} | {'0': 2} |

## Historical-only alignment

| Pool | Historical latest-prior | Fresh previous-cycle | Intersection | Historical old-only |
|---|---:|---:|---:|---:|
| A1 | 51 | 51 | 51 | 0 |
| A2 | 43 | 4 | 4 | 39 |
| A3 | 25 | 3 | 3 | 22 |
| A4 | 9 | 2 | 2 | 7 |

A1 aligns exactly. A2-A4 are smaller because Fresh Anchor accepts only the immediately previous cycle, whereas historical V3 searched the latest eligible frontier from any earlier cycle. The verifier and context gates remain aligned.

## Natural plus anchor (descriptive only)

| Pool | Natural | Anchor | Semantic entries | Exact overlap | Deduplicated | 0/1/2/3/>3 |
|---|---:|---:|---:|---:|---:|---|
| A1 | 1484 | 51 | 1535 | 24 | 1511 | {'0': 498, '1': 952, '2': 245, '3': 31, '>3': 0} |
| A2 | 1550 | 4 | 1554 | 1 | 1553 | {'0': 449, '1': 1003, '2': 271, '3': 3, '>3': 0} |
| A3 | 1721 | 3 | 1724 | 1 | 1723 | {'0': 336, '1': 1057, '2': 332, '3': 1, '>3': 0} |
| A4 | 2056 | 2 | 2058 | 2 | 2056 | {'0': 185, '1': 1026, '2': 513, '3': 2, '>3': 0} |

## π3→π4 regression coverage

Historical 18: {'anchor_coverage': 2, 'anchor_only': 2, 'both': 0, 'current_positive_only': 14, 'denominator': 18, 'natural_coverage': 14, 'neither': 2, 'union_coverage': 16}
Strict all: {'anchor_coverage': 2, 'anchor_only': 2, 'both': 0, 'current_positive_only': 57, 'denominator': 252, 'natural_coverage': 57, 'neither': 193, 'union_coverage': 59}

## Audit

CPU tests: 82 passed in 5.35s

GPU TRAINING CONFIG = UNDECIDED

B=128 scenarios are ILLUSTRATIVE ONLY and NOT APPROVED TRAINING CONFIG.
