V2 π3 EXTERNAL EVAL = COMPLETE

# Matched external evaluation

| metric | π0 | V2 π3 | V2 π4 |
|---|---:|---:|---:|
| Frozen task/reference success | 23 | 94 | 38 |
| Frozen final-state success | 94 | 111 | 82 |
| Frozen dependency edge complete | 98 | 190 | 53 |
| Frozen gold node complete | 136 | 194 | 55 |
| Frozen internal E2E | 98 | 190 | 53 |
| Frozen tool calls | 995 | 932 | 710 |
| Frozen unexpected calls | 214 | 204 | 213 |
| Frozen redundant calls | 19 | 4 | 0 |
| Rich24 official reward | 0.1840277777777778 | 0.109375 | 0.19444444444444445 |
| Rich24 semantic success | 0 | 0 | 1 |
| Rich24 tool events | 28 | 124 | 63 |

Consumer reach is the number of expected dependency edges whose consumer was actually reached; conditional propagation is edge success conditioned on that reach.

π3 consumer reach: 496/710; conditional propagation: 422/496.

π3 clearly retains structural continuation better than π4 on Frozen300 (edge 190 vs 53; gold node 194 vs 55).  The tradeoff remains visible on Rich24: π3 makes 124 tool events and scores 0.109375, versus π4's 63 and 0.19444444444444445.
