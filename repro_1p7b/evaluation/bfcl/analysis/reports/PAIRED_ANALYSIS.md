# BFCL V3 Multi-Turn Paired Analysis

All comparisons pair the same task IDs. McNemar uses the exact two-sided
binomial test over discordant pairs. Delta intervals use 10,000 paired
bootstrap resamples with seed 20260913.

## Overall

- Base: 70 / 800 = 8.75%
- Base Wilson 95% CI: [6.98%, 10.91%]
- SFT: 78 / 800 = 9.75%
- SFT Wilson 95% CI: [7.88%, 12.00%]
- Delta: +8 cases, +1.00 pp
- TT: 25
- TF (regressed): 45
- FT (fixed): 53
- FF: 677
- McNemar exact p: 0.479692499567
- Paired bootstrap 95% CI: [-1.500, +3.375] pp

## Base category

- Base: 25 / 200 = 12.50%
- Base Wilson 95% CI: [8.61%, 17.80%]
- SFT: 27 / 200 = 13.50%
- SFT Wilson 95% CI: [9.45%, 18.93%]
- Delta: +2 cases, +1.00 pp
- TT: 8
- TF (regressed): 17
- FT (fixed): 19
- FF: 156
- McNemar exact p: 0.867939400428
- Paired bootstrap 95% CI: [-5.000, +7.000] pp

## Miss Func category

- Base: 15 / 200 = 7.50%
- Base Wilson 95% CI: [4.60%, 12.00%]
- SFT: 22 / 200 = 11.00%
- SFT Wilson 95% CI: [7.38%, 16.09%]
- Delta: +7 cases, +3.50 pp
- TT: 7
- TF (regressed): 8
- FT (fixed): 15
- FF: 170
- McNemar exact p: 0.210039615631
- Paired bootstrap 95% CI: [-1.000, +8.000] pp

## Miss Param category

- Base: 17 / 200 = 8.50%
- Base Wilson 95% CI: [5.37%, 13.19%]
- SFT: 16 / 200 = 8.00%
- SFT Wilson 95% CI: [4.98%, 12.60%]
- Delta: -1 cases, -0.50 pp
- TT: 6
- TF (regressed): 11
- FT (fixed): 10
- FF: 173
- McNemar exact p: 1
- Paired bootstrap 95% CI: [-5.000, +4.000] pp

## Long Context category

- Base: 13 / 200 = 6.50%
- Base Wilson 95% CI: [3.84%, 10.80%]
- SFT: 13 / 200 = 6.50%
- SFT Wilson 95% CI: [3.84%, 10.80%]
- Delta: +0 cases, +0.00 pp
- TT: 4
- TF (regressed): 9
- FT (fixed): 9
- FF: 178
- McNemar exact p: 1
- Paired bootstrap 95% CI: [-4.000, +4.000] pp

## Interpretation

The overall point estimate is positive, but its paired 95% interval includes
zero and the exact McNemar test is not significant at 0.05. The result is
therefore evidence of an observed improvement under this run, not a
statistically established population-level gain.
The intervals resample task IDs only; they do not include between-run
generation variance from the unlocked stochastic server seeds.
