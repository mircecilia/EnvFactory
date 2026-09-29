# MCPMark Verified Filesystem paired pilot

**PRELIMINARY PAIRED PILOT — NOT FULL MCPMARK SCORE.**

Official commit: `cd45b7f57923b9b3985467f5139927575f83141c`; standard suite, Filesystem, k=1, MCPMarkAgent.
Frozen manifest SHA256: `fc4f2061622451d12ece56ad80299f526a85d3a715d8b67686e26f942e5359ba`. Two independent mutable roots; same task order and protocol.

- GRPO checkpoint: `repro_1p7b/checkpoints_eval/standard_rl_v2_control_pilot128_step64_hf`; model SHA256 `d871a96525ae69a40b57484022b8cde9ec7dd4576e6d7f7ac29cddab42727a40`; tokenizer SHA256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
- Cpi3 checkpoint: `repro_1p7b/graph_frontier/recursive_opd_v3_experiment_run/cycle3/checkpoint`; model SHA256 `1e1022b12a21a730ce6038f1c257c7d75e3ec7e119314ce80e663088c78e3790`; same tokenizer SHA256.
- GRPO identity is tied to the 11.50% historical BFCL Standard-v2 row by `repro_1p7b/graph_frontier/run_guided_rl_v2_benchmarks.sh` and `repro_1p7b/graph_frontier/reports/guided_rl_v2_benchmark_evaluation.md`.
- MCPMark source is the unmodified official GitHub codeload archive pinned to the commit above, outside the EnvFactory repo; direct Git clone timed out on this host.
Official agent temperature=1.0; both arms use the identical 8192-token transport cap to fit the checkpoints' native 40960-token context.

Paired completed: **7/10**. GRPO: **0/7**; Cπ3: **0/7**.
Cπ3 wins 0; GRPO wins 0; tie-pass 0; tie-fail 7.

| Task | GRPO | Cπ3 | Relation |
|---|---|---|---|
| desktop/project_management | INFRA | INFRA | unpaired pending |
| desktop_template/file_arrangement | FAIL | FAIL | tie-fail |
| file_context/file_splitting | FAIL | FAIL | tie-fail |
| file_property/time_classification | FAIL | FAIL | tie-fail |
| folder_structure/structure_analysis | FAIL | FAIL | tie-fail |
| legal_document/individual_comments | FAIL | FAIL | tie-fail |
| papers/author_folders | FAIL | INFRA | unpaired pending |
| student_database/gradebased_score | FAIL | FAIL | tie-fail |
| threestudio/output_analysis | FAIL | FAIL | tie-fail |
| votenet/debugging | FAIL | INFRA | unpaired pending |

Failure classification across all ten attempted tasks:

- grpo: MODEL_FAIL=9; INFRA_FAILURE=1 (desktop/project_management); TIMEOUT=0.
- cpi3: MODEL_FAIL=7; INFRA_FAILURE=3 (desktop/project_management, papers/author_folders, votenet/debugging); TIMEOUT=0.

- desktop/project_management: the official verifier raised FileNotFoundError for a missing progress_tracking directory on both arms; excluded by the predeclared verifier-exception rule.
- papers/author_folders (Cpi3): turn-5 input reached 47,864 tokens, above the native 40,960 context.
- votenet/debugging (Cpi3): turn-41 input 32,962 plus the common 8,192 output budget exceeded 40,960; no per-model retuning or retry was performed.

No directional signal in the valid paired subset.


Diagnostics on paired tasks only:

- grpo: mean turns 8.143; mean tool calls 9.571; mean runtime 148.608 s.
- cpi3: mean turns 13; mean tool calls 13.714; mean runtime 192.916 s.

Representative official verifier failures:

- desktop_template/file_arrangement (grpo): ❌ Missing required files in work/ folder: ['client_list.csv', 'timesheet.csv', 'experiment_results.txt', 'budget_tracker.csv', 'expenses.csv']
- desktop_template/file_arrangement (cpi3): ❌ Missing required files in work/ folder: ['client_list.csv', 'timesheet.csv', 'experiment_results.txt', 'budget_tracker.csv', 'expenses.csv']
- file_context/file_splitting (grpo): ❌ Missing files: ['split_02.txt', 'split_03.txt', 'split_04.txt', 'split_05.txt', 'split_06.txt', 'split_07.txt', 'split_08.txt', 'split_09.txt', 'split_10.txt']

This small, category-stratified pilot cannot support a final MCPMark claim. Raw official meta/messages/log paths are indexed in the companion JSON.

Timing (UTC+8): frozen manifest 21:06, matched-protocol smoke finished 21:40, pilot launched 21:41, last arm finished 22:22 on 2026-09-29. The user relaxed the original one-hour deadline; RESULT_AVAILABLE_WITHIN_60_MIN=NO. The smoke checked tool parsing, MCP execution, and verifier persistence; both models failed that smoke task, so smoke task success was not claimed.

Limitations: only 7/10 tasks have valid paired scores; all seven are tie-fail. No directional signal for either checkpoint, and no claim about full MCPMark or statistical superiority.
