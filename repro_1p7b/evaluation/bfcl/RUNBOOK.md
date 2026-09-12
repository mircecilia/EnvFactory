# BFCL V3 runbook

## 2026-09-13 - Bootstrap

- Restored reproduction worktree `/home/u2024311031/workspace/envfactory_repro_1p7b`, branch `repro/envfactory-1p7b`.
- Cancelled the queued post-check and removed its untracked custom NLL configs.
- Cloned official Gorilla `v1.3` at `ea13468e4423454d0c213704fb87cf7cb3990433` into an isolated clean checkout.
- Read official `README.md`, `TEST_CATEGORIES.md`, `SUPPORTED_MODELS.md`, and `CONTRIBUTING.md`.
- Confirmed `multi_turn` expands to four V3 multi-turn categories and official support exists for `Qwen/Qwen3-1.7B-FC`, local paths, SGLang, run IDs, and partial evaluation.
- Created isolated Python 3.10 environment `envfactory_bfcl_v1p3`. Conda initially still contacted defaults and failed DNS; retry with `--override-channels` and the Tsinghua main/r mirrors succeeded.
- Installed BFCL from the fixed source checkout without editable mode so the checkout stays clean. SGLang is pinned to 0.5.9.
- Selected eight smoke IDs before generation with fixed seed 20260913, two per official category.
- Base/SFT share the official QwenFC handler, SGLang backend, one GPU, temperature 0.7, official dynamic max tokens, and identical case IDs.
- Formal 8K SFT training remains untouched. Its final checkpoint is pending until exit code 0 and completeness checks.

## Required smoke evidence

For each model, retain:

1. non-empty ordinary completion;
2. raw `<tool_call>` output and official-handler decoded call;
3. rendered `<tool_response>` refill;
4. eight official multi-turn result entries with input/state logs;
5. official per-category partial score JSON;
6. explicit scan for error, empty, timeout, truncation, and parse failures.

Do not launch the full `multi_turn` category from Codex.
