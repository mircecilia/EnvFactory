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

## SGLang launch compatibility

BFCL v1.3's built-in SGLang launcher passes the historical `--tp` flag, while the pinned SGLang 0.5.9 CLI exposes `--tp-size`. The repository wrapper therefore starts SGLang itself with the verified current flag and invokes official BFCL generation with `--skip-server-setup`. Model prompting, handler decoding, case execution, and scoring remain official and unmodified.

- First Base smoke launch exited before any case because the wrapper readiness one-liner placed an escaped quote inside an f-string expression. The server cleanup trap ran. The URL construction was replaced with plain concatenation; this infrastructure attempt is not a model result.

- The next Base server attempt loaded the model and exposed `/v1/models`, then the first completion caused SGLang QK-Norm/RoPE JIT to fail with `fatal error: concepts: No such file or directory`. This was not OOM and no BFCL case ran. The new environment already contained GCC/G++ 13.4 and the C++20 header; the JIT had selected the system host compiler. `CC`, `CXX`, `GCC`, and `GXX` are now pinned to the BFCL environment compilers.

- Exporting `CC/CXX` alone did not alter direct nvcc invocations. A minimal `/tmp` compile including `<concepts>` passed only after `NVCC_PREPEND_FLAGS=-ccbin=$CXX`; the wrapper now exports that NVIDIA-supported flag. No GPU inference was attempted until this compiler gate passed.

- Base generation produced all eight predeclared result rows, but stock `bfcl evaluate` failed before scoring because v1.3 reloads 200 prompt/answer rows and has no evaluation-side run-ID filter. A narrow repository-owned adapter now selects exact generated IDs from official data in memory and calls official `multi_turn_runner`; official files and scoring remain unchanged.

## 2026-09-13 - Formal SFT completion and Base smoke

Formal SFT completed with exit code 0. The verified final root checkpoint is `repro_1p7b/checkpoints/baseline_sft_8k_1p7b`. Base produced 8/8 results and official partial score JSON files; category scores were 0/2 each. No inference-error row or top-level empty output was found.
