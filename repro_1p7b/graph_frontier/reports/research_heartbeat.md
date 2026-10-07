# Research heartbeat — 2026-10-07

## Start

- Connected GPU host 10.251.171.6:28143, branch repro/envfactory-1p7b; both A100 idle, no tmux.
- Priority: actual MAIN correction exposure, not candidate-pool membership.
- Extracting successful optimizer attempt stream + dataset identities, excluding the failed cycle0 attempt from trained-policy exposure.
- Existing collector does NOT truncate rollout at first divergence; the miner excludes off-path supervision. Recovery motivation must reflect this distinction.
- Current stage CPU-only; no inference or training launched. Next: exposure-conditioned and selection-adjusted paired audit; related-work novelty check.

## 2026-10-07 18:20 Asia/Singapore — audit complete / prototype gated

- MAIN exposure rebuilt from successful optimizer inputs: 341 task-transition events, each exposed once. Strict crossing 23/309 (7.44%) vs 282/3495 (8.07%); task-cluster difference CI [-3.53,+2.50] pp. Eligible matched and no-other-exposure sensitivity do not reverse the conclusion.
- Structural proxy heldout signal in cycle0 disappears under environment matching (5 treated/7 controls, difference0); later cycles do not reproduce it. Real dependency metadata remains unknown.
- All-task START grows496→734, frontier CDF/transition matrices show no monotonic self-curriculum. Logging/verifier coverage changes are retained as UNCERTAIN, not success.
- Independent100-task recovery CPU preflight:93 exact passes,7 failures. Repeated diagnostics: all7 saved state/post-state roundtrips match;6 observation array order differences and1 process-dependent GitHub comment_id. Original failed gate preserved; no GPU/model request/optimizer update launched.
- Four CPU rollback/mock tests PASS remotely. Local Windows asyncio mock invocations stalled and were interrupted; remote Linux tests are the authoritative test run.
- Next research step: versioned observation-portability protocol, then rerun whole fixed100-task gate before any recovery sampling. No new training authorized or started. Package small analysis/scripts/reports into a scoped commit, preserve existing BFCL change, no push.

## 2026-10-07 19:00 Asia/Singapore — Verified Recovery Phase0 STOP

- 新用户任务只推进 Verified Recovery，不继续 exposure/transfer/curriculum heuristics。
- 修复原七个生产端确定性问题；重复检查额外定位 TravelBooking list(set) 顺序问题，保留失败缓存后新建 environment_v3。8项真实API CPU单测PASS。
- 新epoch fixed100三次均100/100；独立random300第一遍299/300。唯一失败 official-rl-0382-c3b604ead122：ClinicalTrialsGov gold HTTP调用DNS失败；独立诊断再次失败。
- 原始gold动作/参数/顺序及历史gold每步状态不改，verifier/equality/reward不改；新observation cache与旧版本分开。随机组不换样本。
- 硬门槛失败：random300第二遍、Recovery collection、GPU inference与训练均未启动。不是Recovery方法Negative Result，数据层效率/深度/span仍未知。
- 本机reports/verified_recovery保存报告与明确NOT_RUN的指标；不伪造10个case。准备scoped commit，不push，保留既有BFCL工作区修改，runtime产物排除。
