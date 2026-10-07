# Verified Recovery vs FD stopping — 未运行

STATUS = NOT_RUN_PHASE0_FAILED

Phase0 的固定100通过，但独立300首遍299/300失败。没有采集 A/B student trajectories，不能计算 yield、deep yield、paired CI 或自主 span；不能把 CPU gold replay 当 student rollout。

`paired_task_results.jsonl` 和 `recovery_examples.jsonl` 为空，表示未采集，不是零效果。没有10个真实 Recovery case，不能拼接旧轨迹来冒充。

## 若将来放行，必须先固定的比较条件

- 原 Dynamic-v1、相同 task IDs/任务顺序、T=0 与原 decoding；新的 FD-stopping A 臂必须实际首错停止。旧 collector 会首错后继续，只是 miner 排除后续监督，不能冒充 A 臂。
- B 臂在错误执行后的同一个环境中执行一个最小纠正；不偷偷 rollback/truncate history，不免费恢复状态，不切换整条 gold suffix。不可最小恢复时明确 UNRECOVERABLE。
- correction 后状态分布为 intervention-induced HYBRID，而不是纯无干预 on-policy。区分错误点/候选 target/实际成功 intervention/自主 success。
- 逐调用 cost ledger 包括 student 成功与失败调用、gold correction、verifier fork/replay；另列 reset/load/save/list-tools 管理成本、tokens 和 wall-clock。不把 fork/replay 当免费。
- 相同 per-task 上限不等于相同实际 interaction 消耗；300-task paired 全量指标与严格等实际成本比较必须分开。等成本分析需要预注册共同 call budget、任务/调用顺序与在预算切点前已完成验证的 target；不能事后按 outcome 挑任务，或只用 raw points/task 宣称预算优势。
- Deep Yield 的 depth 使用真实已验证 gold frontier/tool-action pointer，不假冒依赖图深度；依赖图无法获得时标 unknown。至少报告 depth>=1/2/3 的 verified targets/全部实际 environment tool calls。
- Span：每次成功局部纠正后至下次错误，学生连续自主正确 action 数；区分 budget-censored span，报告 mean/median/P75/P90/zero-step ratio。
- 候选点按 task_id + verified state + target 去重，浅层重复不算新 frontier。预先锁定 GO 的操作化阈值后再取样。

这些只是避免无效实验的设计约束，不是已实现/已验证的 Recovery 算法，也不是放行训练的理由。
