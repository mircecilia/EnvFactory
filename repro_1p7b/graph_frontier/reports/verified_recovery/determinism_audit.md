# Verified Recovery — Phase0 确定性审计

VERDICT = PHASE0-STOP / DETERMINISM_FAILED

日期：2026-10-07。仅 CPU；没有启动模型推理、Recovery collection 或训练。

## 实测结果

| 固定任务组 | 回放次数 | 严格通过 | 结果 |
|---|---:|---:|---|
| 原 fixed100 | 1 | 100/100 | PASS |
| 原 fixed100 | 2 | 100/100 | PASS，完整输出与第一次 typed_equal |
| 原 fixed100 | 3 | 100/100 | PASS，完整输出与第一次 typed_equal |
| 独立 random300 | 1 | 299/300 | FAILED |
| 独立 random300 | 2 | 未运行 | 第一遍失败后 gate 阻止继续 |

独立随机组从原 1726 train tasks 中排除 fixed100 后，按固定 seed=20261008 均匀抽样。两组不重叠，不更换失败样本，不从 heldout 选取。固定任务组沿用旧 40/40/20 配额及补齐方案，不能视作均匀总体样本。

## 已修复的生产端问题

| 环境 | 修复 |
|---|---|
| GorillaFileSystem | 目录项按名称排序 |
| WhatsApp | chats 按 phone 排序后再 limit；不改消息时间序列 |
| AirDNA | markets 按 market_id 排序 |
| MetMuseum | objectIds 排序 |
| EtsyServer | listing_id 稳定选择后再 limit；保留图片顺序 |
| TickTick | tasks 按 taskId；columns 保留显式 project_columns 顺序 |
| GitHubServer | comment/review ID 的 Python hash 改为 SHA256 后取原相同整数范围 |
| TravelBooking | airport_codes 的 list(set(...)) 改为 sorted(set(...)) |

第八项来自初次修复后 fixed100 第二遍的 99/100：`official-rl-1102-408a0c8be273` 的机场代码数组随进程随机化。三个额外 CPU replay 确认仅机场代码顺序不同。失败记录保留，修复后建立独立 environment_v3 epoch 重跑全部任务。

8 项真实 API 方法 CPU 单测全部 PASS。没有对 verifier/equality/reward 做改动。排序仅用于未声明排名的集合；TickTick 显式顺序、Etsy 图片顺序等未排序。

## 唯一剩余阻塞

- task_id：`official-rl-0382-c3b604ead122`
- environment：`ClinicalTrialsGov`
- source_row_index：382
- 第一条固定 gold action：`ClinicalTrialsGov-clinicaltrials_search_studies(condition="glioblastoma", pageSize=1000)`
- 真实 API：`https://clinicaltrials.gov/api/v2/studies`
- 错误：`Error executing tool clinicaltrials_search_studies: [Errno -3] Temporary failure in name resolution`
- 独立诊断 replay 再次失败，耗时 10.975 秒，未替换原失败记录。
- 远端主进程的 `socket.getaddrinfo('clinicaltrials.gov',443)` 可以解析；不能据此声称 MCP tool 子进程已恢复。故障精确可确认到 MCP 中的 HTTP 请求 DNS 失败，不能进一步无证据猜系统根因。

源码 `ClinicalTrialsGov.py` 的 search/list/results/trends 使用实时 httpx 请求；初始 scenario 的 4 条 cached_studies 并不是搜索、结果与趋势 API 的完整离线 fixture，这些方法也不读取该缓存作 fallback。不能静默用历史 gold observation 替代真实工具响应。

## 不变量与缓存说明

1. 官方 `ground_truth` 的 name、typed arguments、顺序，与旧 gold actions 严格相等。`masked_arguments` 是非执行注释，投影方法与已有 official gold_replay 相同，不是删除动作参数。
2. 初始状态、每一步 gold post-state、完整最终状态均与旧 gold cache 严格相等；官方 final-state verifier 必须为 True。
3. fresh raw observations 不在 verifier 中归一化；独立进程重复回放直接使用原 typed_equal 比较整个 replay payload（包括 schema、状态、observation、verifier decision）。本次 decision probe 是执行完整 gold 后的 SUCCESS，不等于检验了所有错误路径的 verifier 行为。
4. 修复后的新 observation cache 与旧 cache 分版本；fixed100 的 12 个任务、random300 已通过任务中的 9 个存在旧 observation 差异，未改写旧文件。canonicalization/稳定 ID 改变的是环境版本的输出表示；旧输出不能被声称与新输出完全相同。
5. 最早 100/100 的 `gold answer drift` 是审计器把 masked_arguments 注释当执行动作的工程错误，在调用环境之前发生；已保留该失败目录，不能计为 100 个环境不确定性问题。

## 权威证据文件（远端）

根目录：`/home/u2024311031/workspace/envfactory_repro_1p7b/repro_1p7b/graph_frontier/verified_recovery_20261007_run/`

- `environment_v3/plan.json`，SHA256=`90d51a3c546b9e572e9c9f3f4722752997e12418fdf1a795b6cd6d1c1c7d8045`
- `environment_v3/audit_protocol_v3.json`：审计代码、父 plan 和目录版本锁。
- `environment_v3/gate_v3_fixed100_{0,1,2}.json`
- `environment_v3/gate_v3_random300_0.json`
- `environment_v3/determinism_v3/{fixed100,random300}/replicaN/<task_id>.json`：逐任务结果与完整成功 replay。
- `environment_v3/diagnostics/clinicaltrials_external_dependency.json`
- `environment_v3/status`：`DETERMINISM_FAILED:29`
- 旧 `determinism/`、`determinism_answer_schema_v2/`、`diagnostics/additional_failure_1102.json` 保留不覆盖。

全体 tools 源码、官方源数据、协议、历史 episode 与核心 verifier 文件均有锁。runtime 产物不提交 Git。

## 决策

Phase0 整体失败，不能标 READY。依用户硬门槛停止于此：不运行 Recovery/GPU，不以 299/300、跳样本或历史缓存作放行依据。此结果是基础设施/可复现性阻塞，不是 Recovery 数据效率假设的 Negative Result。后续须先解决真实 HTTP 执行条件或明确批准可审计的版本化离线环境协议，再从独立 epoch 完成整个 gate。
