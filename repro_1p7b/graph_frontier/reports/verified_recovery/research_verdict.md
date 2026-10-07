# Verified Recovery 研究判断

VERDICT = PHASE0-STOP

**不是 Recovery NO-GO / Negative Result；当前方法尚未被数据实验检验。** 固定100三次全通过，独立300第一遍299/300，ClinicalTrialsGov 真实 gold HTTP 请求与独立诊断均 DNS 失败。完整确定性 gate 未通过，不启动 GPU。

## 用户要求的九个问题

1. **每单位 interaction 多产生多少监督？** 未知。A/B collection 未运行，不能报0或提升倍数。
2. **新监督是否更深？** 未知。CPU gold replay 不是模型新暴露的错误点。
3. **纠正后学生能否自主继续？** 未检验；没有真实 Recovery span 或10个 case。
4. **intervention 占全部 action 比例？** 未测量。不能以 gold CPU 回放计算该比例。
5. **值得训练吗？** 现在不允许。必须先完整通过 Phase0，再通过真实 paired collection 的至少两项 GO 条件。
6. **最小 GPU pilot？** 现在不进入 Phase5，也不锁定训练配置。若未来 GO，只设计1-cycle matched pilot：同初始化与 heldout432/Frozen300 IDs，分固定交互预算和固定 targets/optimizer/token 预算两种比较，不能因为 B 产样更多就增加更新次数。不是本轮已授权启动的训练。
7. **最可能的 novelty？** 待验证：可执行环境的严格验证触发最小局部 intervention，再交回学生，是否以更低实际交互成本获得更深监督。没有数据就不能写成论文结论，不能宣称首次使用局部纠正或无 teacher LLM。
8. **与 DAgger/teacher correction 的区别？** DAgger 聚合当前策略访问状态的 expert labels，原文也允许混合 expert/student 控制；“不是整条 teacher forcing”本身不是与 DAgger 的本质新颖性。这里拟研究的是可执行 gold oracle + 严格 execution verification + 错误触发的局部执行 + 成本/深度证据，而不是消除了 oracle。干预后轨迹为 HYBRID，不能直接搬用纯 student state-distribution 的理论保证。[DAgger 原文，Algorithm3.1与其后段落](https://arxiv.org/pdf/1011.0686)。RLIF 已研究 intervention 信号与学习，不能声称介入机制本身首创；它以 intervention 信号作 RL rewards，与本任务的数据层 verified CE targets 目标不同。[RLIF 原文](https://arxiv.org/abs/2311.12996)。使用论文检索技能核对上述界限，HF markdown 端点不可读后回退到原论文。
9. **哪些 claim 禁止继续使用？** 已固定的 exposure-conditioned audit 没有显著 correction advantage；缺乏稳定 structural-transfer/automatic-curriculum 证据，不为这些方向追加 heuristic。本轮另外禁止把“固定100通过”写成“完整环境确定性通过”，或把尚未采集的 Recovery 写成提高效率、深度、自主 suffix 或最终能力。

## 下一项所需条件

先解决 ClinicalTrialsGov 在 MCP 执行路径中的真实 HTTP/DNS 依赖；或者另行明确制定不伪造响应、不改变任务/gold 的可审计版本化离线环境协议。现有 cached_studies 不是完整 API fixture，不能自行当 fallback 放行。然后新 epoch 重跑原100×3和固定随机300×2，不覆盖本次失败文件，不删除失败任务。

详细证据见 [determinism_audit.md](determinism_audit.md)。源码/CPU tests/小型报告进入 scoped commit，runtime logs、完整 replay、checkpoints 不提交；不 push，保留无关 BFCL 修改。
