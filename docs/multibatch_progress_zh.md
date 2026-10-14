# 多 Batch 功能进展

截至 2026 年 10 月，[上游 PR #433](https://github.com/JiusiServe/vllm-gr/pull/433) 已合入 GPU Beam 多 Batch 执行能力。离线调用现在可以把多个请求作为一个 Batch 提交，当前 GPU 路径支持 **B=1、2、4**。这让多个请求分摊一次调用中的固定开销，也为后续按实际业务负载调整吞吐和延迟提供了选择。

## 当前设计

- **上层共用**：GPU 和 NPU 在 EngineCore 及以上复用请求接入、Batch 调度和结果返回流程；Worker 侧分别实现各自的设备执行逻辑，保持对外功能一致。PR #433 的性能数据只针对 GPU。
- **固定成员**：调用方一次提交 B 个请求，它们组成一个 Batch Session。运行中不会动态加入、移走或重新配对请求；结果按提交顺序返回。即使某个请求先结束，Batch 中未完成的请求仍继续执行。
- **没有自动组 Batch**：当前系统不负责把分散到达的请求攒成一批，也不会跨多次调用合并请求。调用方需要在调用前确定成员和 B。
- **请求隔离**：Decode 时按 `B × Beam 宽度` 组织计算，但每个请求保有自己的 Beam 状态和 KV。取消、异常、结束后的清理以整个 Batch 为单位处理，避免留下未释放的状态。

目前离线单进程场景下，框架自身开销已降至约 **4 ms**，这部分优化已比较充分。这个数字描述框架开销，与下文包含模型计算的完整请求耗时不是同一个统计口径。

## Batch 大小如何配置

离线 `GRLLM.beam_search_v1(prompts, params)` 中，**一次调用传入的 prompt 数量就是 B**。同一 Batch 使用同一组 Beam 参数。调用方需要自行将请求分成 B=1、2 或 4 的组，剩余请求再用受支持的大小提交；不支持的大小会在接入阶段被拒绝。

```python
B = 2
batch_prompts = prompts[:B]
outputs = llm.beam_search_v1(batch_prompts, params)
# outputs[i] 对应 batch_prompts[i]
```

创建引擎时也要为目标 Batch 留出容量：

| 配置 | 需要关注的关系 |
| --- | --- |
| `max_num_seqs` | 至少为 B；它限制一次可接入的请求数。 |
| `max_num_batched_tokens` | 当前多请求 Prefill 要整批调度，需覆盖该 Batch 的实际输入 Token 总量；可参考 `max(max_model_len, B × input_len)` 设置。预算不足时会拒绝该 Batch。 |
| `long_prefill_token_threshold` | 不要设置成会触发 Batch 内请求切块的值；当前路径会拒绝这种情况。 |
| `beam.max_width` 与 `BeamSearchParams.beam_width` | 前者至少覆盖实际 Beam 宽度 W；设备执行规模随 `B × W` 增加。 |
| `max_model_len`、显存预算 | 覆盖输入和生成长度，并检查目标 B 下的峰值显存。 |

PR 的 GPU 测试使用 V1 Beam、异步调度和 CUDA Graph。上述数值是容量条件，**不是推荐所有业务固定使用 B=4**：prompt 长度、Beam 宽度、生成步数、约束表、设备和延迟目标都会影响合适的 B。

## 为什么暂不采用 CB

这里的 CB 指 Continuous Batching，即运行中按请求到达和完成情况动态调整执行批次。当前主要面向离线单进程、由调用方一次给出请求集合的工况，因此选择固定成员的 Batch：调度和结果归属更明确，Worker 可以按确定的 `B × Beam 宽度` 形状执行。动态加入、替换或独立退出成员需要另一套状态管理，当前没有实现，也没有自动组 Batch 的入口。若后续转向请求持续到达的在线场景，再根据收益评估 CB 和组批能力。

## 为什么当前不切 Prefill chunk

[Issue #5](https://github.com/ZhiYunpenghuawei/vllm-gr/issues/5) 比较了相同的 4 个 4095-token 请求：同步 `generate` 端到端计时，不含引擎初始化和 CUDA Graph 捕获，每组汇总 30 次采样。

| Prefill 方式 | 中位延迟 | 相对整批执行 |
| --- | ---: | ---: |
| 4 个请求一次进入 16K batch-4 图 | 578.74 ms | 基准 |
| 每轮 4×1K，使用 4K batch-4 图，共 4 轮 | 584.45 ms | +5.71 ms（+0.99%） |
| 逐请求执行，使用 4K batch-1 图，共 4 轮 | 586.52 ms | +7.78 ms（+1.34%） |

在这组请求和计算量相同的测试中，整批 Prefill 最快；切 chunk 会多出几轮执行，而没有带来延迟收益。因此当前固定 Batch 路径优先一次完成 Prefill，并尽量选择刚好覆盖实际 Token 数的图 bucket，减少 padding。当前多请求 Beam Prefill 也尚未支持切 chunk，触发 Token 预算或长请求切块条件时会在接入阶段拒绝。

这不是对所有场景都禁用 chunk 的结论。显存不足、需要公平调度或避免长请求阻塞时，切 chunk 仍可能必要；那时需要补齐对应实现，并重新比较延迟与资源收益。

## 已有测试数据

### PR 原始测试

PR #433 使用 OneRec-1.7B、输入长度 1024、Beam 宽度 128、生成 3 个 SID token、CUDA Graph；预热 2 次，测量 5 次。串行列表示把同样的 B 个请求逐个按 B=1 执行，Batch 列表示一次提交 B 个请求。

B=4 的平均每请求分摊耗时比单独执行 B=1 低约 19%，吞吐从 18.19 提升到 22.52 req/s。表中的“总耗时 / B”是分摊成本；如果业务关心一次 Batch 中单个请求何时拿到结果，还应直接看 **Batch 总耗时** 和实际等待时间。

<img width="1113" height="752" alt="d9702c9aaba4847bcbaf86b83ed2a0cb" src="https://github.com/user-attachments/assets/07501dba-9fa8-4d60-bb31-f2392ecd50ff" />


### Review A/B 复测（NVIDIA L20X）

[这条 review](https://github.com/JiusiServe/vllm-gr/pull/433#pullrequestreview-5479329092) 明确标注硬件为 **NVIDIA L20X**。它使用 OneRec-1.7B BF16、Beam 宽度 128、3 个 SID token、4 条不同的 1024-token prompt（加上 `sid_begin` 后模型输入为 1025 token），以及合成的 2048 条约束目录。单卡、离线单进程，关闭 prefix caching 和 chunked prefill；两次进程重复采用 ABBA 顺序，每种模式每次测 20 组，预热组不计入结果。

| 配置 | 分摊耗时 | 吞吐 | 相对本次 Head B1 |
| --- | ---: | ---: | ---: |
| Base B1 | 19.71 ms/请求 | 50.73 req/s | — |
| Head B1 | 19.84 ms/请求 | 50.40 req/s | 1.00× |
| Head B2 | 12.44 ms/请求 | 80.41 req/s | 1.60× |
| Head B4 | 10.63 ms/请求 | 94.11 req/s | 1.87× |

Base 与 Head 的比较只用于看 B1 是否回退；B2/B4 的收益是在同一 Head 内与 B1 比较。B2、B4 的平均整次调用耗时分别约为 24.87 ms 和 42.50 ms，不能把表中的分摊耗时当作单个请求的实际响应时间。B2 的 Prefill 在这组配置下没有命中 CUDA Graph，因此各个 B 的 Prefill 路径也不完全相同。

### 输出一致性

PR 原始测试中，B2、B4 的 Top-1 与逐个执行 B1 一致，Top-32 候选集合重合率分别为 93.75%、92.97%。上述 L20X review 在正常计算模式下得到另一组结果：

| B | 相对 B1 的 Top-1 一致请求数 | Top-32 重合率 | Top-100 重合率 |
| ---: | ---: | ---: | ---: |
| 2 | 2/4 | 96.875% | 98.50% |
| 4 | 3/4 | 97.656% | 98.25% |

Review 的 4 条 prompt 使用合成约束目录，测的是与 B1 的数值一致性，不是对真实推荐目标的准确率。在开启仅用于诊断的 `VLLM_BATCH_INVARIANT=1` 后，B2/B4 的 Token 和分数与该模式下的 B1 完全一致；该模式未计入性能对比。正常模式的 Top-1 结果仍应使用真实约束表和目标业务数据复核，不能仅凭这些小样本数据断言所有场景完全一致。

## 如何为业务选择 B

建议用实际模型、约束表和有代表性的 prompt 长度、Beam 宽度及生成步数，分别测 B=1、2、4。保持硬件、引擎配置和请求集合一致，先预热模型、JIT 与图捕获，再做足够多轮测量。至少比较：

1. **延迟**：Batch 总耗时及业务关心的 P50/P95，是否满足单请求等待时间要求。
2. **吞吐和资源**：req/s、分摊耗时、峰值显存；更大的 B 是否带来值得接受的增益。
3. **结果**：对照逐个执行的 B=1，检查 Top-1、候选集合和分数差异，按业务容忍度判断。

选择满足延迟、显存和结果要求的 B，再看吞吐收益。上表只说明 PR 测试工况的进展，不代表其他设备或业务负载也会得到相同收益。

## 后续工作

继续在真实数据上验证不同 Batch 大小的性能与结果一致性；待 GPU、NPU 功能进一步对齐后，考虑重构 Worker 侧的重复逻辑，保持上层共用、设备侧独立的边界。
