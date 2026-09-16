# decode_graph Multi-Batch 与异步 Beam Search 迁移总结

## 1. 分支与目标

- 分支：`decode-graph-multibatch-port`
- 基线：`0efef35 feat(beam): rebase async stage scheduling on merged A2 (#384)`
- 目标：把已有的 Beam Search multi-batch 能力迁移到 `decode_graph`，并在不破坏同步路径的前提下接入 resident Request 异步调度。

本次工作分成四个实现提交：

| 提交 | 主题 | 作用 |
|---|---|---|
| `9415424` | Beam multi-batch 容量配置 | 建立 B 大于 1 所需的配置、校验和固定资源容量 |
| `459beb2` | 同步 NPU multi-batch | 打通同步调度下 B 个请求、每请求 W 条 beam 的完整执行链路 |
| `c626c12` | 异步 resident multi-batch | 接入单 Request 驻留式异步调度，并扩展到 B 大于 1 |
| `abc6ab4` | 可配置 NPU 异步测试脚本 | 提供隔离进程、可变 batch、循环与 profile 的实际 NPU 验证入口 |

## 2. `9415424`：Beam multi-batch 容量配置

### 修改内容

新增 `beam_max_batch_size`，用于声明单个 Worker 可以同时承载的逻辑 Beam Session 数量：

- `GRConfig.beam_max_batch_size`，默认值为 1；
- additional config 键 `vllm_gr_beam_max_batch_size`；
- CLI 参数 `--beam-max-batch-size`；
- schema v1 的 `beam.max_batch_size` 读写与 round-trip；
- 校验范围为 `1..128`。

该配置被传递到 ModelRunner、Beam attention context、NPU workspace 和 graph capture。资源几何从原先只考虑 W，扩展为同时考虑：

```text
logical requests = B
physical decode rows = B * W
```

其中 B 的上限由 `beam_max_batch_size` 决定，W 的上限由 `beam_max_width` 决定。

### 设计意义

这个提交只建立容量和配置基础，不负责具体的跨请求调度。默认值仍为 1，因此原有单 batch 行为保持不变；只有显式配置更大的容量时，才会分配 multi-batch 所需的固定缓冲区。

### 配套修改

- 补充配置解析、序列化、CLI 和边界校验测试；
- 更新 runtime patch inventory；
- 调整 GPU/NPU ModelRunner 初始化，使固定 workspace 能按 B 维度扩展。

## 3. `459beb2`：同步 NPU Beam multi-batch

### 修改内容

该提交打通同步调度下的完整 B 大于 1 路径。

#### 3.1 批量 step 协议

新增 `BEAM_STEP_BATCH` EngineCore 请求类型以及对应 codec、SyncMPClient 和 InprocClient 接口。Frontend 可以在一次消息中提交同一个 decode step 的 B 个 `BeamRequestStepUpdate`。

#### 3.2 Cohort 屏障

EngineCore 为同一批请求登记 expected/pending cohort。只有同一 step 的所有成员到齐后，才会一起进入 Scheduler，避免部分请求提前执行导致 slot、prefix KV 和 suffix KV 错配。

#### 3.3 `BeamExecutionBatch`

新增 `vllm_gr/v1/beam/execution_batch.py`，把 B 个独立 Beam Session 暴露成一个零拷贝执行视图：

- Scheduler 保留 B 个独立 request ID；
- attention 和 beam decision 看到连续的 `B * W` 物理行；
- 要求 slot 连续为 `0..B-1`；
- 要求 cohort 内的 W、decode step 和 terminal 状态一致；
- 禁止 winner parent 跨越请求边界。

#### 3.4 Worker、attention 和结果拆分

- 将 positions、sampling metadata、logits indices 等从逻辑 B 行映射到物理 `B * W` 行；
- 将多个 session 的 unshared KV 暴露为连续布局，供 xllm attention 使用；
- 在一次 NPU beam decision 中处理整个 cohort；
- 通过 `split_batch_decision()` 把全局结果恢复成 B 份 request-local 结果，并把 parent 索引转换回每个请求内部的 `0..W-1`。

### 安全检查

在进入 kernel 前增加批次顺序、slot 连续性、共享 context、宽度和 step 一致性检查。相比静默地产生错误结果，这些条件不满足时会直接失败。

### 兼容性修复

本提交同时补齐了 EngineCore 协议契约、InputBatchProxy、NPU runner 和上游私有方法签名的测试适配，避免 multi-batch patch 破坏已有单 batch 合约。

## 4. `c626c12`：异步 resident Beam multi-batch

### 修改内容

该提交在同步 multi-batch 基础上，引入 Stage 1 resident Request 异步路径。

#### 4.1 新配置与入口

- 新增 `beam_async` / `vllm_gr_beam_async`；
- 新增 `GRLLM.async_beam_search()`；
- `beam_async=True` 时保留 vLLM async scheduling；
- 原 `GRLLM.beam_search()` CUSTOM 路径仍按同步 Stage 0 运行，不允许误用 async scheduling。

异步入口当前要求：

- `beam_async=True`；
- `async_scheduling=True`；
- CUSTOM attention；
- `ignore_eos=True`；
- 不使用 Trie 约束；
- `concurrency_limit <= beam_max_batch_size`。

#### 4.2 单 Request 驻留生命周期

每个输入只向 EngineCore 提交一次 prefill Request：

1. Frontend 先发送 `BEAM_SESSION_TEMPLATE`；
2. 再通过 `ADD_BATCH` 提交 B 个 prefill Request；
3. prefill sampler 生成 top-W 首 token并初始化 device beam state；
4. 同一个 Request 保留在 AsyncScheduler 的 running 集合中；
5. 后续 decode step 不再由 Frontend 逐步重新注册请求；
6. terminal step 才回传完整 sequence 和 score。

#### 4.3 Resident NPU 快路径

- decode 时把逻辑 B 个请求展开为 `B * W` 物理行；
- `_resident_beam_overwrite_input_ids_npu()` 直接从 device sequence buffer 取 winner token 并覆盖下一步 input IDs；
- x_attention 同时读取共享 prefix KV 与每条 beam 的 unshared suffix KV；
- beam decision 后立即按 winner parent 重排 suffix KV；
- 非 terminal step 只发布轻量 routing stub，避免完整 beam 结果逐 step D2H；
- terminal step 使用紧凑 D2H，并按 request 拆分最终结果。

#### 4.4 异步 B 大于 1

resident 模式复用 `BeamExecutionBatch`，使 B 个长生命周期 Request 可以在同一个 decode step 中一起执行，同时保持独立的 request ID、slot、sequence、score 和结果通道。

#### 4.5 集成问题修复

在迁移和定向测试过程中，还修复了以下兼容问题：

- patch marker 使用明确布尔语义，避免 `MagicMock` 或代理对象的 truthiness 被误判；
- sampling metadata remap 使用独立副本，避免修改调用方持有的原始对象；
- sampling metadata 在 prepare 阶段预热并缓存，避免重复 gather；
- 对上游 ModelRunner 私有方法签名变化保持契约兼容；
- 修复异步输出路径缺失的运行时 import；
- 收紧同步 batch context 与异步 resident 部署组合的错误提示和防护条件。

## 5. `abc6ab4`：可配置 NPU 异步测试脚本

新增：

```text
examples/offline_inference/beam_search/offline_async_beam_search.py
```

原来的两个测试脚本提交已合并为当前一个提交。

### 主要能力

- 每次进程只测试一种 batch geometry，避免 Engine、graph 和 NPU runtime 状态在 B=1/B=2 之间复用；
- `--batch-size` 支持 1 到 128，可测试 B=1、2、4、8、16、32、64 等；
- `--engine-batch-capacity` 可把活动 batch 与固定容量分开配置；
- `--execution graph|eager` 用于对比 graph 与 eager；
- `--runs` 默认 5 轮，只在最后一轮启动 profile；
- 支持配置 model、prompt、beam width、max tokens、显存比例和调度容量；
- JSON 保存每个 item 的全部 beam token、recommendation token、score、SHA-256 token hash 和每轮耗时；
- 启动前检查 `max_num_seqs`、`max_num_batched_tokens >= B * W` 等资源条件；
- 支持通过 `GR_B2_MODEL` 或 `VLLM_GR_TEST_MODEL` 指定模型。

## 6. 验证结果

### 已验证

- multi-batch 基础协议、cohort、codec、InputBatchProxy 和 BeamExecutionBatch 定向单元测试通过；
- 配置、ModelRunner、async scheduler 和上游契约回归测试通过；
- 同步 B=1 与 B=2 NPU 路径可以完整执行；
- 异步 resident B=1 与 B=2 可以完整执行并生成 JSON、日志和 profile；
- 测试脚本可切换 graph/eager，并可扩展到 B=4、8、16 等容量。

### 尚未关闭的问题

1. **B 大于 1 的数值/排序等价性仍需继续验证。** 功能可运行不等于输出已经和两次独立 B=1 完全一致。
2. 多生成 token 时在 eager 模式也观察到差异，因此目前不能把问题只归因于 graph capture；应优先检查 resident token 回填、x_attention 的 suffix KV slot、winner parent 与同 step KV reorder。
3. NPU kernel 完成后仍有明显 Frontend 后处理时间，主要来自全部 beam 的 token 拼接、排序、`batch_decode()`、JSON/日志物化；这部分尚未纳入本轮性能优化。
4. 异步 resident 路径当前固定按 decode step 数终止，尚不支持 Trie 和逐 beam early EOS。

## 7. 最终提交结构

```text
abc6ab4 Add configurable NPU async beam example
c626c12 Port async resident beam scheduling to multi-batch
459beb2 Port synchronized NPU beam multi-batch
9415424 Add beam multi-batch capacity plumbing
0efef35 feat(beam): rebase async stage scheduling on merged A2 (#384)
```

由于测试脚本提交已 squash，且旧文档提交已移除，如果该分支此前已经推送，更新远端时应使用：

```bash
git push --force-with-lease origin decode-graph-multibatch-port
```

不要使用无保护的 `--force`。
