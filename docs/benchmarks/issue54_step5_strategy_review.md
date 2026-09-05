# Issue #54 Step 5：单独提交用户评审的策略差异

团队已达成以下实施/验收共识。**Step 5 的修订验收通过，不代表原文所有
字面条件通过，也不代表整个 Issue 完成。** 详细命令、原始结果、失败记录与
trace 路径见 [Step 5 证据记录](issue54_step5_review.md)。

## 需要重点评审的三项验收口径

| 原始要求 | 本轮采用的口径 | 明确保留的限制 |
| --- | --- | --- |
| 一个 executing + 一个 prepared，同时 `issued-completed <= 1` | 两个已 dispatch 的 stage 都如实计入 issued；`outstanding <= 2`，`lookahead = max(outstanding-1, 0) <= 1` | 原字面不等式 **未通过**；每个 SchedulerOutput 仍只推进一步，不允许第三个在途 stage。 |
| legacy 与 ASYNC 的完整输出严格 parity | 有效 legacy miss token/score 对照保留；损坏的 legacy parent/selected-logprob 参考改用独立 CPU 候选回放，并核对实际下一次 Forward token/position 与最终 ABI | **legacy-hit 严格 parity 未通过**。候选回放不独立验证 sampler/softmax、模型 Forward 或 KV 数学，不能冒充完整模型 oracle。 |
| eager 与 CUDA Graph parity | 主配对隔离 Beam Graph 变量：同一编译配置下 compiled Beam-eager 对 FULL Beam Graph；另行运行真正 `enforce_eager=True` 的候选及 metadata 诊断 | 六组配对均为 `enforce_eager=False`，不能称为完整 uncompiled-eager/Graph 数值矩阵；后者仍属后续正确性工作。 |

## 实现选择与新增边界

- **控制输出完成通知**：增加 GPU 安装时编译的 C++17/pthread native ABI 2
  扩展，配合两个固定 control slot 和一个 CUDA 通知 stream。无需 NVCC/CUDA
  headers，不在 Worker 执行时编译；GPU persistent ASYNC 缺失/版本不符时
  fail-fast。CPU/NPU 安装不构建该扩展。未增加第二套 Future queue。
- **上游 V1 适配方式**：保留原 AsyncGPUModelRunnerOutput 类，只在明确的 GR
  sampling scope 内适配构造/读取；普通请求和非最终 Prefill chunk 委托原实现。
  GPU 在原 `_prepare_inputs()` 后写设备真实 token，避免被原逻辑覆盖；NPU
  的“必须在原 prepare 前写入”约定不变。
- **逻辑/物理行数**：SchedulerOutput 保持一个逻辑请求；Worker execution
  frame 暂时展开 W 行，`finally` 恢复原计数，typed sidecar 表达物理几何。
- **固定 Decode metadata**：设备生成的布局来自 request-group geometry，
  不是写死全零 request index。布局变更会失效旧 buffer resident 标记；多组
  layout 单测不等于允许 B>1 admission，本轮仍明确拒绝 B>1。
- **状态权威及释放**：完整 sequence/score 权威仍在 Beam session decision；
  workspace 仅补充 parent/local-logprob 列和最终回溯 scratch。逻辑 state
  释放不等于允许覆盖尚未完成的 control DMA；只有设备完成与 Host 消费/丢弃
  都满足后才归还 slot，错误路径保留所有权直到成功 drain。
- **异常和 shutdown**：观察实际 release RPC 的完成/异常；失败只退休受影响
  Engine/Executor。Worker drain 成功后再处理 Native/Frontend GR ownership；
  普通请求的健康路径不增加调度队列、同步屏障或全局 sleep 改动。
- **安全等待**：persistent Decode 的原 input-prep Event 已完成时跳过冗余
  synchronize；未完成仍保留原安全 guard。当前 smoke/trace 的 Decode 均走
  ready 分支，不能把这一观测扩大为所有配置永远无等待。

## 测量方式变更

真实 GPU 证明：旧 empty BatchQueue output 可携带下一请求的 cache stats，
而上游 Offline logger 不记录空输出，因此 Prometheus 漏计。本轮不修改普通
logger 行为；验收用绑定实际 UniProc KVCacheManager 的 profile-only observer
记录调用和真实返回 hit，**同时保留原 Prometheus 值、来源标签和差异**。
带这些 hook 的 E2E 样本不用于性能收益结论。

真实构造/add/free/dispatch 另有独立调用观察器，不能把 GR stats 中声明的 1/0
字段当成直接采集。诊断额外 D2H 与 Host 事件存储也不混入产品 transport 或
长期内存测试结果。

## 未解决项与非变更项

- hit/miss 最终输出差异仍存在。已独立复现首层矩阵乘法 M=1024/16 引入 BF16
  差异；实际 eager attention 消费的 Prefix table/slot/length 检查通过，但
  **尚未证明全部最终差异仅由数值误差导致**，没有调整普通 GEMM 设置或容差。
- 最新 trace 有 D→D 调度/计算重叠，P→D 未观察到重叠；未承诺性能百分比。
- NPU 无硬件验收，Step 6 不得写成通过；完整 CI、Step 8 矩阵、Step 9 的
  P99/goodput、E2E 归因与长期 soak 仍未完成。
- GPU/NPU V2、bundled/MULTI_STEP、递归 execute、B>1 admission 均未纳入。
- legacy 路径保留作阶段性对照；其 cache-disabled 生命周期 bug 没有通过
  修改普通请求或强制启用 Prefix Cache 来掩盖。
