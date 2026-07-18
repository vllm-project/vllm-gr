# 性能回退定位与待测试计划

## 1. 目标与当前判断

目标是在 ModelRunner V1、精度一致的前提下，使 vLLM 0.22.1rc1 + vllm-gr 相对
vLLM 0.14.1 + vllm-gr 不发生性能回退。

当前 L20 的 100 请求测试显示：

- TTFT P99 回退 8.16%；
- Beam 总耗时回退 6.17%；
- Prefill 回退 4.49%，Decode 回退 6.84%；
- 500 请求离线吞吐回退约 2%；
- `request-rate=inf`、`max-concurrency=1` 的串行测试中，新版本 Beam 总耗时反而提升
  3.99%，P99 TTFT 提升 3.64%；
- Sort 耗时没有变化，优先级低。

因此当前回退不是稳定存在于所有执行模式：它出现在 0.5 RPS、允许请求交叠的低负载测试，
但在单请求连续执行时消失。后续应先区分“请求间空闲”与“请求交叠/批处理”，再进入底层
kernel 定位。完整数据见
[L20 在线性能对比结果](l20_online_performance_results.md)。

## 2. 定位原则

- 每次只改变一个变量，不能同时更换运行栈、启动参数和 vllm-gr 代码。
- 旧、新版本必须在同一张卡、相同功耗和频率策略下串行测试。
- 两组请求必须由同一个 0.22.1rc1 客户端容器生成，固定数据顺序和 seed。
- 每个筛查配置先跑 100 个性能请求、重复 3 次，取三次中位数；筛查阶段关闭精度测试。
- 候选优化通过性能筛查后，再恢复完整精度和 A100 CI 验证。
- 结论以无统计显著回退为准，不能用一次偶然更快的结果放行。

推荐筛查命令：

```bash
RUN_ACCURACY=0 \
ONLINE_PERF_NUM_PROMPTS=100 \
ONLINE_PERF_NUM_WARMUPS=10 \
PORT=<server-port> \
./do_test.sh <result-tag>
```

每次重启服务后都先确认：

```bash
curl -s http://127.0.0.1:<server-port>/v1/models | python -m json.tool
ss -ltnp | grep ':<server-port>'
nvidia-smi
```

## 3. 第一阶段：先排除测试和环境差异

### 3.1 固化两套运行栈

在两个容器中分别保存以下信息。vLLM 0.14.1 和 0.22.1rc1 支持的 PyTorch、CUDA、
Triton 版本可能不同，不能在分析时默认二进制运行栈完全一致。

```bash
python - <<'PY'
import importlib.metadata as md
import platform

import torch

print("python:", platform.python_version())
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
for name in ("vllm", "vllm-gr", "triton", "flash-attn", "flashinfer-python"):
    try:
        print(f"{name}:", md.version(name))
    except md.PackageNotFoundError:
        print(f"{name}: not installed")
PY

nvcc --version
nvidia-smi --query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu \
  --format=csv
python -m torch.utils.collect_env
python -m pip freeze
```

同时记录两个源码仓的 `git rev-parse HEAD`、`git status --short`、完整服务启动命令和
服务端启动日志。若任何一个容器中 `import vllm_gr` 指向了另一套共享代码目录，结果无效。

### 3.2 核对请求与输出 token

当前两组输入 token 完全相同，但 generated tokens 分别为 62,761 和 64,000。需要从
`performance-detail.json` 比较每个 request 的：

- prompt token 数；
- completion token 数；
- HTTP 响应中的 `usage`；
- 128 个 choice 是否全部达到相同长度；
- EOS、finish reason 和失败重试情况。

增加一组两端都使用相同固定输出策略的诊断测试，例如客户端支持时加入
`--ignore-eos`，用于排除 EOS 和 token 计数差异。该测试只用于归因，不替代正常业务配置。

### 3.3 控制机器状态

每个版本开始前应停止另一套服务，确认 GPU 上没有残留进程。测试期间记录 GPU 功耗、
SM 时钟、温度和利用率，避免 0.14.1 与 0.22.1rc1 分别落在不同的时钟或热状态。
推荐交替执行 `old -> new -> new -> old`，降低固定测试顺序带来的偏差。

## 4. 第二阶段：无排队串行诊断

串行模式用于回答“回退来自单请求执行路径，还是来自请求重叠、排队和调度”。它不是容量测试。

```bash
# 旧服务已在 7001 就绪
PORT=7001 \
ONLINE_PERF_NUM_WARMUPS=10 \
./do_test.sh 0.14.1-serial serial

# 新服务已在 7002 就绪
PORT=7002 \
ONLINE_PERF_NUM_WARMUPS=10 \
./do_test.sh 0.22.1rc1-serial serial
```

判定方式：

| 结果 | 主要怀疑方向 |
| --- | --- |
| 串行回退消失，0.5 RPS 下仍回退 | async scheduling、并发请求交叠、队列或批处理策略 |
| 串行 Prefill 仍回退 | 输入处理、首次 Engine 请求、attention/compile 图或 prefix cache 路径 |
| 串行 Decode 仍回退 | Beam 循环、采样/logprobs、Engine IPC、attention 或 GPU kernel |
| Prefill、Decode 都接近，但 TTFT 仍回退 | API server、chat template、序列化、网络流式首包 |
| 串行和负载模式都回退 | 优先定位每请求固定开销和 GPU 执行，不先调 scheduler 容量参数 |

### 4.1 已完成的串行测试结果

测试参数为 100 个正式请求、10 个内部 warmup、`request-rate=inf`、
`max-concurrency=1`。两组输入 token 相同，均为 238,753。

| 指标 | 0.14.1 | 0.22.1rc1 | 新版本变化 |
| --- | ---: | ---: | ---: |
| Benchmark duration | 27.78 s | 26.74 s | **-3.74%** |
| Request throughput | 3.60 req/s | 3.74 req/s | **+3.89%** |
| Mean TTFT | 272.72 ms | 262.37 ms | **-3.80%** |
| Median/P50 TTFT | 286.68 ms | 275.25 ms | **-3.99%** |
| P90 TTFT | 303.34 ms | 294.68 ms | **-2.85%** |
| P95 TTFT | 308.29 ms | 298.80 ms | **-3.08%** |
| P99 TTFT | 317.21 ms | 305.66 ms | **-3.64%** |
| Avg Prefill Time | 78.22 ms | 73.84 ms | **-5.60%** |
| Avg Decode Time | 153.20 ms | 148.36 ms | **-3.16%** |
| Avg Sort Time | 0.038 ms | 0.036 ms | -0.002 ms |
| Total Beam Time | 231.46 ms | 222.23 ms | **-3.99%** |

新版本的 Prefill、Decode、TTFT 和 Beam 总耗时方向一致，单请求连续执行路径没有复现
之前的性能回退。当前不能把问题归因于 ModelRunner V1 的通用单请求执行，但仍需检查它在
并发 batch、mixed prefill/decode 和不同 CUDA Graph shape 下的表现。

### 4.2 与 0.5 RPS 结果的反向关系

| 模式 | 0.22.1rc1 相对 0.14.1 |
| --- | --- |
| 0.5 RPS、未限制最大并发 | Beam 总耗时回退 6.17%，P99 TTFT 回退 8.16% |
| `request-rate=inf`、`max-concurrency=1` | Beam 总耗时提升 3.99%，P99 TTFT 提升 3.64% |

串行测试同时改变了两个变量：禁止请求重叠，并且让请求持续到达、GPU 不再长时间空闲。
因此“串行变快”还不能直接证明是调度并发问题。目前需要区分两个方向：

1. 低到达率下的 GPU 降频、缓存冷却、周期性后台工作或 idle-to-active 开销；
2. Poisson 请求偶发重叠后产生的调度、batch shape、mixed prefill/decode 或 Engine IPC 开销。

### 4.3 `Peak concurrent requests: 6` 的解释

串行日志虽然输出 `Peak concurrent requests: 6`，但参数解析已经确认
`max_concurrency=1`。该指标不是精确的同时在途请求数：vLLM benchmark 将每个请求按
整数秒分桶，并把请求计入从开始秒到结束秒的所有桶，结束秒也包含在内。多个实际串行的
请求落在同一个一秒桶时会被累加，因此可能显示为 6。

这一字段不能用于否定 semaphore 的并发限制。100 请求耗时 26.74 秒、平均 TTFT
262.37 ms 也与单请求串行执行相符。如需精确并发，应使用详细 JSON 中的 `start_times`、
`ttfts` 和 `itls` 重建请求时间区间，或直接观察服务端 running/waiting request 指标。

### 4.4 Generated token 差异

串行测试再次复现了输出 token 数差异：

| 版本 | Total generated tokens |
| --- | ---: |
| 0.14.1 | 62,761 |
| 0.22.1rc1 | 64,000 |

新版本正好为每请求 640 tokens，旧版本少 1,239。需要比较
`performance-detail.json` 中每个请求的 `output_lens`、生成文本、EOS 和 finish reason，
确认这是部分 Beam 提前结束还是 API token 统计变化。在差异解释清楚前，不能用 output
token throughput 直接判断版本快慢。

新版本在报告更多 token 的同时，串行请求仍然更快，说明其单请求执行效率至少没有表现出
当前低负载测试中的 6% 回退，但最终结论仍需以重复测试和等工作量比较为准。

### 4.5 下一步：隔离空闲与请求交叠

第一组保持 0.5 RPS 的空闲间隔，同时强制最大并发为 1。

旧版本：

```bash
RUN_ACCURACY=0 \
ONLINE_PERF_REQUEST_RATE=0.5 \
ONLINE_PERF_MAX_CONCURRENCY=1 \
ONLINE_PERF_NUM_PROMPTS=100 \
ONLINE_PERF_NUM_WARMUPS=10 \
PORT=7001 \
./do_test.sh 0.14.1-r05-c1 load
```

新版本：

```bash
RUN_ACCURACY=0 \
ONLINE_PERF_REQUEST_RATE=0.5 \
ONLINE_PERF_MAX_CONCURRENCY=1 \
ONLINE_PERF_NUM_PROMPTS=100 \
ONLINE_PERF_NUM_WARMUPS=10 \
PORT=7002 \
./do_test.sh 0.22.1rc1-r05-c1 load
```

判定方法：

- 如果新版本仍回退，主要怀疑请求间空闲、GPU 时钟/功耗状态、缓存冷却或周期性开销；
- 如果新版本恢复为更快，主要怀疑请求交叠、batch 和调度路径。

第二组保持请求连续到达，逐级增加最大并发为 2 和 4：

```bash
RUN_ACCURACY=0 \
ONLINE_PERF_REQUEST_RATE=inf \
ONLINE_PERF_MAX_CONCURRENCY=2 \
ONLINE_PERF_NUM_PROMPTS=100 \
ONLINE_PERF_NUM_WARMUPS=10 \
PORT=<7001-or-7002> \
./do_test.sh <version>-inf-c2 load

RUN_ACCURACY=0 \
ONLINE_PERF_REQUEST_RATE=inf \
ONLINE_PERF_MAX_CONCURRENCY=4 \
ONLINE_PERF_NUM_PROMPTS=100 \
ONLINE_PERF_NUM_WARMUPS=10 \
PORT=<7001-or-7002> \
./do_test.sh <version>-inf-c4 load
```

如果并发 1 时新版本更快、并发 2 或 4 开始变慢，应优先检查 concurrent Beam 调度、
mixed prefill/decode、batch shape、Engine IPC、async scheduling 和 CUDA Graph shape。
每个配置至少运行 3 次，old/new 交替执行，并使用独立结果 tag。

## 5. 第三阶段：低成本单变量 A/B

### 5.1 动态 shape 的 32 位索引

这是目前优先级最高的已确认默认值差异：

- vLLM 0.14.1：`assume_32_bit_indexing=True`；
- vLLM 0.22.1rc1：`assume_32_bit_indexing=False`。

服务启动日志也显示了这个差异。先只在 0.22.1rc1 上恢复旧默认值：

```bash
vllm-gr serve "$ONEREC_MODEL_PATH" \
  <其余基线参数> \
  --compilation-config \
  '{"dynamic_shapes_config":{"assume_32_bit_indexing":true}}'
```

必须从启动日志确认最终配置为 `assume_32_bit_indexing: True`。如果当前 PyTorch 明确拒绝
该配置，则记录错误并跳过，不应通过修改 vLLM 的版本保护强行开启。

### 5.2 FlashInfer sampler

另一个默认值差异是：0.14.1 默认不启用 FlashInfer sampler，0.22.1rc1 默认启用。
新版本日志中出现了 `Using FlashInfer for top-p & top-k sampling`。用以下方式做单变量测试：

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm-gr serve "$ONEREC_MODEL_PATH" <其余基线参数>
```

Beam 请求当前使用 greedy 路径时可能绕过 top-p/top-k sampler，所以该项预期优先级低于
32 位索引，但测试成本很低。若指标无变化，即可排除。

### 5.3 Eager 模式

分别在旧、新服务增加 `--enforce-eager`。如果 eager 下版本差距明显缩小，问题更可能来自
`torch.compile`、CUDA Graph 或动态 shape；如果差距保持，则优先检查普通执行和 Beam
控制路径。Eager 绝对性能较慢是正常的，此项只比较 old/new 相对差值。

### 5.4 推荐测试矩阵

| 顺序 | 版本 | 单一变化 | 目的 |
| --- | --- | --- | --- |
| A0 | 0.14.1 | 无 | 重复基线，估计波动 |
| A1 | 0.22.1rc1 | 无 | 重复当前回退 |
| A2 | 0.22.1rc1 | 32 位索引开启 | 验证动态索引默认值差异 |
| A3 | 0.22.1rc1 | FlashInfer sampler 关闭 | 排除 sampler 默认值差异 |
| A4 | 0.14.1 | `--enforce-eager` | eager 旧基线 |
| A5 | 0.22.1rc1 | `--enforce-eager` | 隔离 compile/CUDA Graph |
| A6 | 0.22.1rc1 | A2、A3 中确认有效的组合 | 验证收益可叠加性 |

每个配置输出独立 tag，例如 `0.22.1rc1-index32-run1`，禁止复用结果目录覆盖原始数据。

## 6. 第四阶段：拆分 vllm-gr Beam 路径

如果低成本 A/B 不能解释 Decode 回退，应在服务端增加低开销计时指标。不要在正式性能测试中
逐请求打印 INFO 日志，日志锁和 I/O 会污染结果。推荐通过 Prometheus histogram/counter 汇总。

### 6.1 Prefill 拆分

至少拆分：

1. API/chat request 转换；
2. `prepare_request` 和 input processor；
3. Engine 请求注册与 IPC；
4. 等待首个 Engine output；
5. 首轮 logprobs 展开和 Beam 对象构造。

0.22.1rc1 的 `prepare_request` 已变为异步 API，而 0.14.1 使用同步路径。如果只有 Prefill
回退，需要重点检查这部分新增的 await、input processor 和 prompt 表示转换。

### 6.2 Decode 拆分

至少拆分：

1. 每轮 prompt/EngineInput 构造；
2. catalog 约束和候选 token 处理；
3. Beam Engine 请求注册与 IPC；
4. 等待模型执行；
5. logprobs 提取、top-k 和排序；
6. BeamSequence 重建和结束条件；
7. 最终 detokenize/响应序列化。

当前代码已经计算 `generation_time` 和 `beam_search_overhead`，但对外汇总主要展示 Prefill、
Decode、Sort 和 Total Beam。应先把这两个内部值暴露为可对比指标，再决定优化 CPU Beam
逻辑还是底层模型执行。

## 7. 第五阶段：Profiler

Profiler 只在小样本串行请求上开启，避免 profile 本身改变正常性能结果。

### 7.1 PyTorch profiler

服务端使用独立输出目录启动：

```bash
vllm-gr serve "$ONEREC_MODEL_PATH" \
  <其余基线参数> \
  --profiler-config \
  '{"profiler":"torch","torch_profiler_dir":"/tmp/onerec-profile"}'
```

客户端以 5 到 10 个串行请求运行，并在 benchmark 支持时增加 `--profile`。分别采集旧、新
版本，比较 CUDA kernel 总时间、CPU self time、同步点、memcpy 和 kernel launch 次数。

### 7.2 Nsight Systems

如果 PyTorch profiler 显示 GPU 侧回退但不能定位具体同步关系，再用 Nsight Systems 比较：

- Prefill attention 和 GEMM；
- Decode 每轮 kernel 数量及空洞；
- CPU 到 GPU launch 间隔；
- `cudaDeviceSynchronize` 或 stream wait；
- top-k/logprobs 相关 kernel。

不要一开始就采集完整 100 请求 Nsight trace，以免文件过大且难以分析。

## 8. 容量与压力测试

当前 0.5 RPS 测试主要测单请求延迟，不能代表服务最大吞吐。修复低负载回退后，再逐级增加
request rate，例如 0.5、1、2、4、8、`inf`，并设置明确的最大并发，观察：

- request throughput 和失败率；
- waiting/running requests；
- TTFT P50/P95/P99；
- Prefill/Decode/Beam 总耗时；
- GPU 利用率和显存；
- 达到 SLO 前的最大稳定 RPS。

两版本必须使用相同流量阶梯和并发限制。若新版本只在高并发回退，再进一步检查 scheduler、
chunked prefill、prefix cache、KV cache 容量和 async scheduling；这些变量不应在第一轮
单请求归因前一起修改。

## 9. 判定树

```text
100 请求、三次重复后仍回退？
├── 否：记录波动区间，在 A100 old/new 上复验
└── 是
    ├── 串行模式不回退
    │   ├── 0.5 RPS + concurrency=1 仍回退
    │   │   └── 定位空闲唤醒、GPU 时钟/功耗状态、缓存和周期性开销
    │   └── 0.5 RPS + concurrency=1 不回退
    │       └── 逐级增加并发，定位 async scheduling、批处理和请求交叠
    └── 串行模式仍回退
        ├── index32 A/B 恢复
        │   └── 固化兼容配置，并验证精度与 A100
        ├── eager 下差距消失
        │   └── 定位 torch.compile、CUDA Graph 和动态 shape
        └── 上述均不能恢复
            ├── Prefill 回退：拆 input processor、IPC、首轮模型执行
            └── Decode 回退：拆 Engine wait、logprobs/top-k、Beam CPU 开销
```

## 10. 完成标准

性能问题只有同时满足以下条件才可关闭：

- 旧、新版本在 L20 上使用同一输入完成至少 3 次重复测试；
- 新版本 TTFT、Beam 总耗时和离线吞吐均无统计显著回退；
- 100 请求精度指标与旧版本一致；
- ModelRunner V1 的实际生效状态可从日志确认；
- 候选修复在 A100 上完成 old/new 同卡复验；
- A100 CI 保存环境 manifest、完整日志和原始 JSON artifact；
- 对 ModelRunner V2 保持禁用，直到它独立通过精度与性能验证。
