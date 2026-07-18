# vLLM 0.14.1 至 0.22.1rc1 升级验证指南

## 1. 目的与适用范围

本文用于验证 vllm-gr 从配套 vLLM 0.14.1 升级到 vLLM
0.22.1rc1 后的功能、精度和性能。

当前 L20 验证的阶段性结果和后续性能定位计划见：

- [L20 在线性能对比结果](l20_online_performance_results.md)；
- [性能回退定位与待测试计划](performance_regression_investigation.md)。

当前本地可使用 L20，目标 CI 使用 A100。因此，本次验证采用以下原则：

- L20 用于同一硬件上的旧版本和新版本 A/B 对比，判断软件升级是否引入回归。
- A100 CI 用于确认新版本在目标硬件上的正确性和性能。
- 不使用理论算力或显存带宽将 L20 结果换算成 A100 结果。
- L20 和 A100 都只能验证 CUDA 路径，不能替代 vLLM Ascend 的 NPU 验证。

本文中的“旧版本”和“新版本”默认指：

| 组件 | 旧版本 | 新版本 |
| --- | --- | --- |
| vllm-gr | `aba0a5f`，即升级提交的父提交 | `ade38d6` |
| vLLM | `v0.14.1` / `d7de043d55d1dd629554467e23874097e1c48993e` | `v0.22.1rc1` / `b284862ea9b8413d21e4b4dd1eb46f8ba3e40627` |
| vLLM Ascend | `v0.14.0rc1` / `52d4acfa51fb868823d1070b81cbd2d97e9e4696` | `v0.22.1rc1` / `5f6faa0cb8830f667266f3b8121cd1383606f2a1` |

以上 commit 是编写本文时本地目录的状态。目录名仅用于组织文件，不能作为版本证据；
每次测试都必须重新记录源码 commit 和实际安装的 Python distribution 版本。

## 2. 验证矩阵

建议至少完成以下四组测试：

| 编号 | 软件栈 | 硬件 | 用途 | 是否必需 |
| --- | --- | --- | --- | --- |
| L20-old | vllm-gr `aba0a5f` + vLLM 0.14.1 | L20 | 建立升级前基线 | 是 |
| L20-new | vllm-gr `ade38d6` + vLLM 0.22.1rc1 | L20 | 检测通用软件回归 | 是 |
| A100-new | vllm-gr `ade38d6` + vLLM 0.22.1rc1 | A100 | 验证目标硬件 | 是 |
| A100-old | vllm-gr `aba0a5f` + vLLM 0.14.1 | A100 | 得到目标硬件上的直接 A/B 结果 | 强烈推荐 |

如果历史部署实际使用过 vLLM 0.14.0rc1，可以增加一个 L20-old-rc
补充测试，但不应替代仓库旧 CI 使用的 vLLM 0.14.1 基线。

最终应优先比较同一硬件上的相对变化：

```text
L20 软件变化  = L20-new / L20-old
A100 软件变化 = A100-new / A100-old
```

只有 `A100-new / A100-old` 才是目标硬件上的最终性能结论。L20 的结果用于提前发现
大部分通用回归，并辅助判断 A100 上的问题是否属于架构或 kernel 特有问题。

## 3. 准备隔离环境

### 3.1 确认独立代码仓

不要在同一个源码目录中反复切换 commit，也不要让两套 editable install 指向同一个
vllm-gr 目录。当前使用的目录关系如下：

```text
/root/PROJECT/Update/
├── vllm_0.14.1/
├── vllm_0.22.1rc1/
├── vllm-gr_0.14.1/
└── vllm-gr_0.22.1rc1/
```

开始测试前确认两个仓库分别指向预期提交：

```bash
git -C /root/PROJECT/Update/vllm-gr_0.14.1 rev-parse HEAD
git -C /root/PROJECT/Update/vllm-gr_0.22.1rc1 rev-parse HEAD
```

预期结果分别为 `aba0a5f` 和 `ade38d6`。目录名本身不能证明代码版本正确。

### 3.2 创建两套 Python 环境

旧版本：

```bash
cd /root/PROJECT/Update/vllm-gr_0.14.1
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install vllm==0.14.1 --torch-backend=auto
VLLM_GR_TARGET_DEVICE=cuda uv pip install -e ".[dev]"
```

如果需要验证本地 vLLM 源码而不是 wheel，可以将 vLLM 安装步骤替换为：

```bash
uv pip install -e /root/PROJECT/Update/vllm_0.14.1 --torch-backend=auto
```

新版本：

```bash
cd /root/PROJECT/Update/vllm-gr_0.22.1rc1
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install vllm==0.22.1rc1 \
  --extra-index-url https://wheels.vllm.ai/b284862ea9b8413d21e4b4dd1eb46f8ba3e40627 \
  --index-strategy first-index \
  --torch-backend=auto
VLLM_GR_TARGET_DEVICE=cuda uv pip install -e ".[dev]"
```

使用本地新 vLLM 源码时，可以将 vLLM 安装步骤替换为：

```bash
uv pip install -e /root/PROJECT/Update/vllm_0.22.1rc1 --torch-backend=auto
```

基线和新版本必须采用相同的安装类型，即都使用 wheel 或都使用本地源码构建。
不要把旧版 wheel 和带有本地修改的新版源码直接进行性能比较。

### 3.3 验证没有环境串用

在两套环境中分别执行并保存输出：

```bash
python -c 'import importlib.metadata as m; print("vllm", m.version("vllm")); print("vllm-gr", m.version("vllm-gr"))'
python -c 'import vllm, vllm_gr; print(vllm.__file__); print(vllm_gr.__file__)'
python -m torch.utils.collect_env
python -m pip freeze
```

源码环境还应记录：

```bash
git -C /root/PROJECT/Update/vllm_0.14.1 rev-parse HEAD
git -C /root/PROJECT/Update/vllm_0.22.1rc1 rev-parse HEAD
git rev-parse HEAD
```

## 4. 固定测试输入和运行条件

### 4.1 固定模型、数据集和 catalog

当前测试机使用以下 OneRec 路径：

```bash
export ONEREC_MODEL_PATH=/home/z00469465/OneRec/OneRec-1.7B
export ONEREC_DATASET_PATH=/home/z00469465/OneRec/OpenOneRec-RecIF/benchmark_data
export ONEREC_CATALOG_PATH=/home/z00469465/OneRec/generated_data/video_catalog.json
```

旧版本和新版本必须使用完全相同的：

- 模型权重和 tokenizer 文件；
- OpenOneRec 数据集 revision；
- `video_catalog.json`；
- chat template；
- 样本子集及其顺序；
- beam width、生成长度和 temperature。

建议为模型、数据集和 catalog 保存 SHA-256 manifest。至少应记录：

```bash
sha256sum "$ONEREC_CATALOG_PATH"
sha256sum "$ONEREC_DATASET_PATH/video/video_test.parquet"
```

OneRec loader 使用 `random_state=42` 生成抽样数据，并会将样本缓存为 parquet。
第一次生成样本文件后，应让 old/new 和 CI 复用同一份缓存文件，而不是在不同环境中
分别生成。

### 4.2 固定 GPU 运行状态

每次运行前记录：

```bash
nvidia-smi
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,power.limit \
  --format=csv
nvidia-smi topo -m
```

还应确保：

- GPU 没有被其他进程使用；
- 未启用 MIG，或 old/new 使用完全相同的 MIG 实例；
- power limit 和时钟策略一致；
- GPU 温度已进入相对稳定状态且没有 thermal throttling；
- 在线 benchmark 的 client、server、CPU 绑核和网络位置一致；
- 测试期间不同时运行下载、编译或其他高负载任务。

### 4.3 同时保留两种配置口径

建议维护两个性能 profile：

1. 默认配置对比：old/new 分别采用各版本默认配置，反映用户升级后的实际体验。
2. 公共配置对比：显式固定 dtype、KV cache dtype、attention backend、
   `max-num-seqs`、`max-num-batched-tokens`、`gpu-memory-utilization` 等参数，
   用于隔离 vLLM 默认值变化。

如果某个公共配置在 L20 上无法运行，应定义一套 L20 和 A100 都能运行的 common
profile，再为 A100 增加一个 target-capacity profile。不能通过降低 L20 的参数后直接
与 A100 的高容量配置比较。

## 5. 正确性和精度验证

### 5.1 单元和契约测试

先在 old/new 环境中运行相同的非 GPU 测试集，排除明显的 API 和调度回归：

```bash
python -m pytest -q \
  tests/test_attention_backend_custom.py \
  tests/test_beam_request_unit.py \
  tests/test_dp_beam_request_unit.py \
  tests/test_offline_beam_search.py \
  tests/test_serving_beam_search.py
```

新版本还应额外运行升级后新增的版本契约测试：

```bash
python -m pytest -q tests/test_runtime_versions.py
```

旧提交不存在的新增测试不应强行回移到旧分支。对于仅存在于新版本的测试，只要求新版本
通过；对于 old/new 都存在的测试，应分别在对应 commit 上运行原始测试文件。

### 5.2 Beam attention 数值正确性

仓库提供了 CUSTOM beam attention 与标准 FlashAttention reference 的逐 tensor 对比：

```bash
python -m pytest -q tests/kernels/attention/test_cascade_beam_attn_gpu.py
```

要求：

- 当前 GPU 支持的全部 case 必须通过；
- 不支持的 FlashAttention 版本可以 skip，但必须保存 skip 原因；
- old/new 的通过和 skip 集合应分别记录；
- 如果 old/new 的 kernel 支持集合发生变化，需要单独解释，不能仅汇总为“测试通过”。

### 5.3 复现现有 CI 精度用例

准备统一的环境变量：

```bash
export ONEREC_MODEL_PATH=/home/z00469465/OneRec/OneRec-1.7B
export ONEREC_DATASET_PATH=/home/z00469465/OneRec/OpenOneRec-RecIF/benchmark_data
export ONEREC_CATALOG_PATH=/home/z00469465/OneRec/generated_data/video_catalog.json
export RESULT_ROOT="${RESULT_ROOT:-$PWD/results}"
mkdir -p "$RESULT_ROOT/accuracy"
```

在 old/new 环境中执行相同命令：

```bash
python -m benchmarks.open_one_rec.one_rec_acc_test \
  --gr \
  --model "$ONEREC_MODEL_PATH" \
  --dataset-path "$ONEREC_DATASET_PATH" \
  --dataset-name onerec \
  --num-prompts 100 \
  --max-concurrency 10 \
  --use-beam-search \
  --task-type video \
  --catalog-path "$ONEREC_CATALOG_PATH" \
  --backend CUSTOM \
  --n 128 \
  --result-dir "$RESULT_ROOT/accuracy"
```

主要指标为：

- `pass@1`；
- `pass@32`；
- `recall@32`；
- `pid_pass@1`、`pid_pass@32`、`pid_recall@32`；
- `elapsed_time` 和 `time_per_sample`。

当前 CI 的最低门槛是 `pass@1 >= 0.09`、`pass@32 >= 0.20`、
`recall@32 >= 0.04`。这些门槛适合 smoke test，不足以判断升级是否引入轻微精度回归。

这些门槛可以通过 `PASS_1_THRESHOLD`、`PASS_32_THRESHOLD` 和
`RECALL_32_THRESHOLD` 环境变量覆盖。修改样本数量时必须重新评估精度门槛，不能在
样本集不同的情况下直接沿用原值。

### 5.4 使用 paired comparison 判断精度回归

100 条固定样本中，一个样本变化就相当于约一个百分点。建议保存每个样本的以下信息：

- sample ID；
- top-1 生成 token IDs 和文本；
- top-32 生成集合；
- 每条 beam 的 cumulative logprob；
- per-sample pass 和 recall；
- 请求是否成功以及错误信息。

然后比较：

- top-1 exact match 比例；
- top-32 集合重合率；
- old 正确但 new 错误的样本；
- new 正确但 old 错误的样本；
- 指标差值的 paired bootstrap 置信区间。

如果当前 benchmark 只保存汇总 metrics，应增加一个可选的逐样本 JSON 输出模式，且保证
该功能本身不改变推理参数或执行顺序。

推荐将正式精度样本量提高到至少 1000；100 条仍保留为快速 CI smoke test。

### 5.5 可重复性检查

vLLM 两个版本的默认 engine seed 都是 0，但在线请求调度仍可能改变执行顺序。
建议：

- offline 精度作为主要精度结论；
- primary profile 保持现有任务默认 temperature，不改变 CI 语义；
- old/new 各重复至少 3 次，确认相同环境内的波动；
- 可以增加 `temperature=0` 的 deterministic shadow profile，但不能用它替代
  primary profile；
- 在线精度只作为 API 和调度路径的补充验证。

## 6. 性能验证

### 6.1 Kernel 微基准

在 old/new 上分别执行：

```bash
python tests/kernels/attention/benchmark_cascade_beam_attn_gpu.py \
  --iter-num 200 \
  --warmup-iters 20 \
  --beam-widths 16 32 128
```

该测试覆盖不同 head size、block size、dtype、soft cap 和 FlashAttention 版本。
每套环境至少运行 5 次，按具体 case 比较中位数，不要只比较所有 case 的总体平均值。

建议保留的字段：

- GPU 型号和 compute capability；
- case 完整参数；
- supported 或 skip 状态；
- 每次运行的平均延迟；
- 5 次结果的 median、最小值、最大值和变异系数。

### 6.2 Offline 端到端吞吐

```bash
mkdir -p "$RESULT_ROOT/offline"

python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --dataset-name onerec \
  --model "$ONEREC_MODEL_PATH" \
  --dataset-path "$ONEREC_DATASET_PATH" \
  --task-types video \
  --num-prompts 500 \
  --max-logprobs 512 \
  --use-beam-search \
  --n 128 \
  --output-json "$RESULT_ROOT/offline/throughput.json"
```

至少记录：

- `requests_per_second`；
- `tokens_per_second`；
- `elapsed_time`；
- 成功和失败的请求数；
- 峰值 GPU 显存；
- GPU utilization、power 和 clock。

当前自定义 offline beam benchmark 在第一次 `beam_search` 调用前开始计时，可能包含
lazy kernel、JIT 或 CUDA Graph 开销。因此它更接近“首轮业务执行”，不能单独代表稳态
吞吐。建议后续将测试结果拆成：

1. cold first run；
2. 固定数量 warmup 后的 steady-state run。

两种结果都应保留，不能在 old 上统计 cold run、在 new 上统计 warm run。

### 6.3 Online 基准

两套容器使用 `--network host`，因此端口在宿主机网络中共享。以下命令约定旧版本使用
7001、新版本使用 7002，并且两套服务在同一张 L20 上串行启动。开始前根据容器内的实际
挂载位置调整 `VLLM_GR_ROOT`；其下应分别存在 `vllm-gr_0.14.1` 和
`vllm-gr_0.22.1rc1`。

#### 6.3.1 启动 vLLM 0.14.1 服务

```bash
export VLLM_GR_ROOT=/home/z00469465/PROJECT/Origin_GR
cd "$VLLM_GR_ROOT/vllm-gr_0.14.1"

export ONEREC_MODEL_PATH=/home/z00469465/OneRec/OneRec-1.7B
export ONEREC_CATALOG_PATH=/home/z00469465/OneRec/generated_data/video_catalog.json
export RESULT_ROOT=/home/z00469465/OneRec/results/online_compare
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=0
export PORT=7001

mkdir -p "$RESULT_ROOT"

python -c 'import importlib.metadata as m; print("vllm:", m.version("vllm")); print("vllm-gr:", m.version("vllm-gr"))'
python -c 'import vllm_gr; print("vllm_gr source:", vllm_gr.__file__)'

vllm-gr serve \
  --port "$PORT" \
  --seed 123 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-logprobs 512 \
  --max-num-seqs 2048 \
  --max-num-batched-tokens 16384 \
  --scheduling-policy priority \
  --catalog-path "$ONEREC_CATALOG_PATH" \
  --attention-backend CUSTOM \
  "$ONEREC_MODEL_PATH" \
  > "$RESULT_ROOT/0.14.1_server.log" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$RESULT_ROOT/0.14.1_server.pid"
echo "0.14.1 server PID: $SERVER_PID"
```

版本检查必须输出 vLLM `0.14.1`，且 `vllm_gr source` 必须指向旧版本仓。等待日志中出现
`Starting vLLM API server` 后检查服务：

```bash
tail -f "$RESULT_ROOT/0.14.1_server.log"
curl -s http://127.0.0.1:7001/v1/models | python -m json.tool
```

`--max-logprobs 512` 不能省略。Beam width 为 128 时，如果服务仍使用默认上限 20，会报
`Requested sample logprobs of 128, which is greater than max allowed: 20`。

#### 6.3.2 启动 vLLM 0.22.1rc1 服务

在 0.22.1rc1 容器内执行：

```bash
export VLLM_GR_ROOT=/home/z00469465/PROJECT/Origin_GR
cd "$VLLM_GR_ROOT/vllm-gr_0.22.1rc1"

export ONEREC_MODEL_PATH=/home/z00469465/OneRec/OneRec-1.7B
export ONEREC_CATALOG_PATH=/home/z00469465/OneRec/generated_data/video_catalog.json
export RESULT_ROOT=/home/z00469465/OneRec/results/online_compare
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=0
export PORT=7002

mkdir -p "$RESULT_ROOT"

python -c 'import importlib.metadata as m; print("vllm:", m.version("vllm")); print("vllm-gr:", m.version("vllm-gr"))'
python -c 'import vllm_gr; print("vllm_gr source:", vllm_gr.__file__)'

vllm-gr serve \
  --port "$PORT" \
  --seed 123 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-logprobs 512 \
  --max-num-seqs 2048 \
  --max-num-batched-tokens 16384 \
  --scheduling-policy priority \
  --catalog-path "$ONEREC_CATALOG_PATH" \
  --attention-backend CUSTOM \
  "$ONEREC_MODEL_PATH" \
  > "$RESULT_ROOT/0.22.1rc1_server.log" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$RESULT_ROOT/0.22.1rc1_server.pid"
echo "0.22.1rc1 server PID: $SERVER_PID"
```

版本检查必须输出 vLLM `0.22.1rc1`，且 `vllm_gr source` 必须指向新版本仓。服务检查：

```bash
tail -f "$RESULT_ROOT/0.22.1rc1_server.log"
curl -s http://127.0.0.1:7002/v1/models | python -m json.tool
```

两组日志都必须确认实际初始化的是 `V1 LLM engine`。本轮不允许启用 ModelRunner V2，
因为它尚未通过 OneRec 精度验证。

#### 6.3.3 停止服务并确认无残留

切换版本前必须先停止当前服务。优先使用启动时记录的 PID：

```bash
# 示例：停止 0.22.1rc1
SERVER_PID=$(cat "$RESULT_ROOT/0.22.1rc1_server.pid")
kill -TERM "$SERVER_PID"
sleep 3

ss -ltnp | grep ':7002' || echo "7002 端口已释放"
nvidia-smi
```

如果 PID 文件失效，先按端口定位，再终止准确的监听进程：

```bash
ss -ltnp | grep ':7002'
# 确认输出中的 PID 后执行：kill -TERM <PID>
```

不要使用宽泛的 `pkill -f vllm`，否则可能误杀另一个容器中的服务。

#### 6.3.4 运行统一客户端

两组请求都从 0.22.1rc1 客户端容器和代码仓发起，避免 benchmark 实现差异污染结果。
先执行至少 20 条 warmup 请求。使用仓库根目录的 `do_test.sh`：

```bash
# 0.14.1 服务监听 7001 时
PORT=7001 ./do_test.sh 0.14.1

# 停止旧服务、启动新服务后
PORT=7002 ./do_test.sh 0.22.1rc1
```

不使用 `do_test.sh` 时，正式测试命令示例：

```bash
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --model "$ONEREC_MODEL_PATH" \
  --dataset-name onerec \
  --dataset-path "$ONEREC_DATASET_PATH" \
  --task-type video \
  --num-prompts 500 \
  --use-beam-search \
  --port 7002 \
  --request-rate 0.5 \
  --n 128
```

建议测试以下负载曲线之一：

| Profile | `request-rate` | 用途 |
| --- | --- | --- |
| latency-low | `0.25` | 低负载延迟 |
| ci-compatible | `0.5` | 与现有 CI 参数对齐 |
| medium | `1` | 中等负载 |
| high | `2` | 高负载 |
| saturation | `inf` | 最大吞吐 |

也可以固定请求速率，测试 `max-concurrency` 为 1、4、8、16 的曲线。

每个 profile 至少运行 5 次，并记录：

- request throughput；
- output token throughput；
- total token throughput；
- TTFT P50、P95、P99；
- TPOT P50、P95、P99；
- ITL P50、P95、P99；
- end-to-end latency；
- failed requests；
- 峰值显存、GPU utilization、power 和 clock。

当前 CI 只使用 5 条 warmup 和 10 条正式请求。它适合快速 smoke test，但 10 个请求的
P99 不适合作为稳定的性能结论。

CI 脚本支持通过环境变量调整负载和门槛：

| 环境变量 | 默认值 | 说明 |
| --- | ---: | --- |
| `KPI_GATING_MODE` | `enforce` | `enforce` 失败退出；`report-only` 只记录告警 |
| `ONLINE_WARMUP_NUM_PROMPTS` | `5` | 在线 warmup 请求数 |
| `ONLINE_PERF_NUM_PROMPTS` | `10` | 在线性能请求数 |
| `ONLINE_PERF_REQUEST_RATE` | `0.5` | 在线性能请求速率 |
| `P99_TTFT_THRESHOLD_MS` | GPU `300` | P99 TTFT 上限 |
| `P99_TPOT_THRESHOLD_MS` | `0.02` | P99 TPOT 上限 |
| `OFFLINE_ACCURACY_ELAPSED_THRESHOLD_SECONDS` | GPU `20` | Offline 精度推理耗时上限 |
| `ONLINE_ACCURACY_ELAPSED_THRESHOLD_SECONDS` | GPU `15` | Online 精度推理耗时上限 |

建立 L20 基线时建议使用 report-only 模式，防止 A100 的旧绝对门槛干扰数据收集：

```bash
BENCHMARK_MODEL="$ONEREC_MODEL_PATH" \
  DATASET_PATH="$ONEREC_DATASET_PATH" \
  CATALOG_PATH="$ONEREC_CATALOG_PATH" \
  KPI_GATING_MODE=report-only \
  ONLINE_WARMUP_NUM_PROMPTS=20 \
  ONLINE_PERF_NUM_PROMPTS=200 \
  ./tests/system/run_benchmark_and_verify.sh
```

CI 的 A100 门槛应在获得 A100-old/new 数据后更新。不能把 L20 的绝对耗时直接写入
A100 CI；如果暂时无法在每次 CI 中运行 A100-old，应保存稳定的 A100-old 历史基线，
再按基线波动和允许的回归比例计算固定门槛。

### 6.4 重复次数和统计方法

推荐流程：

1. 预热 GPU 和模型；
2. 每个 profile 连续运行 5 次；
3. old/new 交替执行，避免温度或机器负载随时间单向漂移；
4. 使用 median 作为主结果；
5. 同时报告最小值、最大值、标准差和变异系数；
6. 对 P99 等尾延迟指标使用更大的请求数量。

性能差值统一定义为：

```text
吞吐变化 = (new - old) / old
延迟变化 = (new - old) / old
```

吞吐变化越大越好，延迟变化越小越好。

## 7. 初始回归门槛

在尚未积累历史数据时，可以使用以下初始门槛：

| 类别 | 初始判定建议 |
| --- | --- |
| Kernel 正确性 | 当前硬件支持的 case 全部通过 |
| 请求正确性 | 无 crash、hang、NaN、OOM 和失败请求 |
| 精度 | paired comparison 无无法解释的显著下降 |
| 稳态吞吐 | 同卡 new 相对 old 下降超过 5% 时告警 |
| 稳态 TPOT | 同卡 new 相对 old 上升超过 5% 时告警 |
| TTFT/ITL P99 | 同卡 new 相对 old 上升超过 10% 时告警 |
| 峰值显存 | 增长超过 5% 时告警，超过目标卡容量时失败 |

这些数值是调查触发线，不应立即等价为发布失败。积累 10 至 20 次稳定历史数据后，
应使用真实波动重新计算门槛，例如：

```text
性能告警阈值 = max(5%, 3 * baseline CV)
```

硬性失败条件应包括：

- 数值正确性测试失败；
- 精度出现可复现且无法解释的下降；
- failed requests 不为 0；
- 出现 crash、hang、NaN 或非预期 OOM；
- A100 target profile 超过约定的性能或显存预算。

## 8. A100 CI 注意事项

CI 结果必须记录 A100 的具体形态，包括：

- A100 40GB 或 80GB；
- PCIe 或 SXM；
- 是否启用 MIG；
- driver、CUDA 和 PyTorch 版本；
- power limit；
- 单卡或多卡及其 topology；
- CI runner 的 CPU 型号、NUMA 和并发任务情况。

L20 和 A100 可能选择不同的编译 kernel 或 FlashAttention 路径，所以即使 L20-new
相对 L20-old 没有回归，A100-new 仍必须完整运行 correctness、accuracy 和 performance
测试。反之，L20 上的性能变化也不能按理论带宽比例外推到 A100。

如果 A100-new 出现问题，建议按以下顺序定位：

1. 查看 L20-old/new 是否出现相同方向的变化；
2. 在 A100 上补跑 old 版本，得到直接 A/B；
3. 比较 old/new 实际选择的 attention backend、FA 版本和 CUDA Graph 配置；
4. 比较 effective engine config，而不是只比较命令行参数；
5. 使用 kernel 微基准判断问题位于 kernel 还是 scheduler/API 层；
6. 检查 CPU、NUMA、MIG、功耗和热降频等机器因素。

## 9. vLLM Ascend 验证

L20 和 A100 都不能覆盖 vLLM Ascend。本次 vLLM 升级涉及 NPU model runner、position
buffer、调度接口和 NPU patch 的变化，因此必须在真实 Ascend 硬件上另行完成：

- NPU beam attention correctness；
- OneRec offline accuracy；
- OneRec online accuracy；
- beam width 128 的 offline 和 online 性能；
- 多轮请求、并发请求和长时间稳定性；
- NPU 显存、内存碎片和 CPU-NPU 同步检查；
- old/new vLLM Ascend 同卡 A/B。

GPU 验证通过只说明 CUDA 路径满足要求，不能作为 vLLM Ascend 升级完成的依据。

## 10. 结果目录和报告格式

建议每次运行使用独立目录：

```text
results/
├── l20-old/
│   ├── manifest/
│   ├── correctness/
│   ├── accuracy/
│   ├── kernel/
│   ├── offline/
│   └── online/
├── l20-new/
├── a100-old/
└── a100-new/
```

每个目录的 `manifest/` 至少包含：

- `git_revisions.txt`；
- `installed_packages.txt`；
- `torch_collect_env.txt`；
- `nvidia_smi.txt`；
- `nvidia_smi_query.csv`；
- `model_dataset_hashes.txt`；
- 完整命令行和环境变量；
- server 日志。

最终报告应至少包含以下表格：

| Profile | L20-old | L20-new | L20 delta | A100-old | A100-new | A100 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pass@1 | | | | | | |
| pass@32 | | | | | | |
| recall@32 | | | | | | |
| offline requests/s | | | | | | |
| offline tokens/s | | | | | | |
| online request throughput | | | | | | |
| TTFT P99 | | | | | | |
| TPOT P99 | | | | | | |
| peak GPU memory | | | | | | |

## 11. 推荐执行顺序

1. 建立并冻结模型、数据集、catalog 和样本 manifest。
2. 确认隔离的 old/new 代码仓和 Python 环境。
3. 运行环境版本检查和非 GPU 单元测试。
4. 在 L20-old 上建立 correctness、accuracy 和 performance 基线。
5. 在 L20-new 上使用完全相同的输入和参数重复测试。
6. 完成 paired accuracy comparison 和同卡性能差值分析。
7. 将同一套 manifest、命令和 profile 迁移到 A100 CI。
8. 尽可能在 A100 上补跑 old 版本，形成目标硬件的直接 A/B。
9. 在 Ascend 硬件上独立完成 vLLM Ascend old/new 验证。
10. 根据历史波动调整 CI 回归门槛，并将原始 JSON 和环境 manifest 作为 CI artifact 保存。
