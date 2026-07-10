# vllm-gr GPU 部署与验证指南

> 本文档面向需要在 GPU 环境部署 vllm-gr、并验证单元测试（UT）与 OneRec Benchmark 的工程师。
>
> 默认 Python 版本：**3.12**

---

## 一、环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA GPU，CUDA 11.8+ | A100 / A800 / H100 / H800 |
| Driver | ≥ 520 | 最新稳定版 |
| Python | **3.12** | 3.12 |
| pip | 23+ | 25+ |
| 内存 | 32GB | 64GB+ |
| 显存 | 24GB | ≥ 40GB（beam=128 场景） |
| 磁盘 | 50GB 可用空间 | SSD |

---

## 二、安装步骤

### 2.1 拉取代码

```bash
git clone <vllm-gr-repo-url> vllm-gr
cd vllm-gr
```

### 2.2 创建虚拟环境

确保系统已安装 Python 3.12：

```bash
python3.12 --version  # 应输出 Python 3.12.x
```

创建并激活虚拟环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2.3 安装 vLLM 0.22.1（GPU 版）

```bash
pip install vllm==0.22.1 --torch-backend=auto
```

验证 GPU 可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
```

### 2.4 安装 vllm-gr

```bash
pip install -e .
```

开发模式（包含 pytest、ruff 等）：

```bash
pip install -e ".[dev]"
```

### 2.5 验证安装

```bash
python -c "import benchmarks.open_one_rec; print('✓ vllm-gr installed')"
```

---

## 三、准备模型与数据集

如果远端**无法直接访问 HuggingFace**，需要提前把模型和数据集下载到本地，再上传到远端。

### 3.1 模型：OpenOneRec/OneRec-1.7B

#### 方式 A：本地下载后上传

在有网络的机器上：

```bash
git lfs install
git clone https://huggingface.co/OpenOneRec/OneRec-1.7B /data/models/OneRec-1.7B
```

打包上传到远端：

```bash
tar czf onerec-1.7b.tar.gz -C /data/models OneRec-1.7B
# 通过 scp / 对象存储 / 内网盘传到远端 /data/models/
tar xzf onerec-1.7b.tar.gz -C /data/models/
```

#### 方式 B：ModelScope（国内）

```bash
pip install modelscope
python - <<'PY'
from modelscope import snapshot_download
snapshot_download("OpenOneRec/OneRec-1.7B", cache_dir="/data/models")
PY
```

#### 方式 C：HF-Mirror

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download OpenOneRec/OneRec-1.7B --local-dir /data/models/OneRec-1.7B
```

### 3.2 数据集：OpenOneRec-RecIF

```bash
git lfs install
git clone https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF /data/OpenOneRec-RecIF
```

目录结构应为：

```text
/data/OpenOneRec-RecIF/data/
├── video/video_test.parquet
├── ad/ad_test.parquet
├── product/product_test.parquet
├── label_cond/label_cond_test.parquet
├── interactive/interactive_test.parquet
├── label_pred/label_pred_test.parquet
├── item_understand/item_understand_test.parquet
└── rec_reason/rec_reason_test.parquet
```

### 3.3 Catalog 文件

Catalog 用于 beam search 候选 token 约束，需与模型/数据一起放到远端：

```bash
export REC_CATALOG=/data/catalog
```

### 3.4 设置环境变量

```bash
export REC_MODEL=/data/models/OneRec-1.7B        # 或 OpenOneRec/OneRec-1.7B
export REC_DATA=/data/OpenOneRec-RecIF/data
export REC_CATALOG=/data/catalog
export HF_HUB_OFFLINE=1                          # 完全离线时建议设置
```

---

## 四、单元测试（UT）验证

### 4.1 安装测试依赖

```bash
pip install -e ".[dev]"
```

### 4.2 全量运行

```bash
cd /path/to/vllm-gr
export HF_HUB_OFFLINE=1
python -m pytest tests/ -x -q
```

### 4.3 推荐分阶段运行

```bash
# 阶段 1：核心逻辑 UT（无需 GPU、无需真实模型）
python -m pytest tests/test_beam_request_unit.py \
                 tests/test_attention_backend_custom.py \
                 tests/test_onerec_patch.py -v

# 阶段 2：serving 相关 UT
python -m pytest tests/test_serving_beam_search.py \
                 tests/test_serving_catalog.py -v

# 阶段 3：offline beam search（可能需要模型，视环境而定）
python -m pytest tests/test_offline_beam_search.py -v
```

### 4.4 跳过 GPU/NPU 相关用例

```bash
python -m pytest tests/ -v -k "not gpu and not npu and not cuda"
```

### 4.5 常见 UT 问题

| 问题 | 解决 |
|------|------|
| `numpy.core._multiarray_umath failed to import` | numpy 2.x 与 tensorflow 冲突，降级 numpy：`pip install "numpy<2"` |
| 无法下载模型导致 test skip | 设置 `REC_MODEL` 指向本地模型路径 |
| GPU 相关 test 在 CPU 环境失败 | 加 `-k "not gpu and not npu and not cuda"` |

---

## 五、Benchmark 验证

### 5.1 一键系统测试（推荐）

```bash
cd /path/to/vllm-gr
source .venv/bin/activate

export REC_MODEL=/data/models/OneRec-1.7B
export REC_DATA=/data/OpenOneRec-RecIF/data
export REC_CATALOG=/data/catalog

bash tests/system/run_benchmark_and_verify.sh
```

脚本会依次执行：

1. Offline accuracy benchmark + 指标校验
2. 启动 `vllm-gr serve`
3. Online throughput benchmark + 指标校验
4. Online accuracy benchmark + 指标校验

### 5.2 GPU 默认阈值

| 指标 | 阈值 | 方向 |
|------|------|------|
| pass@1 | ≥ 0.09 | 越大越好 |
| pass@32 | ≥ 0.20 | 越大越好 |
| recall@32 | ≥ 0.04 | 越大越好 |
| offline elapsed_time | ≤ 20s | 越小越好 |
| P99 TTFT | ≤ 300 | 越小越好 |
| P99 TPOT | ≤ 0.02 | 越小越好 |
| online elapsed_time | ≤ 15s | 越小越好 |

### 5.3 分步手动验证

#### 5.3.1 Offline Accuracy（不启动服务）

```bash
python -m benchmarks.open_one_rec.one_rec_acc_test \
  --gr \
  --model "$REC_MODEL" \
  --dataset-path "$REC_DATA" \
  --dataset-name onerec \
  --task-type video \
  --num-prompts 100 \
  --max-concurrency 10 \
  --use-beam-search \
  --catalog-path "$REC_CATALOG" \
  --backend CUSTOM \
  --n 128
```

关注输出：`pass@1`、`pass@32`、`recall@32`、`elapsed_time`

#### 5.3.2 启动 Online Server

```bash
vllm-gr serve "$REC_MODEL" \
  --port 7000 \
  --seed 123 \
  --no-async-scheduling \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-logprobs 512 \
  --max_num_seqs 2048 \
  --max_num_batched_tokens 16384 \
  --scheduling-policy priority \
  --catalog-path "$REC_CATALOG" \
  --attention-backend CUSTOM
```

验证服务：

```bash
curl http://localhost:7000/v1/models
```

#### 5.3.3 Online Throughput

```bash
python -m benchmarks.open_one_rec.one_rec_main \
  bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --model "$REC_MODEL" \
  --dataset-name onerec \
  --dataset-path "$REC_DATA" \
  --task-type video \
  --num-prompts 10 \
  --use-beam-search \
  --port 7000 \
  --request-rate 0.5 \
  --n 128
```

关注输出：`Failed requests`、`P99 TTFT`、`P99 TPOT`

#### 5.3.4 Online Accuracy

```bash
python -m benchmarks.open_one_rec.one_rec_acc_test \
  --model "$REC_MODEL" \
  --dataset-path "$REC_DATA" \
  --dataset-name onerec \
  --num-prompts 100 \
  --use-beam-search \
  --backend openai-chat \
  --task-type video \
  --endpoint /v1/chat/completions \
  --port 7000 \
  --n 128
```

---

## 六、显存调优

如果运行中出现 OOM，按顺序降级：

```bash
# 1. 减小 beam width
--n 64

# 2. 减小并发/序列数
--max_num_seqs 1024
--max_num_batched_tokens 8192

# 3. 减少测试样本数
--num-prompts 50

# 4. 使用 FP16
--dtype float16

# 5. 限制 GPU 内存利用率
--gpu-memory-utilization 0.85
```

---

## 七、最小验证流程（资源紧张）

如果只有 24GB 显存，先跑最小验证：

```bash
python -m benchmarks.open_one_rec.one_rec_acc_test \
  --gr \
  --model "$REC_MODEL" \
  --dataset-path "$REC_DATA" \
  --task-type video \
  --num-prompts 50 \
  --use-beam-search \
  --backend CUSTOM \
  --n 32
```

确认不 OOM 且指标正常后，再逐步提升到 `--num-prompts 100 --n 128`。

---

## 八、常见问题排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `CUDA out of memory` | beam width / batch 太大 | 按第六节降级 |
| `Failed requests > 0` | 服务端压力过大 | 降低 `--request-rate` 或 `--num-prompts` |
| `attention backend not found` | 未指定 CUSTOM | 加 `--attention-backend CUSTOM` |
| catalog 报错 | `REC_CATALOG` 未设置 | `export REC_CATALOG=/data/catalog` |
| P99 TTFT 超标 | 首 token 延迟高 | 检查 warmup、batch 大小 |
| 模型下载失败 | 无法访问 HuggingFace | 按 3.1 节离线准备 |
| 数据集找不到 | 目录结构不对 | 检查 `data/video/video_test.parquet` 等路径 |

---

## 九、验证 Checklist

```text
□ vllm 0.22.1 GPU 版安装完成
□ vllm-gr 安装完成
□ torch.cuda.is_available() == True
□ 模型权重已放到远端本地路径
□ OneRec 数据集已放到远端
□ catalog 文件已配置
□ offline accuracy 指标达标
□ vllm-gr serve 能正常启动
□ /v1/models 接口可访问
□ online throughput 无 failed requests
□ online accuracy 指标达标
```

---

## 十、参考命令速查

```bash
# 安装
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install vllm==0.22.1 --torch-backend=auto
pip install -e ".[dev]"

# 环境变量
export REC_MODEL=/data/models/OneRec-1.7B
export REC_DATA=/data/OpenOneRec-RecIF/data
export REC_CATALOG=/data/catalog
export HF_HUB_OFFLINE=1

# UT
python -m pytest tests/test_beam_request_unit.py tests/test_attention_backend_custom.py tests/test_onerec_patch.py -v

# Benchmark
bash tests/system/run_benchmark_and_verify.sh
```
