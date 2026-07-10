# 本地跑 CI 用例（UT + ST）指导 Wiki

> 本文档面向需要在本地/CI Docker 中复现 vllm-gr 全部 CI 用例的工程师。
>
> 环境假设：
> - Python 3.12
> - pip 直接安装（不使用 uv）
> - GPU 环境（CUDA 11.8+）
> - 模型：`OpenOneRec/OneRec-1.7B` 或本地路径
> - 数据集：`./OpenOneRec/benchmark_data`（含 `sid2pid.json`）

---

## 一、环境准备

### 1.1 创建虚拟环境

```bash
cd /path/to/vllm-gr

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 1.2 安装 vLLM 0.22.1

```bash
pip install vllm==0.22.1 --torch-backend=auto
```

验证 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 1.3 安装 vllm-gr

```bash
# 基础安装
pip install -e .

# 测试依赖（如果 dev extras 有冲突，手动装这几个）
pip install pytest>=7.0.0 pytest-asyncio>=0.21.0 pytest-cov>=4.0.0 tblib psutil
```

### 1.4 安装 pre-commit 工具（可选）

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

---

## 二、准备模型和数据

### 2.1 设置环境变量

```bash
export REC_MODEL="/path/to/OneRec-1.7B"        # 或 OpenOneRec/OneRec-1.7B
export REC_DATA="/path/to/OpenOneRec/benchmark_data"
export REC_CATALOG="/path/to/generated_data/video_catalog.json"
export HF_HUB_OFFLINE=1
```

### 2.2 目录结构检查

确保 `$REC_DATA` 下有：

```text
benchmark_data/
├── ad/ad_test.parquet
├── interactive/interactive_test.parquet
├── item_understand/item_understand_test.parquet
├── label_cond/label_cond_test.parquet
├── label_pred/label_pred_test.parquet
├── product/product_test.parquet
├── rec_reason/rec_reason_test.parquet
├── sid2iid.json
├── sid2pid.json
└── video/video_test.parquet
```

### 2.3 生成 Catalog

CI 使用 `create_task_catalog.py` 生成 video 任务的 catalog：

```bash
mkdir -p generated_data

python ./benchmarks/open_one_rec/tools/create_task_catalog.py \
  -t video \
  -d "$REC_DATA" \
  -o "$REC_CATALOG"
```

验证生成成功：

```bash
ls -lh "$REC_CATALOG"
head -5 "$REC_CATALOG"
```

输出示例：

```json
[
  ["<|sid_begin|>", "<s_a_346>", "<s_b_6566>", "<s_c_5603>", "<|sid_end|>"],
  ...
]
```

---

## 三、跑 UT（单元测试）

### 3.1 核心 UT（不依赖 GPU/真实模型）

```bash
python -m pytest tests/test_beam_request_unit.py \
                 tests/test_attention_backend_custom.py \
                 tests/test_onerec_patch.py \
                 tests/test_catalog.py \
                 -v
```

### 3.2 Serving 相关 UT

```bash
python -m pytest tests/test_serving_beam_search.py \
                 tests/test_serving_catalog.py \
                 -v
```

### 3.3 Beam Search 相关 UT

```bash
python -m pytest tests/test_offline_beam_search.py \
                 tests/test_dp_beam_request_unit.py \
                 tests/test_prometheus_beam_search_metrics.py \
                 -v
```

### 3.4 Patch 相关 UT

```bash
python -m pytest tests/test_split_qkv_rmsnorm_rope_patch.py \
                 tests/test_yuanrong_backend.py \
                 tests/test_yuanrong_monkey_patch.py \
                 -v
```

### 3.5 全量 UT（按需跑）

```bash
python -m pytest tests/ -v -k "not gpu and not npu and not cuda"
```

### 3.6 跳过 GPU 用例

如果环境没有 GPU：

```bash
python -m pytest tests/ -v -k "not gpu and not npu and not cuda"
```

---

## 四、跑 ST（系统测试）

### 4.1 OneRec Benchmark ST

这是 CI 主流程，执行 `run_benchmark_and_verify.sh`：

```bash
export DATASET_PATH="$REC_DATA"
export CATALOG_PATH="$REC_CATALOG"

bash tests/system/run_benchmark_and_verify.sh
```

脚本会执行：

1. offline accuracy benchmark
2. 启动 `vllm-gr serve`
3. online throughput benchmark
4. online accuracy benchmark
5. KPI 校验

### 4.2 Pangu 1B ST（需要 Pangu 模型）

```bash
export PANGU1B_MODEL_PATH="/path/to/openPangu-Embedded-1B-V1.1"
export PANGU1B_NUM_PROMPTS=8
export PANGU1B_MIN_ACCURACY=0.5

bash tests/system/run_pangu1b_acc_and_verify.sh
```

---

## 五、一键跑全部 CI 用例

创建脚本 `run_all_ci_tests.sh`：

```bash
#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
export REC_MODEL="${REC_MODEL:-/path/to/OneRec-1.7B}"
export REC_DATA="${REC_DATA:-/path/to/OpenOneRec/benchmark_data}"
export REC_CATALOG="${REC_CATALOG:-./generated_data/video_catalog.json}"
export HF_HUB_OFFLINE=1

# Pangu 1B 可选
export PANGU1B_MODEL_PATH="${PANGU1B_MODEL_PATH:-}"

# ---------------------------------------------------------------------------
# 1. 生成 Catalog
# ---------------------------------------------------------------------------
echo "=== Step 1: Generate catalog ==="
mkdir -p "$(dirname "$REC_CATALOG")"
python ./benchmarks/open_one_rec/tools/create_task_catalog.py \
  -t video \
  -d "$REC_DATA" \
  -o "$REC_CATALOG"

# ---------------------------------------------------------------------------
# 2. 单元测试
# ---------------------------------------------------------------------------
echo "=== Step 2: Core unit tests ==="
python -m pytest tests/test_beam_request_unit.py \
                 tests/test_attention_backend_custom.py \
                 tests/test_onerec_patch.py \
                 tests/test_catalog.py \
                 -v

echo "=== Step 3: Serving unit tests ==="
python -m pytest tests/test_serving_beam_search.py \
                 tests/test_serving_catalog.py \
                 -v

echo "=== Step 4: Beam search unit tests ==="
python -m pytest tests/test_offline_beam_search.py \
                 tests/test_dp_beam_request_unit.py \
                 tests/test_prometheus_beam_search_metrics.py \
                 -v

# ---------------------------------------------------------------------------
# 3. 系统测试
# ---------------------------------------------------------------------------
echo "=== Step 5: OneRec benchmark system test ==="
export DATASET_PATH="$REC_DATA"
export CATALOG_PATH="$REC_CATALOG"
bash tests/system/run_benchmark_and_verify.sh

if [ -n "$PANGU1B_MODEL_PATH" ]; then
  echo "=== Step 6: Pangu 1B system test ==="
  bash tests/system/run_pangu1b_acc_and_verify.sh
fi

echo "=== All CI tests passed ==="
```

执行：

```bash
chmod +x run_all_ci_tests.sh
./run_all_ci_tests.sh
```

---

## 六、CI 与本地差异

| 项目 | CI（`.github/workflows/pre-commit.yml`） | 本地 |
|------|----------------------------------------|------|
| Python | 3.11 | 3.12（你的环境） |
| 包管理 | uv | pip |
| vLLM 安装 | `uv pip install vllm==0.22.1` | `pip install vllm==0.22.1 --torch-backend=auto` |
| 数据集下载 | `hf download OpenOneRec-RecIF` | 本地已有或手动下载 |
| Catalog 生成 | `create_task_catalog.py -t video` | 同上 |
| 代码检查 | `pre-commit run --all-files` | 可选 |
| 主要 ST | `run_benchmark_and_verify.sh` | 同上 |

---

## 七、常见问题

### 7.1 `pip install -e ".[dev]"` 冲突

原因：numpy/pyarrow/black/ruff/mypy 版本冲突。

解决：分步安装，跳过 dev extras 里的 lint 工具。

```bash
pip install -e .
pip install pytest>=7.0.0 pytest-asyncio>=0.21.0 pytest-cov>=4.0.0 tblib psutil
```

### 7.2 `create_task_catalog.py` 报错

检查 parquet 中 `metadata.answer` 是否存在：

```bash
python -c "
import pandas as pd
df = pd.read_parquet('$REC_DATA/video/video_test.parquet')
print(df.columns)
print(df.iloc[0].to_dict())
"
```

### 7.3 serving 测试默认下载 HF 模型

设置本地模型路径：

```bash
export REC_MODEL=/path/to/OneRec-1.7B
```

### 7.4 benchmark ST 找不到 `REC_DATA`

`run_benchmark_and_verify.sh` 支持 `REC_DATA` 和 `DATASET_PATH`：

```bash
export REC_DATA=/path/to/OpenOneRec/benchmark_data
export REC_CATALOG=/path/to/generated_data/video_catalog.json
```

### 7.5 GPU OOM

按顺序降级：

```bash
--n 64
--max_num_seqs 1024
--max_num_batched_tokens 8192
--num-prompts 50
--dtype float16
--gpu-memory-utilization 0.85
```

### 7.6 numpy 2.x 与 tensorflow 冲突

```bash
pip install "numpy<2"
```

或升级 tensorflow：

```bash
pip install --upgrade tensorflow ml-dtypes
```

---

## 八、最小可运行 Checklist

```text
□ python3.12 虚拟环境已创建并激活
□ pip 已升级到最新版
□ vllm==0.22.1 安装完成
□ vllm-gr 安装完成（pip install -e .）
□ pytest 等测试依赖已安装
□ REC_MODEL 指向有效模型路径
□ REC_DATA 指向 benchmark_data 目录
□ REC_CATALOG 已生成
□ HF_HUB_OFFLINE=1 已设置（离线环境）
□ 核心 UT 能跑通
□ Benchmark ST 能跑通
```

---

## 九、速查命令

```bash
# 环境
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install vllm==0.22.1 --torch-backend=auto
pip install -e .
pip install pytest>=7.0.0 pytest-asyncio>=0.21.0 pytest-cov>=4.0.0 tblib psutil

# 变量
export REC_MODEL=/path/to/OneRec-1.7B
export REC_DATA=/path/to/OpenOneRec/benchmark_data
export REC_CATALOG=./generated_data/video_catalog.json
export HF_HUB_OFFLINE=1

# Catalog
python ./benchmarks/open_one_rec/tools/create_task_catalog.py \
  -t video -d "$REC_DATA" -o "$REC_CATALOG"

# UT
python -m pytest tests/test_beam_request_unit.py \
                 tests/test_attention_backend_custom.py \
                 tests/test_onerec_patch.py \
                 tests/test_catalog.py -v

# ST
bash tests/system/run_benchmark_and_verify.sh
```
