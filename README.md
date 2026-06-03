# vllm-GR

vLLM-based Generative Recommender Inference Engine

当前仓库聚焦于基于 vLLM 的生成式推荐推理，包含 HSTU 离线推理主链路、示例脚本、测试套件，以及 GPU / NPU 运行时适配代码。

## 当前范围

当前实现面向推理场景，不追求完整训练态功能覆盖。

## 仓库内容

- `vllm_gr/`：GRLLM、输入输出协议、worker、runtime utils、ops
- `examples/`：demo 推理、dataset 推理、带权重加载、benchmark 工具
- `scripts/`：统一测试入口与 checkpoint 转换脚本
- `tests/`：unit、contract、platform smoke 与运行时守护测试
- `requirements-common.txt` / `requirements-gpu.txt` / `requirements-npu.txt`：共享依赖与平台依赖

## 前置依赖

### 1. 通用前提

- Python 版本要求：`>=3.10,<3.14`
- 推荐使用单独的虚拟环境，不要和 golden 仓混用同一个环境
- 以下命令默认都在仓库根目录执行：

### 2. GPU 前置依赖

- 机器上需要有可用的 NVIDIA GPU
- 基础环境需要能正常使用 CUDA 驱动 / runtime
- 本仓的 GPU 安装步骤会通过 `.[gpu]` 或 `requirements-gpu.txt` 安装：
    - `torch==2.9.1`
    - `numpy==2.2.6`
    - `vllm==0.14.1`
    - `torchrec==1.4.0`
    - `fbgemm-gpu==1.4.0`
    - `opencv-python-headless`
    - `safetensors` 可选但强烈建议安装

### 3. NPU 前置依赖

- 机器上需要有可用的 NPU 设备
- NPU 基础环境必须提前准备好：
    - `torch`
    - `torch-npu`
    - NPU 对应 runtime
    - 如需真实 NPU backend 链路，需提前准备对应平台插件，例如 `vllm_ascend`
- 本仓的 NPU 安装步骤只安装仓内依赖，不会安装：
    - `torch == 2.9.0`
    - `torch-npu == 2.9.0`
    - `numpy == 2.2.6`
    - `vllm_ascend == 0.14.0rc1`
- 本仓的 NPU 安装步骤会安装：
    - `vllm==0.14.1`
    - `fbgemm-gpu-cpu==1.4.0`
    - `opencv-python-headless`
    - `safetensors` 可选但强烈建议安装

## 推荐安装顺序

下面的命令按推荐顺序排列。日常开发、跑测试、跑 examples 时，优先使用 editable install。

### 1. 先升级打包工具

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 2A. GPU 环境安装

推荐最常用的开发/验证安装：

```bash
python -m pip install -e ".[gpu,dev,safetensors]"
```

如果你只想安装最小 GPU 运行时：

```bash
python -m pip install -e ".[gpu]"
```

如果你希望一次性装齐完整 GPU 扩展：

```bash
python -m pip install -e ".[all-gpu]"
```

### 2B. NPU 环境安装

前提是你的 NPU 基础环境已经提前装好了 `torch`、`torch-npu` 和对应平台 runtime / plugin。

推荐最常用的开发/验证安装：

```bash
python -m pip install -e ".[npu,dev,safetensors]"
```

如果你只想安装最小 NPU 运行时：

```bash
python -m pip install -e ".[npu]"
```

如果你希望一次性装齐完整 NPU 扩展：

```bash
python -m pip install -e ".[all-npu]"
```

### 3. `requirements*.txt` 方式

如果你更习惯按 requirements 文件安装，可使用下面的等价方式：

GPU：

```bash
python -m pip install -r requirements-gpu.txt
python -m pip install -e .
```

NPU：

```bash
python -m pip install -r requirements-npu.txt
python -m pip install -e .
```

如果你走 `requirements*.txt` 路径且还要跑 `pytest` 测试，再补：

```bash
python -m pip install -e ".[dev]"
```

`.[all]` 仍保留为兼容别名，默认对齐 GPU 路径；新文档中优先使用 `.[all-gpu]` / `.[all-npu]`。

## 安装后预检

下面的预检命令默认你使用的是上面的推荐“开发/验证安装”。如果你只装了最小 runtime `.[gpu]` 或 `.[npu]`，请根据实际情况去掉 `pytest` / `safetensors` 相关 import。

### 1. GPU 预检

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.current_device() if torch.cuda.is_available() else 'no-cuda')"
python -c "import vllm, msgspec, tqdm, pytest, safetensors, torchrec; print('gpu python deps ok')"
```

### 2. NPU 预检

```bash
python -c "import torch; print(torch.__version__); print(hasattr(torch, 'npu')); print(torch.npu.is_available() if hasattr(torch, 'npu') else 'no-npu')"
python -c "import importlib; print(importlib.util.find_spec('torch_npu') is not None); print(importlib.util.find_spec('vllm_ascend') is not None)"
python -c "import vllm, msgspec, tqdm, pytest, safetensors; print('npu python deps ok')"
```

预期检查点：

- `torch` 可正常导入
- `vllm` 可正常导入
- 目标 accelerator 可见
- 没有 import error

## YuanRong KV Pool (当前仅支持NPU)

### 1. 安装yuanrong-datasystem

```bash
pip install https://gitcode.com/openeuler/yuanrong-datasystem/releases/download/v0.7.6.rc1/openyuanrong_datasystem-0.7.6rc1-cp311-cp311-manylinux_2_35_aarch64.whl
```

### 2. 启动 etcd

```bash
ETCD_VERSION="v3.5.12"
ETCD_ARCH="linux-arm64"
wget https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
tar -xvf etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
cd etcd-${ETCD_VERSION}-${ETCD_ARCH}
sudo cp etcd etcdctl /usr/local/bin/

etcd \
  --name etcd-single \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://0.0.0.0:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://0.0.0.0:2380 \
  --initial-cluster etcd-single=http://0.0.0.0:2380 &

etcdctl --endpoints "127.0.0.1:2379" put key "value"
etcdctl --endpoints "127.0.0.1:2379" get key
```

### 3. 启动 Datasystem Worker

```bash
dscli start -w \
  --worker_address "${WORKER_IP}:31501" \
  --etcd_address "${ETCD_IP}:2379" \
  --shared_memory_size_mb 20480
```

停止 worker：

```bash
dscli stop --worker_address "${WORKER_IP}:31501"
```

### 4. 环境变量

```bash
export PYTHONHASHSEED=0
export DS_WORKER_ADDR="${WORKER_IP}:31501"
# 可选（默认 0）
export DS_ENABLE_EXCLUSIVE_CONNECTION=0
# 远端 H2D（R2HD）开关
export DS_ENABLE_REMOTE_H2D=0
```

### 5. 验证

```bash
python examples/yuanrong_datasystem_smoke.py \
  --expect-device npu \
  --device-id 0 \
  --worker-addr "${WORKER_IP}:31501"
```

### 6. 示例运行

```bash
python examples/yuanrong_backend_example.py \
  --model /workspace/ckpt/movielens_1m/800_truncated_new/converted \
  --input-json ./examples/prefix_cache_benchmark.sample_input.json \
  --expect-device npu \
  --max-uih-len 128 \
  --max-candidates 256 \
  --max-cases 2 \
  --lookup-rpc-port 1
```

更多帮助，参见：<https://pages.openeuler.openatom.cn/openyuanrong-datasystem/docs/zh-cn/latest/index.html>

## 推荐执行顺序

下面的命令按推荐顺序排列。建议先跑统一测试入口，再跑 examples，最后再跑带权重路径。

### 1. 查看可用测试套件

```bash
python scripts/run_tests.py --list-suites
```

### 2. 统一入口自验证

推荐命令：

```bash
python scripts/run_tests.py --all --auto-platform
```

说明：

- `--all` 会跑 `unit + platform lane + examples`
- `--auto-platform` 会自动解析当前可用后端：
    - NPU 环境优先跑 `platform-npu`
    - GPU 环境优先跑 `platform-gpu`
    - CPU 环境会跳过平台 smoke，只跑 `unit + examples`
- examples 会自动带 `--expect-device` 做后端一致性检查

如果只想跑默认 unit：

```bash
python scripts/run_tests.py
```

如果要显式指定平台 smoke：

```bash
python scripts/run_tests.py --suite gpu-smoke
python scripts/run_tests.py --suite npu-smoke
```

## 如何把新用例接进当前 CI

当前 `/retest` 对这个仓默认执行的是统一测试入口，而不是直接扫描整个 `tests/` 目录。

可以把当前 CI 理解成下面这条链路：

```text
PR /retest
  -> CI clone 当前 MR 对应代码
  -> python scripts/run_tests.py --suite unit gpu-smoke
  -> 只执行 SUITE_DEFINITIONS 里显式列出的测试文件
```

这意味着：

- 新增 `tests/test_xxx.py` 文件本身还不够
- 还需要把这个文件加入 `scripts/run_tests.py` 里的 `SUITE_DEFINITIONS`
- 如果不加入 suite，CI 默认不会跑到这个文件

### 最稳妥的接入方式

如果你的新用例属于默认门禁，推荐按下面 4 步做：

1. 在 `tests/` 下新增测试文件
2. 在 `scripts/run_tests.py` 的 `SUITE_DEFINITIONS` 里，把文件名加入合适的 suite
3. 先在本地执行统一入口验证
4. push 到 MR 分支后，在 PR 里评论 `/retest`

### 默认门禁到底跑什么

当前默认门禁等价于：

```bash
python scripts/run_tests.py --suite unit gpu-smoke
```

其中：

- `unit` 适合纯逻辑、协议、contract、无外部大依赖的测试
- `gpu-smoke` 适合需要单独脚本跑起来的 GPU 烟测
- 如果你的用例是 NPU 烟测，一般不该塞进 `unit`，而应放进 `npu-smoke` 或新建 suite

### 示例 1：把一个新 contract 用例加入默认 CI

假设你新增了文件：

```text
tests/test_batch_recommendation_contract.py
```

如果你希望它进入默认 `/retest`，最直接的方法是把它加入 `unit`：

```python
SUITE_DEFINITIONS: dict[str, dict[str, object]] = {
    "unit": {
        "label": "Core Unit Tests",
        "runner": "pytest",
        "files": [
            "test_engine.py",
            "test_gr_schema_contract.py",
            "test_processors.py",
            "test_embedding_collection_patch.py",
            "test_gr_runner_core.py",
            "test_dataset_config_parity.py",
            "test_checkpoint_conversion_contract.py",
            "test_output_contract.py",
            "test_positional_encoder_parity.py",
            "test_run_tests_script.py",
            "test_hstu_attention_contract.py",
            "test_batch_recommendation_contract.py",
        ],
    },
}
```

然后先本地验证：

```bash
python scripts/run_tests.py --suite unit
```

如果只想快速确认这个新文件本身，也可以先单独跑：

```bash
python -m pytest tests/test_batch_recommendation_contract.py -q
```

确认通过后：

```bash
git add tests/test_batch_recommendation_contract.py scripts/run_tests.py
git commit -m "Add batch recommendation contract test to unit suite"
git push
```

然后在对应 PR 评论：

```text
/retest
```

### 示例 2：新增一个独立 suite

如果你的测试不适合放进默认 `unit`，也可以单独定义一个 suite，例如：

```python
SUITE_DEFINITIONS: dict[str, dict[str, object]] = {
    "unit": {
        "label": "Core Unit Tests",
        "runner": "pytest",
        "files": [
            "test_engine.py",
            "test_gr_schema_contract.py",
        ],
    },
    "batch-contract": {
        "label": "Batch Recommendation Contract Tests",
        "runner": "pytest",
        "files": [
            "test_batch_recommendation_contract.py",
        ],
    },
}
```

本地验证：

```bash
python scripts/run_tests.py --suite batch-contract
```

注意：

- 这样做以后，本地可以跑到这个 suite
- 但如果 CI 默认仍然执行 `python scripts/run_tests.py --suite unit gpu-smoke`，那么 `/retest` 还不会自动跑 `batch-contract`
- 只有当 CI 的测试命令被改成包含这个 suite，或者你把它并入默认 `unit`，它才会进入默认门禁

### 什么时候不需要改 CI 基础设施

下面这种情况，通常完全不用改 webhook、queue、worker、容器脚本：

- 只是新增 `tests/test_xxx.py`
- 并把它加入现有 `unit` / `gpu-smoke` / `npu-smoke`
- 仍然使用 `python scripts/run_tests.py` 作为统一入口

也就是说，大多数“新增一个普通用例并让 `/retest` 跑到它”的需求，只需要改仓库自身：

- `tests/`
- `scripts/run_tests.py`

### 提交前建议检查

建议至少跑下面其中一组：

```bash
python scripts/run_tests.py --suite unit
python scripts/run_tests.py --suite gpu-smoke
python scripts/run_tests.py --all --auto-platform
```

最小检查建议：

- 新增的测试文件名符合 `test_*.py`
- 测试文件放在 `tests/` 下
- 文件名已经加入 `scripts/run_tests.py`
- 本地 `pytest` 或统一入口能跑通
- push 后再用 PR 评论 `/retest`

### 3. 跑 demo examples

GPU：

```bash
python examples/simple_inference.py --expect-device gpu
python examples/dataset_inference.py --expect-device gpu debug
```

NPU：

```bash
python examples/simple_inference.py --expect-device npu
python examples/dataset_inference.py --expect-device npu debug
```

CPU：

```bash
python examples/simple_inference.py --expect-device cpu
python examples/dataset_inference.py --expect-device cpu debug
```

如果 `debug` 数据集已跑通，再继续：

```bash
python examples/dataset_inference.py movielens-1m
python examples/dataset_inference.py kuairand-1k
python examples/dataset_inference.py streaming
```

### 4. 转换并加载真实权重

先转换训练 checkpoint：

```bash
python scripts/convert_gr_checkpoint.py \
  --input-path /path/to/training_checkpoint \
  --output-dir /path/to/converted_model \
  --dataset movielens-1m
```

再加载转换后的模型：

```bash
python examples/load_converted_model.py \
  --model /path/to/converted_model \
  --expect-device gpu
```

如果你在 NPU 环境加载转换后的模型，把 `--expect-device gpu` 改成 `npu`。

注意：

- `examples/demo-model/` 只有 `config.json`，没有真实权重；它只用于 demo/random-init 路径
- `scripts/convert_gr_checkpoint.py` 依赖 `torchrec` 相关能力，更适合在 GPU 或已具备 `torchrec` 能力的环境中执行
- 纯 NPU 最小依赖环境并不保证能直接执行 checkpoint 转换脚本

### 5. 跑 benchmark / acc 工具

```bash
python examples/acc_test.py \
  --model /path/to/converted_model \
  --input-json /path/to/input.json \
  --output-json /tmp/gr_pred.json
```

如果已经有 GT 结果，再补：

```bash
python examples/analyze_acc_error.py \
  --pred-json /tmp/gr_pred.json \
  --gt-json /path/to/gt.json \
  --input-json /path/to/input.json
```

## Python API 示例

```python
from vllm_gr import GRLLM, GRPrompt

llm = GRLLM(
    model="examples/demo-model",
    max_seq_len=64,
    max_candidates=50,
    allow_missing_weights=True,
)

prompt: GRPrompt = {
    "user_id": "user_001",
    "user_history": [
        {"item_id": 1001, "timestamp": 1609459200},
        {"item_id": 1002, "timestamp": 1609545600},
    ],
    "candidates": [
        {"item_candidate_id": 2001, "query_time": 1610000000},
        {"item_candidate_id": 2002, "query_time": 1610000000},
    ],
}

outputs = llm.recommend([prompt], use_tqdm=False)
print(outputs[0].item_ids)
llm.shutdown()
```

`GRPrompt` 中 `user_history` / `candidates` 的特征名需要和模型配置保持一致。不同数据集的写法可以直接参考 `examples/` 与 `vllm_gr/inputs/data.py`。

## 卸载

```bash
pip uninstall vllm-gr
```
