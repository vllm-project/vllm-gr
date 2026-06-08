# vllm-gr

<div align="center">
  <img src="docs/assets/vLLM-GR.png" alt="vllm-gr logo" width="60%">
</div>

vLLM-based Generative Recommender Inference Engine

This repository focuses on vLLM-based generative recommendation inference, including the HSTU offline inference pipeline, example scripts, test suites, and GPU / NPU runtime adaptation code.

## Current Scope

The current implementation targets inference scenarios and does not aim for full training-phase feature coverage.

## Repository Contents

- `vllm_gr/` — GRLLM, I/O protocols, workers, runtime utils, ops
- `examples/` — demo inference, dataset inference, weight loading, benchmark tools
- `scripts/` — unified test runner and checkpoint conversion scripts
- `tests/` — unit, contract, platform smoke, and runtime guard tests
- `requirements/` — platform dependency lists (auto-selected by `setup.py` as `cuda` / `npu` / `cpu`)

## Installation

**Install vllm / vllm-ascend first, then `pip install vllm-gr`.** Platform dependencies are auto-detected by `setup.py` — no `[gpu]` / `[npu]` extras needed.

For detailed instructions, see [docs/getting_started/installation/](docs/getting_started/installation/README.md).

### Prerequisites

- Python: `>=3.11,<3.14`
- A dedicated virtual environment is recommended
- All commands below assume you are in the repository root directory

### GPU Installation

#### Step 1: Create a virtual environment and install vllm

vllm-gr does not install vllm. Set up the base environment first:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm==0.14.1 --torch-backend=auto
```

#### Step 2: Install vllm-gr

```bash
git clone https://github.com/JiusiServe/vllm-gr.git
cd vllm-gr
uv pip install -e .
```

`setup.py` detects the CUDA backend and automatically loads `requirements/cuda.txt` (includes torchrec and fbgemm-gpu). LMCache is not installed by default for now.

#### Step 3 (optional): Developer tools

```bash
uv pip install -e ".[dev]"
```

#### Step 4: Verify installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import vllm, msgspec, safetensors, torchrec; print('gpu python deps ok')"
```

### NPU Installation

**Step 1: Base environment** (not managed by vllm-gr)

Follow the [vllm-ascend installation guide](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) to set up CANN, torch-npu, vllm, and vllm-ascend. The vllm-ascend pre-built Docker image is recommended.

#### Step 1.5: GR-specific dependencies (manual install)

```bash
pip install /path/to/torchrec-1.4.0-npu.whl
pip install fbgemm-gpu==1.4.0+cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

#### Step 2: Install vllm-gr

```bash
VLLM_GR_TARGET_DEVICE=npu uv pip install -e . --no-build-isolation
```

#### Step 3 (optional): Developer tools

```bash
uv pip install -e ".[dev]"
```

### Developer Quick Start

On a GPU machine:

```bash
uv pip install vllm==0.14.1 --torch-backend=auto
uv pip install -e ".[dev]"
```

On CI or local lint without GPU/NPU:

```bash
VLLM_GR_TARGET_DEVICE=cpu uv pip install -e ".[dev]"
```

### `VLLM_GR_TARGET_DEVICE`

**Usually does not need to be set.** Only specify it explicitly when auto-detection is incorrect, or in CI / wheel builds:

```text
VLLM_GR_TARGET_DEVICE=cuda|npu|cpu
```

### Deprecated Command Migration

The following extras still work for one transitional version but will print a `DeprecationWarning` and will be removed in a future release:

| Old command (deprecated) | New command |
| ------------------------ | ----------- |
| `pip install -e ".[gpu]"` | Install vllm first, then `pip install -e .` |
| `pip install -e ".[npu]"` | Install vllm-ascend stack first, then `pip install -e .` |
| `pip install -e ".[gpu,dev]"` | Install vllm first, then `pip install -e ".[dev]"` |
| `pip install -e ".[all-gpu]"` | `pip install -e ".[dev]"` (platform deps auto-selected) |

## Post-Installation Verification

### NPU Verification

```bash
python -c "import torch; print(torch.__version__); print(hasattr(torch, 'npu'))"
python -c "import vllm, msgspec, safetensors; print('npu python deps ok')"
```

Expected: `vllm` and the target accelerator are importable with no import errors.

## YuanRong KV Pool (NPU only)

### 1. Install yuanrong-datasystem

```bash
pip install https://gitcode.com/openeuler/yuanrong-datasystem/releases/download/v0.7.6.rc1/openyuanrong_datasystem-0.7.6rc1-cp311-cp311-manylinux_2_35_aarch64.whl
```

### 2. Start etcd

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

### 3. Start Datasystem Worker

```bash
dscli start -w \
  --worker_address "${WORKER_IP}:31501" \
  --etcd_address "${ETCD_IP}:2379" \
  --shared_memory_size_mb 20480
```

Stop the worker:

```bash
dscli stop --worker_address "${WORKER_IP}:31501"
```

### 4. Environment Variables

```bash
export PYTHONHASHSEED=0
export DS_WORKER_ADDR="${WORKER_IP}:31501"
# Optional (default: 0)
export DS_ENABLE_EXCLUSIVE_CONNECTION=0
# Remote H2D (R2HD) switch
export DS_ENABLE_REMOTE_H2D=0
```

### 5. Verification

```bash
python examples/yuanrong_datasystem_smoke.py \
  --expect-device npu \
  --device-id 0 \
  --worker-addr "${WORKER_IP}:31501"
```

### 6. Example Run

```bash
python examples/yuanrong_datasystem_smoke.py \
  --expect-device npu \
  --device-id 0 \
  --worker-addr "${WORKER_IP}:31501"
```

For more help, see: <https://pages.openeuler.openatom.cn/openyuanrong-datasystem/docs/zh-cn/latest/index.html>

## Recommended Execution Order

The commands below are listed in the recommended order. Start with the unified test runner, then try examples, and finally load real weights.

### 1. List available test suites

```bash
python scripts/run_tests.py --list-suites
```

### 2. Unified self-validation

Recommended command:

```bash
python scripts/run_tests.py --all --auto-platform
```

Notes:

- `--all` runs `unit + platform lane + examples`
- `--auto-platform` automatically resolves the currently available backend:
    - NPU environment → runs `platform-npu`
    - GPU environment → runs `platform-gpu`
    - CPU environment → skips platform smoke tests, runs `unit + examples` only
- Examples automatically pass `--expect-device` for backend consistency checks

To run only the default unit suite:

```bash
python scripts/run_tests.py
```

To explicitly specify a platform smoke test:

```bash
python scripts/run_tests.py --suite gpu-smoke
python scripts/run_tests.py --suite npu-smoke
```

## How to Add New Test Cases to CI

The current `/retest` for this repository runs the unified test runner rather than scanning the entire `tests/` directory.

The CI pipeline works as follows:

```text
PR /retest
  -> CI clones the MR code
  -> python scripts/run_tests.py --suite unit gpu-smoke
  -> Only executes test files explicitly listed in SUITE_DEFINITIONS
```

This means:

- Adding a new `tests/test_xxx.py` file is not enough on its own
- You also need to add the file to `SUITE_DEFINITIONS` in `scripts/run_tests.py`
- If it is not added to a suite, CI will not run it by default

### Recommended Integration Steps

If your new test case belongs in the default gate, follow these 4 steps:

1. Add the new test file under `tests/`
2. Add the filename to the appropriate suite in `SUITE_DEFINITIONS` in `scripts/run_tests.py`
3. Run the unified test runner locally to verify
4. Push to the MR branch and comment `/retest` on the PR

### What the Default Gate Runs

The current default gate is equivalent to:

```bash
python scripts/run_tests.py --suite unit gpu-smoke
```

Where:

- `unit` is for pure logic, protocol, contract, and dependency-light tests
- `gpu-smoke` is for GPU smoke tests that need to run as standalone scripts
- If your test is an NPU smoke test, do not put it in `unit`. Use `npu-smoke` or create a new suite instead

### Example 1: Add a new contract test to the default CI

Suppose you added:

```text
tests/test_batch_recommendation_contract.py
```

If you want it included in the default `/retest`, the simplest way is to add it to the `unit` suite:

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

Verify locally:

```bash
python scripts/run_tests.py --suite unit
```

To quickly verify just the new file:

```bash
python -m pytest tests/test_batch_recommendation_contract.py -q
```

Once confirmed passing:

```bash
git add tests/test_batch_recommendation_contract.py scripts/run_tests.py
git commit -m "Add batch recommendation contract test to unit suite"
git push
```

Then comment on the corresponding PR:

```text
/retest
```

### Example 2: Create a new standalone suite

If your test does not fit in the default `unit` suite, you can define a separate suite:

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

Verify locally:

```bash
python scripts/run_tests.py --suite batch-contract
```

Note:

- After doing this, the suite can be run locally
- However, if CI still runs `python scripts/run_tests.py --suite unit gpu-smoke` by default, `/retest` will not automatically run `batch-contract`
- It will only enter the default gate when the CI test command is updated to include this suite, or when you merge it into the default `unit` suite

### When CI Infrastructure Changes Are Not Needed

The following scenarios typically require no changes to webhooks, queues, workers, or container scripts:

- You are only adding a new `tests/test_xxx.py`
- And adding it to an existing suite (`unit` / `gpu-smoke` / `npu-smoke`)
- And still using `python scripts/run_tests.py` as the unified test runner

In other words, most "add a new test case and make `/retest` run it" scenarios only require changes within the repository itself:

- `tests/`
- `scripts/run_tests.py`

### Pre-Commit Checklist

It is recommended to run at least one of the following:

```bash
python scripts/run_tests.py --suite unit
python scripts/run_tests.py --suite gpu-smoke
python scripts/run_tests.py --all --auto-platform
```

Minimum checks:

- New test filename follows the `test_*.py` convention
- Test file is placed under `tests/`
- Filename has been added to `scripts/run_tests.py`
- Local `pytest` or unified runner passes
- After pushing, comment `/retest` on the PR

### 3. Run demo examples

GPU:

```bash
python examples/simple_inference.py --expect-device gpu
python examples/dataset_inference.py --expect-device gpu debug
```

NPU:

```bash
python examples/simple_inference.py --expect-device npu
python examples/dataset_inference.py --expect-device npu debug
```

CPU:

```bash
python examples/simple_inference.py --expect-device cpu
python examples/dataset_inference.py --expect-device cpu debug
```

Once the `debug` dataset passes, continue with:

```bash
python examples/dataset_inference.py movielens-1m
python examples/dataset_inference.py kuairand-1k
python examples/dataset_inference.py streaming
```

### 4. Convert and load real weights

First, convert the training checkpoint:

```bash
python scripts/convert_gr_checkpoint.py \
  --input-path /path/to/training_checkpoint \
  --output-dir /path/to/converted_model \
  --dataset movielens-1m
```

Then load the converted model:

```bash
python examples/load_converted_model.py \
  --model /path/to/converted_model \
  --expect-device gpu
```

For NPU environments, replace `--expect-device gpu` with `npu`.

Note:

- `examples/demo-model/` contains only `config.json` with no real weights; it is only used for the demo/random-init path
- `scripts/convert_gr_checkpoint.py` depends on `torchrec` capabilities and is best run in a GPU environment or an environment with `torchrec` already available
- A minimal NPU dependency environment may not support running the checkpoint conversion script directly

### 5. Run benchmark / accuracy tools

```bash
python examples/acc_test.py \
  --model /path/to/converted_model \
  --input-json /path/to/input.json \
  --output-json /tmp/gr_pred.json
```

If ground truth results are available:

```bash
python examples/analyze_acc_error.py \
  --pred-json /tmp/gr_pred.json \
  --gt-json /path/to/gt.json \
  --input-json /path/to/input.json
```

## Python API Example

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

The feature names in `user_history` / `candidates` within `GRPrompt` must match the model configuration. For dataset-specific usage patterns, see `examples/` and `vllm_gr/inputs/data.py`.

## Uninstall

```bash
pip uninstall vllm-gr
```
