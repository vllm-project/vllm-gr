# vllm-gr

<div align="center">
  <img src="docs/assets/vLLM-GR.png" alt="vllm-gr logo" width="60%">
</div>

vLLM-based Generative Recommender Inference Engine

This repository focuses on vLLM-based generative recommendation inference, including the OneRec beam-search / catalog-constrained generation pipeline, the Pangu LLM passthrough path, example scripts, test suites, and GPU / NPU runtime adaptation code.

## Current Scope

The current implementation targets inference scenarios and does not aim for full training-phase feature coverage.

## Repository Contents

- `vllm_gr/` — GRLLM (offline beam search), beam-search/catalog patches, NPU runtime utils, ops
- `examples/` — OneRec beam-search and Pangu LLM passthrough offline examples
- `scripts/` — unified test runner
- `tests/` — unit, contract, and Pangu system tests
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

`setup.py` detects the CUDA backend and automatically loads `requirements/cuda.txt` (shares the common dependencies; no CUDA-only extras). LMCache is not installed by default for now.

#### Step 3 (optional): Developer tools

```bash
uv pip install -e ".[dev]"
```

#### Step 4: Verify installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import vllm, msgspec, safetensors; print('gpu python deps ok')"
```

### NPU Installation

**Step 1: Base environment** (not managed by vllm-gr)

Follow the [vllm-ascend installation guide](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) to set up CANN, torch-npu, vllm, and vllm-ascend. The vllm-ascend pre-built Docker image is recommended.

#### Step 1.5: GR-specific dependencies (manual install)

The NPU runtime no longer requires torchrec or fbgemm-gpu; jagged-tensor
ops run through the native PyTorch fallback path.

```bash
# No additional GR-specific wheels required on NPU.
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

Recommended command (this is what the PR gate runs):

```bash
python scripts/run_tests.py --all
```

`--all` expands to the `unit` suite only. `pangu1b-st` is intended for
nightly / manual runs (it requires the Pangu 1B checkpoint and a GPU/NPU).

To run only the default unit suite explicitly:

```bash
python scripts/run_tests.py
```

## How to Add New Test Cases to CI

The current `/retest` for this repository runs the unified test runner rather than scanning the entire `tests/` directory.

The CI pipeline works as follows:

```text
PR /retest
  -> CI clones the MR code
  -> python scripts/run_tests.py --all
  -> Only executes test files explicitly listed in SUITE_DEFINITIONS["unit"]
```

This means:

- Adding a new `tests/test_xxx.py` file is not enough on its own
- You also need to add the file to `SUITE_DEFINITIONS` in `scripts/run_tests.py`
- If it is not added to a suite, CI will not run it by default

### Recommended Integration Steps

If your new test case belongs in the default gate, follow these 4 steps:

1. Add the new test file under `tests/`
2. Add the filename to the `unit` suite in `SUITE_DEFINITIONS` in `scripts/run_tests.py`
3. Run the unified test runner locally to verify
4. Push to the MR branch and comment `/retest` on the PR

### What the Default Gate Runs

The current default gate is equivalent to:

```bash
python scripts/run_tests.py --suite unit
```

Where:

- `unit` is for pure logic, protocol, contract, and dependency-light tests
  (beam search, catalog, patch helpers, Pangu passthrough)

### Example 1: Add a new contract test to the default CI

Suppose you added:

```text
tests/test_my_new_contract.py
```

If you want it included in the default `/retest`, the simplest way is to add it to the `unit` suite:

```python
SUITE_DEFINITIONS: dict[str, dict[str, object]] = {
    "unit": {
        "label": "Core Unit Tests",
        "runner": "pytest",
        "files": [
            "test_yuanrong_monkey_patch.py",
            "test_yuanrong_backend.py",
            "test_split_qkv_rmsnorm_rope_patch.py",
            "test_batch_and_fork_unit.py",
            "test_dp_beam_search_unit.py",
            "test_catalog.py",
            "test_task_specific_catalog_program.py",
            "test_pangu_llm_passthrough.py",
            "test_my_new_contract.py",
        ],
    },
    # ...
}
```

Verify locally:

```bash
python scripts/run_tests.py --suite unit
```

To quickly verify just the new file:

```bash
python -m pytest tests/test_my_new_contract.py -q
```

Once confirmed passing:

```bash
git add tests/test_my_new_contract.py scripts/run_tests.py
git commit -m "Add my new contract test to unit suite"
git push
```

Then comment on the corresponding PR:

```text
/retest
```

### Example 2: Create a new standalone suite

If your test does not fit in the default `unit` suite (e.g. it needs special hardware or a model download), you can define a separate suite alongside `unit` and `pangu1b-st`:

```python
SUITE_DEFINITIONS: dict[str, dict[str, object]] = {
    # ... existing suites ...
    "my-suite": {
        "label": "My Specialized Tests",
        "runner": "pytest",
        "files": [
            "test_my_specialized.py",
        ],
    },
}
```

Verify locally:

```bash
python scripts/run_tests.py --suite my-suite
```

Note:

- After doing this, the suite can be run locally
- However, since the PR gate runs `--all` (which expands to `unit` only), `/retest` will not automatically run `my-suite`
- It will only enter the default gate when you either merge its files into `unit`, or update the CI `--all` expansion

### When CI Infrastructure Changes Are Not Needed

The following scenarios typically require no changes to webhooks, queues, workers, or container scripts:

- You are only adding a new `tests/test_xxx.py`
- And adding it to an existing suite (`unit` or `pangu1b-st`)
- And still using `python scripts/run_tests.py` as the unified test runner

In other words, most "add a new test case and make `/retest` run it" scenarios only require changes within the repository itself:

- `tests/`
- `scripts/run_tests.py`

### Pre-Commit Checklist

It is recommended to run at least the unit suite:

```bash
python scripts/run_tests.py --suite unit
```

Minimum checks:

- New test filename follows the `test_*.py` convention
- Test file is placed under `tests/`
- Filename has been added to `scripts/run_tests.py`
- Local `pytest` or unified runner passes
- After pushing, comment `/retest` on the PR

### 3. Run demo examples

The OneRec offline beam-search example lives under `examples/offline_inference/beam_search/`,
and the Pangu 1B LLM passthrough example lives under `examples/offline_inference/pangu1b/`.
See the per-example README files for hardware-specific instructions.

### 4. Run OneRec benchmark / accuracy tools

The OneRec benchmark suite is under `benchmarks/open_one_rec/` and provides
its own dataset loaders, evaluators and catalog tooling. See
`docs/benchmarks/` for details.

## Python API Example

`GRLLM` is a thin subclass of vLLM's `LLM` that adds optional catalog-constrained
beam search. After removing the GR adapter / recommend runtime it always goes
through the standard vLLM LLM path, so any HuggingFace checkpoint (OneRec,
Pangu, …) is supported:

```python
from vllm_gr import GRLLM
from vllm.sampling_params import BeamSearchParams

llm = GRLLM(
    model="OpenOneRec/OneRec-1.7B",
    trust_remote_code=True,
    catalog_path="/path/to/catalog.json",  # optional catalog constraint
)

params = BeamSearchParams(
    beam_width=512,
    max_tokens=5,
    temperature=0.0,
    ignore_eos=False,
)
outputs = llm.beam_search([{"prompt": "..."}], params)
print(outputs[0].sequences[0].text)
llm.shutdown()
```

For online serving, use the `vllm-gr serve` subcommand or the vLLM OpenAI API
server (the beam-search / catalog patches are applied automatically via the
vLLM general plugin).

## Uninstall

```bash
pip uninstall vllm-gr
```
