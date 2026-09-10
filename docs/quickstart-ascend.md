
# Installation for Ascend NPU

## Prerequisites

- Rust 1.88.0+ and Cargo for source or editable installs; the latest stable
  toolchain is recommended. Prebuilt wheels do not require Rust. Follow the
  [Rust toolchain instructions](quickstart.md#prerequisites) before running
  `pip install .` or `pip install -e .` from this repository.

- Python 3.11 or 3.12
- Pip 23+
- CANN 9.0.0
- PyTorch / torch-npu 2.10.0
- Triton-Ascend 3.2.1
- Conda 24.11+
- 32GB+ RAM (recommended)

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/JiusiServe/vllm-gr.git
cd vllm-gr
```

2. **Create environment and install CANN**

- Follow miniconda installation procedure on [conda installation](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install) if not yet installed.
- For details about CANN installation on various platforms, please refer to [CANN installation](https://ascend.github.io/docs/sources/ascend/quick_install.html)
- This example was verified for 910B architecture

```bash
export MY_ENV=xGr-install # any valid env name will do
conda create -n ${MY_ENV}  -c conda-forge python=3.11 numactl gcc=10.3.0  gxx=10.3.0
conda activate ${MY_ENV}
```

If device is 910b install 910b ops

```bash
conda install -c https://repo.huaweicloud.com/ascend/repos/conda/ ascend::cann-toolkit==9.0.0 ascend::cann-910b-ops==9.0.0
```

If device is 910c install a3 ops

```bash
conda install -c https://repo.huaweicloud.com/ascend/repos/conda/ ascend::cann-toolkit==9.0.0 ascend::cann-a3-ops==9.0.0
```

For any other devices please follow the official [CANN installation documentation](https://ascend.github.io/docs/sources/ascend/quick_install.html).

```bash
source ${CONDA_PREFIX}/Ascend/ascend-toolkit/set_env.sh
conda install -c https://repo.huaweicloud.com/ascend/repos/conda ascend::cann-nnal==9.0.0
source ${CONDA_PREFIX}/Ascend/nnal/atb/set_env.sh
```

3. **Install vllm**

follow [vllm installation](https://docs.vllm.ai/en/latest/getting_started/installation) for more details.

```bash
python -m pip install vllm==0.22.1
```

This vllm-gr release supports the vLLM 0.22.1 API only.

4. **Install vllm-ascend (base environment)**

vllm-gr does **not** install vllm-ascend. Install the Ascend platform stack first:

```bash
python -m pip install vllm-ascend==0.22.1rc1
```

To install the matching release from source instead:

```bash
cd <working-dir>
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout v0.22.1rc1
pip install .
```

Return to vllm-gr folder for next steps.

See the [vllm-ascend v0.22.1rc1 installation reference](https://github.com/vllm-project/vllm-ascend/tree/v0.22.1rc1).

5. **Install vllm-gr**

```bash
VLLM_GR_TARGET_DEVICE=npu pip install -e . --no-build-isolation
```

For developer mode:

```bash
VLLM_GR_TARGET_DEVICE=npu pip install -e ".[dev]" --no-build-isolation
```

Confirm that the complete Ascend runtime pair was resolved correctly:

```bash
python -c 'import importlib.metadata as m; print("vllm", m.version("vllm")); print("vllm-ascend", m.version("vllm-ascend")); print("vllm-gr", m.version("vllm-gr"))'
python -m pip check
```

6. **Verify installation**

```bash
python -c "import benchmarks.open_one_rec; print('✓ Installation successful')"
```

## Quick Start

Refer to [quickstart](quickstart.md#quick-start) for deployment.

See [getting_started/installation/npu/npu.inc.md](getting_started/installation/npu/npu.inc.md) for the full NPU installation guide.
