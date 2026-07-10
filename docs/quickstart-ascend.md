
# Installation for Ascend NPU

## Prerequisites

- Python 3.11
- Pip 23+
- CANN 8.5.0+
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
source ${CONDA_PREFIX}/Ascend/nnal/asdsip/set_env.sh
```

3. **Install vllm**

follow [vllm installation](https://docs.vllm.ai/en/latest/getting_started/installation) for more details.

```bash
python -m pip install vllm==0.22.1
```

4. **Install vllm-ascend (base environment)**

vllm-gr does **not** install vllm-ascend. Install the Ascend platform stack first:

```bash
python -m pip install vllm-ascend==0.22.1rc1
```

**Note**: For Npu A3 (910C) users - due to bug in vllm-ascend, it is required to install vllm-ascend from source

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

6. **Verify installation**

```bash
python -c "import benchmarks.open_one_rec; print('✓ Installation successful')"
```

## Quick Start

Refer to [quickstart](quickstart.md#quick-start) for deployment.

See [getting_started/installation/npu/npu.inc.md](getting_started/installation/npu/npu.inc.md) for the full NPU installation guide.
