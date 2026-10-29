
# Installation

## Prerequisites

- Python 3.11+
- Pip 23+
- CUDA 13.0 (for GPU inference)
- 32GB+ RAM (recommended)

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/JiusiServe/vllm-gr.git
cd vllm-gr
```

2. **Create virtual environment**

This step is not mandatory. You can use a previously installed environment.

```bash
uv venv --python 3.12
source .venv/bin/activate
python -m ensurepip --upgrade
ln -s -r .venv/bin/pip3 .venv/bin/pip
#make sure pip is installed correctly
pip --version
#pip 25.0.1 from /your-path/vllm-gr/.venv/lib/python3.12/site-packages/pip (python 3.12)
```

3. **Install vllm-gr**

The CUDA package contract pins `vllm==0.30.0`. The platform-specific
requirements are selected automatically; do not install a different vLLM
version into the same environment.

For standard installation:

```bash
pip install -e .
```

For developer mode (includes pytest, ruff, etc.):

```bash
pip install -e ".[dev]"
```

Confirm that the resolved environment contains the supported runtime:

```bash
python -c 'import importlib.metadata as m; print("vllm", m.version("vllm")); print("vllm-gr", m.version("vllm-gr"))'
python -m pip check
```

Shared dependencies come from `requirements/common.txt`, while CUDA adds the
vLLM runtime from `requirements/cuda.txt`. There is no CUDA extra. The Ascend
path is validated on the vLLM-Ascend 0.30.0rc2 vendor image; see the
[Ascend installation guide](quickstart-ascend.md) before attempting an NPU environment.

> **Note:** Installing using `uv` (e.g. `uv pip install`) may cause race conditions during installation. It is recommended to use `pip` directly to ensure a predictable installation.

4. **Verify installation**

```bash
python -c "import vllm_gr; print(vllm_gr.__version__)"
```

## Quick Start

- [Offline OneRec V1 quickstart](onerec_v1_async_quickstart.md): prepare the
  catalog, compile an artifact, and run `GRLLM.beam_search_v1()`.
- [Online V1 serving](../examples/online_serving/beam_search/README.md): start
  `vllm-gr serve` with canonical configuration and send an OpenAI request.
- [Beam configuration reference](../examples/offline_inference/beam_search/README.md):
  device selection, graph policy, and generation limits.

Download the model and catalog separately. Use the same tokenizer snapshot
for artifact compilation and serving. The retired `GRLLM.beam_search()` API
and frontend catalog arguments are not valid substitutes for V1 configuration.
