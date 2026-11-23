# CUDA installation

Install vLLM-GR in a CUDA environment and verify the selected runtime.

## Prerequisites

- Python 3.11, 3.12, or 3.13
- pip 23 or later
- CUDA 13.0
- 32 GB of host RAM (recommended)

For the shared checkout, target-flag, and interpreter steps, see
[Installation](../installation.md).

## Setup

### Create a virtual environment

Create an environment or activate one that already meets the prerequisites.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip --version
```

### Install vLLM-GR

The CUDA install resolves the [supported vLLM version](../installation.md#runtime-compatibility)
from the [CUDA requirements](../../../requirements/cuda.txt). Set the target
explicitly so the build selects the CUDA dependencies even when the build
environment lacks Torch.

```bash
VLLM_GR_TARGET_DEVICE=cuda python -m pip install -e .
python -m pip check
```

For development tools such as pytest and Ruff, use the `dev` extra.

```bash
VLLM_GR_TARGET_DEVICE=cuda python -m pip install -e ".[dev]"
```

### Enable the GR CLI and verify the runtime

Run the environment check in the same shell.

```bash
python -m vllm_gr.entrypoints.cli.main check-env --gr
```

The report should show `platform: cuda`. See [check-env](../../cli/check_env.md)
for the output format and failure conditions.

To use `vllm ... --gr`, follow the
[CLI dispatch patch instructions](../../cli/README.md#enable-gr-dispatch)
for the active environment. Then confirm that the native executable can
dispatch to vLLM-GR.

```bash
vllm check-env --gr
```

## Quick start

[Run OneRec on CUDA](../quickstart.md#cuda) to verify model inference.
