# CUDA installation

Install vLLM-GR in a CUDA environment and verify that the CLI can load the GPU backend.

## Prerequisites

- Python 3.11, 3.12, or 3.13
- Pip 23+
- CUDA 13.0
- 32GB+ RAM (recommended)

For the shared checkout, target-flag, and interpreter steps, see
[Installation](README.md).

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

The CUDA package contract pins `vllm==0.30.0`. The platform-specific
requirements are selected automatically; do not install a different vLLM
version into the same environment.

Choose the development install if you need tools such as pytest and Ruff.

=== "Standard"

    ```bash
    VLLM_GR_TARGET_DEVICE=cuda python -m pip install -e .
    ```

=== "Development"

    ```bash
    VLLM_GR_TARGET_DEVICE=cuda python -m pip install -e ".[dev]"
    ```

Confirm that the resolved environment contains the supported runtime:

```bash
python -c 'import importlib.metadata as m; print("vllm", m.version("vllm")); print("vllm-gr", m.version("vllm-gr"))'
python -m pip check
```

Shared dependencies come from `requirements/common.txt`, while CUDA adds the
vLLM runtime from `requirements/cuda.txt`. There is no CUDA extra. The Ascend
path is validated on the vLLM-Ascend 0.30.0rc2 vendor image; see the
[Ascend installation guide](npu.md) before attempting an NPU environment.

If you have `uv` installed, you can use it with the same target and dependency
contract.

```bash
VLLM_GR_TARGET_DEVICE=cuda uv pip install -e ".[dev]"
```

Set the target explicitly because an isolated build environment may not contain
Torch for backend detection.

### Enable the GR CLI and verify the runtime

Follow the [CLI dispatch patch instructions](../../cli/README.md#command-form-and-installation)
for the active vLLM 0.30.0 installation. For a wheel, apply the patch in the
interpreter's site-packages. For editable vLLM, apply it at the vLLM source
root. Then run the environment check in the same shell.

```bash
vllm check-env --gr
```

The command validates the vLLM version and reports the selected platform. See
[check-env](../../cli/check_env.md) for the checks and the output format.

## Quick start

- [Run OneRec offline](../quickstart.md#cuda) with a
  local model snapshot and the CUDA example command.
- [Serve OneRec over HTTP](../../../examples/online_serving/beam_search/README.md)
  to start `vllm serve MODEL --gr` and send an OpenAI-compatible request.
- [Beam search on CUDA](../../user_guide/beam_search_cuda.md)
  shows how to run Beam search with or without a constraint table.
- [Beam configuration reference](../../../examples/offline_inference/beam_search/README.md):
  device selection, graph policy, and generation limits.

For catalog-constrained serving, use the same tokenizer snapshot for artifact
compilation and serving. Set catalog options in the
[constraint table configuration](../../configuration/constraint_tables.md#configure-the-engine).
