# Installation

Install vLLM-GR from a source checkout. Choose the guide for the accelerator
and vendor runtime you will use.

## Supported hardware

| Device | Guide |
| --- | --- |
| NVIDIA GPU (CUDA) | [CUDA installation](gpu.md) |
| AMD Instinct and Radeon GPU (ROCm) | [ROCm installation](rocm.md) |
| Ascend NPU | [Ascend installation](npu.md) |

## Shared steps

Both guides assume a checkout of this repository:

```bash
git clone https://github.com/JiusiServe/vllm-gr.git
cd vllm-gr
```

!!! important "Select the installation target"

    Set `VLLM_GR_TARGET_DEVICE` to `cuda` or `npu`. When the variable is unset,
    `setup.py` tries to detect the Torch backend. An isolated build may have no
    Torch installation, in which case the build defaults to CUDA.

After installing in the target environment, check the active interpreter and
the installed distribution metadata. A missing distribution raises
`PackageNotFoundError`.

```bash
python --version
python -c 'from importlib.metadata import version; print(version("vllm-gr"))'
```

Dependency and Python constraints belong to `pyproject.toml`, `setup.py`,
`requirements/`, and `vllm_gr/platforms/compat.py`. See the
[dependency and version policy](../../contributing/dependency_policy.md) for
release tags, overrides, and package versions.

## Next

- [Quickstart](../quickstart.md) for the first offline run.
- [check-env](../../cli/check_env.md) to validate the installed runtime.
- [Getting started](../README.md) for the path from installation to serving.
