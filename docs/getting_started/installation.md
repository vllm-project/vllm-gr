# Installation

Install vLLM-GR from a source checkout. Choose the guide for the accelerator
and vendor runtime you will use.

## Supported hardware

| Device | Guide |
| --- | --- |
| CUDA (NVIDIA GPU) | [CUDA installation](installation/gpu.md) |
| Ascend NPU | [Ascend NPU installation](installation/npu.md) |

## Runtime compatibility

The built-in CUDA and Ascend NPU providers require **vLLM 0.30.0**. The
runtime check compares the public package version and accepts a local build
suffix. The CUDA install resolves the pinned vLLM package. The Ascend install
uses vLLM and its companion packages from the vendor image.

| Additional Ascend NPU runtime | Supported version |
| --- | --- |
| `vllm-ascend` | `0.30.0rc2` |
| `triton-ascend` | `3.6.0` |

The executable pins live in [platform compatibility checks](../../vllm_gr/platforms/compat.py)
and the [CUDA requirements](../../requirements/cuda.txt). See the
[Ascend image and runtime details](installation/npu.md#vendor-images) for the rest of the
validated vendor stack.

## Shared steps

Both guides assume a checkout of this repository:

```bash
git clone https://github.com/JiusiServe/vllm-gr.git
cd vllm-gr
```

Set `VLLM_GR_TARGET_DEVICE` to `cuda` or `npu` when installing. When the
variable is unset, `setup.py` tries to detect the Torch backend. An isolated
build may have no Torch installation, in which case the build defaults to
CUDA.

After installing in the target environment, check the active interpreter and
the installed distribution metadata. A missing distribution raises
`PackageNotFoundError`.

```bash
python --version
python -c 'from importlib.metadata import version; print(version("vllm-gr"))'
```

Dependency and Python constraints belong to `pyproject.toml`, `setup.py`,
`requirements/`, and `vllm_gr/platforms/compat.py`. See the
[dependency and version policy](../contributing/dependency_policy.md) for
release tags, overrides, and package versions.

## Next

After verifying the runtime with [check-env](../cli/check_env.md), return
to the [quickstart](quickstart.md) for your first offline request.
