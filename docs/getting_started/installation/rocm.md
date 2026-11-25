# AMD ROCm installation

Install vLLM-GR on AMD Instinct or Radeon GPUs and verify that the CLI selects
the ROCm backend. ROCm support is tracked in
[RFC #885](https://github.com/vllm-project/vllm-gr/issues/885).

## Validated hardware

| Series | GPU | Architecture | FlashAttention backend |
| --- | --- | --- | --- |
| Instinct | MI355X | `gfx950` | ROCm FlashAttention (Composable Kernel) |
| Instinct | MI308X | `gfx942` | ROCm FlashAttention (Composable Kernel) |
| Radeon | Radeon AI PRO R9700 | `gfx1201` | FlashAttention Triton AMD backend |
| Radeon | Radeon PRO W7900 | `gfx1100` | FlashAttention Triton AMD backend |

These GPUs were validated with OneRec offline Beam search (ModelRunner V1 and
V2) and HTTP serving, using vLLM `v0.30.0rc2` built from source on ROCm 10.1
with PyTorch 2.12 (`rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.12.0`).

## Prerequisites

- Python 3.12 with a ROCm build of PyTorch and Triton. Keep the existing ROCm
  PyTorch stack: pass a constraints file that pins `torch`, `torchvision`,
  `torchaudio` and `triton` to every `pip install` below.
- `git` and `cmake`.
- For a pip-installed ROCm SDK, point the build at the SDK root:

```bash
export ROCM_PATH=$(rocm-sdk path --root)
export ROCM_HOME=$ROCM_PATH HIP_PATH=$ROCM_PATH
export CMAKE_PREFIX_PATH=$(rocm-sdk path --cmake)
export PYTORCH_ROCM_ARCH=gfx950   # gfx942, gfx1201 or gfx1100 for the other GPUs
```

vLLM detects ROCm through `amdsmi`. If `python -c 'import amdsmi'` fails with a
pip-installed SDK, add the SDK copy to the interpreter path:

```bash
echo "$ROCM_PATH/share/amd_smi" > "$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/rocm_sdk_amdsmi.pth"
```

## Install vLLM from source

vLLM-GR requires vLLM 0.30.0. The ROCm backend also accepts `v0.30.0rc2`,
which differs from `v0.30.0` only in build scripts for other platforms.
Build a wheel and install it without dependencies:

```bash
git clone --branch v0.30.0 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/rocm.txt -c constraints.txt
VLLM_TARGET_DEVICE=rocm python setup.py bdist_wheel --dist-dir=dist
pip install --no-deps dist/vllm-*.whl
```

!!! warning "Install vLLM as a wheel"

    Do not use `setup.py develop` for vLLM. An editable vLLM install puts the
    vLLM repository root on `sys.path`, and its `examples` package hides the
    vLLM-GR `examples` package that the HTTP quickstart imports.

## Install FlashAttention

Beam attention uses the upstream `flash_attn` package that vLLM loads on ROCm.

=== "Instinct (CK)"

    Use the revision pinned by vLLM's `docker/Dockerfile.rocm_base`:

    ```bash
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    git checkout 0e60e394
    git submodule update --init
    GPU_ARCHS=gfx950 python setup.py bdist_wheel --dist-dir=dist
    pip install --no-deps dist/flash_attn-*.whl
    ```

=== "Radeon (Triton)"

    Composable Kernel FlashAttention does not support RDNA GPUs. Install the
    Triton AMD backend from a recent FlashAttention revision, such as
    `b038b49`, and enable it at run time:

    ```bash
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    git checkout b038b49
    git submodule update --init third_party/aiter
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install --no-build-isolation --no-deps .
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    ```

    Do not use the Triton backend at the older `0e60e394` revision: its
    varlen path ignores `block_table` instead of rejecting it.

Paged Beam attention uses 128-token KV blocks on ROCm. vLLM selects this size
automatically for the GR attention backend.

## Install vLLM-GR

Install the shared requirements, then vLLM-GR without dependencies.
`requirements/cuda.txt` would otherwise replace the ROCm vLLM build with the
CUDA wheel.

```bash
pip install -r requirements/common.txt -c constraints.txt
VLLM_GR_TARGET_DEVICE=cuda pip install --no-deps --no-build-isolation -e .
```

`VLLM_GR_TARGET_DEVICE=cuda` selects the GPU packaging path; ROCm PyTorch
reports no CUDA version, so automatic detection would select `cpu`.

Apply the [CLI dispatch patch](../../cli/README.md#command-form-and-installation)
to the vLLM wheel in site-packages, then check the runtime:

```bash
vllm check-env --gr
```

The report shows the installed vLLM version and `platform: rocm`.

## Quick start

Follow the [quickstart](../quickstart.md) and the
[HTTP serving guide](../../../examples/online_serving/beam_search/README.md).

- On GPUs with about 30 GB of memory, such as the R9700, pass
  `--max-model-len 2048` to the offline example. With the default of 8192,
  graph capture alone needs about 21 GiB.
- Inside containers, the HIP device order can differ from the host `amd-smi`
  order. Match devices by PCI bus ID before setting `HIP_VISIBLE_DEVICES`,
  and exclude integrated GPUs.

## Current limitations

- Short-suffix attention fusion uses NVIDIA-specific instructions and is
  disabled on ROCm. Beam decode uses the generic cascade path.
- Attention sinks, FP8 KV cache and FlashAttention 3 prefill are not
  available on ROCm.
- Performance has not been characterized. The gRPC and Sidecar examples have
  not been validated on ROCm.
