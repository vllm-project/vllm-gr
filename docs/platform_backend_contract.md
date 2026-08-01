# vLLM-GR platform backend contract

## Scope

The platform backend boundary separates hardware-neutral vLLM-GR runtime
orchestration from the CUDA implementation. A backend provider binds an exact
canonical vLLM `device_type` to ModelRunner patch hooks,
Beam Attention execution primitives, environment validation, and any
platform-specific compatibility patches.

The community `vllm-gr` distribution ships the CUDA provider. Internal source
distributions can add providers through the same contract without changing
the public orchestration code.

## Lifecycle and process model

Backend registration and activation are process-local. Each vLLM main,
EngineCore, or Worker process owns an independent registry.

The startup sequence is:

1. Read the canonical `device_type` from `vllm.platforms.current_platform`.
2. Filter the `vllm_gr.platform_backends` entry points by that exact name.
3. Load, validate, register, and activate only the selected provider.
4. Validate the active provider's API version and environment.
5. Apply the provider's engine configuration hooks.
6. Apply the provider's runtime compatibility patches.
7. Apply hardware-neutral vLLM-GR runtime patches.
8. Release memory through the active provider during shutdown.

Provider resolution and activation must happen before any runtime patch that
needs platform-specific behavior. Activation does not itself run environment
checks or apply patches.

Only one backend may be active in a process. Re-activating the same provider
is idempotent; switching to another provider is an error. Backend aliases are
not resolved by the registry. In particular, `gpu` does not mean `cuda`.

An unknown backend is always an error and never falls back to CUDA.

## Provider contract

A provider exposes one canonical `device_type`, one API version, and six lifecycle methods:

```python
class GRBackendProvider(Protocol):
    api_version: int
    device_type: str

    def configure_engine(self, config) -> None: ...

    def check_environment(self) -> dict[str, str]: ...

    def release_accelerator_memory(self) -> None: ...

    def apply_runtime_patches(
        self,
        *,
        force: bool = False,
    ) -> int: ...

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks: ...

    def get_attention_hooks(self) -> BeamAttentionHooks: ...
```

Registration validates `api_version`, `device_type`, and the presence of the
required callable methods. It must not invoke the methods. Entry-point
filtering happens before `entry_point.load()`, so a broken unselected backend
cannot affect the current platform.

Provider objects are registered directly. Entry points must not expose a
provider class or an arbitrary factory.

## Lazy import boundary

Provider modules and their package initializers must not import `torch`,
`vllm`, ModelRunner implementation modules, or Beam Attention execution
modules at module import time.

Hardware-dependent imports belong inside:

- `check_environment()`;
- `configure_engine()`;
- `release_accelerator_memory()`;
- `apply_runtime_patches()`;
- `get_model_runner_hooks()`;
- `get_attention_hooks()`.

Importing `vllm_gr.platforms` therefore has no provider or hardware import side
effects. Loading one entry point imports only that provider module; its heavy
hardware-dependent modules remain lazy until the corresponding hook runs.

## ModelRunner patch hooks

Platform ModelRunner hooks provide:

```python
class ModelRunnerPatchHooks(Protocol):
    def discover_worker_runners(
        self,
    ) -> tuple[tuple[str, type], ...]: ...

    def make_patched_prepare_inputs(self, original): ...

    def make_patched_bookkeeping_sync(self, original): ...
```

`make_patched_execute_model()` is intentionally absent. It is shared logic
owned by `model_runner_common.py` and remains part of the neutral patch
orchestrator.

## Beam Attention hooks

Platform Attention hooks provide the complete capability, cache-layout,
metadata-selection, and execution boundary used by the shared facade:

- `is_available()`;
- `get_flash_attn_version()`;
- `supports_sinks()`;
- `supports_batch_invariance()`;
- `supports_kv_cache_dtype()`;
- `supports_compute_capability()`;
- `validate_combination()`;
- `get_kv_cache_shape()` and `get_kv_cache_stride_order()`;
- `split_kv_cache()` and `get_kv_scales()`;
- `get_fp8_dtype()`;
- `get_metadata_builder_cls()`;
- `reshape_and_cache()`;
- `cascade_attention()`.

The complete call signatures are defined by
`vllm_gr.platforms.protocol.BeamAttentionHooks`. `beam_attn_gpu.py` satisfies
the contract directly; the CUDA provider returns that module rather than
copying or wrapping its kernels.

The provider-selected metadata builder is part of this contract. Shared
request segmentation, prefix/suffix geometry, block-table handling, and CPU
mapping live in `beam_attn_metadata.py`. CUDA scheduler and graph state live in
`beam_attn_metadata_gpu.py`. The shared facade never selects a builder by
inspecting the device.

KV memory management and the shared execute-model wrapper remain outside this
contract.

## In-package provider entry points

The community distribution declares its provider in one entry-point group:

```toml
[project.entry-points."vllm_gr.platform_backends"]
cuda = "vllm_gr.platforms.cuda:CUDA_BACKEND"
```

The entry-point name must exactly equal `provider.device_type`. A selected
provider is loaded once per process. Duplicate matching entry points are
rejected before either one is loaded, and diagnostics identify every source.
An API mismatch or import failure is reported only if that provider is
selected.

## CUDA provider mapping

| Provider method | CUDA mapping |
| --- | --- |
| `device_type` | `cuda` |
| `check_environment()` | Validate current CUDA platform and vLLM version |
| `apply_runtime_patches()` | No-op |
| `get_model_runner_hooks()` | `gpu_model_runner_patch` |
| `get_attention_hooks()` | `beam_attn_gpu` |
| `get_metadata_builder_cls()` | `CUDABeamAttentionMetadataBuilder` |

The source tree and installed distribution use a single package.

## Installation boundary

Users continue to install every supported variant with `pip install .` (or
`pip install -e .`). `setup.py` selects exactly one dependency file in this
order:

1. `VLLM_GR_TARGET_DEVICE`;
2. `READTHEDOCS` (CPU);
3. an installed CUDA PyTorch runtime;
4. CUDA when PyTorch is absent;
5. CPU when PyTorch is present without an accelerator backend.

The selected source distribution must contain
`requirements/<device>.txt`, and that file must resolve to a non-empty set of
dependencies. A missing or empty backend definition is an installation error;
it never falls back to CUDA and never produces a knowingly unusable wheel.
