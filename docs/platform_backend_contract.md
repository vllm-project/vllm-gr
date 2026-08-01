# vLLM-GR platform backend contract

## Scope

The platform backend boundary separates hardware-neutral vLLM-GR runtime
orchestration from CUDA- and NPU-specific implementations. A backend provider
binds an exact canonical vLLM `device_type` to ModelRunner patch hooks,
Beam Attention capability and execution hooks, environment validation, and any
platform-specific compatibility patches.

## Lifecycle and process model

Backend registration and activation are process-local. Each vLLM main,
EngineCore, or Worker process owns an independent registry.

The intended startup sequence is:

1. Register built-in providers.
2. Load external providers from Python entry points.
3. Resolve and activate the provider for `vllm.platforms.current_platform`.
4. Register the hardware-neutral CUSTOM Beam Attention backend in the process.
5. Validate the active provider's environment.
6. Apply the provider's runtime compatibility patches.
7. Apply hardware-neutral vLLM-GR runtime patches.

Provider resolution and activation must happen before any runtime patch that
needs platform-specific behavior. Activation does not itself run environment
checks or apply patches.

Only one backend may be active in a process. Re-activating the same provider
is idempotent; switching to another provider is an error. Backend aliases are
not resolved by the registry. In particular, `gpu` does not mean `cuda` and
`ascend` does not mean `npu`.

An unknown backend is always an error and never falls back to CUDA. During the
first infrastructure-only stage, this guarantee applies to the new registry
APIs. The existing production dispatchers acquire the same guarantee when
they are migrated to the provider boundary.

## Provider contract

A provider declares the core API version, one canonical `device_type`, and
the complete lifecycle surface:

```python
class GRBackendProvider(Protocol):
    api_version: int
    device_type: str

    def configure_engine(self, config: Any) -> None: ...

    def check_environment(self) -> dict[str, str]: ...

    def apply_runtime_patches(
        self,
        *,
        force: bool = False,
    ) -> int: ...

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks: ...

    def get_attention_hooks(self) -> BeamAttentionHooks: ...

    def release_accelerator_memory(self) -> None: ...
```

Core currently defines `BACKEND_API_VERSION = 1`. Registration validates the
exact API version, `device_type`, and the presence of every callable method.
It must not invoke the methods. An incompatible version is an explicit error,
not a CUDA fallback. This preserves lazy imports and allows CUDA and NPU
providers to coexist in the registry without loading both hardware runtimes.

Provider objects are registered directly. Entry points must not expose a
provider class or an arbitrary factory.

## Lazy import boundary

Provider modules and their package initializers must not import `torch`,
`torch_npu`, `vllm`, `vllm_ascend`, ModelRunner implementation modules, Beam
Attention execution modules, or Ascend compatibility code at module import
time.

Hardware-dependent imports belong inside:

- `configure_engine()`;
- `check_environment()`;
- `apply_runtime_patches()`;
- `get_model_runner_hooks()`;
- `get_attention_hooks()`;
- `release_accelerator_memory()`.

Importing `vllm_gr.platforms` or registering the built-in providers therefore
has no hardware import side effects and does not mutate the registry unless
the registration function is explicitly called.

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

`make_patched_execute_model()` is intentionally absent. It is shared GPU/NPU
logic owned by `model_runner_common.py` and remains part of the neutral patch
orchestrator.

## Beam Attention hooks

Platform Attention hooks own both capability policy and execution:

- `is_available()`;
- `get_flash_attn_version()`;
- `supports_batch_invariance()`;
- `supports_kv_cache_dtype()`;
- `supports_sinks()`;
- `supports_compute_capability()`;
- `validate_combination()`;
- `get_kv_cache_shape()`;
- `get_kv_cache_stride_order()`;
- `split_kv_cache()`;
- `get_kv_scales()`;
- `get_fp8_dtype()`;
- `reshape_and_cache()`;
- `cascade_attention()`;
- `get_metadata_builder_cls()`.

The complete call signatures are defined by
`vllm_gr.platforms.protocol.BeamAttentionHooks`. The current
`beam_attn_gpu.py` and `beam_attn_npu.py` modules satisfy the contract directly;
providers return those modules rather than copying or wrapping their kernels.

`beam_attn.py` resolves hooks lazily. Each `BeamAttentionImpl` stores one hook
object during construction, and its `forward()` hot path never reads the
provider registry. Backend class methods resolve the active hooks lazily.
Common Beam segmentation, geometry, mappings, and block-table updates live in
`beam_attn_metadata.py`. CUDA scheduling/device tensors and the NPU causal-mask
lifecycle live in their respective `beam_attn_metadata_gpu.py` and
`beam_attn_metadata_npu.py` builders. The concrete builder is selected through
the hook contract.

Importing the facade neither activates a provider nor imports CUDA/NPU
execution modules. It also does not register the CUSTOM backend. CUSTOM
registration is the hardware-neutral responsibility of
the process initializer through `beam_attn_registration.py`; the explicit
EngineCore patch plan retains a verifier and repair step.

## External provider entry points

External packages register provider objects in this entry-point group:

```toml
[project.entry-points."vllm_gr.platform_backends"]
npu = "vllm_gr.platforms.npu:NPU_BACKEND"
```

The entry-point name must exactly equal `provider.device_type`. Providers are
loaded once per process. A provider cannot silently override a built-in or
another external provider; duplicate device types from different objects are
errors and diagnostics identify both registration sources.

Plugin discovery validates the complete batch before committing registrations,
so a failing entry point does not leave an earlier provider partially
registered.

Core registers only CUDA as a built-in. The current distribution exposes its
NPU provider through this entry point, matching the loading behavior of the
future separately distributed package.

## Current provider mappings

| Provider method | CUDA mapping | NPU mapping |
| --- | --- | --- |
| `api_version` | `BACKEND_API_VERSION` | `BACKEND_API_VERSION` |
| `device_type` | `cuda` | `npu` |
| `configure_engine()` | No-op | Ascend block-size default, preserving explicit and hybrid-model values |
| `check_environment()` | Validate current CUDA platform, then `check_runtime_versions()` | `check_ascend_environment(require_npu_platform=True)` |
| `apply_runtime_patches()` | No-op | Ascend Attention selector patch, then NPUWorker initialization hook; return the summed patch count |
| `get_model_runner_hooks()` | `gpu_model_runner_patch` | `npu_model_runner_patch` |
| `get_attention_hooks()` | `beam_attn_gpu` | `beam_attn_npu` |
| `release_accelerator_memory()` | Release CUDA cache and IPC state | Release NPU cache state |

## NPU provider detachability regression

Regression coverage removes or injects the external `npu` entry point and
exercises the same contract:

1. Importing and registering core built-ins still loads no NPU modules.
2. The external `npu` provider resolves and activates for the canonical NPU
   platform.
3. Its environment check delegates to the complete Ascend validation.
4. Its selector patch runs before the NPUWorker hook and patch counts are
   summed.
5. Its getters lazily return the NPU ModelRunner and Attention hook modules,
   including the NPU Metadata Builder.
6. Configuration and shutdown delegate to the Provider.
7. An NPU process without the external package fails with an explicit
   unsupported-backend error and never falls back to CUDA.
