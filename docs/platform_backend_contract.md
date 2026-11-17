# vLLM-GR platform backend contract

## Scope

The platform backend boundary separates hardware-neutral vLLM-GR runtime
orchestration from CUDA- and NPU-specific implementations. A backend provider
binds declared provider metadata and an identity match against the native
platform vLLM already selected to ModelRunner patch hooks, Beam Attention
execution primitives, environment validation, and any platform-specific
compatibility patches.

Production startup already uses provider activation, environment validation,
runtime compatibility patches, and the constrained TopK execution factory.
GR EngineArgs also publishes an immutable backend plan. Model load binds its
services to the native runner, and EngineCore checks every worker's readiness
before its constructor returns. See [Engine plans](#engine-plans-and-service-binding).
The worker and attention dispatchers still select their implementations
directly; the two provider getters describe available modules but do not
replace those dispatchers.

## Lifecycle and process model

Backend registration and activation are process-local. Each vLLM main,
EngineCore, or Worker process owns an independent registry.

The general-plugin startup sequence is:

1. Configure the ModelRunner default before platform-dependent configuration.
2. `check_environment()` registers built-ins, loads external providers, selects
   the provider matching the current native platform identity, and invokes its
   environment check.
3. `re_apply_patches()` reuses that provider (or activates it in a fresh process).
4. Install the shared Pangu and CUSTOM attention selector compatibility hooks.
5. Invoke `provider.apply_runtime_patches(force=True)`. The NPU provider installs
   spawn protection, then the native worker hook and its early config filter.
6. Install the common frontend Beam plan and sliced-logits hooks.

The provider's environment method must call leaf validation functions, not the
shared `check_environment()` dispatcher, to avoid recursive activation.
Repeated initialization probes live targets and does not stack wrappers.
Provider patch failures retain their original exception, identify the backend,
and stop initialization with a partial-installation/restart diagnostic. This
stage does not implement rollback; a failed process must not continue serving.

EngineArgs still reasserts the Ascend EngineCore config sanitizer on every
`create_engine_config()` call. The sanitized child entry still initializes the
common EngineCore plan before native construction. The NPU worker hook still
retries runtime installation, installs KV/warmup/utility hooks, and delegates to
native `init_device()` before loading constraint tables. These earlier and
later boundaries are not moved into general-plugin initialization.
Every worker patch attempt revalidates the early config filter and its native
caller bindings before checking the device-wrapper marker. This repairs a
restored native config alias without wrapping `init_device()` again.

```mermaid
flowchart TD
    A["General plugin"] --> B["Defaults and provider validation"]
    W["Worker retry"] --> C["Activate current provider"]
    B --> C
    C --> D["Shared Pangu and CUSTOM selector"]
    D --> E{"Selected provider"}
    E -->|CUDA| F["No early backend patch"]
    E -->|NPU| G["Spawn protection and worker hook"]
    F --> H["Common Beam plan and sliced logits"]
    G --> H
    G --> I["Deferred native worker initialization"]
```

Provider resolution and activation must happen before any runtime patch that
needs platform-specific behavior. Activation does not itself run environment
checks or apply patches.

Only one backend may be active in a process. Re-activating the same provider
is idempotent; switching to another provider is an error. Backend IDs are not
aliased by the registry. In particular, `gpu` does not mean `cuda`, and the
built-in Ascend provider's ID is `ascend`, not the `npu` tensor device type.

An unknown backend is always an error in the registry APIs and never falls
back to CUDA. This registry guarantee does not describe every independent
worker or attention dispatch branch.

## Engine plans and service binding

`BackendRequest`, `BackendPlan`, and `PlanData` are import-light values in
`platforms/plan.py`. A request contains normalized GR settings, common native
configuration, compatibility versions, and backend-owned JSON options. Plans
encode their schema and planning API versions separately from the provider
identity API. SHA-256 covers the canonical JSON encoding, including the
provider/adapter versions and admitted choices. Device ordinals, native
objects, handles, and unordered containers are excluded. Nested inputs are
copied into immutable canonical data; decoded views cannot mutate the plan.

EngineArgs writes the envelope to
`additional_config["vllm_gr_backend_plan"]` after normalizing configuration.
The ordinary native configuration transport carries this JSON/pickle-safe
data to workers. Canonical and legacy GR parsing both reserve this key; it is
not a legacy option and does not create a canonical/legacy conflict. Backend
options use the separate `vllm_gr_backend_options` object. The provider validates
its own option schema. Built-ins currently retain their typed `GRConfig`
options and reject additional options; an external provider can add its own
option names and choice values without changing the core encoder.

Provider identity API version 2 remains compatible. Providers opt into the
planning extension by declaring `plan_api_version = 1` and an
`adapter_revision` compatibility string, then implementing:

```python
def resolve(request: BackendRequest) -> BackendPlan: ...
def validate_worker(*, plan: BackendPlan, native_runner): ...
def create_model_runner_adapter(*, plan: BackendPlan, native_runner): ...
def create_attention_executor(*, plan: BackendPlan, device): ...
def create_constrained_topk_execution(*, plan=None, device=None): ...
```

`resolve()` uses only its request. Runtime and installed distribution metadata
are collected before calling it. Factories receive the same engine plan and
the actual owner/device explicitly. A provider never holds `current_plan` or
runner resources. Unknown declared planning versions and missing methods fail
registration. API-v1/v2 providers without this declaration use narrow legacy
delegation: existing getters and the device-only constraint factory are
preserved, extra backend options are rejected, and incidental methods never
count as planning capabilities.

The built-in plan records CUDA V1/V2 or Ascend V1, Prefill/Decode policy,
constraint implementation, stage path, and graph/eager policy. Native defaults
and existing configuration validation remain authoritative during migration.
Ascend's unset Decode mode retains its capability-sensitive `auto` behavior;
readiness also compares the actual bound implementation so workers cannot
choose different modes silently. An explicitly requested mode conflicting
with the existing process-global attention mode requires a separate process.

The runner owns `BackendBinding`. Binding verifies the received plan against
local configuration and runtime versions, runs provider capability validation,
then constructs required services. Partial construction releases services
already created. Reload/shutdown detach the binding, release owned services,
and clear the legacy execution alias. Stateless module bindings do not close
process-wide modules. Canonical constraint artifacts retain their current
loader and lifecycle owner; readiness requires successful load before binding.

EngineCore calls `worker_backend_readiness` through the native executor and
requires its expected number of TP/PP worker reports, an identical plan
fingerprint, and matching implementation identities. A failed or missing
report shuts down the executor and aborts startup. This gate runs before
request acceptance; it is not a request-path synchronization or hash operation.

This is the plan/binding migration for RFC #706, following #748. Actual native
Runner dispatch (#711), stage/operator routing (#712), attention admissibility
and per-layer ownership (#713), constraint identity migration (#714), and
graph/resource integration (#715) remain separate changes. The compatibility
bindings preserve those current implementations; they do not claim those
later migrations are complete.

Host-only contract coverage can run without Torch or vLLM:

```bash
python -m pytest --confcutdir=tests/platforms -q \
  tests/platforms/test_backend_plan.py \
  tests/platforms/test_external_provider_fixture.py::test_installed_provider_plan_round_trip_and_factory_binding
```

`tests/test_backend_plan_engine_args.py` additionally covers the real EngineArgs
wrapper under the normal Torch/vLLM test fixtures. CUDA V1/V2 and Ascend startup,
graph replay, inference accuracy, and performance still require device runs.

## Platform identity and deterministic selection

A backend ID is a stable GR implementation identity, not tensor placement.
NVIDIA CUDA and AMD ROCm both report `device_type == "cuda"`, so selection
cannot key on it.

`vllm_gr.platforms.identity` is the only bridge from the native platform to GR.
It builds an immutable `PlatformIdentity` from the live `current_platform`
instance: the upstream platform enum name, the tensor `device_type`, the
`device_name`, and the qualified class name plus full MRO lineage. Reading it
imports no vendor runtime and instantiates nothing, so the packaging and import
isolation guarantees still hold. Native vLLM remains the only hardware
selection authority, and no availability probe is made here.

Selection follows these rules:

1. Register built-ins that are not explicitly disabled, then discover external
   entry points. Validate the complete batch before committing.
2. Reject a duplicate backend ID from a different provider, reporting both
   origins, before any match is evaluated.
3. Call `provider.matches(identity)` for every registered provider.
4. With an explicit `VLLM_GR_BACKEND`, that ID must exist and match the selected
   platform. An explicit request can resolve ambiguity but cannot suppress a
   duplicate-ID error or make an incompatible platform match.
5. Without an explicit request, exactly one provider must match. Zero matches is
   `UnsupportedBackendError`; several is `AmbiguousBackendError`. Discovery order
   and priority scores are never used as tie-breakers.

Core holds no hardware identity allowlist. `matches()` is the only place a
platform identity is accepted or rejected, so adding a backend never edits a
table under `platforms/`.

## Provider contract

A provider exposes declared metadata, an identity match, and five operations:

```python
class GRBackendProvider(Protocol):
    backend_id: str
    device_type: str            # placement, for diagnostics only
    api_version: int            # currently 2
    gr_version_range: str | None
    vllm_version_range: str | None
    vendor_version_range: str | None
    vendor_distribution: str | None

    def matches(self, identity: PlatformIdentity) -> bool: ...

    def check_environment(self) -> dict[str, str]: ...

    def apply_runtime_patches(
        self,
        *,
        force: bool = False,
    ) -> int: ...

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks: ...

    def get_attention_execution(self) -> BeamAttentionExecution: ...

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
    ) -> "ConstrainedTopKExecution | None": ...
```

Registration validates `backend_id`, the declared `api_version`, and the
presence of `matches` and the five callable operations. It never invokes them.
This preserves lazy imports and keeps CUDA and Ascend providers registered in
one process without loading either hardware runtime.

The built-in mappings are `backend_id="cuda"` for the native CUDA platform and
`backend_id="ascend"` for the supported Ascend platform. The Ascend provider
matches the vendor class lineage (`vllm_ascend.platform.NPUPlatform`) as well as
`device_type == "npu"`, because the vendor platform's upstream enum is `OOT`,
which another out-of-tree platform could also report.

Provider objects are registered directly. Entry points must not expose a
provider class or an arbitrary factory.

## Lazy import boundary

Provider modules and their package initializers must not import `torch`,
`torch_npu`, `vllm`, `vllm_ascend`, ModelRunner implementation modules, Beam
Attention execution modules, or Ascend compatibility code at module import
time.

Hardware-dependent imports belong inside:

- `check_environment()`;
- `apply_runtime_patches()`;
- `get_model_runner_hooks()`;
- `get_attention_execution()`;
- `create_constrained_topk_execution()`.

Importing `vllm_gr.platforms` or registering the built-in providers therefore
has no hardware import side effects and does not mutate the registry unless
the registration function is explicitly called.

## ModelRunner patch hooks

V1 platform ModelRunner hook modules must provide:

```python
class ModelRunnerPatchHooks(Protocol):
    def discover_worker_runners(
        self,
    ) -> tuple[tuple[str, type], ...]: ...

    def make_patched_prepare_inputs(self, original): ...

```

Other factories are optional capabilities discovered by
`model_runner_patch.apply_worker_patches()`. In particular,
`make_patched_bookkeeping_sync(original)` is optional: CUDA retains the native
method and records its identity, while NPU installs its six-item bookkeeping
wrapper. A module without this factory still satisfies `ModelRunnerPatchHooks`.
Optional sample, sample-tokens, state-update, reload, and graph hooks remain
owned by the corresponding platform module.

`make_patched_execute_model()` is intentionally absent. It is shared GPU/NPU
logic owned by `model_runner_common.py` and remains part of the neutral patch
orchestrator.

The CUDA dispatcher selects MR V1 or MR V2 before installing their respective
hooks. The provider's V1 hook getter is not a replacement for this selection;
NPU continues to use its existing runner path.

## Beam Attention execution

Platform Attention execution provides:

- `is_available()`;
- `get_flash_attn_version()`;
- `supports_sinks()`;
- `reshape_and_cache()`;
- `cascade_attention()`.

The complete call signatures are defined by
`vllm_gr.platforms.protocol.BeamAttentionExecution`. The current
`beam_attn_gpu.py` and `beam_attn_npu.py` modules satisfy the contract directly;
providers return those modules rather than copying or wrapping their kernels.

Production attention dispatch also selects the NPU xllm or cascade execution
path. Returning the base `beam_attn_npu` module from a provider must not bypass
that selection. Getter unification is a separate migration.

Metadata construction, KV memory management, and the shared execute-model
wrapper are deliberately outside this contract until their owning migration
stages define stable boundaries.

## External provider entry points

External packages register provider objects in this entry-point group:

```toml
[project.entry-points."vllm_gr.platform_backends"]
npu = "vllm_gr_npu.provider:NPU_BACKEND"
```

The entry-point name must exactly equal `provider.backend_id`. Providers are
loaded once per process. A provider cannot silently override a built-in or
another external provider; duplicate IDs from different objects are errors, and
diagnostics identify both registration sources plus the installed distribution
and version when entry-point metadata provides them.

`VLLM_GR_DISABLE_BUILTIN_BACKENDS` is a comma-separated list of backend IDs. It
is read before built-in registration, so an external distribution can replace a
built-in with no code change. The switch must be set consistently in every
participating process. It removes one built-in; it never suppresses the
duplicate-ID check.

## Provider API versions

An entry point either declares `api_version` explicitly or is adapted through
the deprecated version-1 path, which is selected by `getattr(provider,
"api_version", 1)` and never by duck typing.

- A version-1 object is wrapped in `LegacyProviderAdapter` and is recognized only
  for the two identities that shipped unversioned: `cuda` maps to the CUDA
  provider ID and `npu` to `ascend`, still requiring an exact device-placement
  match. It emits a deprecation warning naming the source, and a warning when it
  matches ROCm through the legacy `cuda` alias.
- A version-1 object is never upgraded in place, and its optional attributes are
  never read as evidence of version-2 capability. Every declared range of the
  adapter is `None`.
- A version-1 object for any other device placement is rejected and must
  declare `api_version=2`, `backend_id` and `matches(identity)`.
- An unknown `api_version` is rejected with `InvalidBackendVersionError`.

Declared ranges are checked against installed distribution metadata before
activation. An unsatisfied range is advisory by default and logged with the
provider identity; setting `VLLM_GR_STRICT_PROVIDER_RANGES=1` makes it a startup
failure. A range that cannot be parsed, or a declared vendor distribution that
is not installed, is reported the same way.

Plugin discovery validates the complete batch before committing registrations,
so a failing entry point does not leave an earlier provider partially
registered.

## Current built-in mappings

| Provider method | CUDA mapping | Ascend mapping |
| --- | --- | --- | --- |
| `backend_id` | `cuda` | `ascend` |
| `device_type` | `cuda` | `npu` |
| `matches()` | `platform_enum == "CUDA"` and `device_type == "cuda"` | `device_type == "npu"` and `vllm_ascend.platform.NPUPlatform` in the lineage |
| `check_environment()` | Validate current CUDA platform, then `check_runtime_versions()` | `check_ascend_environment(require_npu_platform=True)` |
| `apply_runtime_patches()` | No early compatibility hooks | NPU spawn protection, then NPUWorker initialization hook; return the summed patch count |
| `get_model_runner_hooks()` | `gpu_model_runner_patch` | `npu_model_runner_patch` |
| `get_attention_execution()` | `beam_attn_gpu` | `beam_attn_npu` |
| `create_constrained_topk_execution()` | Lazy-loads a fresh `CUDAConstrainedTopKExecution(device=device)` | Lazy-loads a fresh `NPUConstrainedTopKExecution(device=device)` |

The Ascend provider is built in during the initial migration so the contract is
exercised against all current Ascend requirements. A later stage may move it to
a separately distributed package, and `VLLM_GR_DISABLE_BUILTIN_BACKENDS` is what
makes that extraction possible without a code change.

## Regression after removing the built-in NPU provider

When the built-in NPU provider is removed, regression coverage must install or
inject an external `ascend` entry point and rerun the same contract tests:

1. Importing and registering core built-ins still loads no NPU modules.
2. The external `ascend` provider resolves and activates for the supported
   Ascend platform identity.
3. Its environment check delegates to the complete Ascend validation.
4. The shared CUSTOM selector runs before its spawn/worker hooks and patch
   counts are summed. CUDA retains CUSTOM routing without importing Ascend.
5. Its getters lazily return the NPU ModelRunner and Attention execution
   modules.
6. An NPU process without the external package fails with an explicit
   unsupported-backend error and never falls back to CUDA.
