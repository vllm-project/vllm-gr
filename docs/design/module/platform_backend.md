# vLLM-GR platform backend contract

A provider binds GR operations to the active vLLM platform. Registry state is
process-local module state: each frontend, EngineCore and worker process
resolves its own provider, and an in-process EngineCore shares the frontend's
registry. The built-in backend IDs are `cuda` and `npu`, matching their device
types.

If an existing deployment selects or disables the Ascend provider by ID, use
`npu` in place of `ascend` in `VLLM_GR_BACKEND` or
`VLLM_GR_DISABLE_BUILTIN_BACKENDS`. Integrations looking up the built-in
provider in `vllm_gr.platform_backends` use the entry point named `npu`.

## Registration and activation

The [loader](../../../vllm_gr/platforms/loader.py) registers built-ins and
discovers external provider objects in the `vllm_gr.platform_backends`
entry-point group. With provider API v2, the entry-point name must match the
provider's `backend_id`, and a non-legacy provider must declare
`api_version == 2`. The deprecated provider API v1 adapter accepts the original
device-type names from the `vllm_gr.legacy_placements` entry-point group; only
the fixed `cuda` and `npu` identities are adapted.

- `VLLM_GR_DISABLE_BUILTIN_BACKENDS` lists built-in IDs to skip before loading,
  so an external provider can claim one; duplicate checks still apply.
- Providers declare `gr_version_range`, `vllm_version_range`,
  `vendor_version_range` and `vendor_distribution`. The loader validates them
  at registration and enforces them at activation: mismatches warn by default
  and fail when `VLLM_GR_STRICT_PROVIDER_RANGES=1`.
- The loader validates a discovery batch before registration and rejects
  duplicate backend IDs. Activation matches the native platform identity
  through the provider's required `matches(identity)` callable. If multiple
  providers match, select one explicitly with `VLLM_GR_BACKEND`.
- Activation happens once per process. Activating a different backend later
  raises `BackendAlreadyActiveError`.

### Plugin startup

Provider modules defer hardware imports until a method is called, so discovery
can run without loading either accelerator runtime. The `vllm.general_plugins`
entry point (`vllm_gr.plugin:register`) then:

1. Calls `configure_model_runner_default()`, which defaults
   `VLLM_USE_V2_MODEL_RUNNER` to 0.
2. Validates the environment.
3. Installs the shared compatibility hooks: the tied-embedding patch and the
   CUSTOM attention selector patch.
4. Applies backend patches and installs the common Beam frontend and
   sliced-logits hooks.

A partial installation failure stops startup.

## Provider contract

The [interface](../../../vllm_gr/platforms/interface.py) defines the provider
operations. `platforms/protocol.py` keeps the original imports available.

| Operation | Responsibility |
| --- | --- |
| `matches` | Decide whether this provider serves the current native platform identity. |
| `check_environment` | Validate the selected runtime and its dependencies. |
| `apply_runtime_patches` | Install early backend-specific compatibility. On Ascend this includes spawn protection and worker initialization. |
| `get_model_runner_hooks` | Return the backend's ModelRunner installer (`install`, `is_patched`) or, for legacy patch-hook providers, its patch hooks. |
| `get_attention_execution` | Expose its Beam attention primitives. |
| `create_constrained_topk_execution` | Construct the selection implementation for the target device. |
| `create_config_service` | Optionally supply host defaults and admission through `BackendConfigService`. |

## Engine plans

Providers that declare plan API version 2 and an adapter revision can build an
engine plan with backend services and runner hooks. The plan is admitted and
attached to the engine config before worker startup. Each worker then
re-resolves the plan, creates the declared services after model load, and
attests them through the collective readiness check. Built-in patch IDs declare
ownership in the [patch inventory](../patch_inventory.yaml). The
[loader tests](../../../tests/platforms/test_plugin_loader.py) exercise
entry-point registration, version and duplicate rejection, the disable switch,
legacy adaptation and conflicting providers.

The provider interface version and the plan API version are independent.
To migrate a planning provider from plan API v1, read the optional
`constraint` block from `request.config.to_dict()`. Its `mode` is `none` or
`table`; in table mode, `constraint["table"]` contains the artifact and its
placement. An omitted constraint is represented by `None`. Replace reads of
the old top-level `constraint_backend` and `constraint_table` fields before
declaring `plan_api_version = 2`. Registration rejects providers declaring v1
before calling their resolver.

## Validation

| Test | Coverage |
| --- | --- |
| [Loader tests](../../../tests/platforms/test_plugin_loader.py) | Entry-point registration, name and version checks, duplicate and conflicting providers, built-in disabling, legacy adaptation. |
| [Runtime wiring tests](../../../tests/platforms/test_runtime_wiring.py) | Provider activation and patch wiring. |
| [Import isolation tests](../../../tests/platforms/test_import_isolation.py) | Provider discovery without loading the other accelerator runtime. |

## Source map

| File | Responsibility |
| --- | --- |
| [Loader](../../../vllm_gr/platforms/loader.py) | Registers built-ins, discovers external providers and activates one. |
| [Registry](../../../vllm_gr/platforms/registry.py) | Stores providers, enforces identity matching and single activation. |
| [Compatibility ranges](../../../vllm_gr/platforms/compat.py) | Validates declared version ranges. |
| [Provider interface](../../../vllm_gr/platforms/interface.py) | Defines the provider operations and plan types. |
| [Plugin entry point](../../../vllm_gr/plugin.py) | Installs shared hooks and starts provider activation in each process. |
