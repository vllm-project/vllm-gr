# vLLM-GR platform backend contract

A provider binds GR operations to the active vLLM platform. Each frontend,
EngineCore and worker process has its own registry. The built-in backend IDs
are `cuda` and `ascend`, with device types `cuda` and `npu` respectively.

## Registration and activation

The [loader](../../../vllm_gr/platforms/loader.py) registers built-ins and
discovers external provider **objects** in the `vllm_gr.platform_backends`
entry-point group. With provider API v2, the entry-point name must match its
provider's `backend_id`. The deprecated provider API v1 adapter accepts the
original device-type names.
The loader validates a discovery batch before registration and rejects duplicate
backend IDs. Activation matches the native platform identity. If multiple
providers match, select a backend explicitly with `VLLM_GR_BACKEND`.

### Plugin startup

Provider modules defer hardware imports until a method is called so discovery
can run without loading either accelerator runtime. Activation selects the
provider. Plugin startup then validates the environment, installs shared
compatibility hooks, applies backend patches and installs the common Beam
frontend and sliced-logits hooks. A partial installation failure stops startup.

## Provider contract

The [interface](../../../vllm_gr/platforms/interface.py) defines the provider
operations. `platforms/protocol.py` keeps the original imports available.

| Operation | Responsibility |
| --- | --- |
| `check_environment` | Validate the selected runtime and its dependencies. |
| `apply_runtime_patches` | Install early backend-specific compatibility. On Ascend this includes spawn protection and worker initialization. |
| `get_model_runner_hooks` | Expose the backend's runner patch hooks. |
| `get_attention_execution` | Expose its Beam attention primitives. |
| `create_constrained_topk_execution` | Construct the selection implementation for the target device. |

### Engine plans

Providers that declare plan API version 1 and an adapter revision can build
an engine plan with backend services and runner hooks. The selected provider's
plan binds those services before worker startup. Built-in patch IDs declare
ownership in the [patch inventory](../patch_inventory.yaml). The
[loader tests](../../../tests/platforms/test_plugin_loader.py)
exercise entry-point registration and rejection of conflicting providers.
