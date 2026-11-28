# Runtime patch inventory

This page records active patch IDs, ownership, installers, process roles and
the upstream assumption each hook protects. The
[machine inventory](patch_inventory.yaml) records native targets, activation
conditions, validation coverage and removal conditions. The
[upstream patch contract test](../../tests/contract/test_upstream_patch_contracts.py)
checks that the active IDs, ordering, scope and backend IDs match. See
[patch activation and maintenance](#activation-and-maintenance) for the
process lifecycle.

## Activation and maintenance

vLLM loads the GR general plugin in the processes that run inference. The
plugin resolves the ModelRunner policy, activates the selected provider and
validates the environment. See the
[platform backend contract](module/platform_backend.md) for provider selection and
worker binding.

| Process | Activation |
| --- | --- |
| Frontend | The frontend plan installs the GR configuration and request adapters before native engine construction. |
| EngineCore | EngineCore installs and verifies its dispatch and scheduling hooks before it accepts requests. A spawned child initializes its own plan. |
| Worker | The selected provider installs its runner hooks at the native worker boundary. Services bind to the admitted plan after model load. |

Backend ownership identifies the provider that declares a hook. The installer
can run later, for example when a worker initializes its device or when native
configuration becomes available. Startup checks the current native bindings so
it can repair an incomplete installation without stacking wrappers.

## Maintaining a patch

1. Update the [machine inventory](patch_inventory.yaml) and the
   [runtime patch inventory](#active-patches) together when
   changing a target, owner, activation or removal condition.
2. Check the native methods, descriptors, wire values and direct import
   bindings for the [supported runtime](../getting_started/installation.md#runtime-compatibility).
3. Run the [contract test](../../tests/contract/test_upstream_patch_contracts.py)
   and the validation listed for the changed hook. Retired IDs remain in the
   machine inventory for traceability.

## Checked fields

| Field | Meaning |
| --- | --- |
| Scope and backend ID | `core` patches are owned by `vllm_gr`; `backend` patches are declared by one provider. |
| Owner and installer | The module that owns the behavior and the callable that installs it. |
| Risk | `critical` hooks can change upstream control flow or wire values; `high` hooks change scheduling, loading or device setup; `medium` hooks are bounded adapters. |
| Process roles | Processes that must install or verify the hook. |

## Active patches

### GR-PATCH-000 — Runtime Plugin Activation

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/plugin.py`

Installer: `vllm_gr.plugin.register`

Risk: `high`; process roles: `main`, `engine_core`, `worker`

The general plugin sets ModelRunner defaults, activates the current provider,
and validates its environment before passing that provider into patch
installation. The frontend and EngineCore use their own ordered plans. Each
patch module verifies its bindings so repeated initialization can reuse an
existing wrapper.

### GR-PATCH-001 — CUSTOM Attention Selector Compatibility

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/attention/selector_patch.py`

Installer: `vllm_gr.v1.attention.selector_patch.apply_custom_attention_backend_patch`

Risk: `critical`; process roles: `engine_core`, `worker`

An explicit CUSTOM choice, including one restored from serialized GR
configuration, must reach the registered Beam backend. Changes to the private
attention selector or its cache can silently select a different backend.

### GR-PATCH-002 — NPUWorker Initialization Hook

Scope: `backend`; backend ID: `ascend`

Owner: `vllm_gr/v1/worker/npu/startup_patch.py`

Installer: `vllm_gr.v1.worker.npu.startup_patch.patch_npu_worker_init_device`

Risk: `high`; process roles: `worker`

The [NPU startup hooks](../../vllm_gr/v1/worker/npu/startup_patch.py) install
worker utilities before native device initialization constructs the runner.
An early import can precede worker availability, so later initialization
retries the installation. The platform configuration filter can run separately.

### GR-PATCH-003 — FlatLogprobs Fast Append Override

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/engine/logprobs_patch.py`

Installer: `vllm_gr.v1.engine.logprobs_patch.patch_flat_logprobs`

Risk: `medium`; process roles: `offline_frontend`, `api_server`

The frontend appends flattened log probabilities through the GR implementation
of the native fast-append method in
[logprobs_patch.py](../../vllm_gr/v1/engine/logprobs_patch.py).
Its arguments and the upstream flattened representation must stay compatible.
The patch test checks offsets, ranks, and decoded tokens for empty and nonempty
entries and omitted optional columns.

### GR-PATCH-004 — OpenAI Beam Serving Override

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/entrypoints/openai/serving_engine.py and protocol.py`

Installer: `vllm_gr.entrypoints.openai.beam_search_patch.patch_beam_search and patch_sampling`

Risk: `high`; process roles: `api_server`

The serving frontend adapts native OpenAI-compatible Beam requests to
grouped sessions. The startup verifier checks the Beam dispatch, Chat and
Completion sampling bindings and the Completion `vllm_xargs` schema for
integer lists. An incomplete binding or mismatched schema triggers repair.
See the [HTTP Beam contract](../serving/completions.md).

### GR-PATCH-005 — EngineArgs Configuration Override

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/engine/arg_utils.py`

Installer: `vllm_gr.engine.arg_utils.patch_create_engine_config`

Risk: `high`; process roles: `main`, `api_server`

GR serve and GRLLM pass a validated snapshot through native
`EngineArgs.additional_config`. The GR hook on `create_engine_config` resolves
device-dependent settings and admits the backend plan before worker startup.

### GR-PATCH-007 — EngineCore Client and AsyncLLM Extensions

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/engine/core_client.py, core_client_patch.py, beam_session_client.py, and output_processor.py`

Installer: `vllm_gr.patch.install_engine_session_hooks`

Risk: `critical`; process roles: `main`, `api_server`

Client extensions submit a whole Beam session and route its wait, abort and
release operations. A rejected submission must preserve the identity that the
frontend uses to complete the request.

### GR-PATCH-008 — EngineCoreRequestType Extension

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/engine/wire.py`

Installer: `vllm_gr.v1.engine.wire.register_engine_core_request_types`

Risk: `critical`; process roles: `api_server`, `engine_core`

GR adds a wire discriminator for grouped session submission to the native
request type. An upstream enum value collision could send a valid message to
the wrong handler.

### GR-PATCH-009 — EngineCore Lifecycle and Socket Processing

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/engine/core.py, core_socket.py, engine_core_patch.py, and hot_update.py`

Installer: `vllm_gr.v1.engine.engine_core_patch.apply_engine_core_request_patches`

Risk: `critical`; process roles: `engine_core`

EngineCore admits grouped submissions and routes their terminal results and
retirement acknowledgements. Native socket dispatch and output collection must
keep ordinary requests progressing alongside GR work.

### GR-PATCH-010 — Scheduler Beam Override

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/core/sched/scheduler_patch.py and beam_async_scheduler.py`

Installer: `vllm_gr.v1.core.sched.scheduler_patch.apply_scheduler_patch`

Risk: `critical`; process roles: `engine_core`

Scheduling attaches Beam stage and retirement metadata to native scheduler
outputs. Logical token accounting and outstanding worker receipts must agree
before the next owner can be admitted.

### GR-PATCH-022 — Prefix-cache hash memoization

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/core/kv_cache_utils.py`

Installer: `vllm_gr.v1.core.kv_cache_utils.apply_kv_hash_memo_patch`

Risk: `high`; process roles: `engine_core`

Bounded per-process memoization avoids hashing identical prefix blocks and
prompts repeatedly. Reinstallation checks the wrappers and the EngineCore
binding that constructs the hasher.

### GR-PATCH-011 — CUDA ModelRunner V1/V2 Overrides

Scope: `backend`; backend ID: `cuda`

Owner: `vllm_gr/v1/worker/cuda/runner_installer.py, vllm_gr/v1/worker/cuda/v1/gpu_model_runner_patch.py and vllm_gr/v1/worker/cuda/v2/gpu_model_runner_v2_patch.py`

Installer: `vllm_gr.v1.worker.model_runner_patch.apply_worker_patches`

Risk: `critical`; process roles: `gpu_worker`

CUDA ModelRunner V1 and V2 adapt native input preparation, stage execution and
output completion. Request ordering, graph-visible storage and native lifecycle
boundaries have to agree. See the [runner contract](module/model_runner_v2.md).

### GR-PATCH-012 — NPU ModelRunner Override

Scope: `backend`; backend ID: `ascend`

Owner: `vllm_gr/v1/worker/npu/npu_model_runner_patch.py, vllm_gr/v1/worker/npu/stage_projection.py`

Installer: `vllm_gr.v1.worker.model_runner_patch.apply_worker_patches`

Risk: `critical`; process roles: `npu_worker`

The Ascend worker adapts native rows and attention metadata for grouped Beam
execution. Cached prefixes, real row lengths and parent selection must stay
aligned with native block tables. See
[Ascend Beam Decode execution](feature/npu_decode_execution.md).

The [worker stage projection](../../vllm_gr/v1/worker/npu/stage_projection.py)
wraps the instance logits processor, runner reload and native worker weight-update
methods. It invalidates the projection before native writes and refreshes it after
finalization and device synchronization. The
[operator helper](../../vllm_gr/ops/npu/stage_projection.py) owns the fixed-address
weights, biases and proxies.

### GR-PATCH-013 — CUSTOM Attention Backend Registration

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/attention/backends/beam_attn.py`

Installer: `vllm_gr.v1.attention.backends.beam_attn register_backend decorator`

Risk: `high`; process roles: `engine_core`, `worker`

GR registers Beam attention in the upstream CUSTOM slot. Registration and the
selector have separate responsibilities. Another extension claiming that slot
can invalidate the selection.

### GR-PATCH-015 — Sliced Vocabulary Projection

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/worker/sliced_logits_patch.py`

Installer: `vllm_gr.v1.worker.sliced_logits_patch.apply_sliced_logits_patch`

Risk: `medium`; process roles: `worker`

The worker binds the loaded table's token window to each model's logits
processor. Processors without a window use native full-vocabulary projection.
Sliced projection also adapts log-probability collection. Head-derived caches
must stay valid across native weight reload.

### GR-PATCH-016 — NPU-aware Spawn Start Method

Scope: `backend`; backend ID: `ascend`

Owner: `vllm_gr/platforms/ascend/spawn_patch.py`

Installer: `vllm_gr.platforms.ascend.spawn_patch.apply_npu_spawn_patch`

Risk: `medium`; process roles: `main`, `engine_core`

Ascend subprocesses must use a compatible start method when the parent has
initialized the device. The Ascend provider installs spawn protection before
the native worker initialization hook.

### GR-PATCH-018 — Ascend Penalties Warmup OOM Guard

Scope: `backend`; backend ID: `ascend`

Owner: `vllm_gr/v1/worker/npu/startup_patch.py`

Installer: `vllm_gr.v1.worker.npu.startup_patch.apply_penalties_warmup_oom_guard`

Risk: `medium`; process roles: `npu_worker`

Ascend warmup can allocate penalty tensors that exceed available memory. The
[NPU startup hook](../../vllm_gr/v1/worker/npu/startup_patch.py) handles the
warmup allocation failure while preserving other execution errors.
The inventory lists no behavioural test for this guard.

### GR-PATCH-019 — Ascend additional_config Sanitizer

Scope: `backend`; backend ID: `ascend`

Owner: `vllm_gr/platforms/ascend/config_patch.py`

Installer: `vllm_gr.platforms.ascend.config_patch.apply_engine_core_config_sanitizer and install_ascend_config_filter`

Risk: `high`; process roles: `main`, `engine_core`, `worker`

The [Ascend configuration patch](../../vllm_gr/platforms/ascend/config_patch.py)
retains GR settings while filtering keys passed to the native configuration
reader. Fresh EngineCore processes install the common plan before native
construction. Worker startup also refreshes the filter's native bindings.

### GR-PATCH-021 — Contiguous Beam Prefix KV

Scope: `backend`; backend ID: `ascend`

Owner: `vllm_gr/v1/worker/npu/kv_layout_patch.py`

Installer: `vllm_gr.v1.worker.npu.kv_layout_patch.apply_npu_contiguous_kv_cache_patch`

Risk: `high`; process roles: `npu_worker`

The [NPU cache layout patch](../../vllm_gr/v1/worker/npu/kv_layout_patch.py)
lets eligible Ascend xllm Beam layers request contiguous prefix K/V at native
allocation. Attention row selection is a separate reader (`select_beam_rows`)
that checks layout, strides and format before using the paged block table.
An unsuitable layout keeps the gather path.

## Retired patches

Retired IDs stay in the YAML inventory for traceability. They have no active
installer and no entry in the checked list above.

| ID | Historical target | Why it was removed |
| --- | --- | --- |
| `GR-PATCH-006` | `OpenAIServingModels.__init__` | Legacy Catalog serving patch and its tests were removed; canonical V1 constraints load in the worker |
| `GR-PATCH-014` | `LMCache RequestTracker, ReqMeta, and LMCacheConnectorV1Impl` | Experimental LMCache integration removed; no supported version pin or vLLM 0.30 validation |
| `GR-PATCH-017` | `NPUWorker.profile_memory in vllm_ascend.worker.worker or vllm_ascend.worker.worker_v1` | Removal condition met by the 0.30 upgrade; vllm-ascend 0.30.0rc2 replaced profile_memory with log_memory_stats, which reads the two allocator counters directly, and the installer found no native method to wrap on that runtime |
| `GR-PATCH-020` | `vllm.config.ModelConfig.maybe_untie_word_embeddings` | Model-specific tied-embedding compatibility was retired with its deployment recipe |
