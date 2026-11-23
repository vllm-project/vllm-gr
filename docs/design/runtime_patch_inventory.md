# Runtime patch inventory

This page is the checked index of active runtime patches. It lists IDs in
the order enforced by the [upstream patch contract test](../../tests/contract/test_upstream_patch_contracts.py),
with each patch's scope, backend ownership, owner, installer and risk level.
The authoritative data is [patch_inventory.yaml](patch_inventory.yaml); the
rationale for each hook is in [runtime patches](module/runtime_patches.md).

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

### GR-PATCH-001 — CUSTOM Attention Selector Compatibility

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/attention/selector_patch.py`

Installer: `vllm_gr.v1.attention.selector_patch.apply_custom_attention_backend_patch`

Risk: `critical`; process roles: `engine_core`, `worker`

### GR-PATCH-002 — NPUWorker Initialization Hook

Scope: `backend`; backend ID: `ascend`

Owner: `vllm_gr/v1/worker/npu/startup_patch.py`

Installer: `vllm_gr.v1.worker.npu.startup_patch.patch_npu_worker_init_device`

Risk: `high`; process roles: `worker`

### GR-PATCH-003 — FlatLogprobs Fast Append Override

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/engine/logprobs_patch.py`

Installer: `vllm_gr.v1.engine.logprobs_patch.patch_flat_logprobs`

Risk: `medium`; process roles: `offline_frontend`, `api_server`

### GR-PATCH-004 — OpenAI Beam Serving Override

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/entrypoints/openai/serving_engine.py and protocol.py`

Installer: `vllm_gr.entrypoints.openai.beam_search_patch.patch_beam_search and patch_sampling`

Risk: `high`; process roles: `api_server`

### GR-PATCH-005 — EngineArgs Configuration Override

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/engine/arg_utils.py`

Installer: `vllm_gr.engine.arg_utils.patch_create_engine_config`

Risk: `high`; process roles: `main`, `api_server`

### GR-PATCH-007 — EngineCore Client and AsyncLLM Extensions

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/engine/core_client.py, core_client_patch.py, beam_session_client.py, and output_processor.py`

Installer: `vllm_gr.patch.install_engine_session_hooks`

Risk: `critical`; process roles: `main`, `api_server`

### GR-PATCH-008 — EngineCoreRequestType Extension

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/engine/wire.py`

Installer: `vllm_gr.v1.engine.wire.register_engine_core_request_types`

Risk: `critical`; process roles: `api_server`, `engine_core`

### GR-PATCH-009 — EngineCore Lifecycle and Socket Processing

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/engine/core.py, core_socket.py, engine_core_patch.py, and hot_update.py`

Installer: `vllm_gr.v1.engine.engine_core_patch.apply_engine_core_request_patches`

Risk: `critical`; process roles: `engine_core`

### GR-PATCH-010 — Scheduler Beam Override

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/core/sched/scheduler_patch.py and beam_async_scheduler.py`

Installer: `vllm_gr.v1.core.sched.scheduler_patch.apply_scheduler_patch`

Risk: `critical`; process roles: `engine_core`

### GR-PATCH-022 — Prefix-cache hash memoization

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/core/kv_cache_utils.py`

Installer: `vllm_gr.v1.core.kv_cache_utils.apply_kv_hash_memo_patch`

Risk: `high`; process roles: `engine_core`

### GR-PATCH-011 — GPU ModelRunner V1/V2 Overrides

Scope: `backend`; backend ID: `cuda`

Owner: `vllm_gr/v1/worker/cuda/runner_installer.py, vllm_gr/v1/worker/cuda/v1/gpu_model_runner_patch.py and vllm_gr/v1/worker/cuda/v2/gpu_model_runner_v2_patch.py`

Installer: `vllm_gr.v1.worker.model_runner_patch.apply_worker_patches`

Risk: `critical`; process roles: `gpu_worker`

### GR-PATCH-012 — NPU ModelRunner Override

Scope: `backend`; backend ID: `ascend`

Owner: `vllm_gr/v1/worker/npu/npu_model_runner_patch.py, vllm_gr/v1/worker/npu/stage_projection.py`

Installer: `vllm_gr.v1.worker.model_runner_patch.apply_worker_patches`

Risk: `critical`; process roles: `npu_worker`

### GR-PATCH-013 — CUSTOM Attention Backend Registration

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/attention/backends/beam_attn.py`

Installer: `vllm_gr.v1.attention.backends.beam_attn register_backend decorator`

Risk: `high`; process roles: `engine_core`, `worker`

### GR-PATCH-015 — Sliced Vocabulary Projection

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/v1/worker/sliced_logits_patch.py`

Installer: `vllm_gr.v1.worker.sliced_logits_patch.apply_sliced_logits_patch`

Risk: `medium`; process roles: `worker`

### GR-PATCH-016 — NPU-aware Spawn Start Method

Scope: `backend`; backend ID: `ascend`

Owner: `vllm_gr/platforms/ascend/spawn_patch.py`

Installer: `vllm_gr.platforms.ascend.spawn_patch.apply_npu_spawn_patch`

Risk: `medium`; process roles: `main`, `engine_core`

### GR-PATCH-018 — Ascend Penalties Warmup OOM Guard

Scope: `backend`; backend ID: `ascend`

Owner: `vllm_gr/v1/worker/npu/startup_patch.py`

Installer: `vllm_gr.v1.worker.npu.startup_patch.apply_penalties_warmup_oom_guard`

Risk: `medium`; process roles: `npu_worker`

### GR-PATCH-019 — Ascend additional_config Sanitizer

Scope: `backend`; backend ID: `ascend`

Owner: `vllm_gr/platforms/ascend/config_patch.py`

Installer: `vllm_gr.platforms.ascend.config_patch.apply_engine_core_config_sanitizer and install_ascend_config_filter`

Risk: `high`; process roles: `main`, `engine_core`, `worker`

### GR-PATCH-020 — Pangu Tied Embeddings

Scope: `core`; backend ID: `null`

Owner: `vllm_gr/config/pangu_patch.py`

Installer: `vllm_gr.config.pangu_patch.apply_pangu_tied_embeddings_patch`

Risk: `high`; process roles: `main`, `engine_core`, `worker`

### GR-PATCH-021 — Contiguous Beam Prefix KV

Scope: `backend`; backend ID: `ascend`

Owner: `vllm_gr/v1/worker/npu/kv_layout_patch.py`

Installer: `vllm_gr.v1.worker.npu.kv_layout_patch.apply_npu_contiguous_kv_cache_patch`

Risk: `high`; process roles: `npu_worker`

## Retired patches

Retired IDs stay in the YAML inventory for traceability. They have no active
installer and no entry in the checked list above.

| ID | Historical target | Why it was removed |
| --- | --- | --- |
| `GR-PATCH-006` | `OpenAIServingModels.__init__` | Legacy Catalog serving patch and its tests were removed; canonical V1 constraints load in the worker |
| `GR-PATCH-014` | `LMCache RequestTracker, ReqMeta, and LMCacheConnectorV1Impl` | Experimental LMCache integration removed; no supported version pin or vLLM 0.30 validation |
| `GR-PATCH-017` | `NPUWorker.profile_memory in vllm_ascend.worker.worker or vllm_ascend.worker.worker_v1` | Removal condition met by the 0.30 upgrade; vllm-ascend 0.30.0rc2 replaced profile_memory with log_memory_stats, which reads the two allocator counters directly, and the installer found no native method to wrap on that runtime |
