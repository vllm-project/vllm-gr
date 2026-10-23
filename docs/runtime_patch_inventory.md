# Runtime Patch and Upstream Dependency Inventory

This document records the runtime patches currently applied by `vllm-gr`, the upstream APIs and internal state they depend on, and the conditions under which those patches may be removed.

It is intended to remain valid beyond a single refactoring phase or pull request. Historical investigation notes, temporary implementation plans, and change-by-change corrections belong in pull request descriptions and issue discussions.

## Supported Baseline

The current supported runtime baseline is:

- `vllm==0.22.1`
- `vllm-ascend==0.22.1rc1` for the NPU path
- LMCache is optional and must be validated against the version installed in the target environment

The upstream source revisions used for compatibility analysis are:

- `vllm-project/vllm@releases/v0.22.1`
    - commit: `dd72658e7db0c6a674f473f6f9f0f5c2ebe7e523`
- `vllm-project/vllm-ascend@releases/v0.22.1rc`
    - commit: `f528cdd561969730504bfe8b5e96b6b7e69d6c4f`

The vLLM-Ascend release branch is movable, so compatibility conclusions must retain the exact source commit.

## Risk Levels

| Level | Meaning |
| --- | --- |
| Critical | Affects cross-process transport, request completion, scheduling, KV cache state, or ModelRunner execution |
| High | Affects global configuration, initialization order, platform selection, or ordinary request behavior |
| Medium | Localized override that remains coupled to upstream implementation details |
| Low | Narrow auxiliary behavior with a limited failure scope |

## Runtime Patch Inventory

The machine-readable counterpart is `docs/patch_inventory.yaml`. Owner,
installer, activation, validation, and process-role entries below must remain
aligned with it.

| ID | Owner | Installer | Activation | Validator or test | Process role |
| --- | --- | --- | --- | --- | --- |
| GR-PATCH-000 | `vllm_gr/plugin.py` | `vllm_gr.plugin.register` | vLLM general-plugin loading | `test_plugin_exposes_simple_per_patch_status` | main, EngineCore, worker |
| GR-PATCH-001 | `vllm_gr/v1/ascend_compat.py` | `apply_ascend_attention_backend_patch` | CUSTOM is registered and overridden | `_validate` behavior in `test_ascend_compat.py`; Ascend integration test | EngineCore, worker |
| GR-PATCH-002 | `vllm_gr/v1/ascend_compat.py` | `patch_npu_worker_init_device` | active NPU platform, first installation | `test_npu_worker_patch_is_lazy_and_preserves_lifecycle_contract` | worker |
| GR-PATCH-003 | `vllm_gr/logprobs.py` | `logprobs_patch.patch_flat_logprobs` | required frontend plan finds the method absent | `_validate_flat_logprobs_patch`; patch initialization tests | offline frontend, API server |
| GR-PATCH-004 | `entrypoints/openai/serving_engine.py`, `protocol.py` | `beam_search_patch.patch_beam_search`, `patch_sampling` | required frontend plan finds either method absent | `_validate_openai_beam_search_patch`, `_validate_chat_sampling_patch`; serving dispatch tests | API server |
| GR-PATCH-005 | `vllm_gr/engine/arg_utils_gr.py` | `patch_add_cli_args` | required frontend plan finds GR EngineArgs methods absent | `_validate_engine_args_patch`; attention backend configuration tests | main, API server |
| GR-PATCH-007 | `v1/engine/core_client.py`, `core_client_patch.py`, `async_llm.py` | `beam_search_patch.patch_batch_and_fork` | required frontend plan finds V1 client/lifecycle methods incomplete | `_validate_batch_and_fork_patch`; V1 session and admission tests | API server |
| GR-PATCH-008 | `vllm_gr/v1/engine/wire.py` | `register_engine_core_request_types` | frontend and EngineCore require GR wire values | `_validate_request_types_patch`; EngineCore protocol contracts | API server, EngineCore |
| GR-PATCH-009 | `v1/engine/core.py`, `engine_core_patch.py` | `apply_engine_core_request_patches` | EngineCore plan finds request dispatch incomplete | `_validate_engine_core_request_group`, `_validate_engine_core_entrypoint_patch`; V1 request and admission tests | EngineCore |
| GR-PATCH-010 | `v1/engine/engine_core_patch.py`, `gr_async_scheduler.py` | `apply_scheduler_patch` | EngineCore plan finds metadata propagation absent | `_validate_scheduler_patch`; EngineCore protocol contracts | EngineCore |
| GR-PATCH-011 | `v1/worker/gpu_model_runner_patch.py` | `model_runner_patch.apply_worker_patches` | active GPU runner hooks are incomplete | `_validate_model_runner_patch`; GPU and dispatcher tests | GPU worker |
| GR-PATCH-012 | `v1/worker/npu_model_runner_patch.py` | `model_runner_patch.apply_worker_patches` | active NPU runner hooks are incomplete | `_validate_model_runner_patch`; NPU and dispatcher tests | NPU worker |
| GR-PATCH-013 | `v1/attention/backends/beam_attn.py` | `register_backend` decorator | facade import and CUSTOM selection | Beam Attention platform, metadata, and configuration tests | EngineCore, worker |
| GR-PATCH-014 | bundled LMCache adapter | `lmcache_patch.apply_lmcache_adapter` | LMCache environment flag or connector argument | installer result; optional patch-count test | worker |
| GR-PATCH-015 | `vllm_gr/ops/sliced_logits.py` | `sliced_logits_patch.apply_sliced_logits_patch` | hooks install at startup; valid slice enables execution | installer result; sliced-logits and optional patch-count tests | worker |
| GR-PATCH-016 | `vllm/utils/system_utils.py` | `npu_spawn_patch.apply_npu_spawn_patch` | engine config loading via the vllm.general_plugins entry point | installer result; spawn patch tests | main, EngineCore |
| GR-PATCH-017 | `v1/worker/npu_memory_counters.py` | `install_npu_memory_counter_patch` | NPU worker initialization finds the supported native sampler and nested memory API | `tests/test_npu_memory_counters.py`; NPU worker lifecycle test | NPU worker |

### GR-PATCH-000 — Runtime Plugin Activation

- Source: `vllm_gr/plugin.py`
- Entry point: `vllm_gr.plugin:register`
- Process roles: main process, EngineCore subprocess, worker subprocess
- Platforms: GPU and NPU
- Risk: High
- Behavior: initializes the current process and applies the active GR runtime patches.
- Main dependency: vLLM general plugins are loaded in every relevant process.
- Removal condition: no removal is expected while vLLM-GR requires runtime patch installation. The entry point may be simplified when patch installation becomes more narrowly scoped.

### GR-PATCH-001 — CUSTOM Attention Selector Compatibility

- Source: `vllm_gr/v1/ascend_compat.py`
- Orchestration: `vllm_gr/patch.py`
- Target: `vllm.v1.attention.selector._cached_get_attn_backend`
- Mutation: private cached-function replacement
- Platform: primarily NPU
- Risk: Critical
- Behavior: restores the GR Beam Attention backend when vLLM-Ascend clears explicit CUSTOM backend intent.
- Dependencies:
    - `_cached_get_attn_backend` signature and cache behavior;
    - the current vLLM configuration context;
    - KV cache layout setup;
    - CUSTOM backend registration.
- Ordinary request impact: the selector is process-global, although the GR branch is used only for explicit Beam backend intent.
- Removal condition: vLLM-Ascend preserves explicit CUSTOM backend intent and resolves the registered backend without a private selector override.

### GR-PATCH-002 — NPUWorker Initialization Hook

- Source: `vllm_gr/v1/ascend_compat.py`
- Orchestration: `vllm_gr/patch.py`
- Target: `vllm_ascend.worker.worker.NPUWorker.init_device`
- Mutation: lifecycle method wrapper
- Platform: NPU
- Risk: High
- Behavior: ensures required compatibility patches are active before the NPU ModelRunner resolves its Attention backend.
- Dependency: NPU ModelRunner construction occurs during `init_device()`.
- Removal condition: the required Ascend compatibility is installed through a stable worker or platform extension point before ModelRunner construction.

### GR-PATCH-003 — FlatLogprobs Fast Append Override

- Source: `vllm_gr/logprobs_patch.py`
- Target: `vllm.logprobs.FlatLogprobs.append_fast`
- Mutation: existing method replacement
- Platforms: GPU and NPU
- Risk: Medium
- Behavior: replaces the upstream implementation with the GR fast append implementation.
- Required compatibility:
    - identical call signature;
    - identical index, token, logprob, rank, and decoded-token behavior;
    - identical handling of empty and iterable inputs.
- Removal condition: GR no longer needs a custom implementation or the required implementation is available upstream.

### GR-PATCH-004 — OpenAI Beam Serving Override

- Targets:
    - `OpenAIServing.beam_search`
    - `ChatCompletionRequest.to_beam_search_params`
- Mutation: existing method replacement
- Process role: API server
- Risk: High
- Behavior: replaces the upstream Beam serving flow with the grouped-beam vLLM-GR path.
- Compatibility-sensitive behavior:
    - request grouping;
    - logprob handling;
    - EOS and length-penalty behavior;
    - request identifiers;
    - output reconstruction.
- Removal condition: the serving path can use shared GR Beam helpers without replacing the upstream methods, or upstream provides the required grouped-beam extension point.

### GR-PATCH-005 — EngineArgs Configuration Override

- Targets:
    - `EngineArgs.add_cli_args`
    - `EngineArgs.from_cli_args`
    - `EngineArgs.create_engine_config`
- Mutation: method replacement and GR configuration propagation
- Risk: High
- Behavior:
    - registers active GR CLI options;
    - writes GR values into `VllmConfig.additional_config`;
    - applies active NPU compatibility defaults.
- Descriptor assumptions:
    - `add_cli_args` is a `staticmethod`;
    - `from_cli_args` is a `classmethod`;
    - `create_engine_config` is an instance method.
- Removal condition: vLLM provides stable configuration extension hooks that cover all active GR values.

### GR-PATCH-007 — EngineCore Client and AsyncLLM Extensions

- Targets: GR extensions on `AsyncMPClient`, `SyncMPClient`, `InprocClient`, and `AsyncLLM`
- Risk: Critical
- Behavior: atomically submits V1 sessions, installs submit/wait/abort/release
  frontend methods, and preserves admission rejection identities.
- Private/internal dependencies: client `_send_input` / `_send_input_message`,
  native output collectors, and the EngineCore utility channel for retirement.
- `patch_batch_and_fork` is a retained installer name for this V1 pipeline;
  it does not restore the retired frontend batch/fork protocol.
- Removal condition: upstream exposes equivalent session submission and lifecycle hooks.

### GR-PATCH-008 — EngineCoreRequestType Extension

- Target: `EngineCoreRequestType`
- Mutation: Python Enum private-state mutation
- Added values:
    - `SUBMIT_GR_SESSION = b"\x0a"`
- Risk: Critical
- Current compatibility: upstream vLLM 0.22.1 uses values `0x00` through `0x05`.
- Private dependencies:
    - `_value2member_map_`;
    - `_member_map_`;
    - `_member_names_`.
- Removal condition: grouped Beam messages can be carried through a stable upstream extension mechanism without mutating Enum internals.

### GR-PATCH-009 — EngineCore Lifecycle and Socket Processing

- Targets:
    - `EngineCore.__init__`
    - `EngineCoreProc.__init__`
    - `EngineCoreProc.process_input_sockets`
    - `EngineCoreProc.run_engine_core`
    - GR Beam request handlers
- Risk: Critical
- Behavior: integrates grouped Beam request processing into EngineCore.
- Compatibility-sensitive upstream behavior includes:
    - ZMQ socket construction;
    - READY and coordinator handshakes;
    - Tensor IPC setup;
    - Msgpack decoder setup;
    - ADD and ABORT processing;
    - input queue payload layout.
- Removal condition: vLLM exposes a stable control-message or request-dispatch extension point that avoids replacing the complete socket loop.

### GR-PATCH-010 — Scheduler Beam Override

- Target: `Scheduler.schedule`
- Mutation: scheduling hot-path wrapper or replacement
- Risk: Critical
- Behavior:
    - adapts KV cache lookup for grouped Beam requests;
    - builds Beam scheduling metadata;
    - attaches GR metadata to the scheduler result.
- Dependencies:
    - `Scheduler.requests`;
    - `Scheduler.kv_cache_manager`;
    - `num_scheduled_tokens`;
    - the ability to attach metadata to `SchedulerOutput`.
- Removal condition: Beam metadata can be carried through an explicit upstream scheduler extension or wrapper without replacing `schedule()` or adding undeclared attributes.

### GR-PATCH-011 — GPU ModelRunner Override

- Source: `vllm_gr/v1/worker/gpu_model_runner_patch.py`
- Shared helpers: `vllm_gr/v1/worker/model_runner_common.py`
- Installation and verification: `vllm_gr/v1/worker/model_runner_patch.py`
- Targets:
    - `GPUModelRunner.execute_model`
    - `GPUModelRunner._prepare_inputs`
    - `GPUModelRunner.sample_tokens`
- Platform: GPU
- Risk: Critical
- Dependencies:
    - private method signatures;
    - the 2-item `_prepare_inputs` result;
    - native execute/sample separation and asynchronous output consumption;
    - InputBatch, SamplingMetadata, request-row mapping, and logprob tensor internals.
- Removal condition: upstream provides stable hooks for grouped input preparation and output collapsing.

### GR-PATCH-012 — NPU ModelRunner Override

- Source: `vllm_gr/v1/worker/npu_model_runner_patch.py`
- Shared helpers: `vllm_gr/v1/worker/model_runner_common.py`
- Installation and verification: `vllm_gr/v1/worker/model_runner_patch.py`
- Targets:
    - `NPUModelRunner.execute_model`
    - `NPUModelRunner._prepare_inputs`
    - `NPUModelRunner._bookkeeping_sync`
- Platform: NPU
- Risk: Critical
- Dependencies:
    - NPU-specific private method signatures;
    - the 4-item `_prepare_inputs` result;
    - the 6-item `_bookkeeping_sync` result;
    - speculative decoding and token-compression state.
- Removal condition: vLLM-Ascend exposes stable grouped input and output adaptation hooks.

### GR-PATCH-013 — CUSTOM Attention Backend Registration

- Mechanism: `register_backend(AttentionBackendEnum.CUSTOM)`
- Mutation: supported process-global registry override
- Risk: High
- Behavior: registers the GR Beam Attention backend in the official CUSTOM slot.
- Ownership:
    - Beam Attention metadata and builder:
      `vllm_gr.v1.attention.backends.beam_attn_metadata`;
    - GPU execution:
      `vllm_gr.v1.attention.backends.beam_attn_gpu`;
    - NPU execution:
      `vllm_gr.v1.attention.backends.beam_attn_npu`;
    - backend registration and facade:
      `vllm_gr.v1.attention.backends.beam_attn`.
- Compatibility facade:
    - `beam_attn` intentionally continues to export
      `BeamAttentionMetadata`, `BeamAttentionMetadataBuilder`,
      `BEAM_DATA_VAR`, and `_compute_beam_bounds`;
    - new internal code should import those symbols from their owner modules.
- Ownership risk:
    - CUSTOM is one shared process-global slot;
    - later registration may replace an earlier owner;
    - plugin loading order can affect the final registration.
- Removal condition: vLLM supports named third-party backend identities or another non-conflicting registration mechanism.

### GR-PATCH-014 — LMCache Integration Override

- Source: `vllm_gr/3rd_party/lmcache/integration/vllm/vllm_v1_adapter.py`
- Installer: `vllm_gr/lmcache_patch.py`
- Orchestration: `vllm_gr/patch.py`
- Targets:
    - `RequestTracker.from_new_request`
    - `ReqMeta.from_request_tracker`
    - `ReqMeta.__annotations__`
    - `LMCacheConnectorV1Impl.start_load_kv`
    - `save_kv_layer`
    - `wait_for_save`
- Risk: Critical
- Activation: optional; enabled only when the LMCache integration is requested.
- Dependencies include tracker lifecycle, connector parent state, connector metadata requests, slot mappings, and load/save semantics.
- Removal condition: LMCache exposes the required grouped Beam metadata and lifecycle extension points directly.

### GR-PATCH-015 — Sliced Vocabulary Projection

- Source: `vllm_gr/ops/sliced_logits.py`
- Installer: `vllm_gr/ops/sliced_logits_patch.py`
- Orchestration: `vllm_gr/patch.py`
- Targets:
    - `LogitsProcessor.__init__`
    - `LogitsProcessor._get_logits`
    - `Sampler.gather_logprobs`
- Risk: Medium
- Activation: hooks are installed during runtime initialization; sliced projection is
  used only when a valid vocabulary slice is configured.
- Removal condition: vLLM exposes stable output-projection and logprob-width extension points.

### GR-PATCH-016 — NPU-aware Spawn Start Method

- Source / Installer: `vllm_gr/npu_spawn_patch.py`
- Orchestration: `vllm_gr/patch.py`
- Targets:
    - `vllm.utils.system_utils._maybe_force_spawn` (module attribute)
    - `vllm.v1.executor.multiproc_executor._maybe_force_spawn` (from-import binding)
- Risk: Medium
- Activation: installed when engine configuration loads the vllm.general_plugins
  entry point; the NPU branch fires only when `torch.npu.is_initialized()` is true
  in the process that is about to create worker subprocesses.
- Removal condition: upstream `_maybe_force_spawn()` covers NPU (or delegates the
  decision to the platform interface).

### GR-PATCH-017 — NPU Worker Memory Counters

- Source: `vllm_gr/v1/worker/npu_memory_counters.py`
- Installer: `install_npu_memory_counter_patch`
- Orchestration: `vllm_gr/v1/ascend_compat.py::patch_npu_worker_init_device`
- Target: `NPUWorker.profile_memory` in `vllm_ascend.worker.worker` or
  `vllm_ascend.worker.worker_v1`
- Mutation: native method replacement during worker device initialization
- Platform / process role: NPU worker
- Risk: Medium
- Supported runtime: `vllm-ascend==0.22.1rc1`, `torch-npu==2.10.0`.
- Native contract: `profile_memory(self)` sets `torch_reserved` and
  `torch_allocated` from the current device allocator counters, emits the native
  debug log, and returns `None`.
- Behavior: reads the public `torch_npu.npu.memory.memory_stats_as_nested_dict()`
  once and selects `reserved_bytes.all.current` and
  `allocated_bytes.all.current`, preserving native zero defaults for an
  uninitialized allocator or missing counters. Allocator errors propagate.
- Installation guards: only the two native module paths and the self-only
  signature are accepted; custom methods, existing wrappers, missing native
  `logger.debug`, and unavailable or non-callable nested memory APIs keep their
  original method. Repeated installation is idempotent.
- Compatibility fallback: a malformed nested schema or non-integer counter
  calls the original sampler. A future native implementation with the same
  module and signature still requires review of the two-counter contract.
- Validation:
    - `tests/test_npu_memory_counters.py::test_one_snapshot_preserves_counters_and_original_logger`;
    - `tests/test_npu_memory_counters.py::test_uninitialized_and_missing_counters_match_native_zero_defaults`;
    - `tests/test_npu_memory_counters.py::test_each_call_refreshes_fields_and_installation_is_idempotent`;
    - `tests/test_npu_memory_counters.py::test_missing_nested_api_keeps_original_method`;
    - `tests/test_npu_memory_counters.py::test_unknown_worker_profiles_keep_original_semantics`;
    - `tests/test_npu_memory_counters.py::test_changed_nested_schema_falls_back_to_native`;
    - `tests/test_npu_memory_counters.py::test_allocator_errors_are_propagated`;
    - `tests/test_npu_memory_counters.py::test_module_import_does_not_load_npu_runtime`;
    - `tests/test_ascend_compat.py::test_npu_worker_patch_is_lazy_and_preserves_lifecycle_contract`.
- Removal condition: vLLM-Ascend reads the two allocator counters without
  flattening full memory statistics.

## Benchmark-only Runtime Mutations

`benchmarks/open_one_rec/one_rec_patch.py` modifies benchmark request and serving helpers at runtime. These mutations are benchmark-only and must not be treated as production runtime patches.

They should remain isolated from production initialization and should be removed when the benchmark no longer requires compatibility shims.

## Shared Implicit Contracts

The current runtime also relies on undeclared attributes and string-based configuration.

### Dynamic attributes

```text
EngineArgs.vllm_gr_config
SchedulerOutput.gr_stage_metadata
SchedulerOutput.gr_retire_metadata
Runner._gr_gpu_stage_runner
EngineCoreOutput.kv_transfer_params["gr_batch_result"]
```

Device-specific compatibility hooks also retain native request/row metadata.
The removed frontend Catalog object and per-step BeamRequest protocol are not
part of the current frontend contract.

### Configuration keys

```text
additional_config["vllm_gr_catalog_path"]
additional_config["vllm_gr_attention_backend"]
VLLM_GR_LMCACHE_PATCH
VLLM_USE_V2_MODEL_RUNNER
```

These contracts must be updated whenever the corresponding runtime behavior changes. Dynamic attributes and private upstream dependencies should be removed only when their replacement is already implemented and verified.

## Maintenance Rules

When an active runtime patch changes:

1. update this document and `docs/patch_inventory.yaml`;
2. update or add a lightweight contract test for critical upstream assumptions;
3. keep the exact supported upstream version explicit;
4. record an explicit removal condition for compatibility code expected to be temporary;
5. avoid adding historical investigation notes or work-package planning to this document.
