# Runtime Patch and Upstream Dependency Inventory

This document records the runtime patches currently applied by `vllm-gr`, the upstream APIs and internal state they depend on, and the conditions under which those patches may be removed.

It is intended to remain valid beyond a single refactoring phase or pull request. Historical investigation notes, temporary implementation plans, and change-by-change corrections belong in pull request descriptions and issue discussions.

## Historical Baseline and Upgrade Scope

The recorded pre-upgrade runtime baseline is:

- `vllm==0.22.1`
- `vllm-ascend==0.22.1rc1` for the NPU path

The upstream source revisions used for compatibility analysis are:

- `vllm-project/vllm@releases/v0.22.1`
    - commit: `dd72658e7db0c6a674f473f6f9f0f5c2ebe7e523`
- `vllm-project/vllm-ascend@releases/v0.22.1rc`
    - commit: `f528cdd561969730504bfe8b5e96b6b7e69d6c4f`

The vLLM-Ascend release branch is movable, so compatibility conclusions must retain the exact source commit.

The current CUDA release candidate targets vLLM 0.30.0 and has completed GPU
functional, packaging, and performance validation. Its release-level precision
decision remains open. The Ascend 0.30.0rc2 baseline at
`43bc85f9b375281452d0af7005dc9fa752f43884` completed NPU validation
(unit, integration, eager/graph end-to-end, dynamic beam, prefill graphs, and
benchmark gates). The subsequent config-filter refactor has host regression
coverage and still requires NPU startup and inference revalidation.

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
| GR-PATCH-001 | `vllm_gr/v1/ascend_compat.py` | `apply_ascend_attention_backend_patch` | vLLM attention selector is importable and unwrapped | `_validate` behavior in `test_ascend_compat.py`; Ascend integration test | EngineCore, worker |
| GR-PATCH-002 | `vllm_gr/v1/ascend_compat.py` | `patch_npu_worker_init_device` | active NPU platform, first installation | `test_npu_worker_patch_is_lazy_and_preserves_lifecycle_contract` | worker |
| GR-PATCH-003 | `vllm_gr/logprobs.py` | `logprobs_patch.patch_flat_logprobs` | required frontend plan finds the method absent | `_validate_flat_logprobs_patch`; patch initialization tests | offline frontend, API server |
| GR-PATCH-004 | `entrypoints/openai/serving_engine.py`, `protocol.py` | `beam_search_patch.patch_beam_search`, `patch_sampling` | required frontend plan finds either method absent | `_validate_openai_beam_search_patch`, `_validate_chat_sampling_patch`; serving dispatch, online entry, and OneRec HTTP tests | API server |
| GR-PATCH-005 | `vllm_gr/engine/arg_utils_gr.py` | `patch_add_cli_args` | required frontend plan finds GR EngineArgs methods absent | `_validate_engine_args_patch`; attention backend configuration tests | main, API server |
| GR-PATCH-007 | `v1/engine/core_client.py`, `core_client_patch.py`, `async_llm.py` | `beam_search_patch.patch_batch_and_fork` | required frontend plan finds V1 client/lifecycle methods incomplete | `_validate_batch_and_fork_patch`; V1 request, admission, rejection, and online lifecycle tests | API server |
| GR-PATCH-008 | `vllm_gr/v1/engine/wire.py` | `register_engine_core_request_types` | frontend and EngineCore require GR wire values | `_validate_request_types_patch`; EngineCore protocol contracts | API server, EngineCore |
| GR-PATCH-009 | `v1/engine/core.py`, `engine_core_patch.py`, `hot_update.py` | `apply_engine_core_request_patches` | Shared EngineCore plan, including NPU child entry, finds request dispatch or Trie-capacity initialization incomplete; utilities do not depend on the management listener | request validators; V1 admission, NPU multi-Trie routing, hot-update utility contracts and management tests | EngineCore |
| GR-PATCH-010 | `v1/engine/engine_core_patch.py`, `gr_async_scheduler.py` | `apply_scheduler_patch` | EngineCore plan finds metadata propagation absent | `_validate_scheduler_patch`; EngineCore protocol contracts | EngineCore |
| GR-PATCH-011 | GPU V1/V2 runner patches; `worker/hot_update.py` | `model_runner_patch.apply_worker_patches` | selected CUDA runner hooks are incomplete; common installer mounts Worker utilities | runner validator; GPU, utility contract, reload and lifecycle tests | GPU worker |
| GR-PATCH-012 | NPU runner patch; `worker/hot_update.py`, `ascend_compat.py` | `model_runner_patch.apply_worker_patches`; delayed `NPUWorker.init_device` | active NPU runner hooks are incomplete; delayed initialization mounts Worker utilities | NPU runner, delayed-install and unsupported-reload guard tests | NPU worker |
| GR-PATCH-013 | `v1/attention/backends/beam_attn.py` | `register_backend` decorator | facade import and CUSTOM selection | Beam Attention platform, metadata, and configuration tests | EngineCore, worker |
| GR-PATCH-015 | `vllm_gr/ops/sliced_logits.py` | `sliced_logits_patch.apply_sliced_logits_patch` | hooks install at startup; valid slice enables execution | installer result; sliced-logits and optional patch-count tests | worker |
| GR-PATCH-016 | `vllm_gr/npu_spawn_patch.py` | `vllm_gr.npu_spawn_patch.apply_npu_spawn_patch` | engine config loading via the vllm.general_plugins entry point | installer result; spawn patch tests | main, EngineCore |
| GR-PATCH-018 | `vllm_gr/v1/ascend_compat.py` | `apply_penalties_warmup_oom_guard` | NPUWorker `init_device` runs and the native penalties warmup is present | installer result; NPU startup at `--max_num_seqs 2048` | NPU worker |
| GR-PATCH-019 | `vllm_gr/v1/ascend_compat.py` | `apply_engine_core_config_sanitizer`, `patch_npu_worker_init_device` | NPU engine configuration and worker plugin initialization | Ascend config preservation and initialization tests; NPU startup | main, EngineCore, worker |
| GR-PATCH-020 | `vllm_gr/pangu_compat.py` | `apply_pangu_tied_embeddings_patch` | runtime initialization; tied Pangu models only | Pangu compatibility and patch initialization tests | main, EngineCore, worker |
| GR-PATCH-021 | `vllm_gr/v1/npu_kv_layout.py` | `apply_npu_contiguous_kv_cache_patch` | known Ascend 0.30.0rc2 policy, before native KV allocation | NPU KV layout, worker lifecycle and row-view tests | NPU worker |

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
- Behavior: resolves the GR Beam Attention backend when CUSTOM is selected
  explicitly or restored from the serialized GR marker.
- Dependencies:
    - `_cached_get_attn_backend` signature and cache behavior;
    - the current vLLM configuration context;
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
    - `GenerateBaseServing.beam_search` and `beam_search_v1`
    - `ChatCompletionRequest.to_beam_search_params` and `CompletionRequest.to_beam_search_params`
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
- Current compatibility: upstream vLLM 0.30.0 uses values `0x00` through `0x05`.
- Private dependencies:
    - `_value2member_map_`;
    - `_member_map_`;
    - `_member_names_`.
- Removal condition: grouped Beam messages can be carried through a stable upstream extension mechanism without mutating Enum internals.

### GR-PATCH-009 — EngineCore Lifecycle and Socket Processing

- Sources: `vllm_gr/v1/engine/core.py`, `engine_core_patch.py`, `hot_update.py`
- Targets:
    - `EngineCore.__init__`
    - `EngineCoreProc.__init__`
    - `EngineCoreProc.process_input_sockets`
    - `EngineCoreProc.run_engine_core`
    - GR Beam request handlers
    - `EngineCore.service_status`, `pause`, `resume`, `reload_trie`, inherited by
      `EngineCoreProc`
- Risk: Critical
- Behavior: integrates grouped Beam request processing and named control
  utilities into EngineCore. `apply_engine_core_request_patches` calls
  `install_core_hot_update_hooks` even when the optional management listener
  is disabled; listener configuration controls external exposure separately.
  The shared child plan installs the live Trie-capacity initialization wrapper
  before construction. The sanitized NPU process entry invokes this plan too;
  installing it later from Worker initialization would miss the current
  EngineCore constructor and leave request admission without a Trie count.
- Control behavior:
    - `pause` delegates to native `pause_scheduler(mode="wait", clear_cache=True)`
      and preserves its drain Future; `resume` delegates to `resume_scheduler`;
    - `service_status` reports scheduler state and resource metadata; the
      management listener owns the checkpoint reload validity/version latch;
    - `reload_trie` calls the named Worker RPC and publishes the Trie version
      only after success; it requires asynchronous V1 CUDA execution and an
      enabled canonical table;
    - checkpoint loading reuses native `reload_weights` via collective RPC;
      these hooks do not install a replacement Core weight loader.
- Compatibility-sensitive upstream behavior includes:
    - ZMQ socket construction;
    - READY and coordinator handshakes;
    - Tensor IPC setup;
    - Msgpack decoder setup;
    - ADD and ABORT processing;
    - input queue payload layout.
- Contract coverage: `tests/contract/test_upstream_patch_contracts.py::test_core_hot_update_utility_contracts`
  checks named installation, native pause/resume preservation, drain-Future
  forwarding and repeat installation. `tests/sidecar/test_management.py`
  covers status, RPC failures, execution guards and listener/client integration.
  `tests/test_gr_async_scheduler.py::test_prefix_reset_after_drain_preserves_waiting_gr_sessions`
  covers the drained-reset boundary without discarding queued GR sessions.
- Removal condition: vLLM exposes stable control-message, request-dispatch and
  named management utility extension points.

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

### GR-PATCH-011 — GPU ModelRunner V1/V2 Overrides

Cross-batch stage-adapter caching is opt-in through
`beam.adapter_cache_enabled` (default false). The existing same-batch reuse
and Decode capture policy remain active when this cache is disabled.

- Sources:
    - `vllm_gr/v1/worker/gpu_model_runner_patch.py`
    - `vllm_gr/v1/worker/gpu_model_runner_v2_patch.py`
    - `vllm_gr/v1/worker/gpu_model_runner_v2_graph.py`
    - `vllm_gr/v1/worker/gpu_beam_stage_runner_v1.py` (`CudaStageHooksV1`)
    - `vllm_gr/v1/worker/gpu_beam_stage_runner_v2.py` (`CudaStageHooksV2`)
- Shared helpers: `vllm_gr/v1/worker/model_runner_common.py`, `vllm_gr/v1/worker/gpu_prefill_requests_v2.py`, `vllm_gr/v1/worker/hot_update.py`, `vllm_gr/constraints/versioned_table.py`, `vllm_gr/v1/worker/gpu_stage_hooks.py`, `vllm_gr/v1/worker/gpu_beam_stage_runner.py`
- Installation and verification: `vllm_gr/v1/worker/model_runner_patch.py`
- CUDA stage binding: execution and startup entrypoints explicitly select the
  independent V1/V2 implementations of `CudaStageHooks`. The shared core has no
  V1 fallback; graph policies remain version-specific. NPU retains its own
  adapter and lazy imports. `CudaStageHooks` is an abstract base class, so an
  incomplete implementation fails at construction on any device. Binding and
  startup regressions live in `tests/test_model_runner_stage_binding.py` and
  `tests/test_gr_startup.py`; `tests/tools/test_cuda_stage_typing.py` pins the
  abstract method set and both subclasses.
- ModelRunner V1 targets:
    - `GPUModelRunner.execute_model`
    - `GPUModelRunner._prepare_inputs`
    - `GPUModelRunner.sample_tokens`
- ModelRunner V2 targets:
    - `GPUModelRunner.execute_model`
    - `GPUModelRunner.prepare_inputs`
    - `GPUModelRunner.add_requests`
    - `GPUModelRunner.sample_tokens`
    - `GPUModelRunner.initialize_kv_cache`
    - `GPUModelRunner.profile_cudagraph_memory`
    - `GPUModelRunner.capture_model`
- Common Worker target: `vllm.v1.worker.gpu_worker.Worker.reload_trie`, installed
  by `install_worker_utility_hooks` through the active CUDA Worker installer.
  It delegates to `model_runner.constraint_backend.reload(path, version)` and
  leaves native `reload_weights` unchanged.
- Model-load lifecycle: the common `load_model` wrapper publishes a
  `VersionedConstraintTable` around an enabled canonical backend; the wrapper
  delegates normal access to the current table and releases expired versions
  on API access after device synchronization. Shutdown closes retained tables.
  These hooks run independently of the management listener. LM-head post-load
  cache refresh is tracked separately under GR-PATCH-015.
- Platform: GPU
- Risk: Critical
- ModelRunner V1 dependencies:
    - private method signatures;
    - the 3-item `_prepare_inputs` result (`logits_indices`, speculative-decode metadata, maximum sampled-token count);
    - native execute/sample separation and asynchronous output consumption;
    - InputBatch, SamplingMetadata, request-row mapping, and logprob tensor internals.
- ModelRunner V2 dependencies:
    - native `prepare_inputs`/`add_requests` request-state contracts;
    - admitted all-new GR Prefill batches skip unused native sampler registration;
    - fresh, complete Prefill batches on an empty request set fuse native state
      writes only when they own all queued writes and one compatible KV group;
    - pure GR Prefill input fusion preserves native row order, GPU token state,
      CPU mirrors and per-dispatch sampling indices; unsupported batches fall back;
    - instance-local CUDA graph manager and capture registry;
    - graph-memory profiling lifecycle and temporary KV-cache ownership.
- V2 Prefill fast-path validation: `tests/kernels/test_gr_prefill_v2.py` compares
  registration and fused inputs with native CUDA state, including reused slots,
  cached/chunked inputs, padding, pending-write fallback and failed submission
  without queue loss. `gpu_prefill_requests_v2.py` owns registration and the
  packed native-state write contract, separate from stage execution hooks.
- Hot-update coverage: `tests/test_model_runner_patch.py::test_active_cuda_worker_hot_update_utility_contract`
  checks active Worker installation, RPC dispatch, repeat installation and
  preservation of the native weight loader. `tests/sidecar/test_worker_update.py`
  covers table replacement/expiry and native cache reloads;
  `tests/v1/worker/test_constraint_lifecycle.py` covers load/shutdown ownership.
- Removal condition: upstream provides stable hooks for grouped input/output,
  Worker utilities and constraint-backend lifecycle.

### GR-PATCH-012 — NPU ModelRunner Override

- Source: `vllm_gr/v1/worker/npu_model_runner_patch.py`
- Shared helpers: `vllm_gr/v1/worker/model_runner_common.py`, `vllm_gr/v1/worker/hot_update.py`, `vllm_gr/constraints/versioned_table.py`, `vllm_gr/v1/ascend_compat.py`
- Installation and verification: `vllm_gr/v1/worker/model_runner_patch.py`
- Targets:
    - `NPUModelRunner.execute_model`
    - `NPUModelRunner._prepare_inputs`
    - `NPUModelRunner._bookkeeping_sync`
    - `vllm_ascend.worker.worker.NPUWorker.reload_trie`
- Utility activation: `ascend_compat.patch_npu_worker_init_device` installs a
  delayed wrapper; it calls `install_worker_utility_hooks(type(self))` before
  native device initialization. This is independent of the management listener.
  The common model-load wrapper also owns versioned canonical table lifecycle.
- Scope: registering `reload_trie` on the NPU Worker does not enable online NPU
  Trie updates. Core rejects that path before Worker RPC dispatch. Checkpoint
  loading continues to use the native weight loader. Common host-side contract
  coverage does not establish device-level NPU hot-update correctness.
- Platform: NPU
- Risk: Critical
- Dependencies:
    - NPU-specific private method signatures;
    - the 3-item `_prepare_inputs` result (`logits_indices`, speculative-decode metadata, total scheduled tokens);
    - the 8-item `_bookkeeping_sync` result, including async NaN counts and request-index snapshots;
    - speculative decoding and token-compression state.
- Hot-update coverage: `tests/test_ascend_compat.py::test_npu_worker_patch_is_lazy_and_preserves_lifecycle_contract`
  checks delayed installation on a simulated NPU Worker;
  `tests/sidecar/test_management.py::test_trie_rejects_unsupported_execution_before_worker_rpc`
  checks NPU rejection, and shared table/cache behavior is covered by
  `tests/sidecar/test_worker_update.py`.
- Removal condition: vLLM-Ascend exposes stable grouped input/output,
  Worker utility and constraint-backend lifecycle hooks.

- Optional BSND Prefill: `_model_forward` selects fixed `(T,B)` captures only for
  complete prompts without history when `VLLM_GR_NPU_PREFILL_BSND_CAPTURE_SPECS`
  is configured. Other requests retain TND/native fallback. Graph metadata owns
  scratch buffers; device capture errors propagate when this mode is enabled.
- Validation: `tests/test_npu_prefill_bsnd.py` and the explicitly scoped BSND
  constructor cases in `tests/test_prefill_graph_runner.py`.
- Configuration and limitations: [NPU BSND Prefill](npu_prefill_bsnd.md).

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

### GR-PATCH-015 — Sliced Vocabulary Projection

- Source: `vllm_gr/ops/sliced_logits.py`
- Reload helper: `vllm_gr/ops/lm_head_cache.py` installs post-load hooks for dense
  `ParallelLMHead` instances by type, including heads named `output` and aliases.
  Shared-weight owners retain their hooks; existing caches refresh in place.
  Coverage: `tests/sidecar/test_worker_update.py::test_sliced_cache_survives_native_layerwise_reload`.
- Installer: `vllm_gr/ops/sliced_logits_patch.py`
- Orchestration: `vllm_gr/patch.py`
- Targets:
    - `LogitsProcessor.__init__`
    - `LogitsProcessor._get_logits`
    - `Sampler.gather_logprobs`
    - `vllm.v1.worker.gpu.sample.logprob.compute_topk_scores` and its bindings
      in the V2 sampler and rejection sampler
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

### GR-PATCH-018 — Ascend Penalties Warmup OOM Guard

- Source / Installer: `vllm_gr/v1/ascend_compat.py`
- Orchestration: `vllm_gr/v1/ascend_compat.py::patch_npu_worker_init_device`
- Target: `vllm_ascend.model_executor.warmup.kernel_warmup.penalties_triton_warmup`
- Mutation: module attribute replacement
- Platform / process role: NPU worker
- Risk: Medium
- Behavior: wraps the native warmup so an NPU `OutOfMemoryError` skips the
  penalties JIT warmup instead of aborting engine startup, then empties the
  allocator cache and logs the skipped stage. Because vllm-ascend >= 0.30 runs
  the warmup after the KV cache has taken the memory budget, its
  `max_num_seqs x padded_vocab` dummy tensors exceed the remaining budget at the
  benchmark server's `--max_num_seqs 2048`. Requests without penalties — which
  includes every vllm-gr beam request — never execute those kernels, so only
  the JIT warmup is lost. Any other exception propagates unchanged.
- Installation guards: no-op when the module is absent, when the native callable
  is missing, or when the wrapper is already installed; installation is
  idempotent.
- Removal condition: vllm-ascend sizes the warmup dummies from the available
  NPU budget, or vllm-gr no longer runs a large vocabulary beside the KV cache.

### GR-PATCH-019 — Ascend additional_config Sanitizer

- Source / Installer: `vllm_gr/v1/ascend_compat.py`
- Orchestration: `vllm_gr/engine/arg_utils_gr.py::create_engine_config`
  and worker plugin initialization through `vllm_gr/patch.py::_apply_patches`
- Targets:
    - `vllm.v1.engine.core.EngineCoreProc.run_engine_core` (process-global
      staticmethod replacement)
    - `vllm_ascend.ascend_config.init_ascend_config` (function replacement)
- Mutation: entrypoint wrapper plus function wrapper
- Platform / process role: NPU; main, EngineCore, worker
- Risk: High
- Behavior: installs a filter and initializes the shared EngineCore-only patch
  plan before delegating to vllm-ascend's own entrypoint. This keeps the native
  Ascend startup chain while installing GR request admission, including live
  Trie-capacity discovery, before EngineCore construction. During each
  synchronous `init_ascend_config` call, the canonical
  `vllm_gr` key and legacy `vllm_gr_*` keys are excluded from its validation
  mapping. The original `additional_config` mapping is restored in `finally`,
  including on failure. The native config cache retains the original
  `VllmConfig` identity; GR parsing and process serialization retain all GR
  parameters without process-global configuration storage. vllm-ascend's
  `AscendConfig` uses pydantic `extra="forbid"` and is initialised in the
  EngineCore child before the vllm-gr plugin loads. Separate worker processes
  load plugins before `NPUWorker.__init__`, which also validates this config.
  The worker hook installs the function wrapper at plugin load time so fresh
  workers do not depend on inheriting EngineCore's patched modules.
- Installation guards: NPU platform only; wrapper installation is idempotent,
  and the entrypoint is re-asserted by identity check on every
  `create_engine_config` because each `EngineArgs` round-trip reloads the
  general plugins and reinstalls their wrappers.
- Removal condition: vllm-ascend's `AscendConfig` accepts unknown
  `additional_config` keys, or vllm-gr stops carrying its configuration there.

### GR-PATCH-020 — Pangu Tied Embeddings

- Source / Installer: `vllm_gr/pangu_compat.py::apply_pangu_tied_embeddings_patch`
- Orchestration: `vllm_gr/patch.py::_apply_patches`
- Target: `vllm.config.ModelConfig.maybe_untie_word_embeddings`
- Platforms / process roles: GPU and NPU; main, EngineCore, worker
- Risk: High
- Behavior: preserves tied embeddings for `PanguEmbeddedForCausalLM` when the
  checkpoint config declares `tie_word_embeddings`. The official checkpoint's
  zero-filled `lm_head.weight` is a placeholder; treating it as an independent
  head produces zero logits. Other architectures and untied Pangu models keep
  the native loader decision.
- Validation: `tests/test_pangu_compat.py` covers model scope and idempotence;
  `tests/test_patch_initialization.py` checks installation order.
- Removal condition: upstream distinguishes the placeholder from independent
  projection weights while loading the official Pangu checkpoint.

### GR-PATCH-021 — Contiguous Beam Prefix KV

- Source / Installer: `vllm_gr/v1/npu_kv_layout.py::apply_npu_contiguous_kv_cache_patch`
- Orchestration: the NPU worker `init_device` hook, before native runner construction.
- Targets: `vllm_ascend.attention.utils.requires_contiguous_pa_kv_cache` and its
  already imported `vllm_ascend.worker.model_runner_v1` binding.
- Platforms / process roles: NPU worker; risk: High.
- Behavior: extends the native allocation policy for explicitly eligible
  CUSTOM V1 xllm Beam layers with ordinary, unpadded FP16/BF16 KV. Native
  allocation then owns separate dense K/V buffers. The xllm execution module
  verifies the actual NHD shape, strides, device and format before reading
  those buffers through the original paged block table. It never reinterprets
  already interleaved storage; unproven layouts retain the gather path.
- Guards: exact supported vendor version and helper signature; both aliases
  must identify the same original or this wrapper. Other platforms, unknown
  bindings, ordinary attention, quantized/hybrid/shared/sliding/transfer
  configurations and unqualified vendor options retain the native policy.
  The six known Ascend compiler defaults are accepted when injected by the
  supported runtime; unknown keys and nondefault values still fall back.
  The ordinary native `FullAttentionSpec` has no `cache_dtype_str` field;
  the cache-config dtype and actual spec dtype remain mandatory guards.
  Eligibility is evaluated per engine/layer; no engine configuration is cached.
- Validation: policy controls in `tests/test_npu_kv_layout.py`, installation
  order in `tests/test_ascend_compat.py`, and direct/fallback routing plus native
  xllm eager/FULL replay checks in `tests/test_npu_beam_row_views.py`.
- Removal condition: a native backend contiguous-KV capability covers GR xllm
  without this version-specific policy bridge.

## Benchmark-only Runtime Mutations

`benchmarks/open_one_rec/one_rec_patch.py` modifies benchmark request and serving helpers at runtime. These mutations are benchmark-only and must not be treated as production runtime patches.

They should remain isolated from production initialization and should be removed when the benchmark no longer requires compatibility shims.

## Shared Implicit Contracts

The following implicit contracts are shared with the upstream runtime; they
were re-verified during the 0.30 upgrade.

### Dynamic attributes

```text
EngineArgs.catalog_path
EngineArgs._vllm_gr_config_input
SchedulerOutput.beam_data
SchedulerOutput._gr_request_row_configs
SchedulerOutput.gr_stage_metadata
SchedulerOutput.gr_retire_metadata
Runner._current_beam_data
Runner._current_beam_info
Runner._beam_logprobs_cache
Runner._gr_gpu_stage_runner
Runner._vllm_gr_config
EngineCoreOutput.kv_transfer_params["gr_batch_result"]
```

Device-specific compatibility hooks also retain native request/row metadata.
The removed frontend Catalog object and per-step BeamRequest protocol are not
part of the current frontend contract.

### Configuration keys

```text
additional_config["vllm_gr_catalog_path"]
additional_config["vllm_gr_attention_backend"]
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

## Retired Patches

**GR-PATCH-006 — OpenAIServingModels Constructor Override** is retired. The
Catalog serving installer, validator, and legacy tests have been removed. The
canonical V1 constraint artifact loads in the worker, covered by
`test_canonical_worker_enables_vocab_slice_before_first_forward` and
`tests/test_gr_config_v1.py`. The ID is retained in the YAML `retired_patches`
section and must not be counted as an active installer.

**GR-PATCH-014 — LMCache Integration Override** is retired. The experimental
integration had no supported version pin or vLLM 0.30 validation. Its adapter,
installer, CLI activation, and usage guide have been removed. The ID is retained
in the YAML `retired_patches` section and must not be counted as an active installer.

**GR-PATCH-017 — NPU Worker Memory Counters** is retired. Its removal
condition was met by the 0.30 upgrade: vllm-ascend 0.30.0rc2 replaced
`NPUWorker.profile_memory` with `log_memory_stats`, which reads the two
allocator counters directly instead of flattening a full statistics snapshot,
and the installer found no native method to wrap on that runtime. The
installer, its module, and its dedicated tests were removed; the NPU worker
lifecycle contract remains covered by
`tests/test_ascend_compat.py::test_npu_worker_patch_is_lazy_and_preserves_lifecycle_contract`.
The ID is retained in the YAML `retired_patches` section and must not be
counted as an active installer.
