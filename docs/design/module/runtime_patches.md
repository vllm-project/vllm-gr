# Runtime patches and upstream dependencies

Three files describe the same patch set:

| File | Role |
| --- | --- |
| [patch_inventory.yaml](../patch_inventory.yaml) | Authoritative data: installer, target, process roles, risk, tests and removal condition. |
| [Runtime patch inventory](../runtime_patch_inventory.md) | Checked index: active IDs in contract-test order, scope, backend ownership, owner, installer and risk. |
| This page | Rationale: the behavior each hook protects and the upstream assumption it depends on. |

Backend ownership identifies which provider declares the patch. Installation
can still happen later in a worker or during native configuration.

## GR-PATCH-000 — Runtime Plugin Activation

Scope: `core`; backend ID: `null`

The general plugin sets ModelRunner defaults, activates the current provider,
and validates its environment before passing that provider into patch
installation. The frontend and EngineCore use their own ordered plans. Each
patch module verifies its bindings so repeated initialization can reuse an
existing wrapper.

## GR-PATCH-001 — CUSTOM Attention Selector Compatibility

Scope: `core`; backend ID: `null`

An explicit CUSTOM choice, including one restored from serialized GR
configuration, must reach the registered Beam backend. Changes to the private
attention selector or its cache can silently select a different backend.

## GR-PATCH-002 — NPUWorker Initialization Hook

Scope: `backend`; backend ID: `ascend`

The [NPU startup hooks](../../../vllm_gr/v1/worker/npu/startup_patch.py) install
worker utilities before native device initialization constructs the runner.
An early import can precede worker availability, so later initialization
retries the installation. The platform configuration filter can run separately.

## GR-PATCH-003 — FlatLogprobs Fast Append Override

Scope: `core`; backend ID: `null`

The frontend appends flattened log probabilities through the GR implementation
of the native fast-append method in
[logprobs_patch.py](../../../vllm_gr/v1/engine/logprobs_patch.py).
Its arguments and the upstream flattened representation must stay compatible.
The patch test checks offsets, ranks, and decoded tokens for empty and nonempty
entries and omitted optional columns.

## GR-PATCH-004 — OpenAI Beam Serving Override

Scope: `core`; backend ID: `null`

The serving frontend groups prompts into one Beam submission and formats their
ranked candidates for the OpenAI-compatible API. The wrapper depends on the
native serving and protocol methods it replaces. The patch validates the Chat
and Completion sampling methods and the Completion `vllm_xargs` schema for
integer lists. A missing binding or a mismatched schema triggers reinstallation.

## GR-PATCH-005 — EngineArgs Configuration Override

Scope: `core`; backend ID: `null`

EngineArgs registers GR options and serializes the resolved settings for
explicitly selected GR engines. Its wrappers preserve native descriptor types
and explicit caller settings when constructing the engine configuration.

## GR-PATCH-007 — EngineCore Client and AsyncLLM Extensions

Scope: `core`; backend ID: `null`

Client extensions submit a whole Beam session and route its wait, abort and
release operations. A rejected submission must preserve the identity that the
frontend uses to complete the request.

## GR-PATCH-008 — EngineCoreRequestType Extension

Scope: `core`; backend ID: `null`

GR adds a wire discriminator for grouped session submission to the native
request type. An upstream enum value collision could send a valid message to
the wrong handler.

## GR-PATCH-009 — EngineCore Lifecycle and Socket Processing

Scope: `core`; backend ID: `null`

EngineCore admits grouped submissions and routes their terminal results and
retirement acknowledgements. Native socket dispatch and output collection must
keep ordinary requests progressing alongside GR work.

## GR-PATCH-010 — Scheduler Beam Override

Scope: `core`; backend ID: `null`

Scheduling attaches Beam stage and retirement metadata to native scheduler
outputs. Logical token accounting and outstanding worker receipts must agree
before the next owner can be admitted.

## GR-PATCH-022 — Prefix-cache hash memoization

Scope: `core`; backend ID: `null`

Bounded per-process memoization avoids hashing identical prefix blocks and
prompts repeatedly. Reinstallation checks the wrappers and the EngineCore
binding that constructs the hasher.

## GR-PATCH-011 — GPU ModelRunner V1/V2 Overrides

Scope: `backend`; backend ID: `cuda`

CUDA ModelRunner V1 and V2 adapt native input preparation, stage execution and
output completion. Request ordering, graph-visible storage and native lifecycle
boundaries have to agree. See the [runner contract](model_runner_v2.md).

## GR-PATCH-012 — NPU ModelRunner Override

Scope: `backend`; backend ID: `ascend`

The NPU worker adapts native rows and attention metadata for grouped Beam
execution. Cached prefixes, real row lengths and parent selection must stay
aligned with native block tables. See
[NPU Beam Decode execution](../feature/npu_decode_execution.md).

The [worker stage projection](../../../vllm_gr/v1/worker/npu/stage_projection.py)
wraps the instance logits processor, runner reload and native worker weight-update
methods. It invalidates the projection before native writes and refreshes it after
finalization and device synchronization. The
[operator helper](../../../vllm_gr/ops/npu/stage_projection.py) owns the fixed-address
weights, biases and proxies.

## GR-PATCH-013 — CUSTOM Attention Backend Registration

Scope: `core`; backend ID: `null`

GR registers Beam attention in the upstream CUSTOM slot. Registration and the
selector have separate responsibilities. Another extension claiming that slot
can invalidate the selection.

## GR-PATCH-015 — Sliced Vocabulary Projection

Scope: `core`; backend ID: `null`

The worker binds the loaded table's token window to each model's logits processor.
Processors without a window use native full-vocabulary projection. Sliced
projection also adapts log-probability collection. Head-derived caches must stay valid across
native weight reload.

## GR-PATCH-016 — NPU-aware Spawn Start Method

Scope: `backend`; backend ID: `ascend`

NPU subprocesses must use a compatible start method when the parent has
initialized the device. The NPU provider installs spawn protection before
the native worker initialization hook.

## GR-PATCH-018 — NPU Penalties Warmup OOM Guard

Scope: `backend`; backend ID: `ascend`

NPU warmup can allocate penalty tensors that exceed available memory. The
[NPU startup hook](../../../vllm_gr/v1/worker/npu/startup_patch.py) handles the
warmup allocation failure while preserving other execution errors.
The inventory lists no behavioural test for this guard.

## GR-PATCH-019 — NPU additional_config Sanitizer

Scope: `backend`; backend ID: `ascend`

The [NPU configuration patch](../../../vllm_gr/platforms/ascend/config_patch.py)
retains GR settings while filtering keys passed to the native configuration
reader. Fresh EngineCore processes install the common plan before native
construction. Worker startup also refreshes the filter's native bindings.

## GR-PATCH-020 — Pangu Tied Embeddings

Scope: `core`; backend ID: `null`

For tied Pangu embeddings, a placeholder checkpoint projection must not replace
the shared input weight. An independent zero-filled output head would produce
zero logits.

## GR-PATCH-021 — Contiguous Beam Prefix KV

Scope: `backend`; backend ID: `ascend`

The [NPU cache layout patch](../../../vllm_gr/v1/worker/npu/kv_layout_patch.py)
lets eligible NPU xllm Beam layers request contiguous prefix K/V at native
allocation. Attention row selection is a separate reader (`select_beam_rows`)
that checks layout, strides and format before using the paged block table.
An unsuitable layout keeps the gather path.

## Retired patches

Retired IDs stay in the YAML inventory for traceability. They have no active
installer.

| ID | Historical target | Why it was removed |
| --- | --- | --- |
| `GR-PATCH-006` | `OpenAIServingModels.__init__` | Legacy Catalog serving patch and its tests were removed; canonical V1 constraints load in the worker. |
| `GR-PATCH-014` | LMCache request tracking | Experimental LMCache integration removed; no supported version pin or vLLM 0.30 validation. |
| `GR-PATCH-017` | `NPUWorker.profile_memory` | vllm-ascend 0.30.0rc2 replaced the method with `log_memory_stats`; no native method remained to wrap. |

## Maintenance rules

1. Update the YAML entry, the checked index and the rationale here together.
2. Keep the patch ID, scope, backend ID and ordering stable; the
   [upstream contract test](../../../tests/contract/test_upstream_patch_contracts.py)
   verifies them.
3. Run the tests or validators listed for the changed patch. Do not cite a
   test that does not exercise the hook.
4. When upgrading vLLM, recheck private methods, descriptor types, wire tags and
   process-specific bindings.

Retired IDs stay in the YAML inventory for traceability. Benchmark-only shims
belong to their benchmark and do not participate in production startup. See
the [installation guide](../../getting_started/installation/README.md) for
supported runtime requirements.
