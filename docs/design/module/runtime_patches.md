# Runtime patches and upstream dependencies

The [patch inventory](../patch_inventory.yaml) records the installer, target,
process roles, risk, tests and removal condition for each active patch. The
sections below explain the behavior each hook protects. Backend ownership
identifies which provider declares the patch. Installation can still happen
later in a worker or during native configuration.

## Runtime patch inventory

### GR-PATCH-000 — Runtime Plugin Activation

Scope: `core`; backend ID: `null`

The general plugin installs runtime hooks in each process that needs them.
Installation checks live targets so repeated initialization can reuse an existing
wrapper.

### GR-PATCH-001 — CUSTOM Attention Selector Compatibility

Scope: `core`; backend ID: `null`

An explicit CUSTOM choice, including one restored from serialized GR configuration,
must reach the registered Beam backend. Changes to the private attention selector
or its cache can silently select a different backend.

### GR-PATCH-002 — NPUWorker Initialization Hook

Scope: `backend`; backend ID: `ascend`

Ascend needs its configuration filter and worker hooks before native device
initialization constructs the runner. An early import can precede worker
availability, so later initialization retries the installation.

### GR-PATCH-003 — FlatLogprobs Fast Append Override

Scope: `core`; backend ID: `null`

The frontend appends flattened log probabilities through the GR implementation of
the native fast-append method. Its arguments and the upstream flattened
representation must stay compatible.

### GR-PATCH-004 — OpenAI Beam Serving Override

Scope: `core`; backend ID: `null`

The serving frontend groups prompts into one Beam submission and formats their
ranked candidates for the OpenAI-compatible API. The wrapper depends on the native
serving and protocol methods it replaces.

### GR-PATCH-005 — EngineArgs Configuration Override

Scope: `core`; backend ID: `null`

EngineArgs registers GR options and serializes the resolved settings for explicitly
selected GR engines. Its wrappers preserve native descriptor types and explicit
caller settings when constructing the engine configuration.

### GR-PATCH-007 — EngineCore Client and AsyncLLM Extensions

Scope: `core`; backend ID: `null`

Client extensions submit a whole Beam session and route its wait, abort and release
operations. A rejected submission must preserve the identity that the frontend uses
to complete the request.

### GR-PATCH-008 — EngineCoreRequestType Extension

Scope: `core`; backend ID: `null`

GR adds a wire discriminator for grouped session submission to the native request
type. An upstream enum value collision could send a valid message to the wrong
handler.

### GR-PATCH-009 — EngineCore Lifecycle and Socket Processing

Scope: `core`; backend ID: `null`

EngineCore admits grouped submissions and routes their terminal results and
retirement acknowledgements. Native socket dispatch and output collection must keep
ordinary requests progressing alongside GR work.

### GR-PATCH-010 — Scheduler Beam Override

Scope: `core`; backend ID: `null`

Scheduling attaches Beam stage and retirement metadata to native scheduler outputs.
Logical token accounting and outstanding worker receipts must agree before the next
owner can be admitted.

### GR-PATCH-022 — Prefix-cache hash memoization

Scope: `core`; backend ID: `null`

Bounded per-process memoization avoids hashing identical prefix blocks and prompts
repeatedly. Reinstallation checks the wrappers and the EngineCore binding that
constructs the hasher.

### GR-PATCH-011 — GPU ModelRunner V1/V2 Overrides

Scope: `backend`; backend ID: `cuda`

CUDA ModelRunner V1 and V2 adapt native input preparation, stage execution
and output completion. Request ordering, graph-visible storage and native
lifecycle boundaries have to agree. See the [runner contract](model_runner_v2.md).

### GR-PATCH-012 — NPU ModelRunner Override

Scope: `backend`; backend ID: `ascend`

The Ascend worker adapts native rows and attention metadata for grouped Beam
execution. Cached prefixes, real row lengths and parent selection must stay aligned
with native block tables. See [NPU Decode](../feature/npu_decode_execution.md).

### GR-PATCH-013 — CUSTOM Attention Backend Registration

Scope: `core`; backend ID: `null`

GR registers Beam attention in the upstream CUSTOM slot. Registration and the
selector have separate responsibilities. Another extension claiming that slot can
invalidate the selection.

### GR-PATCH-015 — Sliced Vocabulary Projection

Scope: `core`; backend ID: `null`

Sliced vocabulary projection computes logits over a configured token range and
adapts log-probability collection. Head-derived caches must stay valid across
native weight reload.

### GR-PATCH-016 — NPU-aware Spawn Start Method

Scope: `backend`; backend ID: `ascend`

Ascend subprocesses must use a compatible start method when the parent has
initialized the device. The hook supplements the native process-start decision for
NPU.

### GR-PATCH-018 — Ascend Penalties Warmup OOM Guard

Scope: `backend`; backend ID: `ascend`

Ascend warmup can allocate penalty tensors that exceed available memory. The
wrapper handles the warmup allocation failure while preserving other execution
errors.

### GR-PATCH-019 — Ascend additional_config Sanitizer

Scope: `backend`; backend ID: `ascend`

The engine retains GR settings while filtering keys passed to the Ascend
configuration reader. Fresh EngineCore processes must install the common plan
before native construction.

### GR-PATCH-020 — Pangu Tied Embeddings

Scope: `core`; backend ID: `null`

For tied Pangu embeddings, a placeholder checkpoint projection must not replace the
shared input weight. An independent zero-filled output head would produce zero
logits.

### GR-PATCH-021 — Contiguous Beam Prefix KV

Scope: `backend`; backend ID: `ascend`

Eligible Ascend xllm Beam layers can request contiguous prefix K/V at native
allocation. The reader checks layout, strides and format before using the paged
block table. An unsuitable layout keeps the gather path.

## Maintenance rules

1. Update the inventory, the explanation here and the contract test covering the upstream assumption.
2. When upgrading vLLM, recheck private methods, descriptor types, wire tags and process-specific bindings.
3. Run the [upstream contract tests](../../../tests/contract/test_upstream_patch_contracts.py) for active IDs, ordering and core/provider ownership.

Retired IDs stay in the YAML inventory for traceability. Benchmark-only shims
belong to their benchmark and do not participate in production startup.
See the [installation guide](../../getting_started/installation/README.md) for
supported runtime requirements.
