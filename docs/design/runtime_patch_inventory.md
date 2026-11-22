# Runtime patch inventory

The [runtime patch design](module/runtime_patches.md) explains the active hooks.
The entries below match [patch_inventory.yaml](patch_inventory.yaml) so
the upstream contract check can verify patch ordering and backend ownership.

## Active patches

### GR-PATCH-000 — Runtime Plugin Activation

Scope: `core`; backend ID: `null`

### GR-PATCH-001 — CUSTOM Attention Selector Compatibility

Scope: `core`; backend ID: `null`

### GR-PATCH-002 — NPUWorker Initialization Hook

Scope: `backend`; backend ID: `ascend`

### GR-PATCH-003 — FlatLogprobs Fast Append Override

Scope: `core`; backend ID: `null`

### GR-PATCH-004 — OpenAI Beam Serving Override

Scope: `core`; backend ID: `null`

### GR-PATCH-005 — EngineArgs Configuration Override

Scope: `core`; backend ID: `null`

### GR-PATCH-007 — EngineCore Client and AsyncLLM Extensions

Scope: `core`; backend ID: `null`

### GR-PATCH-008 — EngineCoreRequestType Extension

Scope: `core`; backend ID: `null`

### GR-PATCH-009 — EngineCore Lifecycle and Socket Processing

Scope: `core`; backend ID: `null`

### GR-PATCH-010 — Scheduler Beam Override

Scope: `core`; backend ID: `null`

### GR-PATCH-022 — Prefix-cache hash memoization

Scope: `core`; backend ID: `null`

### GR-PATCH-011 — GPU ModelRunner V1/V2 Overrides

Scope: `backend`; backend ID: `cuda`

### GR-PATCH-012 — NPU ModelRunner Override

Scope: `backend`; backend ID: `ascend`

### GR-PATCH-013 — CUSTOM Attention Backend Registration

Scope: `core`; backend ID: `null`

### GR-PATCH-015 — Sliced Vocabulary Projection

Scope: `core`; backend ID: `null`

### GR-PATCH-016 — NPU-aware Spawn Start Method

Scope: `backend`; backend ID: `ascend`

### GR-PATCH-018 — Ascend Penalties Warmup OOM Guard

Scope: `backend`; backend ID: `ascend`

### GR-PATCH-019 — Ascend additional_config Sanitizer

Scope: `backend`; backend ID: `ascend`

### GR-PATCH-020 — Pangu Tied Embeddings

Scope: `core`; backend ID: `null`

### GR-PATCH-021 — Contiguous Beam Prefix KV

Scope: `backend`; backend ID: `ascend`
