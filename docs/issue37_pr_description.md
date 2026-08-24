feat(issue37): EngineCore Beam Batch ingress with child builder, wire dispatch and unified beam_batch facade

<!-- markdownlint-disable -->

## Purpose

This PR implements [issue #37](https://github.com/zhanghanleo10/vllm-gr/issues/37)
(subtask 2/10 of [RFC #35](https://github.com/zhanghanleo10/vllm-gr/issues/35)):
the shared Online/Offline EngineCore Beam Batch ingress. A business batch is
accepted as one immutable `BeamBatchRequest` (contract from PR #292), materialized
as B native child `Request` objects plus one `BeamBatch` host object, registered
as a single atomic unit, and resolved through one final-only handle.

What is included:

- `EngineCoreBeamBatchIngress` with `submit_batch` / `submit_single` / `cancel` /
  `route_terminal_result` / `fail_beam_batch`; a minimal registry keyed by
  `batch_id`; terminal-result dedup; atomic registration with
  enqueue-failure rollback; mixed-mode fail-fast.
- Native child builder (`Request.from_engine_core_request` per item) and the
  `BeamBatch` host object holding the B native children.
- Wire protocol: `BEAM_BATCH_SUBMIT=0x08`, `BEAM_BATCH_ABORT=0x09` registered via
  `wire.py`, decoded in `codec.py`, dispatched in
  `EngineCoreProc.process_input_sockets`.
- EngineCore assembly: patched `EngineCore.__init__` creates the ingress;
  `_handle_beam_batch_submit` / `_handle_beam_batch_abort` route messages into it.
- Client methods: `submit_beam_batch` / `abort_beam_batch` on `AsyncMPClient`,
  `SyncMPClient`, and `InprocClient`, following the existing
  `core_client_patch.py` wiring pattern.
- Unified user entry: `GRLLM.beam_batch(prompts, beam_params, output_options)`
  for offline (blocking, aligned with `LLM.generate`) and
  `AsyncLLM.beam_batch(...)` for online; one entry handles B=1 and B>1.
- Patch inventory registration (GR-PATCH-016) in
  `docs/patch_inventory.yaml` / `docs/runtime_patch_inventory.md`.
- 9 unit tests covering submit/cancel/final routing, atomic rollback, duplicate
  ID / terminal rejection, mixed-mode fail-fast, child materialization, and wire
  value registration.

Explicitly out of scope (owned by later subtasks): #39 Scheduler all-or-none
admission, #40 lifecycle exactly-once, #41 final-output schema and cross-process
result delivery, #43 Worker run-to-completion. The `enqueue` / `cancel` callbacks
and `route_terminal_result` are the documented seams where those subtasks connect.

## Test Plan

```bash
python3 -m pytest tests/test_beam_batch_ingress.py -q
python3 -m pytest tests/test_patch_initialization.py -q
python3 -m py_compile vllm_gr/v1/engine/beam_batch.py \
  vllm_gr/v1/engine/wire.py vllm_gr/v1/engine/codec.py vllm_gr/v1/engine/core.py \
  vllm_gr/v1/engine/engine_core_patch.py vllm_gr/v1/engine/core_client.py \
  vllm_gr/v1/engine/core_client_patch.py vllm_gr/v1/engine/async_llm.py \
  vllm_gr/entrypoints/gr.py
```

Module import smoke check for all touched modules was also run.

## Test Result

`tests/test_beam_batch_ingress.py`: **9 passed**

Cases covered:

- submit batch enqueues once and routes the terminal result;
- submit single builds a single-item batch;
- duplicate `batch_id` rejected;
- enqueue failure rolls back the registry;
- mixed ordinary/beam requests fail fast;
- duplicate terminal result rejected;
- native child materialization (`BeamBatch.children`);
- child-builder failure leaves no registered record;
- wire `0x08` / `0x09` registration.

`tests/test_patch_initialization.py`: **8 passed** (existing patch initialization
is unaffected by the new wiring).

All touched modules compile (`py_compile`) and import cleanly. No GPU/e2e run in
this change: the Worker result path (which calls `route_terminal_result`) is
owned by #43, so terminal routing is exercised by unit tests only.

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [x] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft.
</details>

**BEFORE SUBMITTING, PLEASE READ <https://github.com/vllm-project/vllm-gr/blob/main/CONTRIBUTING.md>** (anything written below this line will be removed by GitHub Actions)
