# Persistent GPU Beam state (A4 PR1)

Suggested PR title: `feat(beam): add persistent GPU beam state and execution primitives`

Implementation baseline: `7125565f6bff30fa86867c0655a2a2d6f14d4d77` on
`scheduler_a4_1`, matching the inspected `upstream/decode_graph` ref. The relevant
tracked files had no changes when implementation started on 2026-09-13.

## Behavior and boundaries

The independent CUDA runner initializes from one Prefill parent, then advances
Beam selection, persistent state and all-layer KV without reading generated
tokens, scores, parents or counters on the host. Invalid candidates retain
fixed physical capacity: masks change, tensor shapes and launch grids do not.

PR1 supports one active session, B=1, fixed beam width W and candidate width W.
It does not install a ModelRunner entrypoint, adapt scheduler metadata, load a
model, implement A3, transfer final results, or change NPU execution. The
existing public selection and session behavior remains unchanged by default.

## Files and ownership

| File | Responsibility |
| --- | --- |
| `vllm_gr/v1/beam/beam_state.py` | Persistent storage, capacity checks, views, row reset and memory accounting |
| `vllm_gr/v1/worker/beam_search_runner.py` | Bind existing session/state/workspace and enqueue one execution step |
| `vllm_gr/ops/cuda/beam_state.py` | Device validation, completion records, sequence/history updates and counter commit |
| `vllm_gr/ops/cuda/beam_selection.py` | Existing deterministic hierarchical selection; optional strict invalid-index propagation |
| `vllm_gr/ops/cuda/beam_kv_reorder.py` | Existing duplicate-safe gather/scatter, with optional device steps and masks |
| `vllm_gr/v1/attention/backends/beam_kernels.py` | Preserve the existing KV wrapper and scalar-step ABI |
| `beam_attention_context.py` / `beam_search_session.py` | Optional decision-row reset; existing slot allocation and release |
| `beam_decision.py` | Existing workspace, with optional persistent-runner scratch and memory accounting |

State ownership is explicit:

| Data | Owner and layout |
| --- | --- |
| Active sequence and accumulated score | Aliases of `BeamDecisionBuffers`, `[Bmax,W,S]` / `[Bmax,W]` |
| Input token, parent, active mask, effective length | `BeamSearchState`, `[Bmax,W]` |
| Generated-token count, Decode count, active count, completed count | State device tensors, `[Bmax]` |
| Initialized, finished, sticky error | State device tensors, `[Bmax]` |
| Immutable token/parent history | State, two `[Bmax,T,W]` tensors |
| Completed fields | State, six `[Bmax,C]` tensors |
| Stop configuration and maximum output length | State, fixed device request parameters |
| Selection outputs, next sequence and selection scratch | Existing `BeamPostprocessWorkspace` |
| Candidate classification and step-entry latches | Optional `workspace.state_scratch` |
| KV pools, per-layer aux and pointer tables | Existing `BeamAttentionContext` |
| Session identity and KV slot | Existing `BeamSearchSession` and registry |

Here T is configured `max_new_tokens`; S is the existing sequence's physical
extent, which may exceed T. The runner respects the original row strides.
It never reads legacy `worker_decode_steps`, session `decode_step`, issued or
completed scheduler counts, or result placeholders.

The leading batch dimension reserves independent counters, history and result
regions. Persistent parent IDs are request-local, not flattened batch indices.
The runner binds a `state_index` independently of the session KV slot and uses
workspace row zero. A test executes state row one with KV slot zero. Future
batch execution still needs execution-to-state/slot mappings and separate or
partitioned KV aux; the current shared aux does not support simultaneous users.

## PR2 interface and sequencing

Construct the existing context/session and workspace, then bind:

```python
state = BeamSearchState(
    context.decision,
    beam_width=width,
    max_new_tokens=max_new_tokens,
    stop_token_ids=stop_token_ids,
    ignore_stop_tokens=False,
)
runner = BeamSearchRunner(session, state, workspace, state_index=0)
runner.initialize_from_prefill(prefill_token_ids, prefill_logprobs, prefill_valid_mask)
runner.advance(decode_token_ids, decode_logprobs, decode_valid_mask)
```

Run `session.begin_session(..., decision_index=state_index)` before initializing
the persistent state. On reuse, order final consumption before `runner.reset()`
and the next initialization. Legacy `begin_session()` still resets all decision
rows; it must not run over another user's persistent buffers.

Callers that just reset this row through `reset_for_session` (batching its rows
into one launch) may pass `reset_decision=False` together with an explicit
`decision_index`; the keyword is rejected without one, because only the full
buffer reset also clears the legacy host counter.

Candidate tensors must be contiguous CUDA int32 IDs, FP32 logprobs and bool
masks on the state's device. Prefill has shape `[1,W]`; Decode has `[W,W]`.
Candidates need not be sorted. Shape/dtype/layout checks inspect tensor metadata
only. A false candidate mask permits arbitrary placeholder token/score values.
A false parent mask overrides all its candidate masks.

Prefill contributes the first token and its score exactly once. Afterwards:

```text
num_generated_tokens = 1
num_decode_steps = 0
```

Before each `advance`, the caller writes this round's KV at device position
`num_decode_steps`. The first Decode therefore writes position zero. The runner
validates candidates on device, computes scores, appends completions, selects
and stages active sequences, writes fixed state rows, gathers/scatters KV, and
finally increments both counts. A same-step parent mapping is applied once;
PR2 must not apply it again before the next forward.

Nonterminal selection uses score descending, flattened parent/candidate index
ascending for ties, then stable grouping by parent for execution order. There
are always W output positions. Inactive positions have mask false, score
negative infinity, safe token/parent zero and length zero. KV operations skip
them, leaving their physical contents unspecified and unread.

At length exhaustion, every valid final candidate is saved and all active masks
become false. EOS-only completion also clears active masks. Such steps have no
next active KV state and perform no KV copy. The kernels may still launch with
fixed grids. Later `advance` calls preserve persistent terminal data and KV.
Protecting the actual model forward after termination is PR2's responsibility.

Runtime device errors are sticky until reset:

| Code | Meaning |
| --- | --- |
| 0 | No error |
| 1 | Decode before initialization |
| 2 | Duplicate Prefill initialization |
| 3 | A valid candidate has an invalid ID, nonfinite score or nonfinite cumulative score |
| 4 | Capacity or device-step invariant violated |
| 5 | No valid candidate and no previously completed result |

All candidate rows are validated before any persistent write. Errors preserve
previous sequence, scores, completed records, counters and KV. The error itself
is reported through `error_code`; consumers must consider it as well as
`is_finished`. This is not a rollback mechanism for CUDA execution failures.

### Completion reconstruction

All valid EOS candidates are retained, including candidates outside active
Top-W. With ignored stop tokens, they participate as ordinary candidates.
The last generation round retains all valid candidates, with stop taking
precedence over length as the finish reason. No final Top-R, length penalty or
stop-token removal is performed in PR1.

Each completion stores six 32-bit fields: last token, raw cumulative score,
parent ID, length, finish reason and tie-break index. Finish reasons are 1 for
stop and 2 for length. Records append in generation-step / candidate-flat-index
order using per-row prefix sums, not unordered atomic reservation.

For length n > 1, start with the record's parent in history layer n-2. Append
that layer's token, follow its parent to the previous layer, and repeat. Reverse
the collected tokens and append the record's last token. Length one refers to
the implicit root. The same traversal reconstructs parent history. Invalid
history positions are never referenced by valid completions.

This preserves generation step, original length, stop token and deterministic
tie information for A5. Completed beams keep no KV. Early histories are never
rewritten when active sequence rows or KV are reordered.

### Streams, graphs and lifetime

Initialization and advance enqueue on the caller's current CUDA stream. Wait on
the producer's event before using cross-stream candidates. A downstream stream
must wait on an event recorded after the Beam step. Reset must follow all final
consumers, using an event dependency when needed.

`runner.end_session(consumer_done=None)` records and waits for a stream event
before returning the slot through the existing session. Pass a final consumer
event for cross-stream consumption. This explicit lifecycle boundary is outside
capture and steady execution; it never synchronizes the whole device. A released
runner rejects subsequent eager initialization, advance and reset.

Captured graphs bind resource addresses, slot, state row, candidate addresses
and fixed configuration. Replay reads changing counts/masks from device state;
it does not specialize a Python step number. Reuse with the same binding and
configuration is supported after reset. Changing bindings needs recapture.
The caller must stop using a graph before releasing its resources.

Active and workspace views are mutable views, not snapshots. Completed records
and history remain valid until reset. The state must live through final A5
consumption; no host transfer is required before that boundary.

## Capacity and memory

For Bmax physical state rows, fixed width W, token capacity T and a stop table
with E physical entries (`max(1, number_of_stop_tokens)`):

```text
C = W + (T - 1) * W * W
new persistent bytes = Bmax * (8*T*W + 24*C + 13*W + 31 + 4*E)
new workspace scratch bytes = workspace.max_batch * (13*W*W + 20*W + 6)
```

C conservatively covers every Prefill candidate and every Decode candidate at
every possible step. This requires no unproven truncation of completed beams.
Sequence/scores, KV, layer pointer tables and existing workspace allocations
are reused and excluded from these new-allocation totals. The formulas count
tensor payload bytes, not allocator rounding or graph runtime storage.

For the recorded W=32, T=40, Bmax=1 run without stop tokens, C=39,968:
state adds 969,923 bytes and optional workspace scratch adds 13,958 bytes.

## Validation and profiler evidence

Environment: NVIDIA L20X, PyTorch 2.11.0+cu130, Triton 3.6.0. GPU tests ran under
`gpu run --gpus 1` reservations. No model or EngineCore was loaded.

Initial suites: 28 passed (10 state tests and 18 real CUDA tests). They cover
Prefill plus four Decode updates with no intermediate result readback;
FP16/BF16 all-layer KV and completion history parity; W=1/3/32/256; ties;
arbitrary duplicate parent gather/scatter; masked holes and inactive parents;
shrink/growth with fixed physical width; partial/all EOS; single-token and
length exhaustion; sticky errors without partial writes; graph replay; row
isolation; event ordering; and slot reuse.

The completion audit also exercises one captured graph with different candidate
values each replay: active counts change 3 -> 1 -> 4 -> 4 -> 0, then two extra
replays follow EOS. All layer KV and every completion are compared with an
independent reference only after the entire chain, including the terminal
no-ops. This verifies mask changes, EOS and KV together under graph replay.

Shared regression: 368 passed, 7 skipped. Suites:

```text
tests/test_beam_search_session.py
tests/test_beam_decision.py
tests/test_beam_postprocess_workspace.py
tests/test_beam_reorder.py
tests/test_beam_session_ops_integration.py
tests/test_beam_attention_pool.py
tests/kernels/test_beam_postprocess_cuda.py
tests/kernels/test_beam_search_group_cuda.py
tests/kernels/test_beam_search_rec_final_select_cuda.py
tests/kernels/test_prefill_beam_result_cuda.py
```

The seven skips require Ascend NPU hardware/runtime. CUDA acceptance is complete;
NPU, full-model and full-repository slow integration runs are outside this PR's
acceptance scope.

The initial combined run reports 396 passed and 7 skipped in
`results/a4_pr1_20260913/acceptance.xml`. The source/test hashes and artifact
summary are recorded in `results/a4_pr1_20260913/validation_manifest.json`.

Static validation passed: Ruff checks/format, mypy for the four new source files
and two test files (`--follow-imports=skip --ignore-missing-imports`, matching
the repository's separate-group policy), SPDX, English-content and torch CUDA
API checks, and `git diff --check`. Mypy used `--no-incremental
--cache-dir=/dev/null` to avoid a local cache-serialization failure.

Reproduce the new tests and traces from the repository root:

```bash
gpu run --gpus 1 --timeout 15m -- env \
  BEAM_STATE_TRACE_DIR=results/a4_pr1_20260913 \
  .venv/bin/python -m pytest \
  tests/test_beam_state.py tests/kernels/test_beam_search_runner_cuda.py -q
```

Local artifacts (generated, not tracked):

- `results/a4_pr1_20260913/beam_state_eager.json`
- `results/a4_pr1_20260913/beam_state_graph.json`
- `results/a4_pr1_20260913/summary_eager.json`
- `results/a4_pr1_20260913/summary_graph.json`
- `results/a4_pr1_20260913/acceptance.xml`
- `results/a4_pr1_20260913/validation_manifest.json`

Both traces cover 16 consecutive W=32 updates after warmup; the graph is captured
once and replayed. Each records zero intermediate Beam D2H / host scalar reads,
zero positive device-allocation events, unchanged allocated bytes, and zero
synchronizations inside `beam_state_steady`. The profiler's own teardown adds a
`cudaDeviceSynchronize` outside that range; this is not a runner dependency.
The tests check these properties and separately verify address stability.

### Capacity-error review follow-up

Review of commit `8eae1185b72b3886191be0d08a53b70b6da8b96f` identified a direct
coverage gap for `ERROR_CAPACITY`. Ten additional real CUDA cases now cover:

- Completed append overflow, a zero generated count, a generated count at its
  configured limit, and Decode/generated counter inconsistency, each in eager
  mode and graph replay. Other capacity conditions remain valid so each case
  isolates the intended check.
- A kernel-level KV capacity boundary, tested both exhausted and within range,
  while generated/Decode counts agree and other capacities are valid. Normal
  runner construction makes this bound redundant with the generation limit;
  the direct operator test demonstrates that the KV check itself is effective.

Each runner-level case captures with valid state, injects the fault afterwards,
and snapshots all persistent state except `error_code`, plus every layer's K/V.
Two failing calls/replays execute before any result readback. They must report
`ERROR_CAPACITY`, disable execute/reorder, and preserve every protected tensor.
The snapshot is taken after fault injection: error handling must not silently
repair counters. Workspace scratch is allowed to change.

The candidates include both EOS and active continuations; without an error they
append three completed records and reorder KV with duplicate parents
`[1, 1, 1, 2]`. Reset followed by the same captured graph must perform this
successful update, ruling out an inert test fixture or permanently disabled
graph. No production code change was needed.

The follow-up state/CUDA suites passed all 38 tests (10 state and 28 CUDA),
including all 10 newly added capacity cases. Ruff, mypy, formatting and the
repository content/API checks also passed.
Evidence is saved separately from the original acceptance run:

```bash
gpu run --gpus 1 --timeout 15m -- env \
  BEAM_STATE_TRACE_DIR=results/a4_pr1_capacity_review_20260913 \
  .venv/bin/python -m pytest \
  tests/test_beam_state.py tests/kernels/test_beam_search_runner_cuda.py \
  --junitxml=results/a4_pr1_capacity_review_20260913/acceptance.xml -q
```

This directory contains the follow-up XML, eager/graph traces and summaries,
and a `validation_manifest.json` with tested source hashes. The original
368 shared-regression passes remain applicable: this follow-up changes only
tests and documentation.

## Completion audit against the task

| Requirement | Evidence |
| --- | --- |
| Persistent ownership and fixed capacity | State alias/reset tests; ownership and exact byte formulas above |
| One-parent Prefill and score inheritance | Continuous independent CPU-reference test; single-token and duplicate-initialization tests |
| Fixed-width invalid nodes | Mask shrink/growth and masked-KV tests; no dynamic output allocation |
| Cross-step sequence, scores and parent history | Four-round FP16/BF16 reference comparison without intermediate readback |
| Same-step duplicate-safe KV | All-layer reference comparison and unsorted duplicate-parent operator test |
| Completed preservation and EOS/length semantics | All valid completions reconstructed and compared; W=256 final candidate coverage |
| Capacity errors cannot partially update persistent state or KV | Four isolated runner-level faults in eager/graph mode, plus independent KV guard boundary tests |
| Device counters, changing masks and termination during replay | Both graph tests, including the varying-mask/EOS test and terminal no-ops |
| Stable addresses and row/slot reuse | State row isolation, repeated graph reset, producer/consumer event and session A/B tests |
| No steady Beam D2H, scalar reads or device allocation | Eager and graph profiler assertions and exported traces |
| Shared behavior retained | 368 shared regression passes; 7 hardware-dependent NPU skips |
| PR2 interface and consumer lifetime | API, counter semantics, graph binding and completion reconstruction contracts above |
| No production integration or new scheduler/KV registry | Runner accepts only session/state/workspace and tensors; production runner entrypoints are unchanged |

The implementation does not claim an end-to-end latency improvement. Production
ModelRunner/A3 adaptation belongs to PR2, and final reconstruction/transport to
A5.
