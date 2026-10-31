# Persistent GPU Beam Worker adapter

## Scope and ownership

Select the Beam interface before creating the engine. In a canonical GR
configuration, set `beam.execution_mode: v1` for `beam_search_v1`. The default
is `legacy` for `beam_search`. Alternatively, `beam_execution_mode="v1"` in
`GRLLM` / `EngineArgs`, or `--beam-execution-mode v1` on the CLI, overrides the
configured mode. This mode override also works with canonical configuration;
the existing restrictions on mixing other GR configuration sources remain.

The selection is fixed for the engine's lifetime and is serialized to every
Worker. Native `async_scheduling=True`, worker decisions and engine-driven
chaining do not select the V1 interface: legacy Beam also uses these features.
V1 requires native async scheduling. A request sent through the wrong Beam
interface is rejected before submission; EngineCore and Worker validate the
mode as well.

The GPU V1 Worker recognizes the imported A3 `GR_STAGE_OUTPUT_ATTR` before
legacy session updates. A native logical Request owns Prefill and its paged
prompt KV. Subsequent invocations execute W physical model rows using the
existing model, Beam Attention context, registry and PR1 runner. Tokens,
scores, parents, histories, counters and completion records remain on GPU.

Supported execution is B=1, one active session, TP/PP/DP=1, fixed W<=1024 with
candidate width K<=256 (K defaults to W; the fused W-way merge only applies
while K==W and W in {64,128,256}), effective N in {1,2,3}, FP16/BF16 and a
loaded canonical CUDA Constraint Table.
The effective budget includes the first Prefill output; begin/end adjustments
already made by A2 are never applied again. Initial chunked Prefill and native
Prefix Cache are supported. Generated-state recompute, heterogeneous/mixed
batches, speculative decoding, encoder-decoder/hybrid models, multimodal
inputs, LoRA, quantization, non-power-of-two head sizes and KV transfer are
rejected. Admission requires ready tables with matching device/vocabulary,
uniform unquantized KV and Beam Attention on every layer. A5 is required before the concrete
public frontend can submit a full-result request. Worker-local completion
views expose terminal device state for the future A5 consumer.

| Concept | Owner and representation |
| --- | --- |
| Request binding | Derived from existing session/registry: session ID, item index, native request ID, state row, KV slot, workspace row |
| Physical execution | Immutable dispatch entries with metadata, model row offset and count |
| Stage identity | A3 session/dispatch/stage references, independent of physical/resource rows |
| Dynamic Beam state | Existing `[Bcap,W,...]` state with per-row generated, Decode, active, finished, error and completed counters |
| Candidates and prefixes | Existing workspace `[Bcap,W,K]` and `[Bcap,W,3]` tensors |
| Attention dynamic values | Execution-local mask, positions, suffix offsets and context step scalar populated from the selected state row |
| Shared selection/KV scratch | Explicit context and workspace execution owner; storage capacity never grants concurrency |
| Control snapshots | Two dispatch slots, one entry per dispatch, three int32 fields per entry |
| Prompt block lifetime | Native Scheduler block-pool references held separately for each physical dispatch |

`BeamSearchRunner(..., workspace_index=0)` is backward compatible. Its output
and state scratch use that row; hierarchical selection and all-layer KV scratch
remain shared. The adapter is a view of `_beam_sessions` and the existing
registry, not another session allocator. Cancellation, completion views and
release require the complete binding.

## Execution and dependency changes

Expected GR admission failures (unsupported mixing, duplicate identity,
request Beam geometry or permanent KV capacity excess) raise
`GRAdmissionError`. In the multiprocess path, the EngineCore ADD consumer
catches only this type and answers the rejected request's client with one
terminal output whose reason travels with it: the rejection is attached on
the output's `kv_transfer_params` as an `admission_rejected` failure, so a
frontend reports the caller's own mistake as a client error instead of
filing it under engine faults. Request-content refusals raised before the
batch is queued take the same path; deployment-configuration failures stay
on the preprocessing error path. Admission stays on the EngineCore thread;
the socket thread only preprocesses and queues ADD. In-process callers
receive the admission exception directly. Other internal exceptions retain
the native engine-failure behavior, and shutdown retains native abort
handling.

Admission validates before native queue insertion, and publishes the GR
session/owner only after native ADD succeeds. Rejections leave existing
requests, in-flight work and resource ownership unchanged. Temporary KV
shortage still applies backpressure; exceeding total capacity is a permanent
request rejection.

Pause and resume retain the native scheduler policy. `keep` drains already
submitted batches and preserves active and waiting GR sessions with their
state and KV; `wait` finishes native running requests while waiting requests
remain queued. GR resource ownership can precede native execution, so owning
the shared buffers does not make a queued request runnable during `wait`.
Pending terminal delivery, draining resources and outstanding Worker release
proofs still count as cleanup work under either pause mode. Retained sessions
alone do not prevent pause completion. Resume continues the preserved queues.

1. Admission checks the complete metadata mapping, native identity/Prefill
   extent, output budget, stage dependencies, configuration and deployment
   limits before dispatch controls or session state are modified.
2. Prefill uses native `execute_model` and the native final-token logits.
   Intermediate chunks run the model but do not initialize Beam state or
   consume a generation token. `session_init` is optional.
3. Decode bypasses native Host token packing and legacy Beam remapping. A
   Triton kernel reads state with explicit row/stride/offset arguments and
   fills fixed token, position, execution mask, suffix length and prefix
   buffers. Query offsets, logits row layout and paged prefix block table
   have fixed addresses. The native `_update_states` only maintains request
   and block bookkeeping; placeholders are never read as model tokens.
4. The canonical CUDA candidate kernel accepts an internal device-step
   pointer. Its original scalar-step API is unchanged. Selection writes
   workspace candidates directly, followed by PR1 `advance()` and one
   same-round parent KV reorder. Only PR1 adds the parent score.
5. GR `sample_tokens` returns `AsyncGPUBeamOutput`. Each dispatch latches its
   own produced/finished/error values before the next update. The copy stream
   waits for the producer event and copies 12 bytes into pinned memory.
   `get_output()` waits only in the executor output consumer and constructs
   the ordinary IPC `ModelRunnerOutput` with `GR_WORKER_RESULT_ATTR` attached.
   Normal production is one logical token, independent of W.

A3 now validates non-output Prefill receipts as well, permits a successful
zero-token chunk, and handles device failures through the existing terminal
failure contract. A missing mapping remains the A3-only compatibility path.

The first Decode after native Prefill waits for the native input-preparation
event before updating CPU bookkeeping. This protects pinned input buffers
still being read by asynchronous H2D. Later Decode stages do not upload those
native buffers and add no Host input-preparation wait.

## Graphs and terminal safety

Both interfaces use the same `BeamDecodeGraph` capture/replay implementation.
Each engine initializes the input adapter for its selected mode and captures
one fixed-width Forward graph over that adapter's buffers. The graph returns
hidden states; LM head and Beam postprocessing are subsequent CUDA launches
on the ordered compute stream. There is no per-step Host wait for logits.

The component has a Beam-only `CUDAGraphWrapper` registry, separate from native
ordinary-attention graphs with the same token-count descriptor. It retains
the fixed input/metadata views and Beam context. Legacy graph dispatch and V1
replay use this same component; the model wrapper delegates non-Beam forwards
to the original native model. Beam graph capture also works with native
compilation/cudagraphs disabled. Capture failure is an error in V1 graph mode;
legacy retains its eager fallback.

Prefill uses the existing `GPUPrefillGraphRunner` in both modes, including
padding into the smallest configured bucket that holds the current scheduled
token count. For example, 900 scheduled tokens replay the 1024-token graph;
the real Attention lengths remain 900 and padded KV slots are masked with
`-1`. Prefix-cache hits and chunking change the scheduled extent, so bucket
selection uses that extent rather than the original prompt length. V1 retains
its live input shape and adapter checks, startup-only capture, and the
runner's slot mapping, static-buffer and captured KV-length-bound checks.
Unsupported inputs or unavailable graphs fall back to native execution.

Debug logs report the scheduled token count, request count, selected bucket
and replay/fallback outcome. The GPU Prefill runner also exposes
`last_replay_reason` for its latest replay attempt, including missing input
metadata and KV lengths exceeding the captured bound. These diagnostics only
use CPU metadata and add no device synchronization. A replay is reported only
after the graph has actually been submitted.

Numerical comparisons must separate legacy-versus-V1 execution at identical
padded geometry from padded-graph-versus-eager execution. BF16 rounding may
change Beam order when the matrix shape changes. Allowing padded replay
preserves the existing graph behavior; it does not assert bitwise equality
with unpadded eager execution or full graph coverage for every workload.

Binding validation covers metadata ABI, batch capacity, W, generation
capacity, state strides, state/workspace rows, KV slot, all-layer pool
addresses/strides, input/prefix addresses, prompt extent, block-table shape
and layer names. Changing block IDs updates the fixed block-table contents.
Changing resource geometry requires a drained new binding and graph capture.
Compatible session reuse retains storage/graph after final consumer drain;
request IDs are not graph addresses. Every producer stream waits on the
previous Beam completion event before consuming persistent state.

Pool writes skip inactive/terminal/error rows and negative/out-of-range
steps. Suffix gather never reads inactive pool rows: it writes zeros to their
fixed packed segments, including dummy terminal successors. Attention can
execute those safe physical rows without modifying valid persistent KV.
Candidate errors preserve committed histories, counters and completed
records. This does not roll back the uncommitted write made by the current
forward; later dummy work is guarded.

## Resource lifetime

Native Scheduler `block_pool.touch` holds every dispatched prompt block until
its corresponding output has drained, including abort/preemption. Python
references in a Worker cannot provide this refcount guarantee. Only complete
prompt blocks are prefix-cacheable; generated placeholders are never cached.
The final lease releases each KV group's blocks from prompt tail to head,
preserving native group order and the preference for retaining shared prefixes.

Release is idempotent and requires all dispatches sampled, all control slots
consumed and an explicit final consumer event. The PR1 runner orders that
event and synchronizes a stream event before releasing the existing registry
slot. Completion views are live device views, not snapshots, and become
invalid after release/reset. An aborted session may still execute already
issued successors, which drain with masked inputs. A5/A6 final transport and
service output remain outside this change.

Cancellation also suppresses candidate generation and state updates for queued
Prefill chunks while preserving their independent control receipts. Device
Prefill initialization ignores a finished, uninitialized row; duplicate
initialization of an initialized row retains the existing error contract.

## Multi-batch expansion order

First partition shared selection/KV scratch and execution-local Attention and
graph buffers. Next implement multiple dispatch entries, model row mappings
and candidate execution. Only then expand admission, batched control receipts
and cross-request EOS/cancellation/release acceptance. A3 public protocol is
unchanged. Capacity>1 tests exercise serial resource isolation and do not
claim B>1 execution support.

## Tests

CUDA adapter tests cover dispatch identity checks, per-row resource isolation,
independent control snapshots, terminal successor safety and resource release.
The scheduler and frontend tests cover asynchronous stage receipts and the A5
availability guard.

```bash
python -m pytest tests/kernels/test_gpu_beam_stage_runner.py \
  tests/kernels/test_gpu_beam_stage_lifecycle.py \
  tests/kernels/test_beam_search_runner_cuda.py tests/test_beam_state.py -q
python -m pytest tests/test_gr_async_scheduler.py \
  tests/test_beam_search_v1_client.py tests/test_beam_search_v1_entrypoint.py \
  tests/test_beam_search_v1_input_processor.py -q
```

## A5/A6 integration

Real-model integration tools and benchmark entrypoints are deferred to A5/A6.
The final-result consumer must finish reading completion views and signal its
consumer event before resource release. Integration must also cover two
in-flight dispatches through the native multiprocess executor, cancellation,
EOS successors and block reuse across request lifetimes.

Performance comparisons require the same model, vocabulary projection,
Prefix Cache numerical history and final-consumption boundary. Full and
chunked BF16 Prefill can produce different rounding histories for cached KV.
Native Prefill graph capture limits can also force eager Prefill even when
the fixed-row Decode graph replays. These distinctions must be recorded when
comparing cache hits, misses, eager execution and graph execution. Worker
measurements alone do not establish service E2E performance or stable
multiprocess tail latency.
