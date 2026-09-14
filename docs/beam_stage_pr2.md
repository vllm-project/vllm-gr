# Persistent GPU Beam Worker adapter

## Scope and ownership

The GPU V1 Worker recognizes the imported A3 `GR_STAGE_OUTPUT_ATTR` before
legacy session updates. A native logical Request owns Prefill and its paged
prompt KV. Subsequent invocations execute W physical model rows using the
existing model, Beam Attention context, registry and PR1 runner. Tokens,
scores, parents, histories, counters and completion records remain on GPU.

Supported execution is B=1, one active session, TP/PP/DP=1, fixed W<=256,
effective N in {1,2,3}, FP16/BF16 and a loaded canonical CUDA Constraint Table.
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
| Candidates and prefixes | Existing workspace `[Bcap,W,W]` and `[Bcap,W,3]` tensors |
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

The Decode adapter captures a complete forward/LM-head CUDA graph with the
real W-row layout. It uses its own graph to avoid collision with native or
legacy graphs keyed only by token count. The existing compiled model and
Attention implementation are reused; Beam postprocessing remains explicit
CUDA launches. Capture failure is an error when this mode is requested.

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
