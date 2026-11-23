# Persistent GPU Beam state

CUDA Beam search keeps candidate scores, parent indices, history and generation
counters on the device across Prefill and Decode. See the
[batch execution contract](../feature/beam_batch_execution.md#scheduling-and-resource-ownership)
for admission and the [scheduler contract](../feature/async_scheduling.md) for
dispatch and retirement.

## Storage and ownership

| Component | Responsibility |
| --- | --- |
| `BeamSearchState` | Request-local counters, active masks, current token and parent indices, append-only per-layer history and completed records. |
| `BeamDecisionBuffers` | Sequence and score allocations aliased by Beam state, plus `group_token_num` and the host-side `worker_decode_steps`. |
| `BeamPostprocessWorkspace` | Candidate selection and state update scratch. Prewarmed once after model load; device bytes count toward `model_memory_usage`. |
| `BeamAttentionContext` | Worker-level singleton owning KV pools, the session registry, suffix buffers and device step buffers. |
| `BeamSearchSession` | Per-request facade exposing `is_active`, `beam_width`, `decode_step`, `pools` and the suffix buffers. |
| `BeamSearchRunner` and `BeamBatchSearchRunner` | Request-local device updates and request-major routing for fixed-width batches. |

The fixed-width batch adapter maps each request to its own state row,
workspace row and KV slot. For Beam width `W`, candidate width `K` and
generation capacity `T`, the state reserves `K + (T - 1) * W * K` completion
records per request. Prefill contributes at most `K` candidates; each Decode
stage at most `W * K`.

Dynamic schedules bypass the fixed-width adapter. `BeamBatchExecutionView`
implements `dynamic_beam_widths`, `per_parent_beam_widths`, `input_width`,
`output_width` and `candidate_top_k`, and dynamic stages run through
`GPUDynamicBeam.advance`.

## Stage ordering and completion

Prefill adds the first generated token and initializes the Decode counter to
zero. The first Decode writes KV at position zero. Each Decode stage writes
current KV, selects successors and applies the parent KV mapping once before
the next Forward. Masks let fixed physical shapes represent inactive rows.

A completion record stores its last token, cumulative score, request-local
parent index, length, finish reason and tie-break index. For length `n > 1`,
follow the record's parent into history at layer `n - 2`. Walk back through the
stored parent indices, reverse the recovered tokens and append the record's
last token. A one-token record refers to the implicit Prefill root. Reordering
active sequences leaves earlier history intact.

History is append-only. The current layer is overwritten in place each step,
and a reset clears it. Cross-stage checks enforce the immutable batch identity,
request order and the rule that a parent cannot come from another request.
Session reuse keeps fixed views in `rebind_session`, and `reset_for_session`
reinitializes state through one Triton kernel with the `-inf` completed-score
sentinel.

## Streams, graphs and retirement

Runner updates execute on the current CUDA stream. Cross-stream producers and
consumers establish ordering through events. Graph replay reads the current
masks and counters at captured addresses; the prewarmed workspace makes those
addresses stable. Reuse requires compatible resource bindings and geometry, and
active views are overwritten during execution. Completed records and history
remain valid until reset.

Reset and KV slot release follow the drain of outstanding dispatches and final
consumers. A device error does not roll back KV writes already issued by the
current model Forward.

## Validation

| Test | Coverage |
| --- | --- |
| [CPU state tests](../../../tests/v1/beam/test_beam_state.py) | Allocation, aliasing, capacity and reset. This module runs on CPU and does not exercise device updates. |
| [CUDA runner tests](../../../tests/kernels/test_beam_search_runner_cuda.py) | Device state updates, ordering, error handling and retirement events. |
| [Worker lifecycle tests](../../../tests/kernels/test_gpu_beam_stage_lifecycle.py) | Completion and resource retirement. |
| [Workspace tests](../../../tests/v1/worker/test_gpu_postprocess_workspace.py) | Workspace prewarming, bucket reuse and memory accounting. |

## Source map

| File | Responsibility |
| --- | --- |
| [Beam state](../../../vllm_gr/v1/worker/gpu/beam_state.py) | Completion-record layout, capacity and reset. |
| [Attention context](../../../vllm_gr/v1/beam/attention_context.py) | KV pools, suffix buffers and session slots. |
| [Session facade](../../../vllm_gr/v1/beam/session.py) | Per-request view of the worker context. |
| [Execution batch](../../../vllm_gr/v1/beam/execution_batch.py) | Fixed and dynamic width schedules and cross-stage validation. |
| [CUDA runner](../../../vllm_gr/v1/worker/gpu/beam_search_runner.py) | Device updates and session reset. |
