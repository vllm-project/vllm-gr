# Persistent GPU Beam state

CUDA Beam search keeps candidate scores, parent indices, history and
generation counters on the device across Prefill and Decode. See the
[batch execution contract](../feature/beam_batch_execution.md#scheduling-and-resource-ownership)
for admission and the [scheduler contract](../feature/async_scheduling.md) for dispatch
and retirement.

## Storage and ownership

| Component | Responsibility |
| --- | --- |
| `BeamSearchState` | Request-local counters, active masks, current token and parent indices, immutable history and completed records |
| `BeamDecisionBuffers` | Sequence and score allocations aliased by Beam state |
| `BeamPostprocessWorkspace` | Candidate selection and state update scratch |
| `BeamAttentionContext` and `BeamSearchSession` | KV storage, attention metadata and session slots |
| `BeamSearchRunner` and `BeamBatchSearchRunner` | Request-local device updates and request-major routing for fixed-width batches |

The fixed-width batch adapter maps each request to its own state row,
workspace row and KV slot. For Beam width `W`, candidate width `K` and
generation capacity `T`, the state reserves `K + (T - 1) * W * K`
completion records per request. Prefill contributes one candidate row per
request. Each Decode stage can contribute up to `W` candidate rows.

## Stage ordering and completion

Prefill adds the first generated token and initializes the Decode counter to
zero. The first Decode writes KV at position zero. Each Decode stage writes
current KV, selects successors and applies the parent KV mapping once before
the next Forward. Masks let fixed physical shapes represent inactive rows.

A completion record stores its last token, cumulative score, request-local
parent index, length, finish reason and tie-break index. For length `n > 1`,
follow the record's parent into history at layer `n - 2`. Walk back through
the stored parent indices, reverse the recovered tokens and append the
record's last token. A one-token record refers to the implicit Prefill root.
Reordering active sequences leaves earlier history intact.

## Streams, graphs and retirement

Runner updates execute on the current CUDA stream. Cross-stream producers
and consumers establish ordering through events. Graph replay reads the
current masks and counters at captured addresses. Reuse requires compatible
resource bindings and geometry. Active views are overwritten during execution.
Completed records and history remain valid until reset.

Reset and KV slot release follow the drain of outstanding dispatches and
final consumers. A device error does not roll back KV writes already issued
by the current model Forward.

## Validation

Check state updates and resource retirement with [the state tests](../../../tests/v1/beam/test_beam_state.py),
[the CUDA runner tests](../../../tests/kernels/test_beam_search_runner_cuda.py) and
[the Worker lifecycle tests](../../../tests/kernels/test_gpu_beam_stage_lifecycle.py).
