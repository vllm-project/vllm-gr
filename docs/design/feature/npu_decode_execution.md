# Ascend Beam Decode execution

On Ascend, beams from the same prompt share its paged KV cache. Each beam has
its own suffix KV for the tokens it generates. At each Decode step, the worker
chooses how to compute attention over both caches and, after selecting
candidates, how to carry the surviving suffix histories into the next step.
Each choice can change from one step to the next.

![Two decision flows: attention chooses xllm or cascade, then KV reorder chooses fused or cascade using the selected parent indices.](../../assets/npu_decode_dispatch.svg)

| Color or shape | Meaning |
| --- | --- |
| Green | Decode rows and session KV. |
| Yellow diamond | Independent dispatch check. |
| Cyan | Fused xllm path. |
| Orange | Cascade path. |
| Pink | Candidate selection between the two decisions. |

## How it works

### Per-step dispatch

The [Ascend stage runner](async_scheduling.md#ascend-stage-execution) expands
each logical request into one row per input beam. The model forward writes
the new suffix KV into the session's pool and reads it alongside the paged
prompt KV for attention. The worker then selects the surviving candidates.
Each candidate's parent index identifies the suffix history that the next
Decode step should read.

### Attention backend selection

| Path | When it runs | What it does |
| --- | --- | --- |
| xllm XAttention | An eligible incremental batch containing only Beam Decode rows, with xllm mode enabled. | Computes prefix and suffix attention in one operator. |
| Cascade | Other valid batches, including batches with both standard and Beam rows. | Computes prefix and suffix attention separately, then merges the results. |

The fused path requires the standard `softmax_scale` of `1 / sqrt(head_dim)`
and a beam row count divisible by the request's beam width. A mismatch raises
an error. Before computing attention, the backend checks for a Beam session
and valid Decode metadata. Ascend CUSTOM attention rejects sliding windows,
ALiBi, nonzero logits softcap, attention sinks, and FP8 KV caches.

### KV reorder

Before the next step, the worker places each selected parent's suffix KV in
the new beam's row.

| Reorder path | Requirements |
| --- | --- |
| Fused `select_unshared_kv` | Requires `xllm` Decode mode, enough survivors to fill the pool's beam dimension (`beam_max_width` for a full-width pool), and parent indices grouped in ascending source-beam order. |
| Cascade reorder | Any other case, including unsorted parents or a shrinking retained width. |

Attention and KV reorder choose their paths independently. For example, a
step can use XAttention and then use cascade reorder if selection shrinks the
retained width. Two winners from the same parent inherit the same suffix KV
and can append different tokens at the next step. Both read the request's
paged prompt KV.

### Caller-owned XAttention output

With the optional output binding, XAttention can write a homogeneous
incremental Beam batch's result directly into the caller's ND-format buffer.
The leading view `output[:num_prefix_indices]` must match the gathered beam
query in shape, dtype (BF16 or FP16), and device. The buffer must be contiguous
and use storage separate from every input. The view can start at a nonzero
storage offset.

| Binding and destination | Action |
| --- | --- |
| Compatible binding and buffer | Invoke XAttention with the caller's buffer. |
| Binding absent or incompatible, or the result needs to be scattered | Use the allocating API, then copy or scatter the result into the destination. |
| Native call fails after it starts | Propagate the error. The destination may already hold partial output. |

The loader verifies the build receipt, Python ABI, runtime versions and the
identities of the binding, operator library and ACLNN provider.

### Decode graph capture

FULL graph replay covers the entire Beam Decode forward pass when the batch
meets these requirements. `W` is the captured Decode input beam width per
request.

- The batch contains only Beam Decode work and uses fused xllm Decode.
- The batch has an active Beam session starting at pool slot zero, where the
  captured addresses match the session's buffers.
- Each request has a complete cached prefix and maps to exactly `W`
  consecutive rows in request order.
- The batch has zero active LoRA adapters.
- Tensor parallelism is 1 and data-parallel ubatching is disabled.
- `max_model_len > W`.

Startup capture uses one configured logical request count. Scheduler capacity
is independent of that count. Dynamic Beam schedules can capture additional
Decode input widths. Replay requires an exact match for the captured request
count, input width, and physical token bucket. If a check fails, the worker
uses native PIECEWISE or eager execution as applicable. The per-step attention
checks also apply in those modes. See the
[ModelRunner dispatch code](../../../vllm_gr/v1/worker/npu/npu_model_runner_patch.py)
for the exact checks.

## Configuration

| `gr.npu_decode_execution` | Backend selection at startup |
| --- | --- |
| `null` (default) | Select xllm when xllm-ops is available and cascade when it is unavailable. |
| `xllm` | Require xllm-ops. Use XAttention for eligible incremental Beam batches. |
| `cascade` | Always use separate prefix and suffix attention with a merge. |

`gr.beam_graph_enabled` allows Beam Decode graph use. Device requirements and
serving settings are in [Beam search on NPU](../../user_guide/beam_search_npu.md).

## Performance

One XAttention call handles both attention contributions and avoids the
separate merge used by cascade. The optional output binding also saves a
temporary allocation and a copy when it can write to the caller's buffer.
Follow the [binding verification steps](../../user_guide/beam_search_npu.md#optional-xattention-output-binding)
to compare outputs and check graph replay on an Ascend device.

## Validation

The [Ascend attention parity tests](../../../tests/kernels/test_beam_attn_npu_xllm_parity.py)
compare xllm with cascade. The [binding tests](../../../tests/ops/npu/test_npu_xattention_out_binding.py)
check how the loader discovers and validates the optional
`vllm_gr_xattention_out_ext` extension.

## Source map

| File | Responsibility |
| --- | --- |
| [xllm attention](../../../vllm_gr/v1/attention/backends/beam_attn_npu_xllm.py) | Implements fused Decode, its dispatch checks and the caller-owned output path. |
| [Cascade attention](../../../vllm_gr/v1/attention/backends/beam_attn_npu.py) | Computes separate prefix and suffix contributions. |
| [Ascend stage runner](../../../vllm_gr/v1/worker/npu/npu_beam_stage_runner.py) | Expands requests into beam rows and selects candidates. |
| [KV reorder dispatcher](../../../vllm_gr/v1/beam/reorder.py) | Checks the decode mode and parent layout to choose fused or cascade reorder. |
