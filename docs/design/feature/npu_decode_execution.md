# Ascend Beam Decode execution

Each Ascend Decode step reads the shared prompt KV and each beam's own suffix
KV. The worker chooses one of two attention paths, then chooses how to reorder
the surviving suffix histories. Both choices are made per step and can differ
between steps of the same request.

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
each logical request into one row per input beam. The forward reads paged
prompt KV and session-owned suffix KV and writes the new KV. Once the worker
selects the winners, their parent indices identify the histories needed by the
next Decode.

### Attention backend selection

| Path | When it runs | What it does |
| --- | --- | --- |
| xllm XAttention | A homogeneous incremental Beam batch on a build with xllm-ops. | Computes prefix and suffix attention in one operator. |
| Cascade | Every other valid batch, including a mixed standard/Beam batch. | Runs separate prefix and suffix attention, then merges the results. |

The fused path additionally requires `softmax_scale` to equal
`1 / sqrt(head_dim)` and the batch's beam rows to be divisible by the request
Beam width; it raises on either mismatch. Missing sessions and invalid Decode
metadata raise before attention runs. Ascend CUSTOM attention rejects sliding
windows, ALiBi, nonzero logits softcap, attention sinks and FP8 KV caches.

### KV reorder

After selection, the worker reorders the surviving suffix histories from the
old beam order to the new one.

| Reorder path | Requirements |
| --- | --- |
| Fused `select_unshared_kv` | Decode mode is `xllm`; the pool's beam dimension is full width (`beam_max_width`); parent indices are grouped by ascending source beam index. |
| Cascade reorder | Any other case, including unsorted parents or a shrinking retained width. |

The two decisions are independent. A step can use XAttention, shrink the
retained width and then use cascade reorder. Two winners with the same parent
inherit the same suffix so far and can append distinct tokens at the next
step; both still read the request's paged prompt KV.

### Caller-owned XAttention output

A homogeneous incremental Beam batch can write directly to a caller-supplied
ND-format output buffer. The leading view `output[:num_prefix_indices]` must
match the gathered beam query's shape, dtype (BF16 or FP16) and device, the
buffer must be contiguous and have storage separate from every input, and the
view may start at a nonzero storage offset.

| Binding and destination | Action |
| --- | --- |
| Compatible | Invoke XAttention with the caller's buffer. |
| Binding absent or incompatible, or the output needs scatter | Use the allocating API and copy or scatter the result. |
| The native call fails after it starts | Propagate the error; the destination may already hold partial output. |

The loader verifies the build receipt, Python ABI, runtime versions and the
identities of the binding, operator library and ACLNN provider.

### Decode graph capture

FULL Beam Decode replay requires all of the following:

- An active Beam session whose pool slot is zero, so the graph's captured
  addresses match the session's buffers.
- Fused xllm Decode and a batch that contains only Beam Decode work.
- Each request mapped to exactly `width` identity rows with a complete prefix.
- Zero active LoRA adapters.
- Tensor parallelism of 1, no data-parallel ubatching, and
  `max_model_len > W`.

Startup captures use an exact logical request count; scheduler capacity is
configured separately. Dynamic Beam schedules can capture additional exact
Decode input widths. Replay checks the batch size, input width and physical
token bucket against the capture. When a condition is unmet, dispatch keeps
native PIECEWISE or eager execution as applicable. See the
[ModelRunner dispatch code](../../../vllm_gr/v1/worker/npu/npu_model_runner_patch.py)
for the exact checks.

## Configuration

| `gr.npu_decode_execution` | Backend selection at startup |
| --- | --- |
| `null` (default) | Use xllm if xllm-ops is available, otherwise cascade. |
| `xllm` | Require xllm-ops. Use XAttention for eligible incremental Beam batches. |
| `cascade` | Always use separate prefix and suffix attention with a merge. |

`gr.beam_graph_enabled` allows Beam Decode graph use. Device requirements and
serving settings are in [Beam search on NPU](../../user_guide/beam_search_npu.md).

## Performance

XAttention combines prefix and suffix attention in one operator. On compatible
batches, the optional output binding also removes a temporary allocation and a
copy. The [binding verification steps](../../user_guide/beam_search_npu.md#optional-xattention-output-binding)
show how to compare outputs on an Ascend device and check graph replay.

## Validation

The [Ascend attention parity tests](../../../tests/kernels/test_beam_attn_npu_xllm_parity.py)
compare xllm with cascade; the [binding tests](../../../tests/ops/npu/test_npu_xattention_out_binding.py)
cover discovery of the optional `vllm_gr_xattention_out_ext` extension.

## Source map

| File | Responsibility |
| --- | --- |
| [xllm attention](../../../vllm_gr/v1/attention/backends/beam_attn_npu_xllm.py) | Implements fused Decode, its dispatch checks and the caller-owned output path. |
| [Cascade attention](../../../vllm_gr/v1/attention/backends/beam_attn_npu.py) | Computes separate prefix and suffix contributions. |
| [Ascend stage runner](../../../vllm_gr/v1/worker/npu/npu_beam_stage_runner.py) | Expands requests into beam rows and selects candidates. |
| [KV reorder dispatcher](../../../vllm_gr/v1/beam/reorder.py) | Checks the decode mode and parent layout to choose fused or cascade reorder. |
