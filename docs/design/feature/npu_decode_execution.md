# Ascend Beam Decode attention

<a id="overview"></a>

Each Ascend Decode step reads shared prompt KV and each beam's own suffix
KV. The worker can compute attention with the fused xllm XAttention operator
or with separate prefix and suffix attention calls followed by a merge
(cascade). After selection, a second decision determines how the worker
reorders the surviving suffix histories.

<a id="table-of-contents"></a>

## Architecture

### Backend selection and fallback

The [Ascend stage runner](async_scheduling.md#ascend-stage-execution) expands
each logical request into one row per input beam. The forward reads paged
prompt KV and session-owned suffix KV and writes the new KV. Once the worker
selects the winners, their parent indices identify the histories needed by
the next Decode.

![Two Ascend dispatch decisions, first for attention on the current rows and then for KV reorder using the selected parent indices.](../../assets/npu_decode_dispatch.svg)

A valid mixed standard/Beam batch uses cascade attention. Fused reorder
requires full-width parents grouped by ascending source beam index. Unsorted parents or
a shrinking width use cascade reorder. For example, a step can use XAttention,
shrink the retained width and then use cascade reorder. Two winners with the
same parent inherit the same suffix so far and can append distinct tokens at
the next step. Both continue to share the prompt pages.

Missing sessions and invalid Decode metadata raise errors. Ascend CUSTOM
attention rejects sliding windows, ALiBi, nonzero logits softcap, attention
sinks and FP8 KV caches.

### Caller-owned XAttention output

A homogeneous incremental Beam batch can write directly to the caller's
contiguous ND-format output. The buffer must match the query's shape and
dtype (BF16 or FP16) and have separate storage from every input. A view may
have a nonzero storage offset.

| Binding and destination | Action |
| --- | --- |
| Compatible | Invoke XAttention with the caller's buffer. |
| Binding absent or incompatible, or output needs scatter | Use the allocating API and copy or scatter the result. |
| Native call fails after it starts | Propagate the error. The destination may already contain partial output. |

The loader verifies the build receipt, Python ABI, runtime versions and
identities of the binding, operator library and ACLNN provider.

## Configuration

| `gr.npu_decode_execution` | Backend selection at startup |
| --- | --- |
| `null` (default) | Select xllm if xllm-ops is available, otherwise cascade. |
| `xllm` | Require xllm-ops. Use XAttention for eligible incremental Beam batches. |
| `cascade` | Use separate prefix and suffix attention and merge their outputs. |

See [Beam search on NPU](../../user_guide/beam_search_npu.md) for device
requirements and serving configuration.

### Decode graph capture

FULL Beam Decode replay requires an active Beam session, fused xllm Decode,
and a batch that contains only Beam Decode work. The pool layout must match
the capture, and the batch must have zero active LoRA adapters.

- `gr.beam_graph_enabled` allows Beam Decode graph use.
- Startup capture uses an exact logical request count. Scheduler capacity is
  configured separately.
- Dynamic Beam schedules can capture additional exact Decode input widths.
  Replay checks batch size, input width, and physical token bucket against the
  capture. See the [packaged defaults](../../../vllm_gr/config/default_config.yaml)
  for the capture settings.

When these conditions are unmet, dispatch keeps native PIECEWISE or eager
execution as applicable. See the
[ModelRunner dispatch code](../../../vllm_gr/v1/worker/npu/npu_model_runner_patch.py)
for the capture and replay checks.

## Performance

XAttention combines the prefix and suffix in one operator. On compatible
batches the optional output binding also removes a temporary allocation and
copy. The [binding verification steps](../../user_guide/beam_search_npu.md#optional-xattention-output-binding)
show how to compare the outputs on an Ascend device and check graph replay.

## Validation

The [Ascend attention parity tests](../../../tests/kernels/test_beam_attn_npu_xllm_parity.py)
compare xllm and cascade. The [binding tests](../../../tests/ops/npu/test_npu_xattention_out_binding.py)
exercise discovery of the optional `vllm_gr_xattention_out_ext` extension.

## Related files

- [xllm attention](../../../vllm_gr/v1/attention/backends/beam_attn_npu_xllm.py) implements fused Decode, its dispatch checks and the caller-owned output path.
- [Cascade attention](../../../vllm_gr/v1/attention/backends/beam_attn_npu.py) computes separate prefix and suffix contributions.
- [Ascend stage runner](../../../vllm_gr/v1/worker/npu/npu_beam_stage_runner.py) selects candidates. The [KV reorder dispatcher](../../../vllm_gr/v1/beam/reorder.py) checks the parent layout.
