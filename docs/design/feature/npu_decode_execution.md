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

### Candidate selection

In the default full-vocabulary mode, each parent row is normalized in FP32 over
its complete finite model vocabulary before Top-K truncation. A temperature
of zero uses unscaled logits. Positive temperatures scale the logits before
normalization. A token's log probability stays the same when only the
requested candidate count changes.

The initial selector is a native PyTorch baseline that sorts the complete
vocabulary. Its purpose is functional support; no performance improvement over
constrained selection is implied. Large vocabularies and wide Beam batches can
make this selection step substantial.

NaN and infinite logits are excluded. When fewer than K finite tokens are
available, remaining slots have `valid_mask=False` and a score of negative
infinity. A completely invalid row produces no valid candidates. Exact score ties
prefer the smaller token ID. These guards make the candidate contract explicit;
they do not imply bitwise parity across different model kernels or devices.

### SID stage selection

With [SID token families enabled](../../configuration/README.md#sid-token-families-without-a-catalog),
Prefill selects SA, the first Decode selects SB, and the second selects SC.
The engine validates the tokenizer's family IDs and retains all members;
family sizes and ID continuity are not hardcoded. A checkpoint with 8192
tokens per family projects to 8192 columns per stage. This path constructs no
catalog artifact or fully connected triples and does not call constrained TopK.

The NPU stage projection owns three weight caches and refreshes them through
the weight-update lifecycle. Ordinary generation and startup dummy calls use
the full LM head.

Each parent is normalized across the complete current SID family before Top-K.
This matches a full projection with other families masked before softmax;
it differs from normalization across the full vocabulary. Compare sliced and
full heads with the same stage domain when checking correctness.

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
serving settings are in [Ascend execution limits](../../configuration/README.md#ascend-npu-execution-limits).

## Performance

XAttention combines prefix and suffix attention in one operator. On compatible
batches, the optional output binding also removes a temporary allocation and a
copy. The [binding verification steps](#optional-xattention-output-binding)
show how to compare outputs on an Ascend device and check graph replay.

## Validation

The experimental full-vocabulary path has five- and six-token model validation
for Qwen3-0.6B-OneRecVocab BF16, B=1/W=128, eight KV heads, head dimension 128,
ModelRunner V1, and eager and FULL Decode Graph execution. This does not
establish support for other model geometries or arbitrary generation horizons.
Six-token generation uses one output-producing Prefill and five Decode
forwards. [Startup admission](../../configuration/README.md#full-vocabulary-generation)
checks capacity independently of a request's length.

Run the focused regressions in the complete Ascend environment:

```bash
python -m pytest tests/v1/engine/test_npu_unconstrained_beam.py \
  tests/v1/engine/test_npu_unconstrained_runner_config.py \
  tests/ops/npu/test_npu_sid_stage_vocab.py \
  tests/engine/test_npu_sid_stage_config.py \
  tests/v1/worker/test_npu_sid_stage_loading.py \
  tests/ops/test_npu_stage_logits.py \
  tests/entrypoints/beam_search/test_beam_search_input_processor.py \
  tests/v1/worker/test_npu_model_runner_patch.py \
  tests/kernels/test_unconstrained_topk_npu.py \
  tests/kernels/test_sid_stage_topk_npu.py
```

The [Ascend attention parity tests](../../../tests/kernels/test_beam_attn_npu_xllm_parity.py)
compare xllm with cascade; the [binding tests](../../../tests/ops/npu/test_npu_xattention_out_binding.py)
cover discovery of the optional `vllm_gr_xattention_out_ext` extension.

## Optional XAttention output binding

The [caller-owned output path](#caller-owned-xattention-output)
uses a separate extension around the installed kernel. The build needs the
xllm-ops helper header `test/python_test/pytorch_npu_helper.hpp` and checks
the runtime versions before compiling.

Run these commands inside the prepared NPU environment, from the vLLM-GR
repository root after completing [Ascend installation](../../getting_started/installation/npu.md).
Set `XLLM_OPS_DIR` to the source checkout used to build the
installed xllm-ops library. Keep the CANN and custom operator environment
loaded in the current shell.

```bash
export XLLM_OPS_DIR=/path/to/the/installed/xllm-ops/source
export XATTENTION_OUT_BUILD="$PWD/build/xattention-out"

# Inspect the build plan. This command does not compile or install anything.
python tools/npu/xattention_out/build.py \
  --xllm-ops-dir "$XLLM_OPS_DIR" \
  --output-dir "$XATTENTION_OUT_BUILD"

# Build the optional binding in a fresh directory.
python tools/npu/xattention_out/build.py \
  --execute \
  --xllm-ops-dir "$XLLM_OPS_DIR" \
  --output-dir "$XATTENTION_OUT_BUILD"

export PYTHONPATH="$XATTENTION_OUT_BUILD/lib:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch_npu, custom_ops_lib, vllm_gr_xattention_out_ext as m; print(m.__file__)'
```

Keep the receipt next to the binary and the output directory on `PYTHONPATH`
for spawned workers. Restart the engine after changing the extension. To use
the allocating route, start the engine without that directory on `PYTHONPATH`.

Validate against the installed allocating kernel in the prepared environment:

```bash
VLLM_GR_TARGET_DEVICE=npu python -m pytest \
  tests/ops/npu/test_npu_xattention_out.py \
  tests/ops/npu/test_npu_xattention_out_binding.py -q
VLLM_GR_TARGET_DEVICE=npu VLLM_GR_TEST_XATTENTION_OUT_REQUIRED=1 \
  python -m pytest tests/kernels/test_x_attention_out_npu.py -q
```

The required-addon device test checks parity and graph replay and fails
when the extension is unavailable. Full-model validation depends on the
intended request geometry and catalog.

## Source map

| File | Responsibility |
| --- | --- |
| [xllm attention](../../../vllm_gr/v1/attention/backends/beam_attn_npu_xllm.py) | Implements fused Decode, its dispatch checks and the caller-owned output path. |
| [Cascade attention](../../../vllm_gr/v1/attention/backends/beam_attn_npu.py) | Computes separate prefix and suffix contributions. |
| [Ascend stage runner](../../../vllm_gr/v1/worker/npu/npu_beam_stage_runner.py) | Expands requests into beam rows and selects candidates. |
| [KV reorder dispatcher](../../../vllm_gr/v1/beam/reorder.py) | Checks the decode mode and parent layout to choose fused or cascade reorder. |
