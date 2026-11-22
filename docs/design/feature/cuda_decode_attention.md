# CUDA Beam Decode attention

<a id="overview"></a>

Each beam attends to two histories at Decode: the paged prompt KV shared by
all beams of that prompt, and the suffix KV for its own selected tokens.
CUDA can split a long prefix across GPU work or fuse a short suffix with the
final merge. Prefix planning and suffix fusion are independent decisions.

<a id="table-of-contents"></a>

## Architecture

### Shared prefix

Native FlashAttention reads the prompt pages. For eligible FA2 shapes, the
prefix planner can divide those pages into segments when device calibration
predicts a gain. It builds a virtual batch of page ranges for one FA call and
merges partial results in FP32. Empty segments are masked and the final
segment can be shorter. Both prefix paths produce attention output in the
activation dtype and its natural-log normalizer (LSE) in FP32.

![CUDA Decode attention combines shared-prefix output and normalization with per-beam suffix KV through a fused or packed path.](../../assets/cuda_decode_attention.svg)

### Per-beam suffix and final merge

For pure incremental Decode with one query per beam and a live, eligible
pool, the fused Triton kernel reads the beam's suffix KV, computes attention
and combines it with the prefix state in one launch. The packed path copies
suffix KV into a packed layout, runs suffix attention and merges the partial
outputs. Both paths handle an empty prefix.

A valid mixed standard/Beam batch uses the packed path. A missing session or
multiple queries per incremental beam is an input error, detected before
attention or cache writes. With fusion enabled, invalid static configurations
fail at initialization. Fusion requires a configured maximum width, a positive
pool capacity, an integral query-to-KV head ratio and unwindowed attention
without ALiBi, softcap or sinks.

### CUDA graphs and fallback

For eligible segmented shapes, both CUDA runners select a page-aligned prefix
bucket from host metadata. The graph key includes batch size, beam width and
prefix bucket. Replay updates lengths, page IDs and active rows at stable
addresses. Calibration stays fixed while serving. A new bucket can incur
first-use capture and graph memory. A model-capacity fallback graph is
available. The suffix kernel reads the current step and active mask from
device buffers during replay.

## Configuration

| GR setting | Effect |
| --- | --- |
| `prefix_policy` | `off` keeps native prefix attention. `auto` allows calibrated segmentation of eligible paged FA2 shapes. |
| `prefix_calibration`, `prefix_calibration_path`, `prefix_min_chain_us` | Control measurement, the profile location and the minimum predicted unsplit chain time for segmentation. |
| `enable_short_suffix_attention` | Defaults to `true` on eligible CUDA CUSTOM engines. Set to `false` under `gr` at startup to use the packed path. |

An explicit FlashAttention version takes precedence. FA3 and FA4 keep native
splitting heuristics. Batch-invariant mode disables splitting. The calibrated
path requires plain paged FP16 or BF16 attention without windows, ALiBi,
softcap or sinks. See the [startup configuration](../../configuration/README.md).

## Performance

A historical A/B run compared the packed and fused suffix paths at
`27ce79160f5bdb098159136747663e83f79bcb7c`. It used Qwen3-0.6B BF16 on an
NVIDIA L20, ModelRunner V1, one 128-token prompt, beam width 128 and three
generated tokens. CUDA graphs were enabled and prefix caching was disabled.
Each fresh engine ran ten warmups and 100 unprofiled requests.

| Independent pair | Packed suffix p50 | Fused suffix p50 |
| --- | ---: | ---: |
| First | 13.334 ms | 11.028 ms |
| Second | 13.605 ms | 11.084 ms |

## Validation

The [prefix planner tests](../../../tests/v1/attention/test_beam_prefix_policy.py)
cover calibration and graph selection. The [segmented attention tests](../../../tests/kernels/attention/test_beam_prefix_segmented.py)
cover ragged prefixes, empty segments and replay. The [suffix tests](../../../tests/kernels/attention/test_beam_short_suffix.py)
compare against packed FA2 and an independent dense reference. The FA2-aligned
path preserves packed rounding for FP16/BF16 pools of one to three steps,
head dimensions 64, 128 or 256 and up to 16 query heads per KV head. Other
supported shapes use generic FP32 accumulation.

## Related files

- [CUDA attention backend](../../../vllm_gr/v1/attention/backends/beam_attn_gpu.py) dispatches prefix and suffix work.
- [Prefix planner](../../../vllm_gr/v1/attention/backends/beam_prefix_policy.py) selects native or segmented prefix attention.
- [Short-suffix kernel](../../../vllm_gr/v1/attention/backends/beam_short_suffix.py) computes suffix attention and the final merge.
