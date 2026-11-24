# CUDA Beam Decode attention

During Decode, beams from the same prompt share its paged KV. Each beam also
keeps suffix KV for the tokens it has selected. CUDA computes attention over
the two parts separately and merges their results. Splitting a long prefix
exposes more parallel work, and fusing a short suffix with the merge avoids
packing its KV. Either suffix path can use the result of a segmented prefix.

![Block diagram: shared prompt pages feed native or segmented prefix attention, which produces prefix output and FP32 LSE; both suffix paths combine the prefix state with per-beam suffix KV to produce per-beam output.](../../assets/cuda_decode_attention.svg)

| Color or shape | Meaning |
| --- | --- |
| Green | Shared input: prompt pages or per-beam suffix KV. |
| Orange | Attention work. |
| Pink | Partial state carried between stages. |
| Cyan | Final per-beam output. |
| Yellow diamond | Independent suffix-path choice. |

## How it works

### Shared prompt prefix

With the default `off` policy, native FlashAttention reads the shared prompt
pages. Set `prefix_policy: auto` to let the planner use device calibration
to segment eligible FA2 prefixes. A segmented plan must meet all of the
following conditions.

- A matching calibration profile is available and predicts at least two
  segments.
- The predicted speedup is at least 1.05x (`native / predicted >= 1.05`).
- The predicted time for the unsplit chain is at least `prefix_min_chain_us`
  and at least 1.5x the larger of the estimated compute and memory costs.
- The bucket's page count divides evenly by the segment count.
- The attention metadata allows splitting (`max_num_splits == 0`), and the
  block table can hold the segment layout.

The segmented path builds a virtual batch of page ranges for one
FlashAttention call, then merges the partial results in FP32. It masks empty
segments and supports a shorter final segment. Both prefix paths return
attention output in the activation dtype and the softmax log-sum-exp (LSE)
in FP32.

### Per-beam suffix and final merge

The fused path serves pure incremental Decode with one query per beam and a
live, eligible pool. It reads suffix KV directly from the pool, computes
suffix attention and merges it with the prefix state in one launch. Eligible
shapes use Gluon, and the remaining fused shapes use the generic Triton
kernel. The packed path copies suffix KV into a packed buffer before running
suffix attention and merging the partial outputs. Both paths handle an empty
prefix.

Valid batches that mix standard and Beam rows use the packed path. The
backend rejects a missing Beam session or multiple queries per incremental
beam before attention or cache writes.

Fusion requires a CUDA device, a configured maximum width, a positive pool
capacity and a model dtype of FP16, BF16 or FP32. The attention layer needs
an available FA2/FA3/FA4 backend, at least one KV head, an integral query-to-KV
head ratio and a head size in `(0, 256]` that is a multiple of 8. It also
requires unwindowed attention without ALiBi, softcap or sinks. Invalid static
configurations fail at initialization.

### Decode graphs and fallback

For eligible segmented shapes, the async metadata builder chooses a prefix
bucket from the longest prefix in the batch. It rounds lengths up to 1024
tokens to a multiple of the block size and rounds longer lengths to a
multiple of 1024, capped at the block-aligned model limit. The generic
builder leaves the bucket at zero. The graph key includes batch size, beam
width and prefix bucket.

The worker captures the startup graph at model capacity with a bucket of
zero. It captures length-bucket graphs on first use, so a new bucket can add
capture time and graph memory. During replay, the worker updates lengths,
page IDs and active rows in buffers at stable addresses. The suffix kernel
reads the current step and active mask from device buffers. Calibration stays
fixed while serving. Decode can run eagerly if graph replay is unavailable.

## Configuration

| GR setting | Effect |
| --- | --- |
| `prefix_policy` | `off` (default) keeps native prefix attention. `auto` allows calibrated segmentation of eligible paged FA2 shapes. |
| `prefix_calibration` | Set to `false` to disable calibration and keep native prefix attention. |
| `prefix_calibration_path`, `prefix_min_chain_us` | Select the calibration profile and the minimum predicted unsplit chain time for segmentation. |
| `enable_short_suffix_attention` | Defaults to `true` on eligible CUDA CUSTOM engines. Set to `false` under `gr` at startup to use the packed path. |

`attention_config.flash_attn_version` affects Decode prefix segmentation.
An explicit FA2 selection allows segmentation for eligible shapes. FA3 and
FA4 keep their native splitting heuristics. `gr.prefill_flash_attn_version`
applies only to Prefill. Batch-invariant mode disables splitting. The
calibrated path requires plain paged FP16 or BF16 attention without windows,
ALiBi, softcap or sinks. See the
[startup configuration](../../configuration/README.md).

## Performance

Measure with the [Beam performance workload](../../cli/bench/beam_performance.md).
Keep the model, prompt, batch size, beam width and Prefill mode fixed across
runs. Record the revision, device, model dtype, runner, Decode length, warmup
count and measured runs with the results. Follow the
[benchmark acceptance policy](../../contributing/ci/benchmark_acceptance.md)
when comparing the paths.

## Validation

The [prefix planner](../../../tests/v1/attention/test_beam_prefix_policy.py),
[segmented attention](../../../tests/kernels/attention/test_beam_prefix_segmented.py)
and [suffix](../../../tests/kernels/attention/test_beam_short_suffix.py) tests
cover calibration, ragged prefixes, empty segments, replay, and parity with
packed FA2 and an independent dense reference.

## Source map

| File | Responsibility |
| --- | --- |
| [CUDA attention backend](../../../vllm_gr/v1/attention/backends/beam_attn_gpu.py) | Dispatches prefix and suffix work and checks fusion eligibility. |
| [Prefix planner](../../../vllm_gr/v1/attention/backends/beam_prefix_policy.py) | Selects native or segmented prefix attention. |
| [Short-suffix kernels](../../../vllm_gr/v1/attention/backends/beam_short_suffix.py) | Dispatch to the fused Gluon kernel or the generic Triton merge. |
| [FA2-aligned suffix kernel](../../../vllm_gr/v1/attention/backends/beam_short_suffix_fa2.py) | Implements the Gluon path for eligible shapes. |
