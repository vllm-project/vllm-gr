# Beam decode prefix optimization

Prefix attention returns activation-dtype `out[tokens, query_heads, head_dim]`
and FP32 natural-log `LSE[query_heads, tokens]`. The existing suffix kernel
consumes this interface without knowing which FlashAttention backend produced it.
Empty paged prefixes are normalized before merging, including FA2 builds that
leave empty-request LSE uninitialized. Prefill and persistent KV writes are independent.

## Selection and compatibility

Without an explicit choice, vLLM selects a supported FA version as usual. The
suffix fusion setting no longer forces FA2. The prefix policy compares:

| Plan | Execution | Capability constraint |
| --- | --- | --- |
| Native | Backend default splitting | Any available FA2/FA3/FA4 adapter |
| Native split | Explicit `num_splits`, including `1` to avoid automatic splitting | FA3/FA4 adapter; FA2 varlen rejects values above 1 |
| Framework segmented | One virtual-batch FA call over request × KV segment, followed by stable LSE merge | Plain paged FP16/BF16 prefix; no window, ALiBi, softcap or sinks |

The default is ON for CUDA/FA2: the segment count comes from the device-local
cost model described in the next section. FA3/FA4 are never overridden, because
their own split-K heuristic already pays a single fused combine. There are no
L20, SM89 or SM90 dispatch branches.

Framework segmentation reuses KV page IDs and copies only queries and compact
metadata. All segments enter a **single** FA call, exposing additional parallel
work; sequential Python calls would not achieve this. The merge uses FP32 online
log-sum-exp accumulation and masks empty segments before loading their outputs
or LSE. It handles uneven final segments and arbitrary request query lengths.
Workspace and merge cost can outweigh the benefit; segmentation is not enabled
just because it is supported.

Batch-invariant mode retains the original version with splitting disabled, and an
explicit `flash_attn_version` captured at start-up always wins. Length buckets are
1024-aligned and clamped to the model bound, so every supported segment count
divides the bucket evenly.

## Device calibration

The four cost-model coefficients (serial chain, attention FLOPs, KV bytes and
per-segment overhead) are measured once per `(device, geometry)` at engine
start-up, before any Beam graph is captured, and cached in a commented YAML file
(`$XDG_CACHE_HOME/vllm_gr/prefix_calibration.yaml`, overridable with
`VLLM_GR_PREFIX_CALIBRATION_PATH`). Every worker measures its own device and
writes the same entry, so no lock is needed; a torn file simply fails to parse
and is re-measured. A plan is installed only when the model predicts at least a
5% win and the unsplit chain is at least 1.5x above the reachable floor, so
compute-bound wide-beam shapes stay native.

`VLLM_GR_PREFIX_CALIBRATION=off` skips measurement and `VLLM_GR_PREFIX_POLICY=off`
disables the whole segmentation path; delete the file to re-measure. Every
failure path logs a warning and continues on the native path.

## CUDA graphs and fallback

The asynchronous V1/V2 Beam stage path obtains lengths from existing CPU
metadata, never from device `.item()` calls. Only a shape present in the matching
profile activates a power-of-two, page-aligned length bucket, capped at model
capacity. Graph signatures include `(B, W, prefix bucket)`; the profile is immutable
so a captured bucket also fixes its backend/split plan. Replay changes lengths,
page IDs, queries and active rows at stable addresses, without selecting or tuning
kernels on the device.

New buckets capture lazily against real request buffers. This introduces a
one-time capture cost and additional graph memory for each measured shape.
Startup retains the original model-capacity fallback graph. Replacing V1 input
storage discards every length bucket for the affected geometry. Dynamic-beam
buffer invalidation continues to clear its graphs; V2 recoverable capture failures
are isolated by geometry and bucket. Legacy paths without length-bucket metadata
retain their original graph bounds and only match profiles for those bounds.

## Validation status

See the PR description and its dated validation note for checks actually run.
Backend capability filtering and CPU graph-cache tests do not establish GPU
performance on untested architectures. FA4 requires its optional CuTe dependencies.
Numerical comparisons use tolerances; changing backend or reduction order does
not promise bitwise equality or identical final beam ranking.
