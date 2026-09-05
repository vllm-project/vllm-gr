# Issue #54 legacy Prefix Cache disabled audit

This audit follows completion of the GPU scope of Step 9. It separates the old
nonpersistent Beam-request defect from the persistent ASYNC path. Reproduction
used clean commit `5fdf454732e9b09018e5668d8ffa458f4eae90b5`, OneRec-1.7B,
RecIF data, W=128, Constraint Tables, and GPU commands submitted through
`gpu run`.

## Reproduction

The fixed B=1/L=1024/S=3 prompt produces legal constraint-table sequences in
both modes, but the complete output changes:

| Legacy compiled case | Full ABI digest | Token digest | Decode Prefix evidence |
| --- | --- | --- | --- |
| Prefix Cache disabled | `737ff9d4c2be…` | `e0bb0528d107…` | 0 queries / 0 hits |
| Prefix Cache enabled, cold before Prefill | `d3805f9c130a…` | `32487d226d59…` | 3,328 queries / 2,048 hits |

The cache-enabled run starts cold, then its reconstructed Decode requests reuse
the same session's Prefill KV. The disabled result is therefore not equivalent
to a real cold-Prefill result on this legacy path.

With FULL Beam Graph requested, the disabled case records zero FULL dispatches;
the enabled case records two. Without cached Prefix blocks, the first legacy
Decode request schedules the 1,024-token Prefix plus W Beam tokens, so it no
longer has the captured W=128 shape.

The real RecIF `ad` 20-sample comparison makes the accuracy effect visible:

| Legacy mode | pass@1 | pass@32 | recall@32 | PID pass@32 | elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefix Cache disabled | 0.00 | 0.00 | 0.000000 | 0.00 | 13.877 s |
| Prefix Cache enabled | 0.00 | 0.15 | 0.065476 | 0.10 | 0.761 s |

Twenty samples are enough to reproduce the defect but not to estimate the full
benchmark's accuracy loss. The disabled case is also about 18x slower here
because it recomputes the Prefix and misses the Beam Graph on every reconstructed
Decode request.

The persistent Graph path with Prefix Cache disabled does not reproduce the
failure: on the same 20 samples it reports pass@1=0.05, pass@32=0.20, and
recall@32=0.040476 in 0.743 s. Its Step 8 fixed-prompt disabled result is exactly
equal to a cold miss, and its 1,000-session Step 9 soak remains stable.

Artifacts:

- `/tmp/vllm-gr-prefix-audit-legacy-compiled-disabled-warm-5fdf454.json`
- `/tmp/vllm-gr-prefix-audit-legacy-compiled-miss-warm-5fdf454.json`
- `/tmp/vllm-gr-prefix-audit-legacy-graph-disabled-5fdf454.json`
- `/tmp/vllm-gr-prefix-audit-legacy-graph-miss-5fdf454.json`
- `/tmp/vllm-gr-prefix-audit-recif-legacy-disabled-5fdf454/`
- `/tmp/vllm-gr-prefix-audit-recif-legacy-enabled-5fdf454/`

## Root cause

The legacy frontend creates a fresh native Request for every Decode step. It
packs the original Prompt and only the newest token from each Beam, with an
explicit assumption that the Scheduler will recover the whole Prefix from
Prefix Cache (`vllm_gr/v1/engine/core.py`, `_apply_beam_step`).

The legacy Scheduler patch only converts a cache hit extending past the exact
Prefix length into a `prefix_len` hit. It also pads and caches the final partial
Prefill block. Neither operation has a cache-disabled branch that transfers KV
ownership to the next native Request
(`vllm_gr/v1/engine/engine_core_patch.py`).

When no hit exists, attention metadata correctly describes the missing Prefix
as standard work. The active incremental Beam session then enters the suffix
pool branch in `BeamAttentionImpl.forward`, writes only the newest Beam-token KV
to the suffix pool, calls cascade attention, and returns. The native
`reshape_and_cache` call that would populate the newly allocated Prefix blocks
is below that early return. Prefix attention can therefore read blocks whose KV
was never populated for this reconstructed Request
(`vllm_gr/v1/attention/backends/beam_attn.py`).

The persistent scheduler avoids all three conditions. It keeps one native
Request, keeps its native Prefix block table, binds that row to the persistent
device state, and schedules one logical control token per Decode. It does not
reconstruct the Request or query Prefix Cache during Decode.

## Repair choices and workload

### 1. Guard the unsupported legacy combination

Reject nonpersistent constrained Beam search when Prefix Cache is disabled and
direct users to persistent ASYNC for B=1. This prevents silent wrong results and
retains the already validated fast path.

Estimated work: **1-2 engineer days**. This includes an admission/config guard,
clear CLI/API error, CPU contracts, one disabled negative GPU case, and one
persistent disabled positive GPU case. It does not add legacy B>1 support.

### 2. Restore legacy correctness by recomputing Prefix KV

Before the incremental suffix-pool branch returns, gather the standard Prefix
rows and their slot mappings and run the platform `reshape_and_cache` operation.
Then use the existing mixed standard/prefix/suffix attention metadata for the
same invocation. The implementation must cover partial blocks, chunked
Prefill, FP8 native KV, mixed ordinary/Beam batches, abort/preemption, eager and
compiled execution, and GPU/NPU platform adapters.

Estimated work: **5-8 engineer days for GPU correctness**, plus **3-5 days for
NPU adaptation and hardware acceptance** when an NPU becomes available. This
keeps correctness but remains slow: every legacy Decode Request is rebuilt and
must recompute the Prefix. The W-only FULL Beam Graph cannot serve that larger
shape.

### 3. Preserve correctness and performance for legacy callers

Give the legacy session explicit ownership of the Prefill native KV blocks
across reconstructed Decode requests, or route its public API internally onto
the persistent Request lifecycle. This requires ownership transfer/refcounting,
block-table rebinding, reset/preemption/abort/shutdown rules, graph-shape
handling, B>1 policy, and the full GPU/NPU lifecycle matrix.

Estimated work: **2-3 engineer weeks**, followed by NPU hardware validation.
Most of this duplicates the persistent path already implemented in Steps 5-7.

## Recommendation

Use option 1 immediately so the invalid combination fails loudly. Treat the
persistent ASYNC path as the supported cache-independent implementation. Only
take option 3 if nonpersistent B>1 compatibility is a product requirement;
otherwise its duplicated ownership and lifecycle surface is not justified.
Option 2 is useful only when legacy correctness is mandatory and its large
latency regression is acceptable.
