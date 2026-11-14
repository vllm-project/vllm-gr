# RecoGEM-inspired CUDA decode optimizations

These changes build on the existing KV ping-pong reorder. The new execution
options are opt-in and are evaluated independently on the same implementation
with one configuration change per A/B comparison.

## Complete Decode step capture

`beam.enable_full_step_graph` retains the native CUDA model Forward graph and
adds a graph covering LM-head projection, canonical constrained Top-K, Beam
selection, sequence/state updates, KV reorder and device progress/error control.
The two graphs execute in stream order. It currently applies to fixed-width,
constrained ModelRunner V1 Decode. Prefill, ModelRunner V2, dynamic widths and
explicit eager execution retain their existing paths.

Reusing the native Forward executable preserves its kernel choices and input
bindings. Warmup of the postprocess graph restores the admitted Beam state,
both KV banks and control buffers. Capture preserves that state; its first
replay advances it once. Capture uses inference mode and retains its native
hidden-state input and all captured outputs for the executable lifetime.

Each step, length bucket and dispatch-control slot has its own allocator pool.
These signatures can replay in different orders, so their model temporaries
must not share an allocator pool. Compatible drained sessions retain captures;
changing Beam state storage clears them. Existing resource validation still
precedes reuse. Cancellation, terminal packing and consumer retirement remain
outside capture and use the existing release protocol.

Enable the option in the engine's existing configuration before startup:

```python
vllm_gr_config["beam"]["enable_full_step_graph"] = True
```

The legacy key is `vllm_gr_enable_full_step_graph`. The default is `False`.
Restart the engine when changing the option. Separate captured pools add memory
and first-use capture work; warm every measured geometry before collecting latency.

## Measurement protocol

The local comparison uses NVIDIA L20, BF16, TP=1, native ModelRunner V1,
PyTorch 2.13.0+cu130, vLLM 0.30.0 and Triton 3.7.1. Every A/B arm uses
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization. Default BF16 comparisons retain the engine's reduction setting. The optional
`--strict-bf16` is a separate numerical configuration and is not a runtime
optimization flag. Runtime code never silently changes the global BF16 setting.

The execution order is off/on/on/off. Each independently initialized process
runs ten warmups and 100 measured requests, yielding 200 requests per side
for each configuration. Qwen profiles five additional requests after timing
in every process; profile samples are excluded from reported latency.

- Qwen uses the existing ID-based benchmark, fixed JSONL workloads, the same
  synthetic constraint artifact and context/beam combinations 128/128 and
  5000/256. It generates three SID-like tokens and records all ordered candidates.
- OneRec uses `benchmarks.open_one_rec.one_rec_main bench throughput`, 100
  real RecIF video rows, 1024 input tokens including SID begin, beam width 128,
  and three generated SID tokens. Text encoding and all Beam text decoding
  are included. Prefix-cache reset and validation happen outside the timer.

Model files come from the official public ModelScope mirrors, with locally
verified file hashes. This local A/B does not establish the pinned Hugging Face
revision required by the official CI recall comparator.

The following switches are available in both model benchmark entrypoints:

```text
--full-step-graph / --no-full-step-graph
--strict-bf16 (requires VLLM_ENABLE_V1_MULTIPROCESSING=0)
```

The JSON records requested optimization settings, implementation commit,
workload/artifact identity and cuBLAS workspace configuration. Compare ordered
token IDs and scores in addition to latency. Capturing work reduces host launch
overhead; it does not remove the underlying model or selection kernels.

## Local results

Measured on 2026-11-14 with the protocol above. Positive percentages mean lower latency.

| Configuration | Off p50 (ms) | On p50 (ms) | p50 reduction |
| --- | ---: | ---: | ---: |

| Qwen, context 128 / beam 128 | 10.970968 | 10.937991 | 0.301% |
| Qwen, context 5000 / beam 256 | 103.551956 | 103.514111 | 0.037% |
| OneRec video, context 1024 / beam 128 | 52.883498 | 52.633979 | 0.472% |

All 400 recorded OneRec requests and 400 short-Qwen requests match ordered
candidate tokens and scores exactly. Long Qwen matches the Top-1 and all 256
candidate token sequences, but one enabled restart differs in ordering and
scores (maximum shared-candidate score delta 0.2227516). Exact long-Qwen model
parity is not established, so this option remains disabled by default. The
kernel/state and independent dense-reference contracts are separate from this
model-level observation. These small latency changes do not establish a
large end-to-end speedup or stable tail-latency gains.

Validation: 202 selected CUDA/state/lifecycle/configuration/benchmark tests
passed. New modules and affected tests passed isolated mypy checks; selected
Ruff, isort, SPDX, character, CUDA API, configuration and whitespace checks passed.

## Streaming constrained LM head

`beam.enable_streaming_constrained_head` uses one reusable projection tile,
activation-dtype legal-candidate storage and FP32 online max/sum state. It
consumes each tile before reusing it. Full-window normalization is retained;
normalizing over only valid continuations would change cross-parent scores.

The existing constraint lookup is shared by full and streamed paths. Candidate
ordering uses the original FP32 score/table-offset keys. The stream uses 8192
columns for smaller windows and 32768 for windows of at least 65536 tokens.
Bias, softcap, scaled/quantized heads, noncontiguous hidden states and dynamic
widths retain their existing paths. Windows at most 8192 tokens, or plans whose
workspace would not be smaller, also retain the original projection.

The option defaults to `False`. Enable it before startup with:

```python
vllm_gr_config["beam"]["enable_streaming_constrained_head"] = True
```

Both model benchmark CLIs accept `--streaming-constrained-head` and its
`--no-streaming-constrained-head` counterpart. Memory results measure the
post-warmup inference peak including model, KV and captures, before profiling.

This incremental ABBA keeps complete-step graph coverage enabled in both arms.
The regular OneRec/Qwen cases use 8192-column tiles or their unchanged narrow
window fallback. A separate wide-Qwen ABBA validates the final 32768-column
tile: context 128, beam 128, fixed original input IDs and synthetic continuation
tokens spanning `[1000, 150000)`. It is an operator stress workload, not a
business recommendation catalog.

| Configuration | Off p50 (ms) | On p50 (ms) | p50 reduction | Peak allocated saved (MiB) |
| --- | ---: | ---: | ---: | ---: |
| OneRec video, 1024 / 128 | 53.069708 | 53.108840 | -0.074% | 7.999 |
| Qwen 128 / 128, narrow fallback | 10.960440 | 10.919202 | 0.376% | 0.000 |
| Qwen 5000 / 256, narrow fallback | 103.536772 | 103.510427 | 0.025% | 0.000 |
| Qwen 128 / 128, wide vocabulary | 12.798563 | 12.673010 | 0.981% | 64.532 |

All 400 recorded requests per case match ordered tokens and scores in these
controls. Wide Qwen mean latency decreases by 1.152%, with 64.532 MiB lower
peak allocated memory. OneRec saves 7.999 MiB while latency is effectively
flat. Narrow-Qwen differences are timing variation of the unchanged fallback
path; no streaming performance gain is attributed to them.

The 8192-column wide-Qwen trial increased p50 by 0.926% and changed scores;
32768 columns reduce launches and pass exact parity in the final wide test.
Blocked GEMM changes can affect BF16 rounding on other shapes/hardware, so
these results do not establish universal model parity.
