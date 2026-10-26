# Beam Search Metrics

## Current V1 measurements

Beam Search V1 owns Prefill and Decode in EngineCore and the Worker. The old
frontend phase loop no longer calls a GR Prometheus collector. Consequently
this version does not automatically expose the old
`vllm_gr_beam_search_*` Prefill/Decode/sort counters or histograms.
The unused `vllm_gr.v1.metrics.beam_search_metrics` collector and
`vllm_gr.v1.metrics.stats.RequestStateStats` extension have been retired.

The native vLLM `/metrics` endpoint is unchanged. Its native request metrics
are not replacements for measured GR Beam stage durations. For reproducible
Beam latency, use the [offline V1 benchmark](../benchmarks/qwen/README.md) or
[paired offline A/B tool](../tools/AB_Tests/README.md), keeping the same model,
input length, width, cache mode, warmup, and output-consumption scope.
The offline A/B tool measures public text materialization as part of E2E.
Online benchmarks additionally include transport and serving overhead.

## Benchmark compatibility with older servers

The OneRec online benchmark still reads historical phase counters when a
server provides them. It requires Prefill, Decode, sort, and request counters
in both snapshots before printing a phase summary. Missing or incomplete
counters are reported as unavailable; they are not interpreted as zero cost.
Non-successful HTTP responses and unavailable endpoints do not prevent the
main benchmark from running. A counter reset invalidates the delta window.

Each sample retains its metric name and complete label set until both
snapshots have been validated. Every label set must have all four counters
in both snapshots; new, disappearing, incomplete, or reset series invalidate
the window. Only validated per-series deltas are summed, so growth in one
engine cannot hide another engine's reset. Label and sample order do not
affect identity. Stable series with no new requests are allowed.
If the total-beams counter appears in either snapshot, it must also pass the
presence, complete label-set, and per-series monotonicity checks. It remains
optional when absent from both snapshots.
The counters describe server-wide activity; isolate benchmark traffic if the
result is intended to represent a particular run. This report is separate
from client-observed E2E and is not a substitute for it.

Future V1 stage metrics require explicit device/host timing boundaries and
aggregation across EngineCore/Worker processes. They must not infer stage
latency from the retired frontend timers or add synchronization to the hot
path solely to reproduce those counters.
