# Issue #54 Step 9 review: GPU E2E, trace, and soak

Step 9 closes the requested GPU-first acceptance scope. The final long soak and
RecIF run used clean commit `6385137cf4ee43a31047ada97da2bee109db5765`.
The integrated trace gate used clean commit
`54864392452b9578206bc206f7a5d754ebc0ef48`; later commits only changed the
RecIF benchmark and documentation. Every GPU command was submitted through
`gpu run`. NPU hardware acceptance remains deferred by project direction.

## Repeated E2E and goodput

The OneRec-1.7B B=1/L=1024/W=128/S=3/BF16 workload used persistent ASYNC,
Constraint Tables built from RecIF data, and Prefix Cache disabled. The FULL
Beam Graph soak completed 1,000 measured sessions after 10 warmups.

| Metric | compiled Beam-eager, 100 sessions | FULL Beam Graph, 1,000 sessions |
| --- | ---: | ---: |
| mean | 51.536 ms | 26.305 ms |
| median | 50.858 ms | 26.039 ms |
| P90 | 53.689 ms | 27.496 ms |
| P99 | 57.512 ms | 29.871 ms |
| min / max | 49.444 / 59.232 ms | 24.883 / 85.745 ms |
| throughput | 19.404 req/s | 38.016 req/s |
| goodput at 50 ms | 2.717 req/s | 37.978 req/s |

The observed mean and throughput ratio is approximately 2.0x in favor of the
FULL Beam Graph. All 1,000 Graph outputs have the same full ABI digest
`a51a5ed2b2fb6e97dda9cb4350a404c91999d6f19d5304c0d65295ac0ee8f230`.
Every one of the 128 returned triples is in the 16,981-entry constraint catalog.
The measured Graph dispatch delta is 2,000, exactly two Decode stages per
session. The compiled control recorded zero Beam Graph dispatches.

Artifacts:

- `/tmp/vllm-gr-step9-compiled-disabled-bf16-w128-r100-5486439.json`
- `/tmp/vllm-gr-step9-graph-disabled-bf16-w128-r1000-6385137.json`

## Memory and lifecycle soak

The 1,000-session run queried 1,024,000 prompt tokens with zero reusable tokens,
confirming the cache-disabled case directly. Before and after the measured
sessions, attributed device storage was exactly stable:

| Component | Bytes |
| --- | ---: |
| native KV reservation | 100,398,792,704 |
| Beam KV pool | 44,040,192 |
| fixed workspace | 371,860 |
| common GR-state peak | 0 |
| Graph pool allocated | 0 |
| Graph pool reserved | 29,360,128 |

The common GR-state value is zero because the GPU backend owns the global fixed
workspace and Beam KV pool; it does not allocate tensors in the shared state
object. Worker counters for every logged session show one state create and one
free, no Decode request objects, no legacy step updates, and no intermediate
Beam payload D2H. Host RSS was 2,474,553,344 bytes after engine warmup,
2,528,055,296 after all measured sessions, and 2,512,543,744 after shutdown.

## Trace attribution

The profiler now requires a single-process Scheduler-to-device marker chain and
parses Chrome device events without loading the complete trace into memory. The
clean Graph trace passed all markers:

```text
gr_async_schedule
gr_prefill_bootstrap
gr_progress_output
gr_final_output
gr_final_d2h
```

It recorded four D2H operations totaling 5,136 bytes. Three are 4-byte control
copies. The only bulk D2H is the 5,124-byte final result/status payload, and its
device timestamp lies inside the `gr_final_d2h` GPU annotation. No bulk D2H
occurs outside that final range. DtoD traffic totals 117,461,280 bytes across 54
events in the profiled session.

Artifacts:

- `/tmp/vllm-gr-step9-graph-trace-5486439.json`
- `/tmp/vllm-gr-step9-torch-trace-5486439/step0_step9-trace_gpu_graph_disabled_rank0.1788627599803462830.pt.trace.json`

Nsight Systems is not installed in this environment. The Torch trace proves
stage and transfer attribution, but it does not add a fresh PtoD-overlap claim.
The earlier Step 5 Nsight evidence established DtoD overlap and explicitly left
PtoD overlap unproven.

## Real RecIF integration

The accuracy entrypoint now exposes persistent ASYNC, Prefix Cache, and Beam
Graph controls. Persistent mode submits each dataset sample as an independent
B=1 session and explicitly shuts down the engine. The clean RecIF `ad` run used
20 samples, W=128, Constraint Tables, FULL Beam Graph, and Prefix Cache disabled.

| Metric | Result |
| --- | ---: |
| pass@1 / pass@32 | 0.05 / 0.20 |
| recall@1 / recall@32 | 0.008333 / 0.040476 |
| PID pass@1 / pass@32 | 0.00 / 0.15 |
| generation elapsed | 0.7426 s |
| generation time/sample | 0.03713 s |

The artifact is
`/tmp/vllm-gr-step9-recif-results-6385137/cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b_ad_bw128_20.json`.
This bounded sample proves the real loader, model, persistent scheduler,
constraint backend, evaluator, and shutdown chain. It is not a statistically
stable release-quality accuracy estimate.

## Regression and exit boundary

The full CPU repository regression after the persistent RecIF submission change
passed `1945 passed, 3121 skipped`; the final shutdown/reporting change passed
the focused 36-test OneRec suite. The profiling transfer gate passed 45 focused
tests. No shared-venv package was changed.

The GPU portion of Step 9 passes: repeated E2E/P99/goodput, real RecIF E2E,
component-memory plateau, lifecycle soak, Scheduler-to-device trace coverage,
and final-only transfer attribution are present. NPU eager/ACL Graph/cascade,
cross-device parity, and NPU soak remain hardware follow-ups. Prefix hit versus
full-Prefill numerical differences and the old nonpersistent cache-disabled KV
ownership defect remain separately tracked correctness investigations.
