# Beam Search V1 validation for #398

This records the 2026-09-23 validation of Core admission reuse and batched
terminal-result validation/construction. The implementation is model independent;
measured performance below is limited to the stated OneRec workload.

## Source and measurement boundary

The tested candidate was `e14afa5af6f776cbffb51f5cd52caf961120854d` plus patch
SHA256 `b1921450b8aa11c9984e279243d20b54d392d604276b110bc7ad793f06c8616e`.
The admission-only control used the same base plus patch
`aa37929b60a7c75f71ab35187680eaa9e66fcbbc8b52c0d9bd87382a6298a65e`.
The upstream control was `upstream/decode_graph` at
`ce4ec3212fb23c4b320a32257a839f452aa97442`.
Subsequent cleanup changes comments, missing-method diagnostic text and test
assertion placement; the successful execution path retains the measured behavior.

The timer covers only the call to and return from `llm.beam_search_v1()`.
Prefix-cache reset and reads of lazy offline tokens/text happen outside the timer.
Result type/value checks, construction and release proof remain inside the call.

[Audited samples and validation metadata](beam_search_v1_398_abba_20260923.json)
retain all 800 measured latencies, 80 warmups, per-leg summaries, source identities,
fixture/output digests, raw-run digests and full-suite/CI results. The original
[historical experiment](beam_search_v1_398_historical_validation.md) includes lazy
materialization in its timer and uses different source/hardware controls; its
latencies are a separate record.

## ABBA protocol and results

- NVIDIA L20 GPU0, OneRec-1.7B BF16, vLLM 0.22.1, PyTorch 2.11.0.
- B=1, input=1024 including SID begin, beam width=128, effective budget=3.
- Cache miss on every request; Prefill and Decode graph replay enabled.
- Frontend CPU24, Worker CPU32; separate background-thread affinity.
- Each comparison runs A1, B1, B2, A2 with a fresh engine for every leg,
  10 warmups and 100 measured requests per leg: 200 samples per version.
- Diagnostic wrappers run for one untimed request and are removed before warmup.
  Every measured request retains output verification outside the timer.

A is the optimized candidate in both comparisons. Each B is a separate control.
Latency deltas below are A minus B, in milliseconds; a negative delta is faster.

| Comparison | Metric | A | B | A minus B |
| --- | --- | ---: | ---: | ---: |
| Admission-only control | Mean | 56.042190 | 56.313099 | -0.270909 |
| Admission-only control | P50 | 56.048521 | 56.325458 | -0.276938 |
| Admission-only control | P90 | 56.095997 | 56.383036 | -0.287039 |
| Admission-only control | P99 | 56.778857 | 57.186393 | -0.407536 |
| Upstream ce4ec32 | Mean | 56.052147 | 55.957904 | +0.094244 |
| Upstream ce4ec32 | P50 | 56.065912 | 55.993557 | +0.072354 |
| Upstream ce4ec32 | P90 | 56.113686 | 56.049063 | +0.064622 |
| Upstream ce4ec32 | P99 | 56.766607 | 56.267537 | +0.499069 |

The incremental mean reduction is **0.270909 ms (0.481%)**. Both adjacent pairs
favor the candidate: A1-B1=-0.286735 ms and A2-B2=-0.255083 ms. The separate
upstream comparison retains a **0.094244 ms (0.168%)** mean gap, with pair gaps
of +0.102775 ms and +0.085712 ms. Means from the two independent comparisons
must not be subtracted to infer another speedup.

All 800 measured requests and 80 warmups have the same complete output signature.
Cache-miss geometry, graph replay, CPU affinity, source snapshots and resource
retirement checks passed. The optimized path consumes one Core admission result
in Scheduler and reuses typed sequence validation in the frontend. Worker
validation still runs once for each of the three dispatches.

Each version has only two independent engine starts per comparison. Requests
within one engine are not independent environment repetitions. Percentiles use
NumPy linear interpolation; P99 from 200 samples is sensitive to a few tail
observations. Neither these results nor Pangu correctness coverage establish a
performance gain for other models, shapes or cache-hit workloads.

## Full tests and CI gates

The full suite ran after both incremental ABBA pairs showed a gain, using the
same frozen candidate:

```bash
CUDA_VISIBLE_DEVICES=0 USE_TF=0 HF_HUB_OFFLINE=1 \
VLLM_GR_TEST_MODEL=/path/to/OneRec-1.7B \
PANGU1B_MODEL_PATH=/path/to/openPangu-Embedded-1B-V1.1 \
PANGU1B_REQUIRE_RUNTIME=1 \
PANGU1B_OFFLINE_ACCURACY_THRESHOLD=1.0 \
PANGU1B_ONLINE_ACCURACY_THRESHOLD=1.0 \
python -m pytest -x --run-slow --durations=10 -s tests --junitxml=ut_st.xml
```

| Check | Result |
| --- | --- |
| Full GPU UT/ST | 3964 passed, 1374 skipped, 22 subtests passed; zero failures/errors |
| Runtime ST | All 8 OneRec offline/online and Pangu accuracy/serving cases executed |
| Beam integration | All 9 supported offline, serving and catalog cases executed |
| Isolated real-vLLM import checks | 2 passed; cover the real-import cases skipped in the combined suite |
| Focused result/frontend/scheduler suite | 440 passed |
| Format and static checks | All applicable pre-commit hooks passed, including Ruff format and mypy 3.11/3.12/3.13 |

Of the 1374 skips, 1344 require FA3 on compute capability 9.x. Other reasons
include unavailable Ascend/NPU dependencies, collection skips, four checks for
historical decode-graph-only helpers, and the two separately covered import cases.
No supported real-model or slow integration case listed above was skipped.
The supplemental and focused tests overlap with full-suite coverage; their
counts should not be added into one total.

All three system scripts returned zero under the unchanged upstream
`development`/`gpu` policy. Accuracy and execution errors are blocking;
performance thresholds are warnings.

The OneRec GPU offline gate explicitly selects `beam_search_v1()`. Its HTTP
online gate uses the existing legacy serving path. Online `beam_search_v1()`
is exercised by the real-model ST, including N=1/2/3, cancellation and subsequent
workspace reuse; that ST is not a full online Video-100 accuracy gate.

| Gate | Measured result | Threshold/outcome |
| --- | --- | --- |
| OneRec offline/online pass@1 | 0.07 / 0.07 | Both >=0.05 |
| OneRec offline/online pass@32 | 0.23 / 0.23 | Both >=0.21 |
| OneRec offline/online recall@32 | 0.03253968 / 0.03253968 | Both >=0.03 |
| OneRec online P99 TTFT / TPOT | 169.61 / 0.01 | <=300 / <=0.02 |
| OneRec online elapsed time | 15.582 s | <=20 s |
| OneRec offline elapsed time | 104.236 s | >20 s; non-blocking performance warning |
| Pangu offline/online accuracy | 1.0 / 1.0 | Both >=1.0 |

The offline CI log contains inference-time JIT compilation. Its aggregate
104.236 s elapsed time has a different boundary from the warmed single-call
ABBA metric and must not be used as that API latency.

## Remaining acceptance scope

Keep this work linked as **Part of #398**. Local CUDA full-suite and system-gate
coverage is now recorded, but this is not a claim that every issue acceptance
gate is complete. The exact merged-#393 comparison, broader real-model width
coverage, full online Video-100 accuracy through `beam_search_v1()`, historical
cold-compile variation and NPU execution support remain separate concerns. The
measured upstream gap and the non-blocking CI performance warning above remain
visible.
