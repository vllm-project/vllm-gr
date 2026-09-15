# Asynchronous scheduling validation

## Scope and base

The implementation is based on `upstream/decode_graph` at `6bca38c`. That
commit and the previous local base `632f7d2` have identical trees and patches;
rebase removes the duplicate local commit without changing the implementation.

Validation covers GPU ModelRunner V1, OneRec-1.7B on NVIDIA L20, BF16,
B=1, fixed beam width 128 and generation budgets 1/2/3. CUDA contract tests
also cover FP16 and widths 1/4/128/256. Public output delivery is outside
this Scheduler/Worker change.

## Correctness and lifecycle

The following configurations were compared with exact ordered device-state
and score equality (`atol=0`), without relaxing the existing precision gate:

- Existing Prefill graph versus asynchronous reuse of the same graph, then
  eager versus graph on the same 1024-token input.
- Consecutive requests with lengths 17/256/512/1008/1024 and budgets 1/2/3.
- Prefix Cache hits with 1008 cached tokens and 16 scheduled tokens.
- Two 512-token Prefill chunks; intermediate chunks do not initialize Beam state.
- Prefill Graph with eager Decode and with Decode Graph.
- One hundred original-length OneRec inputs and one hundred controlled
  1024-token inputs, using the existing recommendation evaluator.

Original-length OneRec SID pass@1/pass@32/recall@32 were
0.06/0.23/0.03279 and passed the existing gate. Only 2/100 original-length
Prefills matched an exact captured bucket; all 100 controlled 1K inputs did.
The controlled 1K dataset changes the input context and is not a replacement
for the original-length accuracy gate.

Runtime tests cover late output consumption, pinned staging reuse, device
step mismatch, EOS, cancellation, owner isolation and delayed retirement.
Real TP=1 Multiproc runs cancel an in-flight request and complete subsequent
requests with no retained Worker session. The startup test rejects any eager
Forward or graph capture during requests while reusing the captured graph
across different prompt lengths and generation budgets.

## Graph coverage and scheduling

Asynchronous Prefill reuses the existing runner and capture geometry only
when the native input shape matches its bucket exactly. Inputs requiring
padding, missing startup graphs or incompatible inputs fall back to eager.
Existing KV-bound and buffer checks remain in the Prefill runner. This does
not change the Attention backend's graph-support contract.

The four-group experiment held Decode Graph, input, dtype, width, budget,
executor and cache conditions fixed. Graph plus asynchronous scheduling
covered D1 CPU preparation, metadata building and binding during Prefill GPU
execution in all five traces. Native BatchQueue already submitted D1 before
consuming P's control output, so no additional scheduler/thread optimization
was necessary for this P-to-D change.

Without the profiler, four modes with three independent runs of 100 measured
requests each used the same 1024-token prompt and ten warmups per run.
Mean per-run service medians were 62.488/59.270/62.212/58.598 ms for eager
serial/eager async/graph serial/graph async. The boundary ends at Worker
retirement proof; these are not public API latency or formal P99 measurements.
The measurements preceded submission cleanup and retain their original source
identity in the local archive. They do not measure the benefit of removing
instrumentation.

## Submission checks

Experimental benchmark modules, trace parsers, diagnostic tests and detailed
experiment reports are excluded from this change. Raw traces and results
remain in the local archive without rewriting. Production dispatch profiling
scopes, observation counters, diagnostic fields and the unused Triton input
preparation/control-snapshot module have been removed.

After this cleanup, the regression set passed 435 tests and skipped three.
Ruff, formatting, isort, source checks, mypy and Markdown checks passed.
A real OneRec-1.7B BF16 run with P=1024, W=128 and N=3 replayed Prefill
graphs for all three measured requests after three warmups. Its complete
device states and scores matched the pre-cleanup run exactly (`atol=0`),
and the Worker released every session.

Run the essential Scheduler, graph and device-lifecycle regressions with:

```bash
USE_TF=0 CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/test_async_scheduling_scheduler.py \
  tests/test_gr_async_scheduler.py tests/test_gpu_model_runner_patch.py \
  tests/test_prefill_graph_runner.py \
  tests/kernels/test_async_scheduling_inputs.py \
  tests/kernels/test_gpu_beam_stage_runner.py \
  tests/kernels/test_gpu_beam_stage_lifecycle.py
```

The existing client, input, configuration, constraint and Beam decision tests
are also retained. No production benchmark switch or alternate scheduler is
introduced by the validation tools.
