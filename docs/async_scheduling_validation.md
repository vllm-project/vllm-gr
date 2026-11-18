# Asynchronous scheduling validation

> Measured on NVIDIA L20 / date not recorded / GPU ModelRunner V1,
> OneRec-1.7B BF16 (runtime versions not recorded). Not a performance guarantee.

The historical runs used batch size 1, Beam width 128 and generation budgets
of one to three tokens. Prefill replay required an exact bucket match in those
runs. See the [current scheduling contract](async_scheduling.md) for padded
Prefill dispatch and retirement behavior.

## Historical correctness and lifecycle results

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
Under that historical policy, the other 98 original-length Prefills used eager
fallback. These counts do not describe the current padded replay coverage.
The controlled 1K dataset changes the input context and is not a replacement
for the original-length accuracy gate.

Historical runtime tests covered late output consumption, pinned staging reuse, device
step mismatch, EOS, cancellation, owner isolation and delayed retirement.
Real TP=1 Multiproc runs cancelled an in-flight request and completed subsequent
requests with no retained Worker session. The startup test checked for graph
reuse without eager Forward or graph capture during requests
across different prompt lengths and generation budgets.

## Historical graph coverage and scheduling results

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
The measurements used the original instrumentation and exact-bucket replay.

## Current system test

The [OneRec Scheduler/Worker system test](../tests/st/test_onerec_v1_scheduler_st.py)
uses public offline and online Beam Search V1 with CUDA ModelRunner V1 and V2.
It checks complete candidates, scheduling and resource retirement. The test
disables Chunked Prefill. Its HTTP batching cases also compare online results
with independent offline results.

```bash
USE_TF=0 CUDA_VISIBLE_DEVICES=0 VLLM_GR_TEST_MODEL=/path/to/OneRec-1.7B \
  python -m pytest --run-slow -v -s tests/st/test_onerec_v1_scheduler_st.py
```
