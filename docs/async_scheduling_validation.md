# Asynchronous scheduling validation

## Scope and base

The implementation is based on `upstream/decode_graph` at `6bca38c`. That
commit and the previous local base `632f7d2` have identical trees and patches;
rebase removes the duplicate local commit without changing the implementation.

The historical validation below covers GPU ModelRunner V1, OneRec-1.7B on
NVIDIA L20, BF16, B=1, fixed beam width 128 and generation budgets 1/2/3. CUDA
contract tests also cover FP16 and widths 1/4/128/256. Public output delivery
is outside this Scheduler/Worker change.

## Current Prefill graph contract and validation status

Asynchronous Prefill reuses the existing runner's padded graph geometry and
static buffers. For example, 900 scheduled tokens can replay a captured
1024-token bucket. Selection uses the scheduled token count after prefix-cache
hits and chunking, while replay preserves the real Attention lengths and masks
padded KV writes. Padding alone does not cause eager fallback.

The native Forward path remains the fallback when the startup runner or matching
captured graph is unavailable, live input shapes or adapters are unsupported,
required inputs, slot mapping or Attention metadata are missing, or captured
KV-length or static-buffer bounds are exceeded. Requests never initialize or
capture a Prefill runner. DEBUG dispatch logs report scheduled tokens, request
count, bucket and replay/fallback reason; `last_replay_reason` records runner-level
rejection. These checks retain the Attention backend's graph-support contract.

The results below were collected with the earlier exact-bucket-only dispatch
policy. They do not establish precision or graph coverage for the final padded
path. Matching full-model precision validation remains outstanding: compare
padded V1 with padded legacy using the same buckets, and separately compare
padded Graph with unpadded eager execution. Padding can change BF16 results and
Beam candidate ordering, so equality across these geometries is not guaranteed.
Use the existing precision gate and record replay/fallback counts on the matching
workload before attributing accuracy or performance results to the padded path.

## Independent-engine reduction policy

CUDA ModelRunner V1 engines using `beam.execution_mode=v1` and Inductor now
default `compilation_config.inductor_compile_config.dynamic_scale_rblock` to
`false`. The native compilation configuration includes this choice in its cache
identity. Existing explicit values take precedence; ModelRunner V2, NPU and
non-Beam engines keep their own defaults.

With the same OneRec inputs, weights and AOT artifact, Inductor could select
either a 64-element or a 128-element reduction block for the generated Q/K norm
and RoPE kernel. Selecting each existing binary reproduced the two previously
observed candidate/score sequences exactly. Differences were already present in
Prefill, before Beam candidate selection. The default disables that dynamic
block-size adjustment while retaining compilation and CUDA graph replay.

This policy addresses the observed independent-engine mismatch. It does not
guarantee invariance across batch shapes, GPUs or compiler versions. Explicitly
setting `dynamic_scale_rblock=true` restores the original tuning behavior and is
outside this consistency guarantee. Business Recall and performance need their
own acceptance measurements on the resulting candidate.

The existing `test_onerec_v1_completion_batch_online` ST retains exact ordered
candidate and score comparisons. Its optional diagnostics are excluded from
performance measurements and disabled by default:

- `VLLM_GR_TEST_STAGE_TRACE=1` saves inputs, logits and Beam state after each
  stage, with CPU readback after the complete request.
- `VLLM_GR_TEST_PREFILL_KV_TRACE=1` additionally saves real prompt KV and hidden
  states for cold completion requests with tracing enabled. The completion ST
  resets the prefix cache before each paired request.
- `VLLM_GR_TEST_MODEL_IDENTITY=1` hashes loaded parameters after the B2 request.
- `VLLM_GR_TEST_PROFILE=1` uses vLLM's native profiler around the B2 request.
- `VLLM_GR_TEST_INDUCTOR_CONFIG` supplies a JSON compilation mapping for
  controlled configuration experiments.

Diagnostic copies and profiling can affect tuning choices. Preserve failing
outputs, verify that each observer reproduces the original outputs, and repeat
the final ST with these diagnostic settings unset.

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

Runtime tests cover late output consumption, pinned staging reuse, device
step mismatch, EOS, cancellation, owner isolation and delayed retirement.
Real TP=1 Multiproc runs cancel an in-flight request and complete subsequent
requests with no retained Worker session. The startup test rejects any eager
Forward or graph capture during requests while reusing the captured graph
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
The measurements preceded submission cleanup and the padded Prefill dispatch
change, and retain their original source identity in the local archive. They
do not measure the benefit of removing instrumentation or broader padded
replay coverage.

## Submission checks

Experimental benchmark modules, trace parsers, diagnostic tests and detailed
experiment reports are excluded from this change. Raw traces and results
remain in the local archive without rewriting. Production dispatch profiling
scopes, observation counters, diagnostic fields and the unused Triton input
preparation/control-snapshot module have been removed.

After the original cleanup, the regression set passed 435 tests and skipped three.
Ruff, formatting, isort, source checks, mypy and Markdown checks passed.
A real OneRec-1.7B BF16 run with P=1024, W=128 and N=3 replayed Prefill
graphs for all three measured requests after three warmups. Its complete
device states and scores matched the pre-cleanup run exactly (`atol=0`),
and the Worker released every session. These are historical checks of the
exact-bucket path, not a fresh validation of the final padded implementation.

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

## OneRec Scheduler/Worker system test

`tests/st/test_onerec_v1_scheduler_st.py` loads real OneRec-1.7B BF16 weights
through the native multiprocess engine with explicit V1 asynchronous scheduling.
It submits successive requests at width 128, prompt lengths 900/1024 and
generation budgets 1/2/3, with chunked Prefill, Prefix Cache and Beam graphs
enabled. After native Worker retirement it checks completed SID legality,
finite scores, generation counts, released ownership and reuse of fixed-address
state. Repeating a request with the same Prefill geometry must reproduce its
device results exactly.

The test submits through the native session API because public A5 output
conversion remains unimplemented. It accepts only the known
`worker_result_unavailable` terminal boundary (or a future successful result),
then independently validates real GPU completion records and retirement.
It does not install a replacement result consumer. This is a scheduling and
lifecycle ST, not public API acceptance or legacy-versus-V1 accuracy parity.

```bash
USE_TF=0 CUDA_VISIBLE_DEVICES=0 VLLM_GR_TEST_MODEL=/path/to/OneRec-1.7B \
  python -m pytest --run-slow -v -s tests/st/test_onerec_v1_scheduler_st.py
```

The existing CI `run-tests` hook collects it through `pytest --run-slow tests`.
CUDA is required; a missing model on a CUDA worker fails instead of skipping.
