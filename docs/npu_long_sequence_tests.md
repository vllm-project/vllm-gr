# Long-sequence NPU functional regressions

These tests extend whole-model coverage to cold 1024-token and 4096-token
requests at batch sizes 1 and 4. They use a fixed beam width of 8 and generate
three SID stages through the public Beam API. They are correctness tests, not
performance benchmarks.

The system-test node is:

```text
tests/st/test_onerec_v1_npu_long_context.py::test_onerec_v1_npu_long_context
```

Its four parameter combinations are:

| Batch | Effective tokens per request | Total Prefill tokens / exact graph bucket | Engine model/token budget |
| --- | --- | --- | --- |
| 1 | 1024 | 1024 | 1152 |
| 1 | 4096 | 4096 | 4224 |
| 4 | 1024 | 4096 | 4224 |
| 4 | 4096 | 16384 | 16512 |

Effective length includes the SID begin token appended by the public API. Raw
prompts are one token shorter. The fixture repeats real prompt tokens with a
different circular offset for each request, rather than comparing identical
batch rows.

The current NPU Prefill bucket selection is also bounded by `max_model_len`.
Therefore both `max_model_len` and `max_num_batched_tokens` are set to
`max(1024, batch_size * input_len + 128)`. This preserves the exact total-token
bucket and room for Decode. It does not increase the actual per-request history
length. Block size remains 128.

## What a passing case proves

Each shape runs two engines sequentially: Eager and FULL graph. Each engine
executes the same logical input twice. Prefix caching and chunked Prefill are
disabled, so every call must process the complete cold history.

The test requires:

- Exactly one Prefill followed by two Decode forwards.
- Worker-side request records showing zero computed tokens and the full
  scheduled length for every request; the scheduler total must equal `B * L`.
- Exact actual/padded token counts, input/position shapes, request boundaries,
  sequence lengths, and unique request IDs. The system test also checks the real
  input token and position values against the submitted prompts.
- A real Prefill replay increment of one in the exact `B * L` bucket. Capture
  configuration or an outer runner mode alone is insufficient.
- Two FULL Decode forwards for graph execution, with the expected `B * 8` rows
  and no fallback. Eager forwards must use NONE mode; the only intentional
  Decode fallback reason is `graph_disabled`.
- Legal, unique, ordered SID outputs and unchanged finish/stop semantics.
  Eager/FULL token IDs must match exactly. Cumulative-score comparison retains
  the original system-test limits: absolute `1e-4` or relative `1e-5` of the
  reference score, whichever is larger. Same-mode repeats must match exactly.
- Stable captured storage addresses across graph repeats and completed request
  cleanup.

The helper sets an explicit Prefill bucket and
`VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS=B`. The current NPU capture implementation
can still use eight dummy requests during warmup. Actual batch coverage is
established from the real worker trace, not inferred from this setting.
`VLLM_GR_PREFILL_GRAPH_KV_BOUND` is cleared: it is not an NPU history-length bound.

## Run on an NPU

From the repository root in a prepared environment, select an available NPU and
run all four shapes:

```bash
VLLM_GR_TARGET_DEVICE=npu \
VLLM_GR_TEST_MODEL=/path/to/OneRec-model \
python -m pytest -v -s --run-slow --strict-markers \
  tests/st/test_onerec_v1_npu_long_context.py::test_onerec_v1_npu_long_context
```

The default model remains `OpenOneRec/OneRec-1.7B` when the model variable is
omitted. A compatible local 0.6B model may be selected explicitly. Each comparison
uses the same model, tokenizer, batch size, shape, and constraint artifact;
results are not compared across models or batch sizes. Runtime setup, model
availability, and memory capacity must support the selected case.

The test selects V1 with spawn, disables fused FFN, NZ, breakable graphs, prefix
caching, and chunked Prefill, and enables the NPU Prefill replay path. On an NPU
target, missing runtime support, a missing device, or unavailable Beam operators
is a failure. Other target platforms may skip this NPU module. A skipped node
does not qualify the NPU case.

The pure CPU acceptance controls are in
`tests/test_npu_long_beam_helpers.py`. They exercise trace and output validation,
including negative controls for cached or partial history, graph fallback,
incorrect request boundaries, lost Beam rows, and output/score differences.
Passing those controls does not constitute a whole-model NPU result.

## Validation scope

No hardware-pass or performance claim follows from adding these tests. Record
the source revision, model identity, runtime versions, exact selected/passed
nodes, replay evidence, output comparisons, and failures for each actual NPU
run. Keep failed attempts and any change in test configuration visible.

This matrix does not cover arbitrary lengths, padded or unequal request
lengths, beam width 128, dynamic Beam, prefix-cache hits, concurrent sessions,
distributed layouts, or ordinary LLM generation parity. In particular, it does
not certify a previously failing configuration merely because a nearby exact
bucket passes. The existing short-context regression and its documentation are
separate and remain unchanged.
