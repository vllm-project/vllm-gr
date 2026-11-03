# NPU whole-model regression tests

## B4 cold Prefill and FULL Decode

The first additional system test is:

```text
tests/st/test_onerec_v1_npu_e2e.py::test_onerec_v1_npu_b4_full_graph
```

It is a bounded functional regression for four requests, beam width 8, and three
SID generation stages. Each request has 128 effective input tokens, including
the SID begin token. The test uses distinct prompts and repeats the same logical
batch to check reuse without relying on a warm prefix-cache hit. The existing B1
and B2 tests and their optional padded-input diagnostic remain unchanged.

Two engines run sequentially: an Eager reference and an engine with Prefill FULL
replay and Beam FULL Decode enabled. They use the same model, tokenizer,
constraint artifact, input tokens, batch size, and decoding parameters. The test
compares all returned beams in order, including token IDs, cumulative scores,
and finish/stop reasons. Scores must match exactly in this case; a tolerance or
an unordered candidate-set comparison does not satisfy this check. Results from
different models or batch sizes are not used as the reference.

The trace observes the model-runner boundary, not individual model operators.
For each measured request batch it checks:

- Cold Prefill: all four requests have zero computed tokens and 128 scheduled
  tokens, with 512 actual tokens in total and no padding to a larger bucket.
- Graph Prefill: the real GR Prefill replay counter increases once for bucket
  512. Configuration flags, successful capture, or the outer runner's runtime
  mode alone do not prove replay.
- Graph Decode: both Decode forwards execute in FULL mode, the FULL dispatch
  counter increases by two, and no fallback is recorded.
- Eager: all three forwards execute in NONE mode, no FULL dispatch occurs, and
  the two Decode dispatches record only the intentional `graph_disabled` reason.
- Outputs: each request returns eight unique, legal SID sequences with finite,
  sorted scores and the expected completion status; repeats and Eager/FULL
  comparisons satisfy the same exact output checks.

The test controls its own configuration rather than depending on inherited shell
settings:

| Setting | Value |
| --- | --- |
| Model runner | V1 (`VLLM_USE_V2_MODEL_RUNNER=0`) |
| Batch / beam / generated SID stages | 4 / 8 / 3 |
| Effective input length per request | 128, including SID begin |
| `max_model_len` / `max_num_batched_tokens` | 1024 / 1024 |
| Prefill graph bucket / captured request count | 512 / 4 |
| Prefix caching / chunked Prefill | Disabled / disabled |
| Fused FFN / NZ optimization | Disabled / disabled |

The token budget is at least the model length, as required by this unchunked
configuration. The 1024 model-length bound also keeps this short exact-bucket
case separate from known padded-input and long-sequence graph diagnostics. It
does not change product defaults or fix those configurations.

## Run the test

Run from the repository root in a prepared Ascend environment with the matching
vLLM, vLLM-Ascend, CANN, and xllm Beam operators. Select an available NPU using
the runner's normal device configuration, then run:

```bash
VLLM_GR_TARGET_DEVICE=npu \
VLLM_GR_TEST_MODEL=/path/to/OneRec-model \
python -m pytest -v -s --run-slow --strict-markers \
  tests/st/test_onerec_v1_npu_e2e.py::test_onerec_v1_npu_b4_full_graph
```

If `VLLM_GR_TEST_MODEL` is omitted, the existing default remains
`OpenOneRec/OneRec-1.7B`. A compatible local 0.6B OneRec model may be supplied for
local qualification. Both engines must load that same model. Model availability
or passing with 0.6B does not establish a 1.7B result.

When `VLLM_GR_TARGET_DEVICE=npu`, missing runtime support, an unavailable NPU, or
missing Beam operators must fail rather than turn this regression into a skip.
On other target platforms the NPU module may be skipped. `--run-slow` is required
to execute this slow system test; a skipped node is not a successful NPU result.

The existing pre-commit `run-tests` hook invokes
`python3 -m pytest -x --run-slow --durations=10 -s tests`, so it recursively
discovers system tests under `tests/st`. Discovery alone is insufficient: an NPU
runner must select the NPU target, provide the dependencies and model, and show
this exact node as **passed**. This document does not claim that an external
Jenkins job has already been configured to do so.

## Required validation evidence and remaining coverage

Record hardware results in the PR validation report or CI artifacts: source
revision, hardware, runtime versions, model identity, exact node result, output
comparison, and Prefill/Decode replay evidence. Re-run the node on the declared
model and runtime when qualifying a change.

This short functional case is not a performance benchmark. It does not qualify
beam width 128, long inputs, padded Prefill, other models, or all cache modes.
In particular, the default 1.7B model has no hardware-pass claim until that model
is actually tested.

| Additional coverage | Status for this test increment |
| --- | --- |
| B4 / W8 / cold128, exact Eager/FULL and repeated outputs | Must execute on the declared model and runtime |
| Default 1.7B model on the target NPU stack | Requires a separate run if local qualification used 0.6B |
| Production W128 shapes and performance | Separate follow-up; not covered |
| Unequal, padded, and long input lengths, including known graph failure cases | Separate follow-up; not covered or fixed |
| Prefix-cache hits, partial history reuse, and eviction | Separate follow-up; cold requests only here |
| Dynamic beam widths, concurrent sessions, and distributed layouts | Separate follow-up; fixed-width single-engine batches here |
| Ordinary LLM generation parity and model-quality metrics | Outside this SID functional regression |

Prior ad hoc validation can provide supporting evidence, but it does not replace
executing this repository test on the declared model and runtime.
