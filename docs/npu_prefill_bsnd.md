# Optional NPU BSND Prefill graphs

BSND Prefill graphs avoid per-layer FIA task parameter updates for complete
prompts without cached history. The feature is disabled by default and uses
the normal vllm-gr modules. No import overlay or `sitecustomize` is required.

## Configuration

Enable the existing Prefill graph facility and select explicit total-token /
request-count pairs before starting the engine:

```bash
export VLLM_ENABLE_PREFILL_CUDAGRAPH=1
export VLLM_GR_NPU_PREFILL_BSND_CAPTURE_SPECS='512,4;1024,4'
```

Each entry is `T,B`, where `P=T/B` is the capacity of each request. The example
supports four requests with individual lengths up to 128 or 256 tokens.
The smallest captured graph that fits every request is selected. A total
length that fits T is insufficient if any individual length exceeds P.

An empty or unset specs variable disables BSND. B must be 1, 2 or 4; T must
be positive, divisible by B, and present in the runner's configured Prefill
buckets. Invalid configuration raises an error during runner initialization.
Restart the engine after changing either variable.

These graphs require the Beam/CUSTOM attention metadata builder for every
attention group. Native attention and mixed-backend models skip the Beam
Prefill captures and retain their native execution path, even when BSND specs
are configured. Capture failures on eligible Beam backends still propagate.

Deployments previously using `tools/bsnd_v41` must remove that entry from
PYTHONPATH and stop using the historical import overlay. Normal package
installation/imports now select the implementation.

## Execution and fallback

Every request must have zero computed tokens and its complete prompt scheduled
in the current batch. Lengths include tokens added by the input processor.
Requests are mapped in actual InputBatch row order, not scheduler dictionary
insertion order.

The runner pads each request on the right into a fixed P-token region. Input
IDs, positions and slot mappings keep stable addresses. Padding slots are -1.
The model still writes real K/V to its paged cache, but BSND FIA reads the
current layer's contiguous Q/K/V rather than paged history. The existing causal
mask isolates padding keys from real queries in this full-prompt setting.

After replay, basic-slice device copies compact the real hidden states into an
independent persistent output buffer. Existing logits indices, stage scheduling,
constraint selection and Decode execution remain unchanged. Workspace and LSE
buffers are retained by captured metadata rather than process-global caches.
Buffer reuse assumes the existing serial compute-stream execution.

Prefix-cache hits, historical KV, mismatched batch sizes and lengths outside
the configured shapes use the existing TND graph/native fallback. Unsupported
BSND attention configurations, including KV sharing, sliding windows, softcap,
sinks, ALiBi and FP8, are rejected by the BSND guard. Device capture/replay
errors are propagated rather than treated as ordinary graph misses. The TND
update-before-replay stream ordering is retained.

Padding and input/output copies add device work and memory use. Reduced FIA
submission overhead does not guarantee lower latency for every length distribution.

## Validation

Dependency-free host tests:

```bash
python tests/test_npu_prefill_bsnd.py
```

In the supported vLLM/Ascend environment:

```bash
python -m pytest -q tests/test_beam_attention_platform_execution.py \
  tests/test_prefill_graph_runner.py tests/test_npu_model_runner_patch.py \
  tests/test_npu_prefill_bsnd.py
python -m ruff check vllm_gr/v1/worker/npu_prefill_graph_runner.py \
  vllm_gr/v1/worker/npu_model_runner_patch.py \
  vllm_gr/v1/attention/backends/beam_attn_metadata.py \
  vllm_gr/v1/attention/backends/beam_attn.py \
  vllm_gr/v1/attention/backends/beam_attn_npu.py \
  tests/test_npu_prefill_bsnd.py tests/test_prefill_graph_runner.py
```

Run `ruff format --check` on the same files. The generic Prefill unit tests
clear deployment BSND specs in their fixture; dedicated configuration tests
explicitly set their own specs. Production bucket validation remains strict.

Run the repository strict type-check wrapper on the host regressions:

```bash
python tools/pre_commit/mypy.py 1 3.12 tests/test_npu_prefill_bsnd.py
```

When collecting replay evidence, ensure the EngineCore logging configuration
attaches a handler to `vllm_gr` as well as `vllm`. If an existing custom logging
configuration is used, preserve its handlers and register `vllm_gr` with an
INFO-level handler. Pass the absolute JSON path through
`VLLM_LOGGING_CONFIG_PATH` before starting each engine. Raising the logging level
alone does not attach a missing handler. Missing replay logs are inconclusive;
capture counts and Decode dispatches do not establish Prefill replay.

Host tests do not establish NPU numerical correctness. Device validation should
cover B=1/2/4, length boundaries, GQA, FP16/BF16, cache-hit fallback, alternating
BSND/TND requests, cancellation and final Beam outputs. Compare attention with
a no-padding reference and check logits, candidate ordering and cumulative scores.

For performance, use fresh engines in A/B/B/A order on the same source revision:
A has BSND disabled, B enables the selected shapes. Keep model, catalog, actual
tokenized inputs, cache policy, B/W/N and Decode configuration identical. Report
capture time and memory separately from warmed-up request latency; collect
profiles separately from timing. Confirm actual BSND hits in logs and zero
per-layer FIA updates for those Prefill replays. The
[historical ABBA results](https://github.com/JiusiServe/vllm-gr/pull/499) record
one round per shape for a specific Ascend 910B4 workload.
