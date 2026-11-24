# CUDA Prefill graph capture

CUDA Prefill graphs capture GPU work at a fixed execution shape. A worker can
reuse a graph for different prompt lengths by padding the live batch to that
shape. Attention uses the real sequence lengths, and the worker writes only
real tokens to the KV cache.

## Capture shape selection

A FULL capture shape pairs a request count `B` with a scheduled-token bucket
`T`. The shape must fit the token budget, request capacity and model context
and allow at least one token per request. During ModelRunner V2 capture, the
pinned vLLM version distributes the dummy tokens across request rows.

V2 installs the GR Prefill plan on a single rank with `enforce_eager` disabled
and without data-parallel ubatching, LoRA, speculative decoding, auxiliary
hidden-state output or multimodal input support. Other configurations keep
vLLM's native graph policy.

When the GR plan is active, V2 picks the smallest bucket that fits the
scheduled tokens. FULL replay requires a compatible Prefill batch and a
successful capture for that bucket and request count. If FULL replay is
unavailable, V2 tries PIECEWISE at the same bucket, then falls back to eager
execution.

![Flow: an eligible batch selects the smallest token bucket; a FULL miss tries PIECEWISE at that bucket, then eager execution.](../../assets/cuda_prefill_dispatch.svg)

| Color or shape | Meaning |
| --- | --- |
| Green | Admitted Prefill batch. |
| Yellow diamond | Dispatch check. |
| Cyan | Graph replay. |
| Orange | Alternative graph replay. |
| Red | Eager fallback. |

## Replay with prompt history

A graph keeps the attention path it used at capture. When dense Prefill is
enabled, FULL replay requires every request to have zero computed tokens.
If a request already has computed tokens, V1 uses native execution and V2
starts with PIECEWISE at the selected bucket. See
[ModelRunner V2](../module/model_runner_v2.md).

For V1 replay, the worker checks that request rows match the scheduled order
and that the current slot mapping, `query_start_loc`, `seq_lens` and block
table are available. If a check fails, the worker falls back to eager
execution.

A prefix-cache hit can move a batch to a smaller bucket. With paged Prefill on
V1, the bucket counts scheduled tokens, and each request's KV length includes
cached tokens too. The default captured `max_seqlen_k` is twice the bucket
size. A request with 64 cached tokens and 128 scheduled tokens can use a
128-token bucket because its KV length of 192 fits the default bound of 256.
To allow replay for a longer KV length, set `gr.prefill_kv_bound` high enough
to cover it. The override also adds buckets of 1 and 16 tokens when they fit
the configured range.

## Where attention reads the prompt

Paged attention reads KV from the cache. When the worker processes a complete
fresh prompt, dense attention can read contiguous K/V directly from the
current layer's projection. Both paths write the new tokens to the KV cache
for Decode. During graph capture, the GR runners allow dense attention only
in their fresh-Prefill capture scope.

Dense Prefill requires disabling prefix caching, chunked Prefill and KV
connectors. The worker checks the current context and scheduled lengths. It
uses paged attention for batch-invariant mode, sliding windows, ALiBi, softcap,
sinks, quantized KV or KV sharing.

## Configuration

| Startup setting | ModelRunner V1 | ModelRunner V2 |
| --- | --- | --- |
| `gr.prefill_token_buckets` is a list (packaged default) | With a configured beam width, each bucket is a batch token total capped at `max_num_batched_tokens`. For a batch size `B` from `gr.beam_batch_sizes` and token total `T`, capture requires `B <= T <= B * max_model_len` and `B <= max_num_seqs`. Without a configured beam width, V1 also caps buckets at `max_model_len`. | Keeps batch sizes up to `max_num_seqs` and token buckets up to `min(max_num_batched_tokens, max(batch_sizes) * max_model_len)`. |
| `gr.prefill_token_buckets: null` | Uses the built-in grid within the token and context limits. | Uses native `cudagraph_capture_sizes`. V2 uses the fallback grid when the native list is absent and honors an explicitly empty list. |

Use `gr.prefill_graph_enabled` to control GR Prefill capture. On V1,
`gr.prefill_kv_bound` widens the KV limit when its value exceeds the bucket's
default bound. Use `gr.dense_prefill` to choose the attention path for
eligible fresh prompts.

| `gr.dense_prefill` | Attention path |
| --- | --- |
| `off` (default) | Paged attention. |
| `fa2` | Dense FlashAttention 2. |
| `cudnn` | cuDNN SDPA for supported single-request shapes of at least 64 tokens, with dense FA2 as the fallback for other eligible shapes. |

## Performance

Closer bucket spacing reduces padding but can increase the number of graphs
the worker has to capture and keep in memory. Prefix-cache hits can reduce the
size of the selected bucket. On V1, compare the resulting KV bound with the
total cached and scheduled tokens per request before setting an override.

## Validation

The [V2 runner tests](../../../tests/v1/worker/test_gpu_prefill_graph_runner_v2.py)
cover bucket selection and replay fallback. The
[dense Prefill tests](../../../tests/kernels/attention/test_beam_dense_prefill.py)
compare attention outputs on CUDA.

## Source map

| File | Responsibility |
| --- | --- |
| [V1 Prefill graph runner](../../../vllm_gr/v1/worker/cuda/v1/gpu_prefill_graph_runner.py) | Manages V1 capture buffers, sets KV bounds and checks replay eligibility. |
| [V2 Prefill graph runner](../../../vllm_gr/v1/worker/cuda/v2/gpu_prefill_graph_runner_v2.py) | Builds the FULL and PIECEWISE plan and selects a descriptor at dispatch. |
| [V1 capture planner](../../../vllm_gr/v1/worker/cuda/v1/prefill_capture_plan.py) | Filters feasible token buckets and request counts before buffer allocation. |
| [Shared Prefill graph utilities](../../../vllm_gr/v1/worker/prefill_graph_common.py) | Resolves bucket limits and KV bounds, allocates buffers and matches live token counts. |
| [Dense Prefill attention](../../../vllm_gr/v1/attention/backends/beam_attn_gpu.py) | Checks whether attention can use projected KV. |

The next generation stage uses [CUDA Decode attention](cuda_decode_attention.md).
