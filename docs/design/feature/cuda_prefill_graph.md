# CUDA Prefill graph capture

CUDA workers capture Prefill at selected request counts and token budgets. A
later batch can replay a captured graph even when its prompt lengths differ
from the capture inputs. Padding fixes the execution shape, attention uses the
real lengths, and only real tokens write to the KV cache.

## Capture shape selection

A FULL shape is the pair of request count `B` and scheduled-token bucket `T`.
It must fit the token budget, request capacity and model context, and have at
least one token per request. ModelRunner V2 pads every request up to the shape;
how the dummy tokens are split across request rows is owned by the pinned vLLM
version. V2 then finds the smallest bucket that fits the scheduled tokens, and
FULL replay additionally requires a successful capture at that request count
and a compatible Prefill batch.

The V2 Prefill plan is installed only for a single rank with no data-parallel
ubatching, LoRA, speculative decoding, auxiliary hidden states, multimodal
input or `enforce_eager`. On other configurations GR Prefill graphs do not
exist and execution stays native.

The diagram follows V2 dispatch when its GR Prefill plan is installed.

![Flow: an eligible batch selects the smallest token bucket; a FULL miss tries PIECEWISE at that bucket, then eager execution.](../../assets/cuda_prefill_dispatch.svg)

| Color or shape | Meaning |
| --- | --- |
| Green | Admitted Prefill batch. |
| Yellow diamond | Dispatch check. |
| Cyan | Graph replay. |
| Orange | Alternative graph replay. |
| Red | Eager fallback. |

## Replay with prompt history

A graph records its attention path at capture. With dense Prefill enabled,
FULL replay requires zero computed tokens for every request. A batch with
history therefore returns to native execution on V1. On V2 it can use
PIECEWISE at the selected bucket, then eager execution if that capture is
unavailable. See [ModelRunner V2](../module/model_runner_v2.md).

V1 replay has its own gates: the request rows must keep their order, a slot
mapping must be present, and `query_start_loc`, `seq_lens` and the block table
must be live. A failed gate falls back to eager execution.

A prefix-cache hit can move a batch into a smaller token bucket. With paged
Prefill on V1, each request's KV length includes both cached and scheduled
tokens. The default captured `max_seqlen_k` is twice the token bucket size.
For example, 64 cached tokens plus 128 scheduled tokens fit a captured
128-token bucket with the default KV bound of 256. If any request exceeds the
default bound, set `gr.prefill_kv_bound` high enough to cover its KV length.
Setting the option also adds small buckets at one sixteenth of each capture
size.

## Where attention reads the prompt

Paged attention reads KV from the cache. For a fresh, complete prompt, dense
attention can read contiguous K/V from the current layer's projection. Both
paths populate the real KV cache for Decode. Only the GR runners' fresh-Prefill
capture scope permits dense attention during graph capture.

Dense Prefill requires prefix caching, chunked Prefill and KV connectors to be
disabled. The worker checks live context and scheduled lengths. Batch-invariant
mode, sliding windows, ALiBi, softcap, sinks, quantized KV and KV sharing retain
paged attention.

## Configuration

| Startup setting | ModelRunner V1 | ModelRunner V2 |
| --- | --- | --- |
| `gr.prefill_token_buckets` is a list (packaged default) | With a configured beam width, values are batch token totals capped by `max_num_batched_tokens`. For a batch size `B` from `gr.beam_batch_sizes` and token total `T`, capture requires `B <= T <= B * max_model_len` and `B <= max_num_seqs`. Without a beam width, buckets are also capped by `max_model_len`. | Filters by the token budget `min(max_num_batched_tokens, max(batch_sizes) * max_model_len)` and filters batch sizes by `max_num_seqs`. |
| `gr.prefill_token_buckets: null` | Uses the built-in grid bounded by token and context capacity. | Uses native `cudagraph_capture_sizes`. An absent list uses the fallback grid; an explicitly empty native list stays empty. |

`gr.prefill_graph_enabled` controls GR Prefill capture. On V1,
`gr.prefill_kv_bound` can widen a bucket's KV bound and adds smaller capture
buckets. The independent `gr.dense_prefill` option selects attention on
eligible fresh prompts.

| `gr.dense_prefill` | Attention path |
| --- | --- |
| `off` (default) | Paged attention. |
| `fa2` | Dense FlashAttention 2. |
| `cudnn` | cuDNN SDPA for supported single-request shapes of at least 64 tokens. Other eligible shapes use dense FA2. |

## Performance

A token bucket can serve several nearby prompt lengths with one capture.
Closer bucket spacing reduces padding and increases the number of graphs the
worker may need to capture and keep. Prefix-cache hits can reduce the selected
bucket size. On V1, check whether the resulting default KV bound covers each
request's cached and scheduled tokens before setting an override.

## Validation

The [V2 runner tests](../../../tests/v1/worker/test_gpu_prefill_graph_runner_v2.py)
cover bucket selection and replay fallback; the [dense Prefill tests](../../../tests/kernels/attention/test_beam_dense_prefill.py)
compare attention outputs on CUDA.

## Source map

| File | Responsibility |
| --- | --- |
| [V1 Prefill graph runner](../../../vllm_gr/v1/worker/cuda/v1/gpu_prefill_graph_runner.py) | Owns V1 capture buffers, KV bounds and replay gates. |
| [V2 Prefill graph runner](../../../vllm_gr/v1/worker/cuda/v2/gpu_prefill_graph_runner_v2.py) | Builds the FULL and PIECEWISE plan and selects a descriptor at dispatch. |
| [V1 capture planner](../../../vllm_gr/v1/worker/cuda/v1/prefill_capture_plan.py) | Filters feasible token buckets and request counts before buffer allocation. |
| [Shared Prefill graph utilities](../../../vllm_gr/v1/worker/prefill_graph_common.py) | Resolves bucket limits and KV bounds, allocates buffers and matches live token counts. |
| [Dense Prefill attention](../../../vllm_gr/v1/attention/backends/beam_attn_gpu.py) | Checks whether attention can use projected KV. |

The next generation stage uses [CUDA Decode attention](cuda_decode_attention.md).
