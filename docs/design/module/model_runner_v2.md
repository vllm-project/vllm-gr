# ModelRunner V2 multi-batch execution

CUDA ModelRunner V2 executes a fixed-width Beam batch as one owner. Each prompt
retains a native request, and the worker produces one `GRWorkerResult` per
session (batch) carrying the per-item results. The scheduler counts logical
requests; the worker allocates the expanded Decode rows. See
[Beam batch execution](../feature/beam_batch_execution.md) for admission,
widths and generation limits; the [OneRec walkthrough](../../user_guide/onerec_quickstart.md)
covers engine setup. Set `VLLM_USE_V2_MODEL_RUNNER=1` before importing vLLM to
select V2 explicitly.

## Reuse of native vLLM

Native `RequestState` and `BlockTables` own the request and KV data, including
Prefix Cache and block tables. For an eligible all-new Prefill batch, `CudaGRModelRunnerV2.add_requests` registers the batch itself through the same native
helpers and skips native sampler initialization. Other dispatches fall through
to native registration. At Decode, GR derives the expanded slot mapping from
native block IDs.

Eligibility is checked for every member before changing slots. Mixed or
unsupported batches retain native registration. Native input preparation can
reorder members; GR gathers the last-token logits in the original request order
before constrained selection.

## CUDA stage hooks contract

The [CUDA hooks](../../../vllm_gr/v1/worker/cuda/gpu_stage_hooks.py) bind
runner-specific input preparation, native output completion and graph policy.
The shared stage runner owns pending dispatches, cancellation and retirement.
Native output bookkeeping completes after submission and before pending state
is cleared. An asynchronous consumer keeps its owner alive until it consumes
the output. Retirement removes all native request IDs in the batch.

### Decode storage by model runner

ModelRunner V1 uses stage-owned inputs and can capture Decode against
replacement buffers. ModelRunner V2 uses views of fixed worker-owned Decode
storage and prepares configured batch shapes at startup. Changing the logical
batch size selects a compatible graph or eager execution. An unsupported
capture can fall back per shape after cleanup and a device-health check. Losing
a graph recorded as successfully captured is an error. Profiled prefix-length
buckets have a separate [capture policy](../feature/cuda_decode_attention.md#decode-graphs-and-fallback).

## Resource lifetime

V2 Decode replay checks the captured model, context and input storage; V1
replay checks the context and the exact batch, width and prefix-bucket
signature. Reload and shutdown require dispatch and output-consumer drain
before they invalidate the Prefill and Decode bindings; a live borrower is
rejected. A subsequent capture uses fresh output storage.
`profile_cudagraph_memory` re-runs the GR capture plan and closes the V2
runtime in its `finally`, which deletes the runtime and the Decode graph.

The V2 runner also has these limits:

- A legacy `beam_data` frame fails fast with a `ValueError` naming both
  remediations.
- Startup Decode capture covers only configured batch sizes that fit
  `max_num_seqs`; a larger configured size never captures.
- Capture is disabled entirely when `beam_max_decode_steps <= 1`, because a
  one-token session completes in Prefill and never replays a Decode graph.
- A runtime capture refusal is recorded per `(batch_size, width,
  prefix_length_bucket)` geometry for the runtime generation; replay of a
  refused geometry stays eager for the rest of that generation.

<a id="v1-compatibility-boundary"></a>

## Configuration and limits

Admission accepts one to `max_decode_steps` generation steps; the reference
OneRec workload uses three. The V2 Prefill plan exists only for a single rank
with no data-parallel ubatching, LoRA, speculative decoding, auxiliary hidden
states, multimodal input or `enforce_eager`. Chunked Prefill is rejected, and
GR enforces `ignore_eos=true` on admitted requests.

## Validation

| Test | Coverage |
| --- | --- |
| [Worker lifecycle tests](../../../tests/kernels/test_gpu_beam_stage_lifecycle.py) | Completion, cancellation and retirement. |
| [Multibatch graph tests](../../../tests/kernels/test_v2_multibatch_graphs.py) | Batch switching and replay. |
| [V2 patch tests](../../../tests/v1/worker/test_gpu_model_runner_v2_patch.py) | Input adaptation and native output completion. |
| [Stage binding tests](../../../tests/v1/worker/test_model_runner_stage_binding.py) | Concrete Runner stage dispatch and execute/sample ownership. |

## Source map

| File | Responsibility |
| --- | --- |
| [Stage hooks](../../../vllm_gr/v1/worker/cuda/gpu_stage_hooks.py) | Define runner-specific input, output and graph policy. |
| [CUDA stage runner](../../../vllm_gr/v1/worker/cuda/gpu_beam_stage_runner.py) | Owns pending dispatches, status copies and retirement. |
| [V2 resources](../../../vllm_gr/v1/worker/cuda/v2/gpu_model_runner_v2_resources.py) | Owns the V2 runtime, Decode graphs and cleanup checks. |
| [V2 runner patch](../../../vllm_gr/v1/worker/cuda/v2/gpu_model_runner_v2_patch.py) | Adapts native input preparation and output completion. |
| [GR startup](../../../vllm_gr/v1/worker/gr_startup.py) | Decides which capture sizes and widths are installed at startup. |
