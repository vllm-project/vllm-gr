# ModelRunner V2 multi-batch execution

CUDA ModelRunner V2 executes a fixed-width Beam batch as one owner. Each
prompt retains a native request and a request-local result. The scheduler
counts logical requests. The worker allocates the expanded Decode rows.
See [Beam batch execution](../feature/beam_batch_execution.md) for admission,
widths and generation limits; the [OneRec walkthrough](../../user_guide/onerec_quickstart.md)
covers engine setup. Set `VLLM_USE_V2_MODEL_RUNNER=1` before importing vLLM
to select V2 explicitly.

## Reuse of native vLLM

Native vLLM owns request registration, Prefix Cache, block tables and slot
mapping. For an eligible all-new Prefill batch, GR can skip unused native
sampler initialization and prepare tokens, positions and last-token logits
indices together. Eligibility is checked for every member before changing
slots. Mixed or unsupported batches retain native registration. Native input
preparation can reorder members. GR gathers the last-token logits in the
original request order before constrained selection.

## CUDA stage hooks contract

The [CUDA hooks](../../../vllm_gr/v1/worker/cuda/gpu_stage_hooks.py) bind
runner-specific input preparation, native output completion and graph policy.
The shared stage runner owns pending dispatches, cancellation and retirement.
Native output bookkeeping completes after submission and before pending state
is cleared. An asynchronous consumer keeps its owner alive until it consumes
the output. Retirement removes all native request IDs in the batch.

<a id="v1-compatibility-boundary"></a>

### Decode storage by model runner

ModelRunner V1 uses stage-owned inputs and can capture Decode against
replacement buffers. ModelRunner V2 uses views of fixed worker-owned Decode
storage and prepares configured batch shapes at startup. Changing the logical
batch size selects a compatible graph or eager execution. An unsupported
capture can fall back per shape after cleanup and a device-health check.
Losing a graph recorded as successfully
captured is an error. Profiled prefix-length buckets have a separate
[capture policy](../feature/cuda_decode_attention.md#cuda-graphs-and-fallback).

## Resource lifetime

Decode replay checks the model, context and input bindings from capture.
Reload and shutdown drain consumers and invalidate Prefill and Decode bindings
before native resources are replaced. A subsequent capture uses fresh output
storage. The [worker lifecycle tests](../../../tests/kernels/test_gpu_beam_stage_lifecycle.py)
exercise completion and cancellation; the [multibatch graph tests](../../../tests/kernels/test_v2_multibatch_graphs.py)
exercise batch switching and replay.

## Configuration and scope

The native request and scheduled-token budgets are separate from the expanded
worker row capacity.

The implemented scope is CUDA, one executing logical batch, fixed beam width,
asynchronous scheduling, and single-rank execution. The reference workload
uses one to three generation steps.
Keep `ignore_eos=True` and Chunked Prefill disabled. This scope does not include
unbounded B, concurrent batch owners, distributed execution, LoRA, multimodal
inputs or speculative decoding. The validation below exercises offline calls;
it does not establish automatic batching through an HTTP service.

V2 uses `gr.prefill_token_buckets` when set. With that field set to `null`,
it consults the native `compilation_config.cudagraph_capture_sizes`.
See [CUDA Prefill capture](../feature/cuda_prefill_graph.md) for bucket
selection, capacity limits, and fallback behavior.

## Regression coverage

The [unconstrained Beam test](../../../tests/entrypoints/gr/test_offline_beam_search.py)
checks offline candidate structure and runner behavior.

| Dimension | Coverage |
| --- | --- |
| Runners | CUDA ModelRunner V1 and V2 |
| Modes | Eager, default Graph and an explicit FULL Prefill grid |
| Batch sizes and width | One and four prompts, both at beam width 128 |
| FULL Prefill replay | 1024 tokens, and 4102 tokens padded to 4224 |

The test checks the replayed shapes, runner identity, Decode execution,
candidate structure and scores, request retirement and clean EngineCore
shutdown. The V2 graph cases also reject request-time capture.
