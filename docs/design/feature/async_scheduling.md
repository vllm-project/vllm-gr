# Asynchronous scheduling

## Table of Contents

1. [Overview](#overview)
2. [Performance](#performance)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [Validation](#validation)
6. [Related files](#related-files)

## Overview

Each beam Decode consumes token IDs selected by the previous stage. EngineCore
can issue the next stage before the worker acknowledges the previous one. This
lets the CUDA worker prepare the next input layout while the GPU is still
selecting beams. The selected tokens stay on the device. CUDA and Ascend share
the scheduler and implement the handoff differently.

## Performance

Use the [Beam performance workload](../../cli/bench/beam_performance.md) to
measure a specific scheduler revision. Keep the model, prompt, batch size,
beam width and Prefill mode fixed across runs. Collect a separate timeline
to check whether host preparation overlaps the previous GPU stage.

## Architecture

### Native progress and execution ownership

The GR scheduler gives one fixed batch at a time ownership of Beam execution
in arrival order. Up to two stages of that batch can await acknowledgement.
The native scheduler tracks one request per prompt and allocates the prompt
KV cache. Prefill selects the first generated token. Each subsequent Decode
advances the native request by one position while the worker evaluates its
surviving beams.

For a 128-token prompt and a three-token output, the reservation covers 130
positions: the prompt plus two Decode inputs. Cached prefix blocks count
toward that reservation.

### Dispatch timeline

On CUDA, preparing the layout for step *k+1* depends on the scheduled shape.
Running its forward also depends on the tokens selected at step *k*. The
figure follows two nonterminal steps from left to right. Arrows show
dependencies. The status copy has its own stream.

![CUDA timeline showing next-step host preparation overlapping beam selection and a status copy branching from completed selection.](../../assets/async_scheduling_timeline.svg)

| First Decode after Prefill | When the worker can update native request state |
| --- | --- |
| CUDA ModelRunner V1 | After the input event, when present. Prefill can still be reading pinned host buffers that the update will reuse. |
| CUDA ModelRunner V2 | As soon as the layout is ready. Each step gets a fresh input batch. |

### Ascend stage execution

One Ascend runner handles every stage of the batch. It samples its pending
forward before accepting another stage, returns a receipt to EngineCore and
keeps the selected token IDs on the NPU. Two prompts with input width three
advance one position each in the scheduler and produce six physical Decode
rows. See [Ascend Decode attention](npu_decode_execution.md).

### Completion and cancellation

EngineCore delivers the terminal result after issued stages drain and the
worker confirms release for the matching session and owner generation. On
Ascend, final sampling builds the result, releases the Beam state and sets
`resources_released` in the same receipt. If stages have drained but the
worker still owns resources, the scheduler issues a zero-token cleanup and
waits for its acknowledgement.

A queued cancellation can finish immediately. For an active cancellation,
the scheduler stops issuing stages, drains outstanding work, discards a
late success and releases the owner before admitting the next batch. See
[Beam state lifetime](../module/beam_state.md#streams-graphs-and-retirement).

## Configuration

The packaged Beam configuration uses these native vLLM options at startup.
Batch and token budgets are described in the
[configuration reference](../../configuration/README.md#asynchronous-scheduling).

```yaml
async-scheduling: true
enable-chunked-prefill: false
```

## Validation

The [OneRec scheduler system test](../../../tests/st/test_onerec_v1_scheduler_st.py)
checks offline and online execution on both CUDA runners, including retirement
at result delivery. On Ascend, the [worker tests](../../../tests/v1/worker/test_npu_model_runner_patch.py)
check logical and physical row counts, the [cleanup tests](../../../tests/v1/worker/test_npu_stage_preparation_rollback.py)
check failure recovery, and the [dynamic-width system test](../../../tests/st/test_npu_dynamic_beam_st.py)
exercises eager and graph execution.

## Related files

- [GR scheduler](../../../vllm_gr/v1/core/sched/beam_async_scheduler.py) controls stage issue, acknowledgements and retirement.
- [CUDA stage runner](../../../vllm_gr/v1/worker/cuda/gpu_beam_stage_runner.py) orders device work and status copies. The [V1](../../../vllm_gr/v1/worker/cuda/v1/gpu_beam_stage_runner_v1.py) and [V2](../../../vllm_gr/v1/worker/cuda/v2/gpu_beam_stage_runner_v2.py) hooks implement the host-buffer handoff.
- [Ascend stage runner](../../../vllm_gr/v1/worker/npu/npu_beam_stage_runner.py) executes and samples a pending stage.
- [Beam batch execution](beam_batch_execution.md#validation-boundaries) defines admission. The [OneRec walkthrough](../../user_guide/onerec_quickstart.md) shows a complete request.
