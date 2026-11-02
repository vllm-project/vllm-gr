# Asynchronous scheduling

Asynchronous scheduling runs Beam Search V1 on the native vLLM 0.30.0 `execute_model → sample_tokens`
interface. The next dispatch can prepare inputs and submit GPU work after the
preceding sampling submission, before EngineCore consumes its receipt.

For a runnable OneRec-1.7B setup before reading this execution contract, see
[the OneRec-1.7B V1 asynchronous scheduling quickstart](onerec_v1_async_quickstart.md).

## Supported execution contract

- GPU ModelRunner V1, FP16/BF16, uniform text decoder with Beam Attention and a
  canonical CUDA constraint table; the existing model, KV and head-size checks apply.
- B=1, one execution owner, Q=2 in-flight stage dispatches. Q is not request concurrency.
- W is fixed at startup, 1–256. Effective N is 1, 2 or 3 and includes the first
  token sampled from Prefill. TP/PP/DP/DCP/PCP are all 1.
- `ignore_eos=True` is required. A legal EOS remains an ordinary candidate;
  generation ends by length after N outputs. Disable Chunked Prefill; partial
  Prefill and recompute after generation are rejected. Prefix-cache hit/miss is supported.
- Native async scheduling and an Executor batch queue of at least two entries.
  No ordinary/legacy Beam request mixing while an asynchronous scheduling session owns resources.
- No speculative decoding, LoRA, KV/EC transfer, quantization, sliding-window
  attention, multimodal inputs, hybrid models or microbatching.

Public offline and online `beam_search_v1()` return complete sequences and
cumulative scores only after matching resource-release confirmation.
`GRLLM.beam_search()` raises a migration error; the legacy frontend step loop
is retired.

## Native progress and execution ownership

One native Request is added once and remains resident across Prefill and Decode.
After native `_update_after_schedule`, the stage index is
`num_computed_tokens - num_prompt_tokens`: zero is Prefill, and positive values
are Decode with `decode_step = stage_index - 1`. The native Scheduler counts one
logical token per Decode; the Worker runs W physical rows using its own mutable
input view. GR stores dispatch/generation identities, dependency records and KV
leases, without a separate token/placeholder budget.

For P=900 and N=3, a cache miss schedules 900 Prefill tokens; a hit reusing
896 tokens schedules 4. Both reach computed=900, stage=0. Subsequent Decode
stages reach computed=901/902 and schedule one logical token each. Worker
snapshots carry the pre-schedule count and validate this same token extent.

The refactor retains the baseline's native BatchQueue/Executor/Futures, two-slot
lookahead, GPU Beam state, streams, staging, graph capture, kernels and output
packing. The responsibilities are:

1. **Admission before native selection.** A view of native waiting queues exposes
   only the FIFO execution owner, retaining the original Request objects and queue
   ordering. A zero native token budget blocks new stage dispatches when
   either physical slot is unavailable. Admission reserves P+N-1 native KV positions
   through `allocate_slots(num_lookahead_tokens=...)`. Temporary shortage applies
   backpressure; impossible capacity is rejected. Recompute after generation is rejected.
2. **Complete per-step CPU preparation.** `GRInputBuffers.prepare_cpu` rebuilds
   positions, offsets, suffix lengths, prefix blocks and logits rows from Host
   metadata and bindings. No generated device state is read on the CPU. The
   metadata builder's asynchronous entrypoints bypass its geometry, layout,
   view, copy and schedule caches. Missing CPU mirrors are errors.
3. **Ordered device binding.** Ordinary PyTorch operations select actual device
   tokens and constraint prefixes, validate generated count and planned step, and
   mask terminated successors. Active step mismatches produce device error 6.
   Two pinned staging slots each have an H2D completion event. H2D, state binding
   and Forward run on the compute stream; control snapshots use separate slots
   and D2H events. The first Decode retains the native input-preparation Host fence.
4. **Startup graph capture.** A dedicated fixed-W Forward graph is captured
   at startup; LM head and Beam postprocessing run outside it. Device input, mask,
   metadata and KV addresses stay fixed while their contents change each step and
   request. Metadata descriptions are fresh each step.
   Prefix bounds are block-aligned model capacity; suffix bounds are pool capacity.
   A requested but unavailable graph is an error; requests never capture graphs.
   Asynchronous Prefill reuses the existing project Prefill runner, including padded
   graph replay: 900 scheduled tokens can use a captured 1024-token bucket. Bucket
   selection uses the scheduled token count after prefix-cache hits.
   Replay refreshes static buffers, preserves real Attention lengths and masks padded
   KV writes. Missing startup graphs, unsupported live inputs, missing metadata or
   slot mapping, and exceeded captured KV-length or buffer bounds still fall back
   to the native Forward path; requests never create or capture a new runner.
   Padding alone does not trigger eager fallback. It can change BF16 results and
   Beam candidate ordering; the historical exact-bucket results do not establish
   precision for the padded path, whose matching validation remains outstanding.
5. **Matching release proof.** Terminal data waits until all in-flight dispatches
   drain and resource release is confirmed. The last terminal output consumer can
   report `resources_released=True`, preserving the existing normal path. When
   that proof is unavailable, a zero-token SchedulerOutput carries session,
   owner generation and retirement ID. Worker drains consumers/GPU work and
   clears session/workspace ownership before acknowledging retirement. Pending
   terminal data survives this round trip and is delivered once, then the next
   owner can start. Cancelled waiting requests allocate no Worker resources;
   running cancellation drains and releases without sending a late success.
6. **Native queue progress and errors.** Retirement counts as pending work even
   for the last request. Submissions, receipt consumption, retirement and admission
   increment Host progress and avoid the fixed idle sleep. Each active dispatch
   requires a matching Worker receipt; absent receipts cannot confirm tokens.
   Async conversion failures
   become request errors; fatal Worker/CUDA failures reach native Executor failure
   handling, including exceptions raised in Multiproc's output consumer thread.

```mermaid
sequenceDiagram
    participant E as EngineCore
    participant W as Worker CPU
    participant G as GPU compute
    participant C as Receipt consumer
    E->>W: execute_model(k)
    W->>G: Input / Forward(k)
    E->>W: sample_tokens(k)
    W->>G: Beam update / control snapshot(k)
    E->>W: execute_model(k+1)
    W->>W: CPU layout and metadata build
    W->>G: Ordered upload / bind / Forward(k+1)
    G-->>C: Independent control copy(k)
    C-->>E: Retire dispatch(k)
    alt Last terminal consumer released resources
        C-->>E: Terminal receipt with matching release proof
    else Independent cleanup required
        E->>W: Zero-token retirement frame after drain
        W-->>E: Matching release proof after actual cleanup
    end
    E->>E: Deliver terminal result once (unless cancelled)
    E->>W: Admit next session
```

### First Decode after Prefill

Native scheduling records complete Prefill progress when it is scheduled;
its real output need not have reached the CPU. Once Worker has submitted both
Prefill execution and sampling, EngineCore can submit D0 into the second slot.
Worker begins D0's CPU layout preparation inside `execute_model(D0)`. It then
checks the native pinned-input fence, updates native request state, uploads the
layout, builds attention metadata, and submits device binding and Forward.
The fence protects Host buffer reuse; it does not wait for Prefill Forward.
Actual token/Beam-state reads remain ordered after Prefill sampling on the GPU.

Prefill Graph reduces the time spent submitting model operations on the Host.
That leaves a larger GPU execution window for D0 preparation in the same Worker
thread. Graph replay and asynchronous queue progress are both needed to explain
the observed benefit; graph replay alone does not move CPU preparation earlier
when the benchmark deliberately waits for each receipt. Short eager fallbacks
can still finish before the next dispatch's CPU preparation starts.

## Validation

The [validation summary](async_scheduling_validation.md) records historical
execution work. Experimental benchmarks,
trace analysis and diagnostic instrumentation are maintained outside the
production change.
