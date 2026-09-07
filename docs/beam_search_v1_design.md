# `beam_search_v1` Asynchronous Scheduling Framework

## A1 Scope

A1 defines the interfaces, state, and fixed control flow for the new Beam
Search path. Follow-up tasks implement these interfaces without depending on
the step-by-step submission flow in the legacy `beam_search` implementation.

The initial implementation supports B=1: each GR session contains one input.
The EngineCore asynchronous queue may advance multiple sessions concurrently,
so B=1 does not limit request concurrency.

## Server Layer

The offline entry point is `GRLLM.beam_search_v1`, and the online entry point
is `OpenAIServing.beam_search_v1`. They use different input and output formats
but follow the same control flow:

1. `prepare_request` creates a shared `BeamBatchRequest`.
2. `submit_once` submits the complete GR session to EngineCore once.
3. `wait_final` waits once for the terminal `BeamBatchResult`.
4. `build_output` converts the terminal result to the Server output format.
5. `close` releases Server-side session resources.

The Server does not submit Prefill or Decode steps and does not receive
intermediate Beam results. The online adapter waits asynchronously so that it
does not block the service event loop, while the offline adapter may wait
synchronously. Both adapters ultimately call the same
`EngineCore.submit_gr_session(request)` entry point.

## EngineCore Layer

`GRSessionState` is the persistent EngineCore/Scheduler state for one session.
It records:

- the original `BeamBatchRequest` and session ID;
- the current session status;
- the number of issued and completed stages;
- `output_placeholders` for scheduled outputs that the Host has not confirmed;
- whether the terminal result is waiting to be returned.

The stages of one session are fixed:

- stage 0 is Prefill and is dispatched exactly once;
- stages 1 through N are Decode steps, each dispatched exactly once.

EngineCore exposes or uses the following interfaces:

- `submit_gr_session` accepts the single online or offline submission;
- `step_gr_with_batch_queue` is the shared asynchronous scheduling loop;
- `issue_gr_stage` asks the Scheduler for one stage and dispatches it to the
  Worker without blocking;
- `complete_gr_stage` consumes one Future and updates the session and its
  placeholders;
- `finish_gr_session` creates the terminal result and releases resources;
- `abort_gr_session` terminates a failed or cancelled session.

`step_gr_with_batch_queue` follows this sequence:

```text
accept new sessions
    ↓
while the queue has capacity, select a runnable session
    ↓
create one Prefill or Decode command
    ↓
execute_gr_stage(non_block=True) and enqueue its Future
    ↓
continue scheduling while capacity remains
    ↓
consume the oldest Future and update its session
    ↓
return a terminal result, or make the session eligible for another step
```

## Asynchronous Execution

The framework covers three sources of device idle time:

1. **Before Prefill:** while the Server or CPU prepares one request, the Worker
   can execute stages from other sessions. A new request joins scheduling only
   after it reaches EngineCore.
2. **Within one request:** the Scheduler uses `output_placeholders` for bounded
   look-ahead. The Worker keeps the previous sampling result and Beam state on
   the device, avoiding a full Host round trip between Prefill and Decode or
   between consecutive Decode steps.
3. **After the final step:** while the CPU consumes a terminal Future and
   converts its output, EngineCore can continue dispatching stages from other
   sessions to the Worker.

`GRAsyncBatchQueueInterface` stores non-blocking Futures.
`can_issue_gr_stage` decides whether a session may be scheduled ahead. Queue
capacity limits total concurrency, while placeholders limit look-ahead for one
session.

## Follow-up Task Boundaries

- **A2:** convert Server arguments into a B=1 `BeamBatchRequest`, create the
  session, and submit it once.
- **A3:** implement Scheduler session state, placeholders, the batch queue, and
  EngineCore `step_gr_with_batch_queue`.
- **A4:** implement Worker `execute_gr_stage`, keeping Beam and sampling state
  on the device for the next stage.
- **A5:** consume the final stage result, create `BeamBatchResult`, release
  resources, and build the Server output.

A2, A4, and A5 depend only on the A1 types and interfaces. A3 connects them
through the shared EngineCore scheduling loop.
