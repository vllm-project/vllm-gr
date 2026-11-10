# Online load control and overload semantics

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the vLLM project -->

This page covers what a beam-search deployment should do when offered load
exceeds what one engine instance can serve, and which knobs the serving process
exposes for it.

## Why overload needs an explicit answer

A GR beam session holds the engine's execution owner until it retires, and one
request is a short chain of stages (prefill, then a few decode steps). The
engine therefore has a hard service rate of roughly `1 / single_request_latency`
per instance. Offering more than that rate leaves two possible behaviours:

- **Queue everything.** Latency grows without bound, clients hit their own
  timeouts, and the work they abandon keeps occupying the engine. Throughput
  stops rising while p99 explodes.
- **Reject early.** Requests beyond capacity get a retryable `429`, admitted
  requests keep a bounded latency, and throughput stays at the service rate.

The serving process implements the second behaviour. It is off by default so
existing deployments keep their current semantics.

## Frontend admission knobs

| Environment variable | Meaning | Default |
| --- | --- | --- |
| `VLLM_GR_BEAM_MAX_INFLIGHT` | Beam sessions this process may have in flight at once | unset (unlimited) |
| `VLLM_GR_BEAM_MAX_WAITING` | Requests allowed to queue for a slot behind them | `0` |
| `VLLM_GR_BEAM_ADMISSION_TIMEOUT_S` | How long a queued request may wait for a slot | `0` |
| `VLLM_GR_BEAM_OUTPUT_TOP_K` | Return at most this many beam results | unset (all) |

Rejections are answered with HTTP `429` and a body whose `code` is `429`. They
are logged at warning level on the first rejection and then every 100th one,
together with the current in-flight/waiting counters.

`VLLM_GR_BEAM_MAX_INFLIGHT` is the capacity decision. Measure the single-request
latency `T` at the target beam width and prompt length, then set the budget to
the concurrency the instance can actually sustain:

```text
max_inflight <= acceptable_throughput * T
```

Setting `VLLM_GR_BEAM_MAX_WAITING` and `VLLM_GR_BEAM_ADMISSION_TIMEOUT_S` to
small, non-zero values absorbs short bursts without letting a backlog form.
Leave them at `0` when the caller would rather shed than wait.

`VLLM_GR_BEAM_OUTPUT_TOP_K` only changes the number of returned choices; the
engine still runs the full beam width. Use it when the caller reads a handful of
candidates out of a wide beam, so the server does not detokenize and serialize
results nobody looks at. Prefer reducing the request's beam width when the
smaller width still gives the quality the business needs.

## Client and gateway contract

Admission control only works if callers cooperate:

1. **Set client timeouts above the p99 you accept.** A timeout shorter than the
   server's own stage chain turns healthy requests into cancelled work.
2. **Treat `429` as backpressure, not as a failure.** Retry with exponential
   backoff and jitter; a fixed-interval retry converts a queue into a retry
   storm.
3. **Bound retries.** Do not retry a request that already consumed a bounded
   budget of attempts; report the failure upstream instead.
4. **Keep the gateway's own queue shallow.** A load balancer that queues deeply
   in front of an admission-controlled server moves the same problem one hop
   earlier.
5. **Scale out, not up, for throughput.** Each replica adds roughly one service
   rate; per-replica admission keeps the latency of each replica predictable.

## Related settings

- `--max-num-seqs` and the beam width together decide how many sessions one
  engine can run; keep per-instance admission consistent with that number.
- `--max-logprobs` sizes the per-step logprob payload. It should match what the
  workload reads, not the vocabulary size.
