# Serving metrics

Use the HTTP server's native vLLM `/metrics` endpoint to monitor request
activity.

## Read Prometheus metrics

Use this command against a local instance.

```bash
curl --fail http://127.0.0.1:8000/metrics
```

## Measure Beam latency

The GR Beam path executes Prefill and Decode in EngineCore and the Worker.
Its current exporter does not publish separate `vllm_gr_beam_search_*` phase
counters. To measure end-to-end latency, use the [Beam performance workload](../cli/bench/beam_performance.md)
or the [online throughput benchmark](../benchmarking/throughput.md). Hold the
model, prompt length, width, cache mode and warmup constant. Offline CUDA
timing includes candidate text materialization. Online timing also includes
transport and serving overhead.

## Benchmark compatibility with older servers

The OneRec online benchmark can report historical Prefill, Decode and sort
times when the server supplies those counters and the request count. Both
snapshots must have complete, matching label sets and nondecreasing counters.
A missing or reset series makes the phase summary unavailable. An optional
total-beams counter, when present, must pass the same checks.

!!! note "Benchmark isolation"

    These counters cover all server traffic. Isolate the workload before
    attributing server-wide deltas to one benchmark run.
