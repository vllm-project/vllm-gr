# Serving, binary transport and metrics

Use the [HTTP Beam example](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/beam_search/README.md)
for basic serving. This guide covers gRPC binary transport, independent replicas
behind Nginx, and the available Beam measurements.

- [gRPC and NumPy transport](#grpc-and-numpy-transport)
- [Independent replicas and load balancing](#independent-replicas-and-load-balancing)
- [Beam metrics](#beam-metrics)

## Choose an interface

| Interface | Use it for | Complete recipe |
| --- | --- | --- |
| Python | Offline calls and application-owned engine lifetime | [Offline Beam](https://github.com/JiusiServe/vllm-gr/blob/main/examples/offline_inference/beam_search/README.md), [API reference](../api/README.md) |
| HTTP | OpenAI-compatible Beam completion requests | [Start and verify a server](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/beam_search/README.md) |
| gRPC | Token-based batches and optional binary result columns | [gRPC example](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/grpc/README.md) |

All paths use the same resolved [startup configuration](../configuration/README.md).
For catalog output, prepare the [canonical table](../features/constraint-tables.md)
with the serving tokenizer and verify candidate membership after deployment.
HTTP prompt-list batching follows `gr.beam_batch_sizes`; it does not collect
separate requests into a session.

## gRPC and NumPy transport

GR gRPC serving accepts token inputs, submits Beam batches through the existing
Engine Core session lifecycle, and returns token-based results. NumPy prompt
arrays avoid repeated Python-list conversion across the frontend and internal
transport. Core storage and device staging still copy data; this is not an
end-to-end zero-copy interface.

### Inputs and request lifecycle

- Prompts accept repeated token IDs or `input_ids_i64_le`: a contiguous buffer
  of little-endian int64 values, with no header. Invalid or conflicting inputs
  are rejected; truncation and vocabulary validation follow the input contract.
- `Generate`, `GenerateBatch`, and `Abort` reuse Beam session admission,
  completion and cancellation. Batch members share generation options and can
  select different Tries through `trie_id` when Multi-Trie execution is enabled.
- Frontend, Engine Core and Workers must use compatible code and restart
  together when the internal wire contract changes.

See the [Beam batch contract](../features/beam-search.md#batch-execution-contract) for ownership
and lifecycle rules, and [Multi-Trie tables](../features/constraint-tables.md#multiple-trie-roots)
for constraint selection and execution limits.

### Usage and tests

Apply the [CLI dispatch patch](../cli/README.md#enable-gr-dispatch), install the
optional gRPC dependencies, and supply a prepared model and GR configuration:

```bash
pip install -e '.[grpc]'
vllm serve /path/to/model --gr --config /path/to/gr-config.yaml --grpc --host 127.0.0.1 --port 50051
python examples/online_serving/grpc/client.py --address 127.0.0.1:50051
```

The gRPC listener has no built-in authentication or transport encryption; the
HTTP API key does not protect it. Keep backend access restricted, and use a
trusted gateway for remote TLS and access control. See [SECURITY.md](https://github.com/JiusiServe/vllm-gr/blob/main/SECURITY.md).

The [smoke script](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/grpc/run_smoke.py) exercises the
serving entry point in a prepared GPU environment. Dependency versions and
binding generation are maintained in the package configuration and
[generation tool](https://github.com/JiusiServe/vllm-gr/blob/main/tools/generate_grpc_proto.py).

### Binary Beam results

The default response uses ordinary Beam fields. Clients can opt into
`BatchGenerateRequest.binary_beam_output = true` for column-based results:
`output_ids_i32_le` contains little-endian int32 tokens with shape
`[count, steps]`; requested scores and parent indices use the corresponding
FP32 and int32 columns.

Clients must accept both `binary_beams` and ordinary `beams`, because small,
stopped or variable-length results can use the ordinary representation.
This option changes the output representation, not generation semantics.

## Independent replicas and load balancing

Nginx can route HTTP requests to independent Beam Search V1 replicas.
Each replica has its own EngineCore, GPU and KV cache.

### Start two replicas

Prepare the [canonical V1 configuration](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/beam_search/README.md#prepare-the-environment-and-inputs).
Use the same model, served name and config for both replicas. In two terminals
run the [V1 server command](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/beam_search/README.md#start-the-server)
with these additions:

| Replica | Environment | Additional arguments |
| --- | --- | --- |
| First | `CUDA_VISIBLE_DEVICES=0` | `--host 127.0.0.1 --port 9001` |
| Second | `CUDA_VISIBLE_DEVICES=1` | `--host 127.0.0.1 --port 9002` |

### Configure Nginx

#### Install Nginx

```bash
sudo apt install nginx
```

#### Configure the upstream

Create a configuration file at `nginx_vllm.conf`. The `least_conn` algorithm is recommended as it accounts for the varying execution times of LLM generation requests.

```nginx
events {
    worker_connections 1024;
}

http {
    upstream vllm_cluster {
        # Directs traffic to the instance with the fewest active connections
        least_conn;

        server 127.0.0.1:9001;
        server 127.0.0.1:9002;
    }

    server {
        listen 127.0.0.1:8000; # Entry point for your application

        location / {
            proxy_pass http://vllm_cluster;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;

            # Optimized for Server-Sent Events (Streaming)
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_cache off;
            proxy_read_timeout 600s;
        }
    }
}
```

---

### Test the proxy

Run Nginx in the foreground with your configuration. The packaged Nginx
installation may require elevated privileges for its default PID and log paths:

```bash
sudo nginx -c /absolute/path/to/nginx_vllm.conf -g 'daemon off;'
```

Use the [V1 client request](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/beam_search/README.md#send-a-request-and-verify-the-result)
against `http://127.0.0.1:8000/v1/completions`. For load testing, use the
[HTTP throughput benchmark](../usage/benchmarks/throughput.md) with the proxy's host
and port.

## Beam metrics

### Current V1 measurements

Beam Search V1 owns Prefill and Decode in EngineCore and the Worker. The old
frontend phase loop no longer calls a GR Prometheus collector. Consequently
this version does not automatically expose the old
`vllm_gr_beam_search_*` Prefill/Decode/sort counters or histograms.
The unused `vllm_gr.v1.metrics.beam_search_metrics` collector and
`vllm_gr.v1.metrics.stats.RequestStateStats` extension have been retired.

The native vLLM `/metrics` endpoint is unchanged. Its native request metrics
are not replacements for measured GR Beam stage durations. For reproducible
Beam latency, use the [offline V1 benchmark](https://github.com/JiusiServe/vllm-gr/blob/main/benchmarks/qwen/README.md) or
[paired offline A/B tool](https://github.com/JiusiServe/vllm-gr/blob/main/tools/AB_Tests/README.md), keeping the same model,
input length, width, cache mode, warmup, and output-consumption scope.
The offline A/B tool measures public text materialization as part of E2E.
Online benchmarks additionally include transport and serving overhead.

### Benchmark compatibility with older servers

The OneRec online benchmark still reads historical phase counters when a
server provides them. It requires Prefill, Decode, sort, and request counters
in both snapshots before printing a phase summary. Missing or incomplete
counters are reported as unavailable; they are not interpreted as zero cost.
Non-successful HTTP responses and unavailable endpoints do not prevent the
main benchmark from running. A counter reset invalidates the delta window.

Each sample retains its metric name and complete label set until both
snapshots have been validated. Every label set must have all four counters
in both snapshots; new, disappearing, incomplete, or reset series invalidate
the window. Only validated per-series deltas are summed, so growth in one
engine cannot hide another engine's reset. Label and sample order do not
affect identity. Stable series with no new requests are allowed.
If the total-beams counter appears in either snapshot, it must also pass the
presence, complete label-set, and per-series monotonicity checks. It remains
optional when absent from both snapshots.
The counters describe server-wide activity; isolate benchmark traffic if the
result is intended to represent a particular run. This report is separate
from client-observed E2E and is not a substitute for it.

Future V1 stage metrics require explicit device/host timing boundaries and
aggregation across EngineCore/Worker processes. They must not infer stage
latency from the retired frontend timers or add synchronization to the hot
path solely to reproduce those counters.
