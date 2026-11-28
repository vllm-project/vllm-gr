# Online serving

<a id="api-server"></a>
<a id="choose-an-endpoint"></a>

Serve OneRec with [HTTP Beam completions](completions.md) via
`POST /v1/completions` or [experimental gRPC](grpc.md) for tokenized requests.

## Start and verify the server

To launch OneRec HTTP, follow the
[OneRec server example](../../examples/online_serving/beam_search/README.md#prepare-a-persistent-service)
for catalog and startup setup. Check readiness and retrieve the model name
after startup.

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

## Deployment and monitoring

- [Inspect serving metrics](#read-prometheus-metrics).
- [Update model weights or the catalog](../user_guide/weight_update.md#serving-prerequisites)
  through a gRPC service with an HTTP management listener. The HTTP
  inference entrypoint does not enable these update endpoints.

### Read Prometheus metrics

After the HTTP server is ready, inspect its native vLLM metrics:

```bash
curl --fail http://127.0.0.1:8000/metrics
```

For end-to-end Beam latency and HTTP throughput, use the
[benchmark guide](../benchmarking/throughput.md). Historical phase counters
are covered under [benchmark compatibility](../benchmarking/throughput.md#benchmark-compatibility-with-older-servers).

### Independent HTTP replicas

Start each CUDA replica on its own device and port with the same served model
name and compatible catalog configuration. Each process owns its own engine and
KV cache. Route requests through your HTTP proxy and send a
[`POST /v1/completions` request](completions.md#request) through it to verify
the deployment.
