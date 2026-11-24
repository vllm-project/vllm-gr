# API server

Choose an API based on the input your client already has and the output it
needs. OneRec can serve ranked Beam candidates over HTTP or experimental gRPC.

## Start and verify the server

For catalog-constrained HTTP serving, [prepare the OneRec catalog and start
the server](../../examples/online_serving/beam_search/README.md). In a second
terminal, check readiness and the model name your client should send.

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

For gRPC, follow the [OneRec gRPC example](../../examples/online_serving/grpc/README.md)
to start a token-input server and check channel readiness.

## Choose an endpoint

| Client input | Endpoint | Output |
| --- | --- | --- |
| Text prompt | [`POST /v1/completions`](completions.md#request) | One text choice per returned candidate |
| Conversation turns | [`POST /v1/chat/completions`](completions.md#chat-completions) | One assistant message per returned candidate |
| Token IDs | [gRPC `GenerateBatch`](grpc.md) | Token sequences and Beam metadata grouped by input |

## Deployment and monitoring

- [Route HTTP requests across CUDA replicas](replica_deployment.md).
- [Inspect serving metrics](metrics.md).
- [Update model weights or the catalog](../user_guide/weight_update.md).
