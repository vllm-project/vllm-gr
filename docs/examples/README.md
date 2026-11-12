# Examples

Use a complete recipe for the interface you plan to deploy. Runnable scripts
and their detailed instructions stay together in the source checkout.

| Goal | Recipe | Inputs to prepare | What to verify |
| --- | --- | --- | --- |
| Make one offline request | [First inference](../getting_started/first-inference.md) | OneRec model snapshot and supported accelerator | Returned candidates and successful shutdown |
| Integrate Beam from Python | [Offline reference](https://github.com/JiusiServe/vllm-gr/blob/main/examples/offline_inference/beam_search/README.md) | Model, request parameters and optional table | Token sequences, cumulative scores and width limits |
| Serve HTTP requests | [HTTP Beam](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/beam_search/README.md) | Unified YAML and model snapshot | Readiness, response shape and returned SIDs |
| Require catalog membership | [Catalog serving](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/catalog/README.md) | Catalog, compiled artifact and both digests | Candidate membership with the catalog verifier |
| Send binary token batches | [gRPC Beam](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/grpc/README.md) | gRPC dependencies, token inputs and startup YAML | Batch ordering, token columns and scores |
| Evaluate Pangu | [Pangu workflow](../contributing/tools/pangu1b.md) | Prepared model, vendor runtime and evaluation inputs | Task-specific evaluation results |

## Choose the right evidence

An example demonstrates a public path. For acceptance evidence, use the
[benchmark guides](../usage/benchmarks/README.md): accuracy needs identified
evaluation inputs, latency needs a matching warmup/cache protocol, and graph
claims need actual replay observations.

Start with [Getting started](../getting_started/README.md) if the matching
runtime is not yet installed. Keep credentials and private datasets outside
the repository.
