# User guide

## Inference and serving

| I want to | Guide |
| --- | --- |
| Run Beam search with or without a constraint table on CUDA | [Beam search on CUDA](beam_search_cuda.md) |
| Run Beam search without a constraint table on Ascend | [Unconstrained Beam search on Ascend](beam_search_npu.md) |
| Check which features apply to my device | [Features and compatibility](feature_compatibility.md) |
| Configure Beam width, candidate selection, or a constraint table | [Configuration](../configuration/README.md) |
| Build a catalog artifact and select a Trie per request | [Constraint tables](../configuration/constraint_tables.md) |
| Start a server and send a request | [Serving APIs](../serving/README.md) |
| Read the candidate SIDs in an HTTP response | [Completions response](../serving/completions.md#response) |
| Read candidate tokens and cumulative log probabilities from an offline run | [OneRec example output](../../examples/README.md#expected-output) |
| Decode gRPC binary Beam outputs and optional scores | [Binary Beam results](../serving/grpc.md#binary-beam-results) |
| Look up `vllm serve`, `bench`, or `check-env` options | [CLI reference](../cli/README.md) |
| Find a model and its requirements | [Supported models](../models/supported_models.md) |
| Run an offline inference or online serving example | [Examples](../examples/README.md) |

## Deployment and operations

- [Deploy HTTP replicas behind Nginx](../serving/replica_deployment.md).
- [Update weights or switch catalogs on running instances](weight_update.md).
- [Monitor serving metrics](../serving/metrics.md) and interpret historical Beam counters.

<a id="related"></a>

## See also

- [Benchmarking](../benchmarking/README.md) for accuracy and throughput tasks.
- [Design](../design/index.md) for the mechanisms behind these tasks.
