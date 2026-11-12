# Features

Choose a feature by the behavior you need, then read its supported configuration
and backend limits. Start with [first inference](../getting_started/first-inference.md)
before combining optimizations. The [architecture](../design/main-arch.md)
explains the execution owners shared by these features.

## Generation and constraints

| Goal | Guide | Key boundary |
| --- | --- | --- |
| Retain and rank multiple continuations | [Beam Search](beam-search.md) | One atomic batch, EngineCore ownership, Worker decisions |
| Require catalog-valid SIDs | [Constraint tables](constraint-tables.md) | Three-level tables bound to an artifact and tokenizer digest |
| Select among multiple catalogs | [Multiple Trie roots](constraint-tables.md#multiple-trie-roots) | One root selector per logical prompt |
| Generate without a catalog | [Unconstrained Beam](unconstrained-beam-search.md) | Full-vocabulary scoring with backend-specific width limits |

## Execution and memory

| Goal | Guide | Check before enabling |
| --- | --- | --- |
| Reduce repeated CUDA launch overhead | [CUDA graphs](cuda-graphs.md) | Runner, supported batch shapes, fixed storage bindings and replay evidence |
| Optimize shared-prefix or short-suffix Attention | [Attention](attention.md) | Selected FlashAttention path, calibration and supported geometry |
| Use Ascend BSND Prefill | [NPU BSND](attention.md#npu-bsnd-prefill) | Fresh uncached prompts, capture specifications and fallback |
| Use the optional XAttention output binding | [NPU XAttention](attention.md#npu-xattention-output-binding) | Matching vendor runtime and separately validated addon |

## Serving and operations

| Goal | Guide | Key boundary |
| --- | --- | --- |
| Choose Python, HTTP or gRPC | [API](../api/README.md), [Serving](../serving/README.md) | Public request and result contracts |
| Route a prompt list as one session | [HTTP batching](../getting_started/quickstart.md#batching-several-prompts-in-one-http-request) | Configured list sizes and native capacity; separate requests are not combined |
| Replace weights or catalog contents | [Hot updates](hot-update.md) | Distinct checkpoint and Trie update processes |
| Measure quality and latency | [Benchmarks](../usage/benchmarks/README.md) | Matched inputs, cache conditions and explicit acceptance policy |

## Backend selection

CUDA supports native ModelRunner V1 and explicit V2 integration. Ascend uses its
prepared V1 runtime. A common API does not imply identical graph policies,
candidate limits or device operators. Use each feature's execution-support table
and the model's documented recipe when selecting combinations.

The CPU reference is a selection oracle and host-test tool. It is not a
production Beam inference backend.
