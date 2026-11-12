# API reference

Use Python for application-owned offline inference, HTTP for completion-style
serving, or gRPC for token-based batches. All three enter the same Beam session
contract; response representation and engine lifetime differ by interface.

## Python entrypoint

`from vllm_gr import GRLLM` exposes the offline engine. Native engine settings
use their normal Python names; GR settings are supplied through `gr` or a
shared `config` YAML. Configure Beam capacity when constructing the engine,
then call `beam_search()` with `BeamSearchParams`.

The default public output is full output with lazily materialized text.
`output_mode="compact"` explicitly requests eager CPU token/score results.
Call `shutdown()` when the application is finished; request completion still
requires its own device retirement and output-consumer rules.

Use the [complete offline example](https://github.com/JiusiServe/vllm-gr/blob/main/examples/offline_inference/beam_search/README.md)
for a runnable constructor, request and shutdown sequence. Widths and generation
limits are in [Configuration](../configuration/README.md#beam-settings).

## Contract reference

| Interface | Definition | Guide |
| --- | --- | --- |
| Offline Python entrypoint | [GRLLM](https://github.com/JiusiServe/vllm-gr/blob/main/vllm_gr/entrypoints/gr.py) | [Offline example](https://github.com/JiusiServe/vllm-gr/blob/main/examples/offline_inference/beam_search/README.md) |
| Beam V1 requests and results | [Protocol](https://github.com/JiusiServe/vllm-gr/blob/main/vllm_gr/entrypoints/beam_search/protocol.py) | [Batch execution contract](../features/beam-search.md#batch-execution-contract) |
| OpenAI-compatible requests | [Protocol extensions](https://github.com/JiusiServe/vllm-gr/blob/main/vllm_gr/entrypoints/openai/protocol.py) | [Online example](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/beam_search/README.md) |
| gRPC service | [Protobuf definition](https://github.com/JiusiServe/vllm-gr/blob/main/vllm_gr/entrypoints/grpc/proto/vllm_engine.proto) | [Transport guide](../serving/README.md#grpc-and-numpy-transport) |

Shared configuration is described in [Configuration](../configuration/README.md).

## Batches, failures and cancellation

A logical batch fixes its membership before execution. Results preserve item
identity and order. A terminal result contains either a complete output or a
failure; it never combines failure with a partial successful batch. Cancellation
still drains issued work before the resources can be reused.

See the [Beam contract](../features/beam-search.md#batch-execution-contract)
for the exact request/result types and [gRPC transport](../serving/README.md#grpc-and-numpy-transport)
for binary token ownership and column formats.
