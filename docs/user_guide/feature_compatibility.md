# Feature compatibility

To run Beam search, use the [offline quickstart](../getting_started/quickstart.md)
or send an [HTTP Beam request](../serving/completions.md#request). Use the
matrix to choose settings for your device.

The [ModelRunner selection reference](../configuration/README.md#modelrunner-selection)
describes how to configure the runner for a Python integration or server.
CPU packages support packaging and host checks only.

## Support matrix

| Feature | CUDA | Ascend NPU | Details |
| --- | --- | --- | --- |
| vLLM ModelRunner for Beam | V2 when compatible, with automatic V1 fallback | V1 required and selected by default | [ModelRunner selection](../configuration/README.md#modelrunner-selection) |
| Catalog-constrained Beam (constraint table) | Supported | Supported | [Configure a constraint table](../configuration/README.md#constraint-table) |
| Full-vocabulary Beam without a catalog | Supported | Supported within the NPU execution limits | [Ascend execution limits](../configuration/README.md#ascend-npu-execution-limits) |
| Dynamic Beam widths | Supported | Supported with or without a table | [Beam settings](../configuration/README.md#beam-settings), [Ascend execution limits](../configuration/README.md#ascend-npu-execution-limits) |
| Independent per-parent candidate widths | Supported | Supported with a constraint table | [Beam settings](../configuration/README.md#beam-settings) |
| Beam Decode graphs | Supported | FULL replay for eligible captures | [CUDA ModelRunner V2](../design/module/model_runner_v2.md), [Ascend capture and fallback](../design/feature/npu_decode_execution.md#decode-graph-capture) |
| Prefill graphs | Supported; explicit capture configuration | Opt-in BSND Prefill graphs | [Prefill capture](../design/feature/cuda_prefill_graph.md), [BSND Prefill](../design/feature/npu_prefill_bsnd.md) |
| Asynchronous scheduling | Supported | Supported | [Configuration](../configuration/README.md#asynchronous-scheduling) |
| Multi-Trie selection per request | Supported | Supported | [Constraint tables](../configuration/constraint_tables.md) |
| gRPC transport | Experimental, requires the `grpc` extra | Ascend-specific validation remains to be documented | [gRPC token inputs and CUDA example](../serving/grpc.md) |
| Checkpoint updates on running instances | Drain the instance; requires gRPC and an admin listener | Drain the instance; gRPC deployment validation remains undocumented | [Serving prerequisites](weight_update.md#serving-prerequisites) |
| Online catalog replacement | Asynchronous Beam with gRPC and an admin listener | Compile and load the new table at startup | [Serving prerequisites](weight_update.md#serving-prerequisites) |
| NPU decode execution modes (`xllm`, `cascade`) | Not applicable | Supported | [Decode execution](../design/feature/npu_decode_execution.md) |
| SID stage projection for catalog-free generation | Not applicable | Supported | [SID token families](../configuration/README.md#sid-token-families-without-a-catalog) |
| Caller-owned XAttention output binding | Not applicable | Optional, separate build | [XAttention output](../design/feature/npu_decode_execution.md#caller-owned-xattention-output) |

## How to read this matrix

- "Supported" describes a documented execution path. The linked guides record
  the validation scope, including model and graph configurations where applicable.
- A feature can be supported on a device and still be excluded for a specific
  model geometry or execution mode. Check the linked page before relying on a
  combination that is not listed here.
- Startup and request validation reject unsupported configurations. A valid
  request can still fall back to native dispatch when it misses a graph capture.

<a id="related"></a>

## See also

- [Platform backend contract](../design/module/platform_backend.md) for how a device
  provider is registered and activated.
- [Supported models](README.md#models) for the model list.
