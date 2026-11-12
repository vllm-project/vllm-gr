# Getting started

Start with a supported accelerator runtime, run one offline request, then choose
a serving interface. No catalog is needed for the first unconstrained run;
catalog-valid recommendations require a compiled constraint table.

## Choose your runtime

| Target | Prepare | First request | Execution choices |
| --- | --- | --- | --- |
| NVIDIA CUDA | [CUDA installation](installation/cuda.md) | [First inference](first-inference.md) | Native ModelRunner V1 or explicit V2; direct CUDA examples default to V2 |
| Ascend NPU | [Ascend installation](installation/ascend.md) | [First inference](first-inference.md), then the [HTTP example](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/beam_search/README.md) | V1 with the documented cascade or xLLM Decode path |

Both targets use vLLM 0.30.0. Prepare the Ascend vendor runtime before installing
GR; selecting an NPU package does not provision CANN or the vendor operators.
The [version policy](../configuration/dependency_version_policy.md) records the
complete compatibility requirements.

## Your first successful run

1. Follow the installation guide for your accelerator.
2. Run [First inference](first-inference.md) with one local OneRec snapshot.
3. For catalog-valid SIDs, follow the [OneRec catalog quickstart](quickstart.md).
4. Reuse the artifact in the [HTTP](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/beam_search/README.md)
   or [gRPC](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/grpc/README.md) example.

## Before changing settings

Use [Configuration](../configuration/README.md) for YAML inheritance and Python
or CLI overrides. Beam widths reserve engine capacity at startup; changing a
client request does not resize the Worker. Use the [feature overview](../features/README.md)
to check graph, Attention and backend restrictions before tuning.
