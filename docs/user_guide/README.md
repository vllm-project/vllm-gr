# User guide

For a first request, follow [Getting started](../getting_started/quickstart.md).
Once your environment is ready, choose a guide for your task.

## Models

[OneRec-1.7B](../models/onerec.md) generates ranked recommendation candidates
on CUDA and Ascend NPU. A catalog can restrict which items it recommends.

## Inference and serving

| I want to | Guide |
| --- | --- |
| Check device and runner support | [Feature compatibility](feature_compatibility.md) |
| Configure Beam width and candidate selection | [Configuration](../configuration/README.md#beam-settings) |
| Recommend items from my catalog | [Constraint tables](../configuration/constraint_tables.md) |
| Configure Ascend Beam limits or SID token families | [Ascend configuration](../configuration/README.md#ascend-npu-execution-limits) |
| Start a server and send a request | [Online serving](../serving/README.md) |
| Look up a command or option | [CLI reference](../cli/README.md) |

## Deployment and operations

| I want to | Guide |
| --- | --- |
| Update weights or switch catalogs on running instances | [Weight and catalog updates](weight_update.md) |
| Monitor a running server | [Serving metrics](../serving/README.md#read-prometheus-metrics) |
| Measure accuracy or throughput | [Benchmarking](../benchmarking/README.md) |

<a id="related"></a>

## See also

- [Design documents](../design/index.md) explain the implementation behind these tasks.
