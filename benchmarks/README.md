# Benchmarks

Use a [source checkout that matches the installed runtime](../docs/cli/bench.md).
Follow the [GR dispatch setup](../docs/cli/README.md#enable-gr-dispatch) to run
`vllm bench --gr --list` and invoke the registered scenarios.

| Directory | Workloads | Guides |
| --- | --- | --- |
| `beam_search/` | `beam-performance`, `beam-performance-ab`, `beam-recall` | [Benchmark guides](../docs/benchmarking/README.md#choose-a-benchmark), [Recall gate](../docs/contributing/ci/benchmark_acceptance.md#gpu-five-task-recall) |
| `open_one_rec/` | `open-one-rec` throughput, OneRec accuracy, dataset and catalog tools | [OneRec benchmarks](../docs/benchmarking/README.md), [Accuracy](../docs/benchmarking/accuracy.md), [Catalog tools](open_one_rec/tools/README.md) |
| `qwen/` | `qwen-v1` offline Beam performance and profiling | [Qwen benchmark](qwen/README.md) |

[`metrics.py`](metrics.py) contains shared percentile calculations for the Beam
latency gate and Qwen benchmark.
