# vllm bench

Use `vllm bench --gr --list` to list the GR workloads. Run benchmarks from a
vLLM-GR source checkout matching your installed version, or from one of its
subdirectories. The workloads use scripts from that checkout, including when
you installed vLLM-GR from a wheel.

## Workloads

| Command | What it measures or produces | Backend | Reference |
| --- | --- | --- | --- |
| `vllm bench open-one-rec throughput --gr` | Offline OneRec throughput | CUDA for the catalog-constrained Beam workload, selected vLLM backend for other modes | [OneRec throughput](../benchmarking/throughput.md#cuda-dataset-throughput) |
| `vllm bench serve --gr` (alias for `vllm bench open-one-rec serve --gr`) | Throughput and latency of a running HTTP server | Server backend | [OneRec serving](../benchmarking/throughput.md#online-serving) |
| `vllm bench beam-performance MODE --gr` | Prepare inputs, measure offline Beam latency, or replay requests online | CUDA | [Beam performance](../benchmarking/throughput.md#controlled-beam-latency) |
| `vllm bench beam-recall --gr` | OneRec Recall on one or more tasks | CUDA | [Beam Recall](../benchmarking/accuracy.md#cuda-recall-runner) |
| `vllm bench qwen-v1 --gr` | Synthetic Beam performance and profiling | CUDA | [Synthetic Beam benchmark](bench.md#qwen-beam) |
| `vllm bench beam-performance-ab --gr` | Compare recorded Beam latency runs | Host | [Comparison tool](../../benchmarks/beam_search/compare_latency_ab.py) |
| `vllm bench ab-offline --gr` | Measure paired cold/warm OneRec requests | CUDA | [Offline A/B guide](../../tools/AB_Tests/README.md) |

## Command syntax

Put `--gr` before the optional `--` separator. Arguments after the separator
go to the workload in their original order.

| Workload | Example |
| --- | --- |
| Synthetic Beam benchmark | `vllm bench qwen-v1 --gr -- --help` prints the runner's options. |
| Beam performance | `vllm bench beam-performance offline --gr --help` puts the mode before the options. |

OneRec also accepts the earlier `open-one-rec bench throughput` spelling.

## Offline A/B

Run `vllm bench ab-offline --gr` with a local CUDA environment and model to
record paired cold/warm latency and prefix-cache observations for individual
OneRec requests. Both the dataset directory (`--data-dir`) and the constraint
triples file (`--catalog`) are required.

Follow the [local A/B guide](../../tools/AB_Tests/README.md) to rebuild the pinned
catalog, measure two revisions in separate checkouts and virtual environments,
and turn the raw JSON files into summaries and a comparison report.

For dataset preparation and measurement procedures, see [Benchmarking](../benchmarking/README.md).

<a id="qwen-beam"></a>

## Synthetic CUDA Beam

`vllm bench qwen-v1 --gr` dispatches the CUDA synthetic Beam runner.
Follow the [benchmark guide](../../benchmarks/qwen/README.md#direct-python-invocation)
for workload generation, token bounds, constraints, profiling, and output.
Use `vllm bench qwen-v1 --gr --help` for the runner's current arguments.
