# vllm bench

Use `vllm bench --gr --list` to list the GR workloads. Run benchmarks from a
vLLM-GR source checkout matching your installed version, or from one of its
subdirectories. The workloads use scripts from that checkout, including when
you installed vLLM-GR from a wheel.

## Workloads

| Command | What it measures or produces | Backend | Reference |
| --- | --- | --- | --- |
| `vllm bench open-one-rec throughput --gr` | Offline OneRec throughput | CUDA for the catalog-constrained Beam workload, selected vLLM backend for other modes | [OneRec throughput](bench/open_one_rec.md#offline-throughput) |
| `vllm bench serve --gr` (alias for `vllm bench open-one-rec serve --gr`) | Throughput and latency of a running HTTP server | Server backend | [OneRec serving](bench/open_one_rec.md#online-serving) |
| `vllm bench beam-performance MODE --gr` | Prepare inputs, measure offline Beam latency, or replay requests online | CUDA | [Beam performance](bench/beam_performance.md) |
| `vllm bench beam-recall --gr` | OneRec Recall on one or more tasks | CUDA | [Beam Recall](bench/beam_recall.md) |
| `vllm bench qwen-v1 --gr` | Synthetic Beam workload on a Qwen model | CUDA | [Qwen Beam benchmark](bench/qwen_v1.md) |
| `vllm bench beam-performance-ab --gr` | Compare recorded Beam latency runs | Host | [Comparison tool](../../benchmarks/beam_search/compare_latency_ab.py) |
| `vllm bench ab-offline --gr` | Run an offline engine performance comparison workload | CUDA | [Offline A/B guide](../../tools/AB_Tests/README.md) |

## Command syntax

Put `--gr` before the optional `--` separator. Arguments after the separator
go to the workload in their original order.

| Workload | Example |
| --- | --- |
| Qwen Beam benchmark | `vllm bench qwen-v1 --gr -- --help` prints the runner's options. |
| Beam performance | `vllm bench beam-performance offline --gr --help` puts the mode before the options. |

OneRec also accepts the earlier `open-one-rec bench throughput` spelling.

## Offline A/B

Run `vllm bench ab-offline --gr` with a local CUDA environment and model.
Pass the OneRec dataset directory with `--data-dir`. Pass the constraint
triples with `--catalog` when the checkout has no bundled catalog.
The [catalog guide](../../examples/online_serving/catalog/README.md#build-a-catalog-from-the-onerec-dataset)
shows how to prepare the triples. Follow the [local A/B guide](../../tools/AB_Tests/README.md)
to measure two revisions in separate environments and compare the results.

For dataset preparation and measurement procedures, see [Benchmarking](../benchmarking/README.md).
