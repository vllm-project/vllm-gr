# OneRec benchmark overview

The OneRec benchmarks measure performance and recommendation quality using
[OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) tasks. Install the
matching runtime with the [CUDA](../quickstart.md) or
[Ascend NPU](../quickstart-ascend.md) quickstart, then run commands from the
repository root.

| Guide | Use it to |
| --- | --- |
| [Dataset preparation](dataset.md) | Download OpenOneRec-RecIF, select a task, and prepare a catalog. |
| [Throughput and latency](throughput.md) | Measure offline CUDA cache-miss latency or HTTP serving performance. |
| [Accuracy](accuracy.md) | Evaluate recommendation quality through the offline runner or HTTP server. |
| [Troubleshooting](troubleshooting.md) | Diagnose data, tokenizer, memory, and Beam configuration errors. |

## CI gates and other benchmarks

- [Video-100 accuracy gate](parent_score_gate.md) and
  [Beam V1 five-task Recall gate](beam_v1_recall_gate.md): acceptance criteria
  for the CI checks. See the [CI benchmark policy](../ci/benchmark-policy-approvals.md)
  for approved inputs and thresholds.
- [OneRec offline cache-miss benchmark](../../benchmarks/open_one_rec/V1_BENCHMARK.md):
  timing protocol and historical CUDA measurements.
- [Standalone Qwen Beam V1 benchmark](../../benchmarks/qwen/README.md):
  offline performance and timeline profiling.

## Open One Rec Benchmark Dataset

### Get Dataset

The [dataset download instructions](dataset.md#get-dataset) and
[Parquet schema](dataset.md#parquet-schema) are in the dataset guide.

[Back to the documentation index](../index.md)
