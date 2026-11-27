# OneRec benchmarks

Use the [common benchmark setup](common.md) to prepare the runtime and OneRec
data. Run commands from the repository root in the checkout that matches your
installed vLLM-GR version.

<a id="scenario-command-contracts"></a>

## Choose a benchmark

| What do you want to measure? | CUDA GPU | Ascend NPU |
| --- | --- | --- |
| Recommendation accuracy | Run the [accuracy benchmark](accuracy.md) with a CUDA constraint table or against a CUDA server. | Use the same [accuracy benchmark](accuracy.md) with an NPU constraint table or against an Ascend server. |
| HTTP throughput and latency | Run the [HTTP client](throughput.md#online-serving) against a CUDA server. | Run the same client against an [Ascend HTTP server](common.md#online-server-mode). |
| Offline Beam performance | Measure [dataset throughput](throughput.md#cuda-dataset-throughput) with catalog-constrained generation. | Measure [batch latency and requests/s](throughput.md#ascend-batch-latency) with a fixed prompt and batch size. |
| Change in performance between two commits | Repeat the [Beam performance workload](../cli/bench/beam_performance.md) on the same device with identical inputs and settings. | Repeat the [Ascend latency example](throughput.md#ascend-batch-latency) on both commits with the same device, model, prompt, Beam settings, and warmup. |

!!! note "Comparing devices"

    The two offline examples measure different workloads. For a cross-device
    comparison, hold the model, inputs, catalog mode, Beam width, batch size and
    timing scope constant. The [cache-miss benchmark](../../benchmarks/open_one_rec/V1_BENCHMARK.md)
    records the CUDA timing scope and historical results.

After applying the [vLLM CLI dispatch patch](../cli/README.md#command-form-and-installation),
run `vllm bench --gr --list` and see the
[CLI benchmark reference](../cli/bench.md) for arguments and backend support.
For pull request acceptance criteria, see the [CI benchmark gates](../contributing/ci/benchmark_acceptance.md#benchmark-gates).

## Compare individual Beam Recall outputs

Run the [Beam Recall benchmark](../contributing/ci/benchmark_acceptance.md#reproduce-the-cuda-gate)
with `V1_RECALL_DIAGNOSTICS=1` for each run. Compare the
`diagnostic_samples.json` files for the same task from the repository root.
Only Python is needed for the comparison.

```bash
python -m benchmarks.beam_search.compare_recall_diagnostics \
  reference/video/diagnostic_samples.json candidate/video/diagnostic_samples.json \
  --top-k 10 --details > comparison.json
```

The comparison matches sample IDs and prompt token digests. It reports exact
order, equal sets with changed order, changed sets, shared SIDs and Top-1
agreement. Top-1 counts include samples with a candidate on both sides. Omit
`--top-k` to use every returned SID. Use `--details` to include per-sample
results and the token IDs added to or removed from each list.

Batch fields compare the ordered members of each submitted batch and the order
in which the batches were submitted. Older diagnostics without batch records
still support ranking comparisons and report the batch context as
`unrecorded`.

The `beam_search_params_status` field compares the submitted search parameters,
including temperature and beam width. If either file lacks these parameters,
the status is `unrecorded`. Parameter differences also appear in
`changed_metadata`.

Absolute score differences cover shared SIDs with scores on both sides. The
report lists changed runtime, model and catalog metadata. Interpret score
differences with the [score normalization rules](../design/module/constraint_tables.md)
for the compared runs. The command produces diagnostics for investigation and
can be used alongside the Recall acceptance gates.

<a id="open-one-rec-benchmark-dataset"></a>

## OneRec benchmark dataset

### Download the dataset

Request access on Hugging Face before cloning the dataset.

<https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF>

```bash
git lfs install
git clone https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF ./data
```

### Locate the task data

After the clone above, the video test split is at
`./data/benchmark_data/video/video_test.parquet`. Pass
`--dataset-path ./data/benchmark_data` to the accuracy and throughput clients.
If you use another location, pass the directory that directly contains
`video/`, `ad/`, and the other task directories.

For catalog-constrained Beam search, [prepare the catalog and constraint
table](common.md#prepare-the-catalog)
using the matching OneRec tokenizer and the dataset's `benchmark_data/sid2pid.json`.

### Parquet schema

Each Parquet file must contain these columns:

| Column | Type | Description |
| -------- | ------ | ------------- |
| `messages` | str/list | Chat messages in JSON format or list of dicts |
| `metadata` | str/dict | JSON metadata containing answer and context |

For example, a Parquet row can store both columns as JSON strings:

```json
{
  "messages": "[{\"role\":\"user\",\"content\":\"Recommend a video\"}]",
  "metadata": "{\"answer\":\"Video ID: ABC123\"}"
}
```

The actual answer format depends on the task. Use the released dataset to
evaluate recommendation quality.

## Tasks

The task registry includes recommendation, understanding and classification
tasks. The Beam examples use the video task and its three generated SID
tokens. Tasks with longer generation budgets require a different execution
path. The [accuracy guide](accuracy.md) describes the additional Judge setup
for `rec_reason` and the label-probability requirement for `label_pred`.

| Task Type | Category | Description | Sample Size |
| ----------- | ---------- | ------------- | ------------ |
| `rec_reason` | Recommendation | Explain a recommendation | 470 |
| `item_understand` | Understanding | Generate a caption from a video SID | 500 |
| `ad` | Recommendation | Predict ad engagement | 27677 |
| `product` | Recommendation | Product recommendation reasoning | 27910 |
| `label_cond` | Recommendation | Conditional label prediction | 34891 |
| `video` | Recommendation | Video recommendation scoring | 38781 |
| `interactive` | Recommendation | Interactive engagement prediction | 1000 |
| `label_pred` | Classification | User engagement classification | 346190 |
