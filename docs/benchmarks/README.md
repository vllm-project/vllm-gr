# OneRec benchmarks

Install the runtime using the [CUDA](../quickstart.md) or
[Ascend](../quickstart-ascend.md) guide and prepare the dataset below.

See the [CLI reference](../cli.md) for workload options and the vLLM 0.30.0
patch required by `vllm ... --gr`.

Run `vllm bench --gr --list` to list the available workloads. For example,
`vllm bench qwen-v1 --gr --help` shows the Qwen scenario options. Workloads
and their data live in the source tree. With a wheel installation, run the
command from the repository checkout or one of its subdirectories.

- [Throughput and latency](throughput.md) covers offline and HTTP benchmarks.
- [Accuracy](accuracy.md) covers offline and HTTP evaluation.
- [OneRec cache-miss benchmark](../../benchmarks/open_one_rec/V1_BENCHMARK.md)
  describes text-to-text timing and records historical CUDA results.
- [Catalog preparation](../../benchmarks/open_one_rec/tools/README.md) covers
  converting the dataset catalog before building a Constraint Table.
- [Video-100](parent_score_gate.md) and [five-task Recall](beam_v1_recall_gate.md)
  describe the CI gates.

## Open One Rec Benchmark Dataset

### Get Dataset

Request access on Hugging Face before cloning the dataset.

<https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF>

```bash
git lfs install
git clone https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF ./data
```

### Data Structure

The loader expects one Parquet file per task, such as
`data/video/video_test.parquet`. Pass the directory containing the task
subdirectories as `--dataset-path`.

### Parquet Schema

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
path.

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

## Scenario command contracts

- `vllm bench serve --gr` is a short form of
  `vllm bench open-one-rec serve --gr`, with the same online benchmark options.
- `vllm bench open-one-rec throughput --gr --help` displays the upstream
  throughput options with the OneRec extensions loaded. For a measurement,
  supply the model and dataset required by that subcommand. Device support
  follows the selected vLLM backend and model.
- `qwen-v1`, `beam-performance`, `beam-recall`, and `ab-offline` require CUDA.
  The offline A/B workload fixes the constraint-table backend to `cuda`.
  Use `vllm bench <scenario> --gr --help` to see the model and artifact inputs.
- To compare existing measurement directories on the host, use the Git commit
  that produced each set of results. The
  `tests/system/run_beam_v1_performance_ab.sh` harness collects the CUDA ABBA runs.

  ```bash
  vllm bench beam-performance-ab --gr \
    --base results/base --base-sha "$(cat results/base/1/commit.txt)" \
    --pr results/pr --pr-sha "$(cat results/pr/1/commit.txt)" \
    --output-dir results/report
  ```
