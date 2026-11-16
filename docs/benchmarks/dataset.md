# OneRec dataset preparation

[Benchmark overview](README.md)

## Get Dataset

Request access to
[OpenOneRec-RecIF](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF)
on Hugging Face and configure Git credentials for your approved account.
From the repository root, use Git LFS to download the Parquet files:

```bash
git lfs install
git clone https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF ./data
```

Pass the directory containing the task subdirectories as `--dataset-path`.
For example, the default test file for `video` is
`./data/video/video_test.parquet`.

## Parquet schema

The [loader](../../benchmarks/open_one_rec/open_one_rec_loader.py) expects two
columns:

| Column | Accepted types | Contents |
| --- | --- | --- |
| `messages` | JSON string or list | Chat messages with roles and content. |
| `metadata` | JSON string or mapping | Task metadata, including a nonempty `answer` used as ground truth. |

For example, a row can store both columns as JSON strings:

```json
{
  "messages": "[{\"role\":\"user\",\"content\":\"Recommend a video.\"}]",
  "metadata": "{\"answer\":\"Example answer\"}"
}
```

This illustrates the storage format. The expected answer format depends on
the task; use the released data for accuracy measurements.

## Task types

| Task | What it measures |
| --- | --- |
| `ad` | Next ad engagement |
| `product` | Next clicked product |
| `label_cond` | Next video under a specified behavior |
| `video` | Next video recommendation |
| `interactive` | Interactive engagement |
| `label_pred` | Binary engagement prediction |
| `item_understand` | Caption generation from a video SID |
| `rec_reason` | Recommendation explanation |

See the loader's task registry for the generation and evaluation settings for
each task. The Beam commands in the [throughput](throughput.md) and
[accuracy](accuracy.md) guides use `video`, which generates three SID components.
Tasks that require a longer generation budget use a different execution path.

## Prepare the constraint catalog

For catalog-constrained Beam Search, follow the
[catalog preparation guide](../../benchmarks/open_one_rec/tools/README.md) to
prepare the task catalog. The
[Constraint Table instructions](../../examples/offline_inference/beam_search/README.md#constraint-table-artifact)
show how to generate triples and compile them with the same model and tokenizer
snapshot used for inference. The offline CUDA throughput benchmark takes the
triples JSON and builds its own artifact; the offline accuracy runner and the
HTTP server take a configuration that points to the compiled artifact.
