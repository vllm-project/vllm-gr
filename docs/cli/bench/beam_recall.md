# Beam recall

`vllm bench beam-recall --gr` measures Recall on the OneRec dataset with a
constraint catalog. It requires CUDA.

## Usage

```bash
vllm bench beam-recall --gr [options]
```

Choose exactly one task selector. Supported tasks are `video`, `ad`, `product`,
`label_cond` and `interactive`. For multiple tasks, pass their names as
separate arguments, for example `--task-types video ad`.

Video, ad, label_cond and interactive share the SID-to-PID catalog. Run
product separately with its SID-to-IID catalog. Each invocation takes one
catalog and builds one constraint artifact for that group.

## Arguments

| Option | Purpose |
| --- | --- |
| `--model`, `--model-revision` | Local model snapshot and optional immutable revision for provenance. |
| `--dataset-path`, `--catalog-path` | Required OneRec data directory and matching constraint triples JSON. |
| `--constraint-artifact` | Optional destination for the artifact compiled for the group. |
| `--num-prompts` | Maximum samples per task. Omit to use the full test split. |
| `--batch-size`, `--n`, `--temperature` | This runner requires batch size 4. Beam width defaults to 128 and temperature to 0.6. |
| `--result-dir` | Required output directory. With `--task-types`, results are grouped by task. |
| `--kv-pool-ping-pong`, `--no-kv-pool-ping-pong` | Toggle the optional second CUDA KV bank. Off by default. |
| `--validate-only` | Load and tokenize the data before allocating an engine. |

Use `vllm bench beam-recall --gr --help` for memory and context options.

## Example

```bash
vllm bench beam-recall --gr \
  --model /models/OneRec --dataset-path /data/onerec \
  --catalog-path /data/triples.json --task-type video \
  --result-dir /results/recall --num-prompts 100 --batch-size 4 --n 128
```

For a local KV bank comparison, hold the model, source data, selected sample
IDs, constraint artifact, tokenizer, batch size, graph profile and
temperature fixed. The selected bank policy is recorded in task metrics and
diagnostics. Local comparisons do not replace acceptance on the pinned CI
model revision.

The Recall gate and its thresholds are described in the
[GPU five-task Recall gate](../../contributing/ci/benchmark_acceptance.md#gpu-five-task-recall).
