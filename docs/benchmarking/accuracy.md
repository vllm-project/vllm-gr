# OneRec accuracy benchmark

Run the [OneRec evaluator](../../benchmarks/open_one_rec/one_rec_acc_test.py)
with a local engine or a running chat server. Prepare the runtime and dataset
with the [common benchmark setup](common.md) before running either workload.

## Offline evaluation

Follow the [common setup](common.md#prerequisites) through
[writing the YAML](../../examples/online_serving/catalog/README.md#write-the-server-configuration).
Use the same tokenizer for the compiled table and evaluation. The generated
`gr.constraint_table.backend` must match the device (`cuda` or `npu`).
The command below selects ModelRunner V1 on both devices.

Set `--n` to the maximum value in `gr.beam_width`. For the video task, the
table constrains three generated SID tokens. Together with the two boundary
tokens, the request budget is five tokens. Without a table, this offline
accuracy runner accepts one to three generated tokens within engine capacity.

```bash
VLLM_USE_V2_MODEL_RUNNER=0 python -m benchmarks.open_one_rec.one_rec_acc_test \
  --model "$ONEREC_MODEL" \
  --dataset-path ./data/benchmark_data \
  --task-type video \
  --config "$ONEREC_WORKDIR/vllm_gr_config.yaml" \
  --n 128 \
  --num-prompts 1000 \
  --result-dir results/acc_test_results
```

<a id="2-online-api-server-mode-production-like"></a>

## Online serving

Start the server as described in [common benchmark setup](common.md#online-server-mode).
The accuracy client uses `--model` both as the API model name and to load the
tokenizer, so use the same accessible model ID or snapshot path on both sides.
Use the same client command for CUDA and Ascend servers.

From a second terminal, run the client.

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
python -m benchmarks.open_one_rec.one_rec_acc_test \
  --endpoint /v1/chat/completions \
  --model "$ONEREC_MODEL" \
  --dataset-name onerec \
  --dataset-path ./data/benchmark_data \
  --task-type video \
  --n 128 \
  --num-prompts 1000 \
  --result-dir results/acc_test_results
```

For an authenticated deployment, pass the configured key with `--api-key`.
Use `--request-timeout` to override the default 180-second HTTP connect and
read timeout. HTTP failures, malformed responses and unexpected choice counts
fail the run. Each choice needs non-blank answer or reasoning text. The client
prefers `reasoning` and falls back to `reasoning_content` for older servers. With
`--result-dir`, the client saves affected sample IDs and errors to
`<task>-failed.json`.

## Task-specific requirements

The `label_pred` task requires label-token probabilities. This entry point
currently exits before inference for that task. The evaluator reports an
undefined AUC as `null` with valid and invalid sample counts and a reason
when zero or one class has usable probabilities. Judge-enabled tasks also
fail when the Judge is unavailable or cannot produce valid scores. The
recommendation-reason Judge requires a score for every selected sample and
fails on empty predictions. WIP matching must account for every occurrence
of each reference and model point. Saved Judge results are for diagnostics
and are evaluated afresh on subsequent runs.

### Recommendation reasoning

For `rec_reason`, set `llm_judge_model` in the task's `evaluation_config` to
the model name exposed by the Judge service. `vllm_endpoint` and `vllm_api_key`
configure its OpenAI-compatible API. `vllm_async=True` uses the asynchronous
SDK inside scoring workers. Each sample score must be an integer from 1 to 5;
the reported mean can be fractional.

## Command-line arguments

### Model

| Argument  | Type | Default                | Description              |
|-----------|------|------------------------|--------------------------|
| `--model` | str  | OpenOneRec/OneRec-1.7B | Model ID or path to load |

### Benchmark options

| Argument           | Type  | Default          | Description                                                               |
| ------------------ | ----- | ---------------- | ------------------------------------------------------------------------- |
| `--dataset-name`   | str   | onerec           | Must be `onerec` for OneRec benchmark                                     |
| `--dataset-path`   | str   | ./data           | Task directory, e.g. `./data/benchmark_data`                              |
| `--num-prompts`    | int   | all              | Number of prompts to sample                                               |
| `--task-type`      | str   | -                | Task to run (ad, product, label_cond, video, interactive)                 |
| `--result-dir`     | str   | -                | Directory to save debug evaluation files                                  |
| `--debug`          | -     | -                | Enable debug mode for evaluators                                          |
| `--config`         | path  | -                | Offline engine YAML. Use `--gr` for JSON overrides.                       |
| `--batch-size`     | int   | 1                | Number of prompts per offline Beam batch (1, 2 or 4).                     |
| `--n`              | int   | dataset-specific | Override the dataset default beam width (number of beams) for beam search |
| `--temperature`    | float | dataset-specific | Override the dataset default temperature                                  |

### Online options

| Argument     | Type | Default       | Description                                                            |
|--------------|------|---------------|------------------------------------------------------------------------|
| `--endpoint` | str  | -             | API endpoint URL for the online benchmark (e.g., /v1/chat/completions) |
| `--api-key`  | str  | -             | API Key for OpenAI-compatible API                                      |

## Example output

The video evaluator reports recall and pass rates for the selected cutoffs.
An evaluation report has the following form.

```json
{
  "total_samples": 100,
  "pass@1": 0.03,
  "position1_pass@1": 0.0,
  "recall@1": 0.004523809523809524,
  "pass@32": 0.16,
  "position1_pass@32": 0.04,
  "recall@32": 0.024341269841269844,
  "pid_pass@1": 0.03,
  "pid_position1_pass@1": 0.0,
  "pid_recall@1": 0.004523809523809524,
  "pid_pass@32": 0.16,
  "pid_position1_pass@32": 0.04,
  "pid_recall@32": 0.024341269841269844,
  "select_k_strategy": "first_k",
  "evaluation_mode": "both",
  "sid_to_pid_strategy": "most_popular_after_downsampling"
}
```
