# OneRec accuracy

[Benchmark overview](README.md) | [Troubleshooting](troubleshooting.md)

Use [`one_rec_acc_test.py`](../../benchmarks/open_one_rec/one_rec_acc_test.py)
to evaluate recommendation quality offline or over HTTP.

## Prerequisites

- Install the [CUDA](../quickstart.md) or [Ascend NPU](../quickstart-ascend.md)
  runtime and run commands from the repository root.
- Download the [OneRec dataset](dataset.md#get-dataset).

## Running the Benchmark

### Offline accuracy

Prepare the matching model snapshot and a canonical V1 configuration with a
compiled Constraint Table. Follow the [artifact and engine setup](../../examples/offline_inference/beam_search/README.md#constraint-table-artifact).
Use the same model and tokenizer snapshot for evaluation and artifact
compilation. Set `--n` to match `beam.max_width` in the config. The offline
V1 evaluator accepts one to three generated SID tokens. The video task below
uses three.

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
python -m benchmarks.open_one_rec.one_rec_acc_test \
  --model "$ONEREC_MODEL" \
  --dataset-path ./data \
  --task-type video \
  --vllm-gr-config ./vllm_gr_config.yaml \
  --n 128 \
  --num-prompts 1000 \
  --result-dir results/acc_test_results
```

### Online HTTP benchmark

Start the server with the [canonical online Beam configuration](../../examples/online_serving/beam_search/README.md#start-the-server).
The accuracy client uses `--model` both as the API model name and to load the tokenizer,
so use the same accessible model ID or snapshot path on both sides. For this
example, pass `--served-model-name "$ONEREC_MODEL"` to the server.

Run the client in another terminal:

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
python -m benchmarks.open_one_rec.one_rec_acc_test \
  --endpoint /v1/chat/completions \
  --model "$ONEREC_MODEL" \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-type video \
  --max-concurrency 1 \
  --n 128 \
  --num-prompts 1000 \
  --result-dir results/acc_test_results
```

The example uses one in-flight HTTP request at a time. For an authenticated
deployment, pass the configured key with `--api-key`.

## Usage

### Command Line Arguments

#### Core Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--model` | str | OpenOneRec/OneRec-1.7B | Model ID or path to load |

#### Benchmark related parameters

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset-name` | str | onerec | Must be `onerec` for OneRec benchmark |
| `--dataset-path` | str | ./data | Path to benchmark data directory |
| `--num-prompts` | int | all | Number of prompts to sample |
| `--task-type` | str | - | Required; see the [task list](dataset.md#task-types) |
| `--result-dir` | str | - | Directory for metrics JSON and evaluator artifacts |
| `--debug` | - | --- | Enable debug mode for evaluators |
| `--vllm-gr-config` | path | - | Required offline. Path to the canonical V1 config |
| `--batch-size` | int | 1 | Offline V1 batch size (1, 2 or 4) |
| `--n` | int | dataset-specific | Override the dataset default beam width (number of beams) for beam search |
| `--temperature` | float | dataset-specific | Override the dataset default temperature |

#### Arguments for Online benchmark

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--endpoint` | str | - | HTTP endpoint path (e.g., `/v1/chat/completions`) |
| `--host` | str | localhost | Server hostname |
| `--port` | int | 8000 | Server port |
| `--max-concurrency` | int | 16 | Maximum number of in-flight requests |
| `--api-key` | str | `EMPTY` | Key for an authenticated HTTP server |

## Evaluation and results

The runner loads a task's reference answers, generates candidates, and invokes
the evaluator registered for that task. Recommendation tasks such as `video`
report `pass@k` and `recall@k` using SIDs and, when configured, mapped PIDs.
Other tasks use their own metrics; see the
[task registry](../../benchmarks/open_one_rec/open_one_rec_loader.py).

The runner prints the metrics as JSON and saves a copy when `--result-dir`
is set. This excerpt illustrates the output for a recommendation task:

```text

...
...
--- Evaluation Results ---
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
