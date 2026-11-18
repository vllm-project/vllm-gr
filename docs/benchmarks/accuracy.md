# Accuracy Benchmark

## Overview

This guide explains how to run the accuracy benchmark using the [`one_rec_acc_test.py`](../../benchmarks/open_one_rec/one_rec_acc_test.py) script.

## Prerequisites

- Ensure that **vllm-gr** is installed. See the [Quickstart](../quickstart.md) guide.

- Ensure that you have downloaded the [OneRec benchmark dataset](README.md#open-one-rec-benchmark-dataset).

---

## Running the Benchmark

### 1. Offline Accuracy Mode (Simple, Local)

Prepare the matching model snapshot and a canonical V1 configuration with a
compiled Constraint Table. Follow the [artifact and engine setup](../../examples/offline_inference/beam_search/README.md#constraint-table-artifact).
Use the same model and tokenizer snapshot for evaluation and artifact
compilation. Set `--n` to match `beam.max_width` in the config. An enabled
Constraint Table fixes generation to exactly three generated SID tokens,
including the token sampled during Prefill; with both OneRec boundary tokens
configured that is public `max_tokens=5`. The video task below uses three. The
unconstrained path accepts one to three, subject to the decode-step capacity
limit.

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

### 2. Online, API Server Mode (Production-like)

Start the server with the [canonical online Beam configuration](../../examples/online_serving/beam_search/README.md#start-the-server).
The accuracy client uses `--model` both as the API model name and to load the tokenizer,
so use the same accessible model ID or snapshot path on both sides. For this
example, pass `--served-model-name "$ONEREC_MODEL"` to the server.

#### Terminal 2 — Run the accuracy benchmark against the server

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
python -m benchmarks.open_one_rec.one_rec_acc_test \
  --endpoint /v1/chat/completions \
  --model "$ONEREC_MODEL" \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-type video \
  --n 128 \
  --num-prompts 1000 \
  --result-dir results/acc_test_results
```

For an authenticated deployment, pass the configured key with `--api-key`.

## Usage

### Command Line Arguments

#### Core Arguments

| Argument  | Type | Default                | Description              |
|-----------|------|------------------------|--------------------------|
| `--model` | str  | OpenOneRec/OneRec-1.7B | Model ID or path to load |

#### Benchmark related parameters

| Argument           | Type  | Default          | Description                                                               |
| ------------------ | ----- | ---------------- | ------------------------------------------------------------------------- |
| `--dataset-name`   | str   | onerec           | Must be `onerec` for OneRec benchmark                                     |
| `--dataset-path`   | str   | ./data           | Path to benchmark data directory                                          |
| `--num-prompts`    | int   | all              | Number of prompts to sample                                               |
| `--task-type`      | str   | -                | Task to run (ad, product, label_cond, video, interactive, label_pred)     |
| `--result-dir`     | str   | -                | Directory to save debug evaluation files                                  |
| `--debug`          | -     | -                | Enable debug mode for evaluators                                          |
| `--vllm-gr-config` | path  | -                | Required offline. Path to the canonical V1 config                         |
| `--batch-size`     | int   | 1                | Offline V1 batch size (1, 2 or 4)                                         |
| `--n`              | int   | dataset-specific | Override the dataset default beam width (number of beams) for beam search |
| `--temperature`    | float | dataset-specific | Override the dataset default temperature                                  |

#### Arguments for Online benchmark

| Argument     | Type | Default       | Description                                                            |
|--------------|------|---------------|------------------------------------------------------------------------|
| `--endpoint` | str  | -             | API endpoint URL for the online benchmark (e.g., /v1/chat/completions) |
| `--api-key`  | str  | -             | API Key for OpenAI-compatible API                                      |

## Expected Output

The benchmark prints accuracy metrics summarizing model performance.
Example:

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
