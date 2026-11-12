# Throughput Benchmark

## Overview

This guide explains how to run the throughput benchmark using the [`one_rec_main.py`](../../benchmarks/open_one_rec/one_rec_main.py) script.

## Prerequisites

- Ensure that **vllm-gr** is installed. See the [Quickstart](../quickstart.md) guide.

- Ensure that you have downloaded the [OneRec benchmark dataset](README.md#open-one-rec-benchmark-dataset).

---

## Running the Benchmark

### 1. Offline Throughput Mode (Simple, Local)

```bash
# Single command — everything runs in one process
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --model OpenOneRec/OneRec-1.7B \
  --max-logprobs 1024 \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-types video \
  --num-prompts 100 \
  --output-json results/throughput.json  \
  --use-beam-search \
  --n 128
```

### 2. Online, API Server Mode (Production-like)

Run the server and client separately for a production‑style setup.

#### Terminal 1 — Start the vLLM API server

```bash

  vllm-gr serve OpenOneRec/OneRec-1.7B \
  --max-logprobs 1024 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 16384 \
  --scheduling-policy priority
```

#### Terminal 2 — Run the throughput benchmark against the server

```bash

python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat  \
  --model OpenOneRec/OneRec-1.7B  \
  --dataset-name  onerec \
  --dataset-path ./data  \
  --task-types video  \
  --num-prompts 100 \
  --use-beam-search \
  --n 128
```

## Usage

### Command Line Arguments

#### Server Core Arguments

| Argument         | Type | Default | Description                                                     |
|------------------|------|---------|-----------------------------------------------------------------|
| `--model`        | str  | -       | Model ID or path to load                                        |
| `--max-logprobs` | int  | -       | Max log Probabilities output by model -- to support beam search |

#### Benchmark related parameters

| Argument             | Type | Default | Description                                             |
|----------------------|------|---------|---------------------------------------------------------|
| `--dataset-name`     | str  | -       | Must be `onerec` for OneRec benchmark                   |
| `--dataset-path`     | str  | -       | Path to benchmark data directory                        |
| `--num-prompts`      | int  | -       | Number of prompts to sample                             |
| `--task-types`       | str  | -       | Comma-separated list of task types to run               |
| `--custom-input-len` | int  | -       | Prompt length; prompts are trimmed or repeated to match |
| `--output-json`      | str  | -       | Path to save benchmark results as JSON                  |

### OpenOneRec Length Caveats

- For OpenOneRec, effective prompt length comes from each task input after task formatting, not from a strict benchmark-spec template length.
- `--custom-input-len` normalizes length by trimming or repeating prompt tokens per sample.
- This can distort semantic content; therefore user-defined prefix length is intended for throughput/latency experiments and is not recommended for accuracy measurement.

#### Beam Search Parameters (Optional)

| Argument            | Type | Default | Description                          |
|---------------------|------|---------|--------------------------------------|
| `--use-beam-search` | flag | `False` | Enable beam search during generation |
| `--n`               | int  | -       | Number of beams for beam search      |

#### Arguments for Online benchmark

| Argument            | Type  | Default       | Description                                                                                      |
|---------------------|-------|---------------|--------------------------------------------------------------------------------------------------|
| `--endpoint`        | str   | -             | API endpoint URL for online benchmark                                                            |
| `--backend`         | str   | 'openai-chat' | Must be `openai-chat` for OneRec online benchmark                                                |
| `--max-concurrency` | int   | `None`        | Maximum number of concurrent requests; if not set, requests are sent as fast as possible         |
| `--request-rate`    | float | `inf`         | Number of requests per second to send; `inf` means all requests are sent as soon as possible     |

## CPU SIMD and Latency

On some CPUs, including older Intel Xeon generations, NumPy AVX-512 kernels
can trigger frequency changes that also slow subsequent scalar CPU work.
Small token-array reductions may therefore increase end-to-end Beam latency
despite being fast themselves. Local tests on a Xeon Gold 6133 with NumPy
2.2.6 showed lower offline latency when NumPy AVX-512 dispatch was disabled;
CPU frequency changes were not measured directly. This is CPU- and
workload-dependent, so keep the default unless an A/B test shows a benefit.
See Intel's [power level and frequency transition explanation][intel-avx512].

For diagnosis with NumPy 2.2.6, prefix the same benchmark command with the
setting below. Use your actual dataset directory and constraint JSON path.

```bash
NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL \
  python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --model OpenOneRec/OneRec-1.7B --dataset-name onerec --dataset-path ./data \
  --task-types video --num-prompts 100 --use-beam-search --n 128 \
  --constraint-triples /path/to/video_constraint_triples.json \
  --output-json results/throughput-no-avx512.json
```

Set the variable before NumPy is imported; restart the inference process for
each setting. For online tests, apply it to the server running inference.
This limits NumPy dispatch only, not other CPU libraries or GPU execution.
Feature names vary by NumPy version; check startup warnings and the
[NumPy runtime dispatch documentation][numpy-dispatch]. Compare default and
restricted runs using the same inputs, model, warmup, cache protocol, and
otherwise idle hardware, and verify output agreement. Record the CPU model
and NumPy version; a result on one machine does not establish a general tuning
recommendation.

[intel-avx512]: https://builders.intel.com/docs/networkbuilders/intel-avx-512-instruction-set-for-packet-processing-technology-guide-1645717553.pdf#page=13
[numpy-dispatch]: https://numpy.org/doc/2.2/reference/simd/build-options.html#runtime-dispatch

## Expected Output

The output will include throughput metrics that can be used to evaluate the model's performance under load conditions.

```bash

============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Benchmark duration (s):                  49.75
Total input tokens:                      241086
Total generated tokens:                  64000
Request throughput (req/s):              2.01
Output token throughput (tok/s):         1286.55
Peak output token throughput (tok/s):    4096.00
Peak concurrent requests:                100.00
Total token throughput (tok/s):          6132.96
---------------Time to First Token----------------
Mean TTFT (ms):                          43927.85
Median TTFT (ms):                        44508.24
P99 TTFT (ms):                           49620.72
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.02
Median TPOT (ms):                        0.01
P99 TPOT (ms):                           0.22
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.05
Median ITL (ms):                         0.01
P99 ITL (ms):                            0.03
==================================================
```
