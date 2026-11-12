# Throughput Benchmark

## Overview

This guide explains how to run the throughput benchmark using the [`one_rec_main.py`](https://github.com/JiusiServe/vllm-gr/blob/main/benchmarks/open_one_rec/one_rec_main.py) script.

## Prerequisites

- Ensure that **vllm-gr** is installed. See the [Quickstart](../../getting_started/installation/cuda.md) guide.

- Ensure that you have downloaded the [OneRec benchmark dataset](README.md#open-one-rec-benchmark-dataset).

---

## Running the Benchmark

### 1. Offline Throughput Mode (Simple, Local)

Prepare the model snapshot and [constraint triples](https://github.com/JiusiServe/vllm-gr/blob/main/examples/offline_inference/beam_search/README.md#constraint-table-artifact).
The offline Beam runner builds a CUDA artifact from the triples.

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
export ONEREC_TRIPLES=/absolute/path/to/video_constraint_triples.json
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --model "$ONEREC_MODEL" \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-types video \
  --num-prompts 100 \
  --output-json results/throughput.json \
  --constraint-triples "$ONEREC_TRIPLES" \
  --output-len 5 --beam-search \
  --n 128
```

### 2. Online, API Server Mode (Production-like)

Start the server using the [online V1 guide](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/beam_search/README.md#start-the-server).
The client uses a local tokenizer from the same model snapshot as the server.
Pass `--chat-template benchmarks/open_one_rec/qwen3_soft_switch.jinja2` to
the server to match the offline renderer. The client sends the original
chat turns and the task's thinking setting. A per-run setting in
`--extra-body` takes precedence over the task default.

Streaming TTFT starts at the first non-empty reasoning or answer fragment.
ITL measures the intervals between successive text fragments, including
whitespace fragments. Request latency runs through the end of the response
stream. A stream with no text output fails the request.

#### Terminal 2 — Run the throughput benchmark against the server

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --model onerec --tokenizer "$ONEREC_MODEL" \
  --dataset-name onerec \
  --dataset-path ./data \
  --task-types video \
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
- `--custom-input-len` trims or repeats the locally rendered prompt per sample.
  The online chat workload sends that adjusted text as user content. Chat
  framing adds tokens, so the server's prompt length can differ from the local
  target. Online totals use the server's `usage.prompt_tokens` when available.
- This can distort semantic content; therefore user-defined prefix length is intended for throughput/latency experiments and is not recommended for accuracy measurement.

#### Beam Search Parameters (Optional)

| Argument               | Type | Default | Description                          |
| ---------------------- | ---- | ------- | ------------------------------------ |
| `--beam-search`        | flag | `False` | Select the offline V1 Beam benchmark |
| `--constraint-triples` | path | -       | Required for offline Beam V1         |
| `--use-beam-search`    | flag | `False` | Enable beam search during generation |
| `--n`                  | int  | -       | Number of beams for beam search      |

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
despite being fast themselves. One historical comparison found lower offline latency with NumPy AVX-512
dispatch disabled. CPU frequency was not measured directly. This is CPU- and
workload-dependent, so keep the default unless an A/B test shows a benefit.
See Intel's [power level and frequency transition explanation][intel-avx512].

> Measured on Intel Xeon Gold 6133 / date not recorded / NumPy 2.2.6.
> Not a performance guarantee.

For diagnosis with NumPy 2.2.6, prefix the same benchmark command with the
setting below. Use your actual dataset directory and constraint JSON path.

```bash
NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL \
  python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --model "$ONEREC_MODEL" --dataset-name onerec --dataset-path ./data \
  --task-types video --num-prompts 100 --beam-search --n 128 \
  --output-len 5 --constraint-triples "$ONEREC_TRIPLES" \
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

The report includes successful/failed requests, benchmark duration, request and
token throughput, concurrency, and latency distributions such as TTFT, TPOT and
inter-token latency. Interpret each metric using the selected serving protocol
and workload. Save raw results with the effective configuration; sample output
from another run is not a performance target.
