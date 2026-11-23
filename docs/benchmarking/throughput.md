# OneRec throughput benchmark

Measure offline Beam throughput or HTTP serving latency with the
[OneRec benchmark](../../benchmarks/open_one_rec/one_rec_main.py). Prepare the
runtime and data with the [common benchmark setup](common.md).

## Offline measurements

### CUDA dataset throughput

The OneRec offline Beam runner builds a constraint table from the
[video catalog triples](../../examples/online_serving/catalog/README.md#extract-constraint-triples-from-the-catalog)
and selects the CUDA constraint-table backend. Use the triples from the
[common setup](common.md#prerequisites).

```bash
export ONEREC_TRIPLES="$ONEREC_WORKDIR/constraint_triples.json"
python -m benchmarks.open_one_rec.one_rec_main bench throughput \
  --model "$ONEREC_MODEL" \
  --dataset-name onerec \
  --dataset-path ./data/benchmark_data \
  --task-types video \
  --num-prompts 100 \
  --output-json results/throughput.json \
  --constraint-triples "$ONEREC_TRIPLES" \
  --output-len 5 --beam-search \
  --n 128
```

The CUDA JSON report includes `cache_miss_e2e_ms` and `requests_per_second`.
The runner times one request at a time, including text encoding and Beam output
decoding. It resets the prefix cache before each timed request.

### Ascend batch latency

Use the [Ascend offline Beam latency example](../../examples/offline_inference/beam_search/README.md#measure-latency-without-profiling).
It runs full-vocabulary Beam search on a fixed prompt at batch sizes 1, 2, and 4.
After warmup, it records batch latency (including P50 and P95), requests/s, and
returned beams/s. For a revision comparison, keep the same workload and save
a separate JSON file for each revision and batch size.

<a id="2-online-api-server-mode-production-like"></a>

## Online serving

Start the server as described in [common benchmark setup](common.md#online-server-mode).
The HTTP client works with either a CUDA or an Ascend server. The example
uses the model snapshot path as the served name and loads the tokenizer from
that same path.

### Online metrics

| Metric | Timing boundary |
| --- | --- |
| Streaming TTFT | Ends at the first non-empty reasoning or answer fragment. |
| ITL | Intervals between successive text fragments, including whitespace fragments. |
| Request latency | Ends when the response stream finishes. |

A stream with no text output fails the request.

### Run the HTTP client

From a second terminal, run the client.

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --endpoint /v1/chat/completions \
  --backend openai-chat \
  --model "$ONEREC_MODEL" --tokenizer "$ONEREC_MODEL" \
  --dataset-name onerec \
  --dataset-path ./data/benchmark_data \
  --task-types video \
  --num-prompts 100 \
  --use-beam-search \
  --n 128
```

## Command-line arguments

### Server options

| Argument         | Type | Default | Description                                                     |
|------------------|------|---------|-----------------------------------------------------------------|
| `--model`        | str  | -       | Model ID or path to load                                        |
| `--max-logprobs` | int  | -       | Max log Probabilities output by model -- to support beam search |

### Benchmark options

| Argument             | Type | Default | Description                                             |
|----------------------|------|---------|---------------------------------------------------------|
| `--dataset-name`     | str  | -       | Must be `onerec` for OneRec benchmark                   |
| `--dataset-path`     | str  | -       | Path to benchmark data directory                        |
| `--num-prompts`      | int  | -       | Number of prompts to sample                             |
| `--task-types`       | str  | -       | Comma-separated list of task types to run               |
| `--custom-input-len` | int  | -       | Prompt length; prompts are trimmed or repeated to match |
| `--output-json`      | str  | -       | Path to save benchmark results as JSON                  |

### Prompt length

- For OpenOneRec, effective prompt length comes from each task input after task formatting, not from a strict benchmark-spec template length.
- `--custom-input-len` trims or repeats the locally rendered prompt per sample.
  The online chat workload sends that adjusted text as user content. Chat
  framing adds tokens, so the server's prompt length can differ from the local
  target. Online totals use the server's `usage.prompt_tokens` when available.
- This can distort semantic content; therefore user-defined prefix length is intended for throughput/latency experiments and is not recommended for accuracy measurement.

### Beam options

| Argument               | Type | Default | Description                          |
| ---------------------- | ---- | ------- | ------------------------------------ |
| `--beam-search`        | flag | `False` | Run the offline CUDA Beam benchmark. |
| `--constraint-triples` | path | -       | Required for offline CUDA Beam.      |
| `--use-beam-search`    | flag | `False` | Enable beam search during generation |
| `--n`                  | int  | -       | Number of beams for beam search      |

### Online options

| Argument            | Type  | Default       | Description                                                                                      |
|---------------------|-------|---------------|--------------------------------------------------------------------------------------------------|
| `--endpoint`        | str   | -             | API endpoint URL for online benchmark                                                            |
| `--backend`         | str   | 'openai-chat' | Must be `openai-chat` for OneRec online benchmark                                                |
| `--max-concurrency` | int   | `None`        | Maximum number of concurrent requests; if not set, requests are sent as fast as possible         |
| `--request-rate`    | float | `inf`         | Number of requests per second to send; `inf` means all requests are sent as soon as possible     |

## CPU SIMD and latency

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
  --model "$ONEREC_MODEL" --dataset-name onerec --dataset-path ./data/benchmark_data \
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

## Example output

Illustrative output from the HTTP benchmark. The values show the report format.

```text
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

## Historical HTTP and gRPC comparison

> Measured on NVIDIA H20 / date not recorded / ModelRunner V1 eager, FP16
> (runtime versions not recorded). Not a performance guarantee.

The comparison used revision `2b84265` for HTTP and the same experimental
gRPC build for both gRPC inputs. The paths shared random Qwen3 MoE weights (4.149B total / 0.311B
active), batch size 4, 4096 input tokens per prompt and Beam width 256. Each
request ran one Prefill and three Decode stages. Prefix Cache, Chunked Prefill
and constraints were disabled. Each path used 10 warmups and 200 measured
requests, alternating GPU probes. All 600 outputs matched.

| Per batch (ms) | Baseline HTTP/JSON/list | gRPC binary | gRPC repeated |
| --- | ---: | ---: | ---: |
| Python frontend, input + output | 16.755 | 1.897 | 2.117 |
| Model forward, excluding logits | 186.342 | 187.474 | 187.372 |
| Beam search, including logits | 11.475 | 11.491 | 11.508 |
| Client RTT minus Python server time | 1.339 | 0.719 | 0.646 |
| Other server/framework time | 5.465 | 8.345 | 4.836 |
| Client RTT | 221.375 | 209.925 | 206.479 |
| RTT median, GPU probes off | 220.944 | 206.324 | 206.476 |
| RTT P95, GPU probes off | 223.107 | 206.931 | 207.105 |

The first six rows are means from the 100 GPU-probed requests per path; the
first five sum to RTT. The last two rows use 100 GPU-probe-off controls;
Python server boundary hooks remain enabled. Client request construction,
encoding and reply parsing are outside RTT. Model and beam times sum four
batch forwards/stages, without multiplying again by batch size.

Other framework time is server time minus frontend, model and beam. The
client/server difference includes transport/runtime scheduling, not just wire
protocol work. HTTP returns text while gRPC returns token arrays, so frontend
savings include decoding and response formatting. All samples are retained;
isolated gRPC wall-time spikes affect means, so median/P95 are also reported.
