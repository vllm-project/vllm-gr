# OneRec throughput benchmark

Measure offline Beam throughput or HTTP serving latency with the
[OneRec benchmark](../../benchmarks/open_one_rec/one_rec_main.py). Prepare the
runtime and data with the [common benchmark setup](README.md#setup).

## Offline measurements

### CUDA dataset throughput

The OneRec offline Beam runner builds a constraint table from the
[video catalog triples](../../examples/online_serving/catalog/README.md#extract-constraint-triples-from-the-catalog)
and selects the CUDA constraint-table backend. Use the triples from the
[common setup](README.md#prerequisites).

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

Start the server as described in [common benchmark setup](README.md#online-server-mode).
The HTTP client works with either a CUDA or an Ascend server. The example
uses the model snapshot path as the served name and loads the tokenizer from
that same path.

### Online metrics

| Metric          | Timing boundary                                                              |
| --------------- | ---------------------------------------------------------------------------- |
| Streaming TTFT  | Ends at the first non-empty completion text fragment.                        |
| ITL             | Intervals between successive text fragments, including whitespace fragments. |
| Request latency | Ends when the response stream finishes.                                      |

A stream with no text output fails the request.

Beam requests use a non-streaming Completions response: the native endpoint
rejects `use_beam_search: true` with `stream: true`. The client records TTFT
as the complete request latency and counts output tokens from the response's
usage across all returned candidates. Beam exposes no token arrival intervals;
ITL is empty and the benchmark's displayed TPOT is zero, not a measured Decode
latency. Compare Beam runs using request latency/TTFT and throughput. Ordinary
sampling continues to use the streaming metrics above.

### Run the HTTP client

From a second terminal, run the client.

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
python -m benchmarks.open_one_rec.one_rec_main bench serve \
  --endpoint /v1/completions \
  --backend openai \
  --model "$ONEREC_MODEL" --tokenizer "$ONEREC_MODEL" \
  --dataset-name onerec \
  --dataset-path ./data/benchmark_data \
  --task-types video \
  --num-prompts 100 \
  --use-beam-search \
  --n 128
```

## Command-line arguments

The [CLI dispatcher](../cli/bench.md) exposes the same workload as
`vllm bench open-one-rec throughput --gr` or `vllm bench serve --gr`.
Use `--help=all` on the selected command for all installed options.
`--enable-thinking` and `--disable-thinking` override the task's thinking
setting. The offline Beam runner excludes `--warmup-requests` (default 10)
before recording samples.

### Server options

| Argument           | Type   | Default   | Description                                                       |
| ------------------ | ------ | --------- | ----------------------------------------------------------------- |
| `--model`          | str    | -         | Model ID or path to load                                          |
| `--max-logprobs`   | int    | -         | Max log Probabilities output by model -- to support beam search   |

### Benchmark options

| Argument               | Type   | Default   | Description                                               |
| ---------------------- | ------ | --------- | --------------------------------------------------------- |
| `--dataset-name`       | str    | -         | Must be `onerec` for OneRec benchmark                     |
| `--dataset-path`       | str    | -         | Path to benchmark data directory                          |
| `--num-prompts`        | int    | -         | Number of prompts to sample                               |
| `--task-types`         | str    | -         | Comma-separated list of task types to run                 |
| `--custom-input-len`   | int    | -         | Prompt length; prompts are trimmed or repeated to match   |
| `--output-json`        | str    | -         | Path to save benchmark results as JSON                    |

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

| Argument              | Type    | Default         | Description                                                                                        |
| --------------------- | ------- | --------------- | -------------------------------------------------------------------------------------------------- |
| `--endpoint`          | str     | -               | API endpoint URL for online benchmark                                                              |
| `--backend`           | str     | 'openai'        | Must be `openai` for OneRec online benchmark                                                       |
| `--max-concurrency`   | int     | `None`          | Maximum number of concurrent requests; if not set, requests are sent as fast as possible           |
| `--request-rate`      | float   | `inf`           | Number of requests per second to send; `inf` means all requests are sent as soon as possible       |

## Controlled Beam latency

Use `vllm bench beam-performance MODE --gr` on CUDA to prepare one fixed
workload, record an offline baseline, and replay the same prompts over HTTP.
Use the same model, triples, prompt file, and output directory for each stage.
Remote model IDs require an immutable `--revision` during preparation; use
that revision in every stage and the server.

```bash
vllm bench beam-performance prepare --gr \
  --model /models/OneRec --triples /data/triples.json \
  --prompt /data/prompt.txt --output-dir /results/beam-performance
vllm bench beam-performance offline --gr \
  --model /models/OneRec --triples /data/triples.json \
  --prompt /data/prompt.txt --output-dir /results/beam-performance \
  --beam-width 32 --batch-sizes 1 2 4 --warmup-runs 2 --measured-runs 5
```

Start the HTTP server in a separate terminal using the generated configuration.

```bash
vllm serve /models/OneRec --gr --port 7001 \
  --config /results/beam-performance/gr-config.yaml \
  --seed 0 --async-scheduling --max-logprobs 32 --max-num-seqs 4 \
  --max-num-batched-tokens 8192 --max-model-len 8192 \
  --gpu-memory-utilization 0.7 --no-enable-prefix-caching --no-enable-chunked-prefill
```

After the server is ready, replay the recorded workload.

```bash
vllm bench beam-performance online --gr \
  --model /models/OneRec --triples /data/triples.json \
  --prompt /data/prompt.txt --output-dir /results/beam-performance --port 7001
```

This workload requires three generated SID tokens. Offline batch sizes count
prompts in one Beam call; online batch sizes count concurrent single-prompt
HTTP requests. Use `vllm bench beam-performance MODE --gr --help` for each
stage's options and the [benchmark evidence policy](../contributing/ci/benchmark_acceptance.md#changing-a-policy)
when comparing revisions.

## Benchmark compatibility with older servers

The current exporter lacks the separate `vllm_gr_beam_search_*` phase
counters that older servers supplied. The OneRec online benchmark can
calculate historical Prefill, Decode, and sort times from those servers.

- Keep the server isolated from other traffic during the measurement.
- Use before and after snapshots with complete, matching labels and
  nondecreasing phase counters and request counts. Check the total-beams
  counter the same way if the server publishes it.
- A missing or reset series makes the phase summary unavailable.

For current measurements, use the end-to-end workloads above. Keep the
model, prompt length, width, cache mode, and warmup constant when comparing
runs. Offline CUDA timing includes candidate text materialization; online
timing includes transport and serving overhead. See
[Serving](../serving/README.md#read-prometheus-metrics) for the metrics endpoint.

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
