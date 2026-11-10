# Online QPS benchmark

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the vLLM project -->

`qps_bench.py` drives a serving endpoint at a **fixed offered request rate** and
reports what the caller experienced: latency percentiles, success rate, achieved
throughput and how often the client itself ran out of slots. It is the tool to
compare GR with a competitor on the same workload shape (long prompt, wide beam,
short decode).

The load is open loop: requests are released on a schedule regardless of how
fast the server answers. A closed-loop concurrency sweep hides the behaviour
that matters here, because it slows down when the server slows down.

## Targets

| `--targets` | Endpoint | Payload |
| --- | --- | --- |
| `gr-openai` | `POST /v1/chat/completions` | text prompt, `use_beam_search`, `n = beam width`, server-side tokenization |
| `gr-grpc` | `VllmEngine/GenerateBatch` | pre-tokenized `input_ids`; one RPC carries `--batch-size` requests |
| `sid-generate` | `POST /generate` | SGLang-compatible `input_ids` + `sampling_params.n` |

Any subset can be selected in one run; each target gets the same prompt, beam
width, decode budget and schedule.

## Starting the servers

GR, OpenAI chain (needs V1 execution and a constraint artifact):

```bash
vllm-gr serve OpenOneRec/OneRec-1.7B --port 8000 \
  --max-logprobs 1024 --max-num-seqs 1024 --max-num-batched-tokens 16384 \
  --scheduling-policy priority
```

GR, gRPC:

```bash
vllm-gr serve OpenOneRec/OneRec-1.7B --grpc --port 50051
```

A SID-GR style competitor (its own repository serves `/generate`):

```bash
GR_BEAM_WIDTH=128 GR_MAX_BATCH_SIZE=4 scripts/serve_qwen3_gr_http.sh
```

## Running

```bash
# One target, 25/50/75/100/150 offered QPS, 60 s each.
python -m tools.online_bench.qps_bench \
  --targets gr-openai \
  --openai-url http://127.0.0.1:8000 \
  --model /models/OneRec-1.7B \
  --prompt-file tests/resources/single_one_rec_prompt.txt \
  --beam-width 128 --max-tokens 5 \
  --request-rate 25,50,75,100,150 --duration 60 \
  --output-json /tmp/gr_openai.json

# Both services in one run, pre-assembled batches of 4 over gRPC.
python -m tools.online_bench.qps_bench \
  --targets gr-grpc,sid-generate \
  --grpc-address 127.0.0.1:50051 --sid-url http://127.0.0.1:8001 \
  --tokenizer /models/OneRec-1.7B \
  --prompt-file tests/resources/single_one_rec_prompt.txt \
  --beam-width 128 --max-tokens 5 --batch-size 4 \
  --request-rate 100,150 --duration 60 --timeout 30
```

Useful options:

- `--input-len N` trims or repeats the prompt so every request has `N` tokens.
- `--max-inflight N` bounds client slots; the `backpressure` column counts
  requests that had to wait for one, which means the client, not the server,
  was the limit.
- `--warmup-shots N` sends untimed batches first (startup, graph capture, prefix
  cache fill).
- `--timeout S` counts a slow request as a `timeout` failure instead of waiting
  forever. Pick it from the SLO being evaluated, not from the server.

## Reading the result

```
| service | offered qps | achieved qps | success | mean ms | p50 ms | p90 ms | p99 ms | max ms | timeouts | backpressure |
```

- `achieved qps` counts successful requests only; a server that rejects or times
  out reads lower than the offered rate.
- Latency is measured from client release to the last byte of the response, so
  it includes queueing, batching and network, which is what a caller sees.
- `--output-json` stores every sample (rate, start offset, latency, status), so a
  run can be re-aggregated or plotted without repeating it.

## Caveats

- The same prompt is reused, so prefix caching behaves as it would for a hot
  request key. Use `--input-len` and a different prompt to measure cold hits.
- `gr-openai` tokenizes on the server; `gr-grpc` and `sid-generate` receive token
  ids. That is the real cost of each entry point, not a benchmark artefact.
- Run the client on a machine that can sustain the offered rate; check the
  `backpressure` column before blaming the server.
