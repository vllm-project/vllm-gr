# Beam performance

`vllm bench beam-performance MODE --gr` measures Beam inference on CUDA.
Choose `prepare`, `offline`, or `online` for `MODE`.

## Usage

```bash
vllm bench beam-performance prepare --gr [options]
vllm bench beam-performance offline --gr [options]
vllm bench beam-performance online --gr [options]
```

Use the same output directory for all three stages. The offline run writes the
prompts and baseline results that the online run replays.

## Beam performance arguments

| Argument | Default | Purpose |
| --- | --- | --- |
| `MODE` | Required | Select `prepare`, `offline`, or `online`. |
| `--model` | Required | Local checkpoint directory or model ID. |
| `--triples`, `--prompt` | Both required | Existing catalog triples JSON and prompt text file. |
| `--output-dir` | Required | Shared directory for setup, prompt IDs and measurements. |
| `--revision` | Required in prepare for a remote model ID | Model/tokenizer revision. Use the same revision for all stages and the server. |
| `--served-model-name` | Model name | Model identifier sent to the HTTP endpoint. |
| `--input-len` | 1024 | Input token count, including SID begin. |
| `--beam-width`, `--generated-steps` | 32, 3 | Beam width and generated steps. This workload requires exactly three steps. |
| `--batch-sizes` | `1 2 4` | Offline prompts per Beam call. Online concurrent HTTP requests, one prompt each. |
| `--warmup-runs`, `--measured-runs` | 2, 5 | Warmup and measured repetitions per batch size. |
| `--max-model-len` | 8192 | Context length limit in tokens. |
| `--gpu-memory-utilization` | 0.7 | Fraction of GPU memory available to the engine. |
| `--port`, `--request-timeout` | 7001, 180 | HTTP port and request timeout in seconds. |
| `--ab-prefill-grid` | Off | Capture prefill for the A/B workload's batch sizes 1, 2 and 4. |
| `--ab-legacy-config` | Off | Use the legacy startup schema for offline A/B. Requires `--ab-prefill-grid`. |

## Measure a workload

### Prepare inputs and record the offline baseline

```bash
vllm bench beam-performance prepare --gr \
  --model /models/OneRec --triples /data/triples.json \
  --prompt /data/prompt.txt --output-dir /results/beam-performance

vllm bench beam-performance offline --gr \
  --model /models/OneRec --triples /data/triples.json \
  --prompt /data/prompt.txt --output-dir /results/beam-performance \
  --beam-width 32 --batch-sizes 1 2 4 --warmup-runs 2 --measured-runs 5
```

### Start the HTTP server

Run it in a separate terminal using the configuration written by
`prepare`.

```bash
vllm serve /models/OneRec --gr --port 7001 \
  --config /results/beam-performance/gr-config.yaml \
  --seed 0 --async-scheduling --max-logprobs 32 --max-num-seqs 4 \
  --max-num-batched-tokens 8192 --max-model-len 8192 \
  --gpu-memory-utilization 0.7 --no-enable-prefix-caching --no-enable-chunked-prefill
```

### Replay the requests online

Once the server is ready, run the online stage from the benchmark terminal.

```bash
vllm bench beam-performance online --gr \
  --model /models/OneRec --triples /data/triples.json \
  --prompt /data/prompt.txt --output-dir /results/beam-performance --port 7001
```
