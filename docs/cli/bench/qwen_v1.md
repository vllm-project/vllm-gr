# Qwen Beam benchmark

<a id="qwen-v1"></a>

`vllm bench qwen-v1 --gr` runs a synthetic Beam workload against a Qwen model. It
requires CUDA.

## Arguments

| Option | Purpose |
| --- | --- |
| `--model`, `--workload-jsonl`, `--output-json` | Required model, request inputs and result destination. Each JSONL row needs a nonempty integer `input_ids` list. |
| `--token-id-start`, `--token-id-end` | Required bounds for input and synthetic output tokens. The upper bound is exclusive. |
| `--constraint-triples`, `--constraint-artifact` | Supply catalog triples or choose where to write the compiled table. |
| `--context-len` | Declared length in the report. The runner measures the actual lengths of the JSONL inputs. Set the generator's `--context-len` to change those inputs. |
| `--requests`, `--beam-width`, `--beam-steps` | Request count, width and generated stages per request. Defaults are 1, 256 and 3. |
| `--warmup-runs`, `--repeat` | Untimed warmup and measured run counts. Defaults are 1 and 3. |
| `--timeline-output` | Write a gzip trace from separate profiled runs. |

Use `vllm bench qwen-v1 --gr --help` for engine, tokenizer and profiling options.

## Example

From the repository root, generate a workload using the local Qwen model's
`config.json` with the [workload generator](../../../benchmarks/qwen/make_workload.py).
Choose a token interval within the model vocabulary and pass the same bounds
to both commands. The upper bound is exclusive.

```bash
python benchmarks/qwen/make_workload.py \
  --model-dir /models/Qwen --token-id-start 1000 --token-id-end 30000 \
  --context-len 5000 --requests 1 --output-jsonl results/workload.jsonl

vllm bench qwen-v1 --gr \
  --model /models/Qwen --workload-jsonl results/workload.jsonl \
  --output-json results/qwen.json --token-id-start 1000 --token-id-end 30000 \
  --context-len 5000 --requests 1 --beam-width 128 --beam-steps 3
```

With `--constraint-triples` omitted, the runner generates synthetic constraints
from the model tokenizer.
