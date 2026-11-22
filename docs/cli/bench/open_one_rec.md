# OneRec workloads

Both workloads take `--dataset-name onerec` and a `--dataset-path` pointing
to the directory containing the task subdirectories. Select tasks with
`--task-types`, a comma-separated list such as `video,ad`. The default is
`ad,product,label_cond,video,interactive,label_pred`. Use `--custom-input-len`
to trim or repeat prompts to a target length. Use `--enable-thinking` or
`--disable-thinking` to override the selected task's thinking setting.

## Offline throughput

`vllm bench open-one-rec throughput --gr` runs the model locally. For the
CUDA Beam workload, supply a local model directory and a catalog for the
selected task.

### Arguments

| Option | Use in offline CUDA Beam |
| --- | --- |
| `--beam-search` | Select the Beam workload. |
| `--n` | Set the beam width. |
| `--constraint-triples` | Required. Path to the catalog triples JSON. |
| `--output-json` | Required. Destination for the measured results. |
| `--output-len` | Use 5 for the three-level constraint table: SID begin, three catalog tokens and SID end. |
| `--warmup-requests` | Number of requests to exclude from measurements. Defaults to 10. |

### Example

```bash
vllm bench open-one-rec throughput --gr \
  --model /models/OneRec --dataset-name onerec --dataset-path ./data/benchmark_data \
  --task-types video --num-prompts 100 --beam-search --n 128 --output-len 5 \
  --constraint-triples /data/video-triples.json --output-json results/throughput.json
```

## Online serving

`vllm bench serve --gr` is an alias for `vllm bench open-one-rec serve --gr`.
Start the [OneRec HTTP server](../../benchmarking/common.md#online-server-mode)
first. The client uses the model name served by that endpoint.

### Arguments

| Option | Use with the HTTP client |
| --- | --- |
| `--backend`, `--endpoint` | Use `openai-chat` and `/v1/chat/completions` for the OneRec chat workload. |
| `--base-url` | Server URL including scheme and port. |
| `--use-beam-search` | Send Beam search requests. |
| `--n` | For fixed-width GR Beam serving, match the beam width configured on the server. |

### Example

```bash
vllm bench serve --gr \
  --model /models/OneRec --dataset-name onerec --dataset-path ./data/benchmark_data \
  --backend openai-chat --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
  --num-prompts 100 --task-types video --use-beam-search --n 128
```

Use `--help=all` on the chosen subcommand for its native vLLM options. See
[throughput](../../benchmarking/throughput.md) for workload preparation and metrics.
