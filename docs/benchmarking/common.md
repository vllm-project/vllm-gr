# Common benchmark setup

The [accuracy](accuracy.md) and [throughput](throughput.md) benchmarks share the
runtime, dataset, and online-server setup described here. Each page keeps its own
script, metrics, and argument table.

## Prerequisites

- Install vLLM-GR for [CUDA](../getting_started/installation/gpu.md) or
  [Ascend](../getting_started/installation/npu.md).
- Download the OneRec-1.7B model snapshot and tokenizer. Set `ONEREC_MODEL` to
  the local snapshot path.
- [Download the dataset](README.md#onerec-benchmark-dataset). The examples
  assume you cloned it into `./data` and use `./data/benchmark_data` as the task directory.

## Prepare the catalog

For catalog-constrained benchmarks, set the dataset and output directories
from the repository root.

```bash
export ONEREC_DATA="$PWD/data"
export ONEREC_WORKDIR="$PWD/results/onerec-benchmark"
mkdir -p "$ONEREC_WORKDIR"
```

Use a fresh work directory when changing the model or dataset. Then build the
[full catalog from the dataset](../../examples/online_serving/catalog/README.md#build-a-catalog-from-the-onerec-dataset).
Follow the [remaining table and YAML steps](../../examples/online_serving/catalog/README.md#extract-constraint-triples-from-the-catalog).
Choose `ONEREC_BACKEND=cuda` or `ONEREC_BACKEND=npu` before generating the
configuration. Keep `catalog.json`, `constraint_triples.json`,
`constraint_table.safetensors` and `vllm_gr_config.yaml` in `ONEREC_WORKDIR`.

## Online server mode

Both benchmarks can run against a live server instead of an in-process engine.
Follow the [HTTP server example](../../examples/online_serving/beam_search/README.md#start-the-server)
for your device. For the accuracy and throughput commands, replace the served
name assignment in that example with `export ONEREC_SERVED_MODEL_NAME="$ONEREC_MODEL"`
and add
`--chat-template benchmarks/open_one_rec/qwen3_soft_switch.jinja2` to the
server command so it renders the benchmark chat turns. Set the server's context
and batched-token limits to accommodate the prompts and output budget for
your task.

The client sends the source chat turns and the task's thinking setting. A
per-run override takes precedence over the task default:

- The accuracy client accepts `--chat-template-kwargs`.
- The throughput client accepts `--extra-body`.

!!! important "Match the served model name"

    Use the same model identifier in the client and server. Choose either the
    model ID or a local snapshot path that both sides can resolve.
