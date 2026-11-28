# OneRec benchmarks

Use the [common benchmark setup](README.md#setup) to prepare the runtime and OneRec
data. Run commands from the repository root in the checkout that matches your
installed vLLM-GR version.

<a id="scenario-command-contracts"></a>

## Choose a benchmark

| What do you want to measure? | CUDA GPU | Ascend NPU |
| --- | --- | --- |
| Recommendation accuracy | Run the [accuracy benchmark](accuracy.md) with a CUDA constraint table or against a CUDA server. | Use the same [accuracy benchmark](accuracy.md) with an NPU constraint table or against an Ascend server. |
| HTTP throughput and latency | Run the [HTTP client](throughput.md#online-serving) against a CUDA server. | Run the same client against an [Ascend HTTP server](README.md#online-server-mode). |
| Offline Beam performance | Measure [dataset throughput](throughput.md#cuda-dataset-throughput) with catalog-constrained generation. | Measure [batch latency and requests/s](throughput.md#ascend-batch-latency) with a fixed prompt and batch size. |
| Change in performance between two commits | Repeat the [Beam performance workload](throughput.md#controlled-beam-latency) on the same device with identical inputs and settings. | Repeat the [Ascend latency example](throughput.md#ascend-batch-latency) on both commits with the same device, model, prompt, Beam settings, and warmup. |

**Comparing devices.** The two offline examples measure different workloads.
For a cross-device comparison, hold the model, inputs, catalog mode, Beam width,
batch size and timing scope constant. The
[cache-miss benchmark](../../benchmarks/open_one_rec/V1_BENCHMARK.md)
records the CUDA timing scope and historical results.

After applying the [vLLM CLI dispatch patch](../cli/README.md#command-form-and-installation),
run `vllm bench --gr --list` and see the
[CLI benchmark reference](../cli/bench.md) for arguments and backend support.
For pull request acceptance criteria, see the [CI benchmark gates](../contributing/ci/benchmark_acceptance.md#benchmark-gates).

<a id="open-one-rec-benchmark-dataset"></a>

## OneRec benchmark dataset

### Download the dataset

Request access on Hugging Face before cloning the dataset.

<https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF>

```bash
git lfs install
git clone https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF ./data
```

### Locate the task data

After the clone above, the video test split is at
`./data/benchmark_data/video/video_test.parquet`. Pass
`--dataset-path ./data/benchmark_data` to the accuracy and throughput clients.
If you use another location, pass the directory that directly contains
`video/`, `ad/`, and the other task directories.

For catalog-constrained Beam search, [prepare the catalog and constraint
table](README.md#prepare-the-catalog)
using the matching OneRec tokenizer and the dataset's `benchmark_data/sid2pid.json`.

### Parquet schema

Each Parquet file must contain these columns:

| Column | Type | Description |
| -------- | ------ | ------------- |
| `messages` | str/list | Chat messages in JSON format or list of dicts |
| `metadata` | str/dict | JSON metadata containing answer and context |

For example, a Parquet row can store both columns as JSON strings:

```json
{
  "messages": "[{\"role\":\"user\",\"content\":\"Recommend a video\"}]",
  "metadata": "{\"answer\":\"<|sid_begin|><s_a_346><s_b_6566><s_c_5603><|sid_end|>\"}"
}
```

This is an illustrative SID answer for a recommendation task. It shows the
column encoding, not a realistic user history or a complete SID-to-item
evaluation record. Other tasks use different answer formats. Use the
released dataset, including its task metadata and mappings, to evaluate
recommendation quality.

## Tasks

The registry includes tasks with different inference and scoring requirements.
The table describes the current **accuracy entrypoint**, not a validation
claim for every model and device combination. The Beam examples use `video`.

| Task | Output | Offline accuracy | Online accuracy | Additional requirements |
| --- | --- | --- | --- | --- |
| `ad` | Recommendation SIDs | Short SID Beam | Beam completions | Matching task data and catalog |
| `product` | Recommendation SIDs | Short SID Beam | Beam completions | Matching task data and catalog |
| `label_cond` | Recommendation SIDs | Short SID Beam | Beam completions | Matching task data and catalog |
| `video` | Recommendation SIDs | Short SID Beam | Beam completions | Matching task data and catalog |
| `interactive` | Recommendation SIDs | Short SID Beam | Beam completions | Matching task data and catalog |
| `label_pred` | Label-token probabilities | Unsupported by this entrypoint | Unsupported by this entrypoint | Exits before inference; text output cannot supply the required probabilities |

The three-step catalog Beam setup below is for SID recommendation. See
[task-specific requirements](accuracy.md#task-specific-requirements)
before choosing a task. Throughput measurements do not establish
accuracy-task support or run its evaluator.

## Setup

### Prerequisites

- Install vLLM-GR for [CUDA](../getting_started/installation/gpu.md) or
  [Ascend](../getting_started/installation/npu.md).
- Download the OneRec-1.7B model snapshot and tokenizer. Set `ONEREC_MODEL` to
  the local snapshot path.
- [Download the dataset](#onerec-benchmark-dataset). The examples
  assume you cloned it into `./data` and use `./data/benchmark_data` as the task directory.

### Prepare the catalog

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

### Online server mode

Both benchmarks can run against a live server instead of an in-process engine.
Follow the [HTTP server example](../../examples/online_serving/beam_search/README.md#start-the-server)
for your device. For the accuracy and throughput commands, replace the served
name assignment in that example with `export ONEREC_SERVED_MODEL_NAME="$ONEREC_MODEL"`
and use `/v1/completions`. Set the server's context
and batched-token limits to accommodate the prompts and output budget for
your task.

The clients render source messages locally with
`benchmarks/open_one_rec/qwen3_soft_switch.jinja2` and send the resulting
`prompt`, with `add_special_tokens: false`. A per-run template override is
applied locally and takes precedence over the task default:

- The accuracy client accepts `--chat-template-kwargs`.
- The throughput client accepts `--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'`.

For fixed-length throughput prompts, set `--enable-thinking` or
`--disable-thinking` before normalization; template overrides require the
original messages. See [prompt templates](../serving/completions.md#model-prompt-templates).

**Match the served model name.** Use the same model identifier in the client
and server. Choose either the model ID or a local snapshot path that both sides
can resolve.
