# OneRec-1.7B V1 asynchronous scheduling quickstart

This guide runs the new EngineCore-owned asynchronous Beam Search V1 path
with OneRec-1.7B. One offline request exercises Prefill, asynchronous Decode,
GPU Beam decisions, and the CUDA Constraint Table. Keep the values in this
guide unchanged for the first run.

> The V1 reproduction entry point is offline `GRLLM.beam_search_v1()`.
> The `vllm-gr serve` and OpenAI Chat API examples are separate online and
> compatibility paths. Do not add the V1 configuration from this guide to
> those commands.

## 1. Check the environment

Use a CUDA GPU, Python 3.11 or later, vLLM 0.22.1, and the matching
`vllm-gr` installation. See [quickstart](quickstart.md) for installation.
From the repository root, verify the environment:

```bash
nvidia-smi
python -c 'import importlib.metadata as m; print("vllm", m.version("vllm")); print("vllm-gr", m.version("vllm-gr"))'
python -c 'from vllm_gr.entrypoints.gr import GRLLM; print("v1 import OK")'
```

Set local paths for the model, dataset, and generated files:

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
export ONEREC_DATA=/absolute/path/to/OpenOneRec-RecIF
export ONEREC_WORKDIR="$PWD/results/onerec-v1"
mkdir -p "$ONEREC_WORKDIR"
```

`ONEREC_DATA` must contain `sid2pid.json`. See the
[OneRec dataset instructions](benchmarks/README.md#open-one-rec-benchmark-dataset).
Use the same model and tokenizer version when building the Constraint Table.

If the model and dataset are not available locally, accept the required access
terms on Hugging Face and sign in first. Then download them, reserving enough
disk space for the dataset:

```bash
huggingface-cli download OpenOneRec/OneRec-1.7B --local-dir "$ONEREC_MODEL"
git lfs install
git clone https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF "$ONEREC_DATA"
```

## 2. Build the V1 Constraint Table inputs

V1 uses a CUDA Constraint Table rather than a plain catalog JSON. Convert
`sid2pid.json` to a SID catalog, then create the triples document:

```bash
python benchmarks/open_one_rec/tools/convert_catalog.py \
  --dataset-path "$ONEREC_DATA" \
  --output-path "$ONEREC_WORKDIR/video_catalog.json"

python tools/build_constraint_triples.py \
  --catalog "$ONEREC_WORKDIR/video_catalog.json" \
  --output "$ONEREC_WORKDIR/video_constraint_triples.json" \
  --begin '<|sid_begin|>' \
  --end '<|sid_end|>'
```

Rebuild these files only when the catalog changes. The standalone example
compiles a temporary `.safetensors` artifact from the triples document and the
model tokenizer. To create a reusable artifact for an integration, run:

```bash
python tools/build_constraint_table.py \
  --triples "$ONEREC_WORKDIR/video_constraint_triples.json" \
  --model "$ONEREC_MODEL" \
  --output "$ONEREC_WORKDIR/video_constraint_table.safetensors" \
  --trust-remote-code
```

The command prints `artifact_digest` and `tokenizer_digest`. Use both values
with the artifact absolute path in an integration's `constraint_table`
configuration. The standalone example in the next section deliberately builds
a temporary artifact and does not read this persisted file. See the
[artifact configuration reference](../examples/offline_inference/beam_search/README.md#constraint-table-artifact)
for the full configuration.

## 3. Run the minimal V1 reproduction

Create a UTF-8 prompt file. The following ASCII-only input is enough to check
the scheduling path:

```text
Video history: <|sid_begin|><s_a_346><s_b_6566><s_c_5603><|sid_end|>. Recommend the next video.
```

Save it as `$ONEREC_WORKDIR/prompt.txt`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 USE_TF=0 \
python examples/offline_inference/beam_search/offline_beam_search_v1.py \
  --model "$ONEREC_MODEL" \
  --catalog "$ONEREC_WORKDIR/video_constraint_triples.json" \
  --prompt "$ONEREC_WORKDIR/prompt.txt" \
  --input-len 1024 \
  --generated-steps 3 \
  --beam-width 128
```

The first run loads the model, compiles the temporary table, and captures the
fixed-width CUDA graph, so it is slower than later runs. A successful run
prints `returned_beams=128` followed by the first five candidates. Each
`sid_tokens` value contains a begin marker, three SID components, and an end
marker. Scores and ranking can differ across GPU, CUDA, weights, and prompts;
do not use one fixed score as the pass condition.

## 4. Why these values are fixed

The new scheduler has an execution contract, rather than being a performance
switch for arbitrary requests. Keep these values for the first reproduction:

| Setting | First-run value | Reason |
| --- | ---: | --- |
| Scheduling | `async_scheduling=True` | EngineCore submits the next stage early. |
| Beam width | 128 | Startup allocates the KV pool and captures a fixed-shape graph; request width must match exactly. |
| Generation steps `N` | 3 | Only 1, 2, and 3 are currently supported. |
| `max_tokens` | 5 | The example has begin and end positions, so `N = 5 - 2 = 3`. |
| Constraint | CUDA Constraint Table | Produces catalog-valid SIDs. |
| Parallelism and requests | TP/PP/DP/DCP/PCP are 1; one request | One execution owner holds the physical dispatch slots. |

The example script sets `execution_mode: v1`, `worker_decision: true`,
`graph_enabled: true`, Prefix Cache, and `max_num_seqs == beam_width`.
When changing the Beam width, also update the request, `beam.max_width`,
Constraint Table `max_top_k`, and engine capacity. Do not change only the
client `beam_width`.

## 5. Validate and troubleshoot

With real model weights, run the system test to cover Prefill, Prefix Cache,
graph replay, and resource release:

```bash
USE_TF=0 CUDA_VISIBLE_DEVICES=0 \
VLLM_GR_TEST_MODEL="$ONEREC_MODEL" \
python -m pytest --run-slow -v -s tests/st/test_onerec_v1_scheduler_st.py
```

| Symptom | Resolution |
| --- | --- |
| `async_scheduling=True` is required | Use the V1 example and keep this `GRLLM` initialization option. |
| Width or generation-step error | Keep request width equal to `max_width` and use 1 through 3 effective generation steps. |
| Constraint Table digest or tokenizer mismatch | Rebuild the artifact from the current model directory and triples document. |
| CUDA graph or out-of-memory error | Start with width 128 and the default `--gpu-memory-utilization 0.5`; check for competing GPU processes. |
| HTTP reproduction is needed | Verify the offline V1 path first. Existing online examples have different parameters and output boundaries. |

## Further reading

- [Asynchronous scheduling execution contract](async_scheduling.md): scheduling, Prefill/Decode overlap, and retirement.
- [Offline Beam Search V1 reference](../examples/offline_inference/beam_search/README.md#beam-search-v1): full configuration fields and script arguments.
- [Asynchronous scheduling validation](async_scheduling_validation.md): system tests and precision or graph-replay boundaries.
- [OneRec benchmark](benchmarks/README.md): dataset, online benchmarks, and accuracy evaluation.
