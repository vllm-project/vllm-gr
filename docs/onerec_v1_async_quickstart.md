# OneRec-1.7B V1 asynchronous scheduling quickstart

This guide runs the new EngineCore-owned asynchronous Beam Search V1 path
with OneRec-1.7B. One offline request exercises Prefill, asynchronous Decode,
GPU Beam decisions, and the CUDA Constraint Table. Keep the values in this
guide unchanged for the first run.

> This reproduction uses offline `GRLLM.beam_search_v1()`. Online requests
> also use V1; see the [online serving example](../examples/online_serving/beam_search/README.md)
> for the matching canonical configuration and server command.

## 1. Check the environment

Use a CUDA GPU, Python 3.11 or later, vLLM 0.30.0, and the matching
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
USE_TF=0 \
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
| Generation steps `N` | 3 | N includes the first token from Prefill; only 1, 2, and 3 are supported. |
| EOS | `ignore_eos=True` | Legal EOS candidates are retained; finish after N outputs. |
| Prefill | `enable_chunked_prefill=False` | The complete remaining prompt must fit the native token budget. |
| Dispatch capacity `Q` | 2 | Two stages may be in flight for the same logical owner; this does not make B=2. |
| `max_tokens` | 5 | The example has begin and end positions, so `N = 5 - 2 = 3`. |
| Constraint | CUDA Constraint Table | Produces catalog-valid SIDs. |
| Parallelism and requests | TP/PP/DP/DCP/PCP are 1; one request | One execution owner holds the physical dispatch slots. |

The example script sets `execution_mode: v1`, `worker_decision: true`,
`graph_enabled: true`, Prefix Cache, and `max_num_seqs == beam_width`.
When changing the Beam width, also update the request, `beam.max_width`,
Constraint Table `max_top_k`, and engine capacity. Do not change only the
client `beam_width`.

### Batching several prompts in one HTTP request

The V1 completions route renders a prompt **list** once. When the list length is
one of the configured fixed batch sizes (2 or 4 by default), it runs the whole list
as **one** beam session. For example:

```bash
curl -X POST http://localhost:8000/v1/completions   -H "Content-Type: application/json"   --data '{
    "model": "OpenOneRec/OneRec-1.7B",
    "prompt": ["<prompt 0>", "<prompt 1>", "<prompt 2>", "<prompt 3>"],
    "use_beam_search": true,
    "n": 128,
    "temperature": 0.0,
    "max_tokens": 5,
    "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"}
  }'
```

The engine admission contract supports logical B=1 through B=8. The default
`beam.online_batch_sizes` policy is `[1, 2, 4]`; engine capability does not
automatically enable every size for HTTP batching or CUDA graph capture.
Arbitrary-length partitioning remains a separate follow-up feature.

To enable other exact HTTP batch sizes, add this field to the existing
`--vllm-gr-config` YAML and restart the service:

```yaml
beam:
  execution_mode: v1
  online_batch_sizes: [1, 3, 7]
```

Keep the existing model/constraint-table settings. Use `--max-num-seqs 7` or
higher for this example. The list must contain unique integers in 1..8 and
include 1; order is normalized. A non-default policy exceeding `max_num_seqs`
is rejected during engine configuration, not repaired by increasing capacity.
The default policy is limited by the configured capacity, preserving B=1-only
deployments. Prefill token and NPU Decode B*beam_width limits still apply.

CUDA V1/V2 Prefill capture follows this policy by default. The existing
`VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS` environment variable can override
Prefill capture independently (for example `1,2,4` to avoid larger captures).
It accepts increasing sizes in 1..8; uncaptured Prefill shapes retain fallback.
CUDA ModelRunner V2 Decode capture follows the policy too: unselected sizes use
eager execution, while a missing graph for a selected, successfully captured
size remains an error. V1's single Decode startup shape is unchanged and uses
`VLLM_GR_V1_STARTUP_CAPTURE_BATCH_SIZE` (default 1). NPU's exact Decode graph
shape remains controlled by `beam.graph_batch_size`.

These settings take effect at startup, not per HTTP request. Wider capture
sets increase startup cost and graph memory. Persistent workspace capacity
still covers the engine's 1..8 capability and remains independent of the policy.

By default, two prompts run as one session of 2; four as one session of 4. Any other count
keeps the original route, which is one session per prompt; a single prompt is
one session either way. The list is not reordered, and a list that this route
does not claim behaves exactly as it did before.

The batch must fit `max_num_seqs` and `max_num_batched_tokens`, including the
begin token appended to every prompt. If configured, a smaller
`max_num_scheduled_tokens` also limits it. Engine KV admission remains
authoritative; the frontend check does not guarantee device capacity.

V1 requires Chunked Prefill to be **off**. Enabling it is refused when the engine
starts (`beam_search_v1 requires enable_chunked_prefill=False`), not split:
there is no partial Prefill on this path, and the complete prompt has to fit the
token budget.

A batch runs one `max_tokens`, so prompts whose effective generation budgets
differ cannot share a session. No prompt's budget is reduced to match another;
such a list keeps the original route instead.

The response is a single object whose `choices` are flat-indexed prompt-major,
so `k` prompts x `n` beams give `k x n` choices, and `usage.prompt_tokens` sums
across the entire HTTP request. This is the same shape the original route
returns, so a caller cannot tell from the response alone whether the list ran as
one session or as many.

A list that is not batched logs the reason at INFO level. Search the server log
for `Beam batch declined` to see the prompt count and which condition applied.

#### Ownership

The HTTP list is a response boundary; the session formed here is the GPU
execution-batch boundary. `GRExecutionCapacity.active_sessions` remains 1: the
session fixes its members before it starts, no members join or leave it, and the
frontend waits for its iterator and cleanup before anything else runs. This is
not cross-request coalescing or continuous batching, and it introduces no new
endpoint or client-side batch-size requirement.

Running a list as one session does not guarantee identical beam rankings to
separate B=1 calls. GPU batch shapes and prefix-cache execution paths can produce
different candidates or scores. Validate model quality for the batch sizes you
deploy.

## 5. Validate and troubleshoot

With real model weights, run the system test for actual offline and online V1
calls, N=1/2/3, Prefix Cache, graph replay, cancellation/reuse and release at return.
Select an available GPU using the environment's existing device allocation rules:

```bash
USE_TF=0 \
VLLM_GR_TEST_MODEL="$ONEREC_MODEL" \
python -m pytest --run-slow -v -s tests/st/test_onerec_v1_scheduler_st.py
```

The same file also carries one completions-route check configured with
`[1,2,3,4,8]`. It starts one engine and posts 1-, 2-, 3-, 4- and 8-prompt lists,
checking actual session sizes, response indices, usage and cleanup. This covers
common batch sizes, a non-power-of-two size and the supported batch-size limit
without starting a second engine for a separate policy. Lightweight configuration
and routing tests retain default `[1,2,4]` and sparse `[1,3,7]` policy coverage.
The test configuration does not change production defaults.

```bash
python -m pytest --run-slow -v -s tests/st/test_onerec_v1_scheduler_st.py \
  -k completion_batch --durations=0
```

The HTTP case writes `completion-batch-evidence.json` in its pytest temporary
directory: phase durations, request checks, and observed Decode/Prefill graph
shapes. Phase timings and request rows are persisted incrementally, including
on ordinary exceptions; a hard process kill can leave only partial evidence.
`subprocess-evidence.json` records the outer process wall time and exit code.
Request timing includes probes, assertions, cache resets and any first-use JIT;
it is not a steady-state inference benchmark. Keep cache conditions consistent
when comparing runs.

| Symptom | Resolution |
| --- | --- |
| `async_scheduling=True` is required | Use the V1 example and keep this `GRLLM` initialization option. |
| Width or generation-step error | Keep request width equal to `max_width` and use 1 through 3 effective generation steps. |
| Constraint Table digest or tokenizer mismatch | Rebuild the artifact from the current model directory and triples document. |
| CUDA graph or out-of-memory error | Start with width 128 and the default `--gpu-memory-utilization 0.5`; check for competing GPU processes. |
| Online V1 validation is needed | The system test calls `OpenAIServing.beam_search_v1()` with a real asynchronous multiprocess engine. Legacy HTTP gates remain separate. |
| EOS or Prefill admission error | Set `ignore_eos=True`, disable Chunked Prefill and allow the complete prompt in `max_num_batched_tokens`. |
| A prompt list did not batch | Check `Beam batch declined`: the count must be in `beam.online_batch_sizes`, and sequence capacity, the token budget and equal generation budgets must all allow it. |

## Further reading

- [ModelRunner V2 multi-batch execution](modelrunner_v2_multibatch.md): native integration, B=1/2/4 configuration, validation and remaining accuracy work.
- [Asynchronous scheduling execution contract](async_scheduling.md): scheduling, Prefill/Decode overlap, and retirement.
- [GPU Prefill capture configuration](prefill_graph_capture.md): V1 grid compatibility, V2 native capture sizes, and multi-request capacity.
- [Offline Beam Search V1 reference](../examples/offline_inference/beam_search/README.md#beam-search-v1): full configuration fields and script arguments.
- [Asynchronous scheduling validation](async_scheduling_validation.md): historical precision and graph-replay results.
- [OneRec benchmark](benchmarks/README.md): dataset, online benchmarks, and accuracy evaluation.
