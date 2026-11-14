# Offline Beam Search Demo

This directory contains a standalone example demonstrating how to run beam search in **offline mode** using the `vllm-gr` framework for generative recommendation models.

The entrypoint is `offline_beam_search_v1.py`, the EngineCore-owned
asynchronous V1 path. The old `offline_beam_search.py` example is retired.

The V1 example accepts a model, Constraint Table triples document, and prompt.
It compiles a temporary CUDA or NPU Constraint Table artifact, builds the canonical V1
configuration, calls `GRLLM.beam_search_v1`, and prints the leading Beam results.
Run `python offline_beam_search_v1.py --help` for its arguments.

New to V1 scheduling? Follow the end-to-end
[OneRec-1.7B asynchronous scheduling quickstart](../../../docs/onerec_v1_async_quickstart.md)
first; this document is the detailed configuration reference.

## Installation

Follow the instructions in  
[`docs/quickstart.md`](../../../docs/quickstart.md)  
to install the required dependencies and set up your environment.

## Beam Search V1

Beam Search V1 is the EngineCore-owned asynchronous Beam path. The parameter
combinations described here follow the working local profiling and A/B scripts.

### Public entrypoint

Use `GRLLM.beam_search_v1()` with `beam.execution_mode: v1`.
`GRLLM.beam_search()` raises a migration error instead of falling through to
upstream vLLM's implementation. Legacy frontend step orchestration is retired.

The V1 flow is:

```text
beam_search_v1()
  -> prepare and validate one request
  -> submit one EngineCore session
  -> keep Prefill, Decode, Beam state, and decisions on the Worker/GPU
  -> rank and pack the final result on the GPU
  -> copy one compact payload to the host
  -> create lazy public Beam outputs
  -> release the session
```

Execution mode is selected when the engine is initialized. Native non-Beam
inference remains available in ordinary engine configurations; do not mix it
with an active V1 Beam session.

### Current restrictions

The constrained example requires:

- CUDA or NPU and an enabled canonical Constraint Table matching the device.
- `async_scheduling=True`.
- Logical B=1..8 with shared generation parameters; online batch/capture policy
  defaults to `[1, 2, 4]` and must fit engine capacity.
- TP, PP, DP, DCP, and PCP sizes equal to 1.
- Request `beam_width == beam.max_width`.
- Effective generation budget `N=3`, matching the three-layer `constraint_table_v1`.
  Other lengths are rejected at request ingress.
- Token/text prompts rather than embedding prompts.
- No speculative decoding, LoRA, quantization, hybrid model, encoder-decoder
  model, or multimodal model.
- `catalog_strategy=None`, because the Worker-side table controls validity.

### Constraint Table artifact

If the source catalog is not already a triples document, convert it first:

```bash
python tools/build_constraint_triples.py \
  --catalog /path/to/video_catalog.json \
  --output /path/to/video_constraint_triples.json \
  --begin '<|sid_begin|>' \
  --end '<|sid_end|>'
```

The standalone `offline_beam_search_v1.py` example accepts this triples document
and compiles a temporary artifact automatically. For a reusable artifact,
run:

```bash
python tools/build_constraint_table.py \
  --triples /path/to/video_constraint_triples.json \
  --model /path/to/OneRec-1.7B \
  --output /path/to/video_constraint_table.safetensors \
  --trust-remote-code
```

The command prints `artifact_digest` and `tokenizer_digest`. Both values belong
in the canonical configuration and prevent loading a table built from different
constraint data or a different tokenizer.

### Engine configuration

The standalone example builds the following CUDA configuration in memory. The
same mapping can be stored as a YAML file for other applications. For NPU, use
`constraint_table.backend: npu`:

```yaml
schema_version: 1
attention_backend: CUSTOM

beam:
  execution_mode: v1
  graph_enabled: true
  worker_decision: true
  max_width: 128
  max_decode_steps: 3

constraint_table:
  enabled: true
  backend: cuda
  max_top_k: 128
  path: /absolute/path/to/video_constraint_table.safetensors
  format: constraint_table_v1
  artifact_digest: sha256:REPLACE_WITH_ARTIFACT_DIGEST
  tokenizer_digest: sha256:REPLACE_WITH_TOKENIZER_DIGEST
```

#### Beam configuration parameters

| Parameter | Meaning | V1 value |
| --- | --- | --- |
| `execution_mode` | Selects the engine-lifetime Beam interface | `v1` |
| `graph_enabled` | Enables the fixed-shape Beam CUDA graph | `true` |
| `worker_decision` | Performs Beam selection on the Worker | `true` |
| `engine_driven` | Retired input; V1 always owns scheduling | Omit; old `true` warns and is discarded, `false` is rejected |
| `max_width` | Engine Beam capacity | Same as request `beam_width` |
| `max_decode_steps` | Startup capacity for effective generation length | Positive integer; at least `3` for this constrained example |

On CUDA, with `constrained_generation: false` and
`constraint_table.enabled: false`, one engine can accept per-request effective
lengths from `1` through its configured `beam.max_decode_steps` (for example,
`7`). Existing NPU startup limits still apply (`max_decode_steps <= 3`).
Increasing capacity does not change the depth of an enabled
`constraint_table_v1`: constrained requests still require exactly three
effective tokens.

#### Constraint Table parameters

| Parameter | Meaning |
| --- | --- |
| `enabled` | Must be `true` for V1 |
| `backend` | `cuda` or `npu`, matching the device |
| `max_top_k` | Candidate capacity; must be at least the candidate width |
| `candidate_width` | Candidates per Beam row (1..256); omit to follow the Beam width. Required when the Beam width exceeds 256 |
| `path` | Constraint Table `.safetensors` artifact |
| `format` | Must be `constraint_table_v1` |
| `artifact_digest` | Digest printed by `build_constraint_table.py` |
| `tokenizer_digest` | Tokenizer digest printed by the same command |

Use an absolute artifact path. Do not combine this canonical configuration with
legacy `vllm_gr_*` keys in `additional_config`.

#### Important `GRLLM` parameters

| Parameter | Purpose |
| --- | --- |
| `vllm_gr_config` | Canonical YAML path or equivalent mapping |
| `async_scheduling` | Required by the V1 EngineCore session path |
| `enable_prefix_caching` | Allows Prefill prefix reuse; correctness does not depend on it |
| `enable_chunked_prefill` | The example disables it for a fixed input case |
| `max_model_len` | Must cover the prompt plus effective generation budget |
| `max_num_batched_tokens` | Must cover the prompt when chunked Prefill is disabled |
| `max_num_seqs` | The example sets it to the fixed Beam width |
| `max_logprobs` | Must be at least the candidate width (`candidate_width`) |
| `enforce_eager` | `false` allows graph execution |

### Request parameters

V1 accepts `vllm_gr.sampling_params.BeamSearchParams`:

| Parameter | Meaning |
| --- | --- |
| `beam_width` | Must exactly equal `beam.max_width` |
| `max_tokens` | Public budget, including configured begin/end positions |
| `temperature` | Sampling temperature; the example uses `0.0` |
| `ignore_eos` | Whether stop/EOS tokens are ignored |
| `include_stop_str_in_output` | Keeps the configured boundary token in public output |
| `begin_token` | Appended to the prompt before Prefill |
| `end_token` | Stop boundary or appended to a length-finished result |
| `length_penalty` | Ranks final candidates of different lengths |
| `catalog_strategy` | Must remain `None` for this Constraint Table path |

The effective generation budget is:

```text
N = max_tokens - begin_token_position - end_token_position
```

With both OneRec boundary tokens, `max_tokens=5` represents three generated SID
tokens and an effective V1 budget of 3.

### Run the V1 example

The standalone example compiles a temporary Constraint Table, constructs the V1
engine configuration, calls `GRLLM.beam_search_v1`, and prints the leading Beam
results:

```bash
CUDA_VISIBLE_DEVICES=0 python offline_beam_search_v1.py \
  --model /path/to/OneRec-1.7B \
  --catalog /path/to/video_constraint_triples.json \
  --prompt /path/to/prompt.txt \
  --input-len 1024 \
  --generated-steps 3 \
  --beam-width 128
```

Run `python offline_beam_search_v1.py --help` to see optional memory, length
penalty, output-count, and measurement arguments.

This constrained example accepts only `--generated-steps 3` and rejects `1` or
`2` during argument validation, before model loading or artifact construction.
On NPU, this runs Prefill followed by two Decode stages and requires the
requested Beam graph shape to be captured and replayed in graph mode. The `fallbacks`
counter means Beam FULL graph was not used: native FULL is rejected, while
native PIECEWISE/NONE is retained. It does not count exclusively eager forwards.

### Optional thinking before SID generation

The same `--prompt` file can be used in three modes:

| Options | Execution |
| --- | --- |
| Neither thinking flag | Existing direct constrained SID Beam Search |
| `--enable-thinking` | Ordinary full-vocabulary LLM generation only; no catalog or SID Beam phase |
| `--cot` | Ordinary thinking generation, followed by constrained SID Beam Search |

`--cot` and `--enable-thinking` are mutually exclusive. Both require a model
that supports the requested thinking behavior. The combined mode also requires
dedicated `<think>` and `</think>` tokenizer tokens and a compatible SID catalog.
It exposes a checkpoint's capabilities; it does not add reasoning proficiency
to a checkpoint that was not trained for it.

#### CUDA setup and commands

Run from the repository root in the configured vLLM CUDA environment. Replace
the model and prompt paths; the combined workflow uses the same complete prompt
for thinking and SID generation.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
export USE_TF=0

MODEL=/absolute/path/to/OneRec_Qwen3_1.7B
CATALOG=tools/AB_Tests/constraints/video_constraint_triples.json
PROMPT=/absolute/path/to/recommendation_prompt.txt
mkdir -p results/cot

python -u examples/offline_inference/beam_search/offline_beam_search_v1.py \
  --model "$MODEL" --catalog "$CATALOG" --prompt "$PROMPT" \
  --constraint-backend cuda --batch-size 1 --beam-width 8 --generated-steps 3 \
  --cot --cot-max-tokens 1536 --max-model-len 4096 \
  --temperature 0 --top-p 1 --top-k -1 --seed 0 --print-beams 8 \
  --output-json results/cot/cuda_beam.json
```

`--batch-size 4` repeats the prompt across four independent requests. A loaded
CUDA Constraint Table uses full-vocabulary projection for ordinary generation;
eligible V1 Beam execution uses the catalog window for its SID scoring. When
the cached sliced head cannot be used, Beam can project the full vocabulary and
then select the window, with higher projection cost. The mode does not transfer
KV ownership between phases. Direct Beam retains its existing window scoring.

In this example, `--execution` selects the NPU Beam execution mode. CUDA Beam
keeps its existing graph behavior; passing `--execution eager` does not force a
CUDA Beam eager run. A graph capture or configuration alone does not establish
replay for an individual request.

#### NPU setup and commands

Run from the repository root in the configured vLLM-Ascend environment. Set
`ASCEND_ENV` to the installed CANN environment script, and replace the model and
prompt paths. All three examples below use the same input file.

```bash
ASCEND_ENV=/absolute/path/to/CANN/set_env.sh
source "$ASCEND_ENV"
# Preserve paths installed by CANN, including the Python acl module.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
export USE_TF=0

MODEL=/absolute/path/to/OneRec_Qwen3_1.7B
CATALOG=tools/AB_Tests/constraints/video_constraint_triples.json
PROMPT=/absolute/path/to/recommendation_prompt.txt
BATCH_SIZE=1
mkdir -p results/cot
python -c "import acl; print(acl.__file__)"
```

Set `BATCH_SIZE=4` to run four requests instead. This CLI repeats the supplied
input for each row, with independent row cache salts in the combined workflow.
It does not accept a separate thinking prompt.

Direct SID Beam Search, with the existing fixed-length input preparation:

```bash
python -u examples/offline_inference/beam_search/offline_beam_search_v1.py \
  --model "$MODEL" --catalog "$CATALOG" --prompt "$PROMPT" \
  --constraint-backend npu --execution eager \
  --batch-size "$BATCH_SIZE" --beam-width 128 \
  --input-len 128 --generated-steps 3 --print-beams 5
```

Ordinary thinking-only generation with constraints disabled. Plain text uses the
model's chat template; serialized Qwen chat retains its existing messages and
opens the terminal thinking prefix as described below. `--max-tokens` limits the
complete generated continuation, including any thinking and final answer:

```bash
python -u examples/offline_inference/beam_search/offline_beam_search_v1.py \
  --model "$MODEL" --prompt "$PROMPT" \
  --constraint-backend npu --execution eager \
  --batch-size "$BATCH_SIZE" --enable-thinking \
  --max-tokens 1024 --max-model-len 8192 \
  --temperature 0 --top-p 1 --top-k -1 \
  --output-json "results/cot/thinking_b${BATCH_SIZE}.json"
```

Thinking followed by SID Beam Search:

```bash
python -u examples/offline_inference/beam_search/offline_beam_search_v1.py \
  --model "$MODEL" --catalog "$CATALOG" --prompt "$PROMPT" \
  --constraint-backend npu --execution eager \
  --batch-size "$BATCH_SIZE" --beam-width 128 --generated-steps 3 \
  --cot --cot-max-tokens 1024 --max-model-len 8192 \
  --temperature 0 --top-p 1 --top-k -1 --print-beams 5 \
  --output-json "results/cot/beam_b${BATCH_SIZE}.json"
```

#### Input, budget and handoff contract

Both thinking modes retain the complete input and ignore `--input-len`. Plain
input uses the model's chat template. If that template emits a terminal empty
`<think>...</think>` span, both modes reopen it without changing earlier token
IDs. Thinking-only otherwise retains the template's output; the combined CoT
mode additionally opens a missing thinking tag and rejects nonempty, already
closed template reasoning rather than deleting it.

Qwen-style serialized ChatML is accepted either as text with real line breaks,
or as one complete JSON string whose decoded value is ChatML. JSON is decoded
exactly once, only for a string beginning with `<|im_start|>` (apart from leading
whitespace); other plain text, including quoted text, is kept as plain text.
Both modes preserve the existing messages, open the terminal empty assistant/
thinking prefix, and change only a terminal `/no_think` directive to `/think`.
This serialized path requires dedicated `<think>` and `</think>` tokenizer
tokens. Nonempty final assistant content is rejected rather than silently
removed. Input files are never modified.

Legacy bare single-line ChatML containing literal `\n` is rejected: there is no
safe way to distinguish escaped separators from body text such as `C:\new` or
an intentional literal `\n`. This includes the existing
`tests/resources/single_one_rec_prompt.txt` fixture when used directly in a
thinking mode. Supply the intended conversation with real line breaks, or
serialize that intended text with `json.dumps` instead of globally replacing
backslashes. For example:

```bash
python - <<'PY'
import json
from pathlib import Path

chatml = (
    "<|im_start|>user\n"
    r"Keep the path C:\new and the literal \n unchanged."
    "\n/no_think<|im_end|>\n<|im_start|>assistant\n"
)
Path("thinking_chatml.json").write_text(
    json.dumps(chatml, ensure_ascii=False), encoding="utf-8"
)
PY
```

Pass `--prompt thinking_chatml.json` in either thinking mode. JSON encodes a
structural newline as `\n` and a body backslash as `\\`, so the path and literal
escape above survive decoding unchanged. The direct SID Beam mode keeps its
existing fixed-length preparation and literal file-text encoding; it does not
apply this thinking-input JSON decoding or normalization.

`--cot-max-tokens` defaults to 1024 and applies only to the thinking phase.
`--generated-steps` remains the SID generation budget. The full input, thinking
budget, separator and SID boundary positions must fit `--max-model-len`; the
actual handed-off length is checked again before Beam dispatch.

The default `--cot-on-limit error` requires the model to emit `</think>` before
starting Beam. An unclosed span, including EOS without `</think>`, fails the
whole batch before the SID phase. To explicitly continue after the thinking
budget is exhausted, add `--cot-on-limit truncate` to the combined command.
Only an actual length-budget stop permits this policy: every generated token
is retained, a closing token is appended, and the report records
`thinking_truncated=true` and `forced_close=true`. This is forced closure,
not a naturally completed thought. Other incomplete stop conditions still fail.

The SID phase receives the original input token IDs, the thinking continuation
IDs and a separator. It does not decode and re-encode the continuation. Both
phases use one loaded engine, but have separate cache salts: this version
hands off token IDs and performs a fresh Beam Prefill, without cross-phase KV
reuse. The JSON output includes raw generated thinking IDs, effective handed-off
IDs, truncation status, per-row context lengths and final Beam results.

These are functional examples, with no performance threshold or recommendation
quality improvement claimed. Optional existing timing/profiling arguments are
diagnostics. Change `--execution eager` to `--execution graph` to request the
existing NPU graph path; normal runtime fallbacks remain applicable. A graph
configuration or a SID Decode replay does not prove FULL Prefill replay, and
this feature does not extend Prefill graph coverage. GPU correctness and
performance should be assessed against the target hardware and model.

### NPU graph correctness regression

The focused CPU tests cover flat EngineArgs B=1/2/4 preservation, Prefill-only
example output, native NONE/PIECEWISE/FULL shape misses, and captured-context
reuse across B=2 -> B=1 -> B=2. They do not load a model or execute NPU kernels.

On a configured Ascend server, run the opt-in public-API regression:

```bash
VLLM_GR_TEST_MODEL=/path/to/OneRec_Qwen3_1.7B \
  python -m pytest --run-slow -s -v --tb=short \
  tests/st/test_onerec_v1_npu_e2e.py::test_onerec_v1_npu_graph_shape_switch
```

This test uses W=8, N=3 and a small catalog, loads eager, native-without-Beam-FULL
and graph engines sequentially, then checks B=2 -> B=1 -> B=2 on the same graph
Worker. By default it uses two **different raw 127-token prompts**. The API adds
one SID begin token to each, giving **128+128=256 actual Prefill tokens** (not
two raw 128-token prompts, which would give 258). The test subprocess enables
independent Prefill graphs with buckets `[128, 256]`, overriding inherited
Prefill bucket settings for this bounded regression. It checks actual input
IDs/positions, request lengths, and a real `{"256": 1}` Prefill replay delta;
native compilation and Beam FULL Decode remain enabled. Captured B=2 outputs must
match the eager B=2 baseline (ordered tokens; scores within `abs=1e-4, rel=1e-5`).
The uncaptured B=1 must remain valid without Beam FULL replay; cross-B exact
determinism is not asserted. Captured context/KV/state-buffer addresses must stay
unchanged, replay must resume for the final B=2, and N=1 must finish without Decode
dispatches. This is a correctness test, not a profile or performance measurement.

#### Open precision limitation: NPU-PREFILL-PADDING-001

The exact-bucket regression narrows coverage; it **does not fix or certify padded
Prefill precision**. Clear `VLLM_GR_TEST_PADDED_PREFILL` for the default test.
The temporary layer/operator probes are not part of this submission; only small
forward metadata and real replay counters are observed, without model hooks.
The unequal-length eager-only public API test is also unchanged.

Observed on 2026-10-15 with OneRec Qwen3 1.7B, BF16, TP=1, B=2, W=8, N=3,
prefix caching enabled with independent request salts, and native compilation:

| Actual Prefill / configuration | Observed strict comparison |
| --- | --- |
| 290 tokens, independent Prefill graph disabled; Beam FULL Decode retained | Passed |
| 290 tokens, exact 290 bucket replay | Passed |
| 290 tokens, single 384 bucket or multiple buckets selecting 384 | Failed; eager/graph ranks and scores differ; request 1 shares 7/8 candidates |
| 290 tokens padded to 384, direct compiled execution without independent Prefill replay | Matches graph; still differs from eager |
| 128+128 actual-token / 256-bucket regression | Passed: both requests have identical 8/8 ordered Beams and zero eager/graph score difference |

The earliest observed mismatch is at layer 0 attention input Q/K; V is identical.
Padding/shape-dependent execution reproduces the difference without replay, but
the responsible operator is not isolated. The later layer-0 probe stopped on a
missing `vllm.unquantized_gemm` observation in eager; fusion-on/off is **unverified**.
The 290/384 evidence uses identical inputs, unlike the new equal-length regression.
Passing the new test alone cannot prove a unique root cause or resolve this issue.
Evidence logs: `npu_prefill_off_20261015_095103.log`,
`npu_prefill_exact290_20261015_104950.log`,
`npu_prefill_single384_20261015_110649.log`,
`npu_prefill_384_noreplay_20261015_113506/run.log`, and
`npu_layer0_384_20261015_144555/run.log` in the local `test_results` archive.

The server run `npu_exact128x2_20261015_150212/run.log` finished with
`1 passed, 16 warnings in 286.65s`. The native control and all three graph B=2
runs reported an actual `256` Prefill replay. Prefill and both Decode stages'
recorded arrays matched eager exactly; Decode FULL counts progressed 2 -> 4 -> 6.
The intervening B=1 produced two expected `shape_miss` fallbacks, captured buffer
addresses stayed stable, and N=1 completed without additional Decode dispatches.
Cross-B eager B=1/B=2 remained diagnostic: 8/8 candidate overlap but different
ordering, with maximum score difference 0.405815125. This is separate from same-B
eager/graph parity. The elapsed 286.65s is whole-suite time, not inference latency.
This hardware log predates diagnostic cleanup and carries no server Git SHA;
the final cleaned submission must not be described as independently NPU-retested.

Retain the original strict failing regression as an explicit opt-in (no xfail,
skip, warning-only comparison, or relaxed tolerance). Run:

```bash
VLLM_GR_TEST_MODEL=/path/to/OneRec_Qwen3_1.7B \
VLLM_GR_TEST_PADDED_PREFILL=1 VLLM_ENABLE_PREFILL_CUDAGRAPH=1 \
VLLM_GR_PREFILL_GRAPH_BUCKET_STEP=384 VLLM_GR_PREFILL_GRAPH_MAX_BUCKET=384 \
  python -m pytest --run-slow -s -v --tb=short \
  tests/st/test_onerec_v1_npu_e2e.py::test_onerec_v1_npu_graph_shape_switch
```

For a same-input exact-bucket control, change both `384` values to `290`.
Before merging, reviewers must explicitly accept this unresolved padded case;
a green bounded regression is not a claim of full Prefill parity. Production
padding behavior is unchanged by this test-scope adjustment.

### Measure latency without profiling

Use `--repeat 20 --warmup-runs 3` to time 20 complete API calls after three
untimed warmup batches, without configuring a profiler. Each call submits
`batch-size` copies of the prompt. Prefix caching stays enabled, but each row
and run uses an independent cache salt, so Prefill is included in every sample.

The `beam_benchmark` JSON log reports batch latency in milliseconds (mean, P50,
P95, min/max), requests per second, actual returned Beams per second, and every
sample. Percentiles use linear interpolation. `--output-json` optionally saves
the same data and test conditions to a file. Only the last batch's Beams are
printed. The timer covers `beam_search_v1` until output objects return; engine
startup, graph capture, warmup, pre-call prompt preparation, post-call result
inspection, statistics collection and printing are excluded. Work performed
inside the API is included. `--repeat` cannot be combined with `--profile-dir`.
Without either option, the script still runs one batch.

For NPU graph B=1/2/4, run this loop from the repository root. Each B starts a
separate engine and captures its own fixed `(B, W)` shape:

```bash
set -o pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
BENCH_RUN="$PWD/test-results/graph_latency_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCH_RUN"
for B in 1 2 4; do
  python -u examples/offline_inference/beam_search/offline_beam_search_v1.py \
    --model /path/to/OneRec_Qwen3_1.7B \
    --catalog examples/offline_inference/beam_search/interactive_constraint_triples.json \
    --prompt tests/resources/single_one_rec_prompt.txt \
    --constraint-backend npu --execution graph \
    --batch-size "$B" --beam-width 128 --input-len 128 \
    --generated-steps 3 --print-beams 5 \
    --warmup-runs 3 --repeat 20 \
    --output-json "$BENCH_RUN/graph_b${B}_w128.json" \
    2>&1 | tee "$BENCH_RUN/graph_b${B}_w128.log" || exit 1
done
```

Keep the model, prompt, generation budget, memory setting and device identical
across B values. Request throughput is `B * repeat / total_timed_seconds`; batch
latency divided by B is only amortized cost, not individual request latency.
NPU graph counters saved in the report include both warmup and measured runs;
check the captured shape and `fallbacks` when comparing graph performance.

### Per-stage Beam widths through the Python API

CUDA and NPU V1 support a retained Beam width for each generated token, plus a
separate final result limit. CUDA can additionally set a per-parent candidate
count for each stage. Omitting it preserves the existing `K=W` behavior.

For an existing NPU V1 engine `llm` and a B=1, 2, or 4 batch `prompts`, use:

```python
from vllm_gr.sampling_params import BeamSearchParams

params = BeamSearchParams(
    beam_width=128,
    dynamic_beam_widths=[128, 64, 32],
    result_width=16,
    max_tokens=5,
    begin_token="<|sid_begin|>",
    end_token="<|sid_end|>",
    temperature=0.0,
    ignore_eos=True,
    include_stop_str_in_output=True,
    length_penalty=1.0,
)
outputs = llm.beam_search_v1(prompts, params)
```

For CUDA, K and W can be configured independently after the first stage:

```python
params = BeamSearchParams(
    beam_width=4,
    dynamic_beam_widths=[2, 4, 4],  # W1, W2, W3
    per_parent_beam_widths=[2, 2, 1],        # K1, K2, K3 per parent
    result_width=2,
    max_tokens=3,
    ignore_eos=True,
)
```

The first stage requires `K_1=W_1`. Each later stage must satisfy
`W_i <= W_(i-1) * K_i`. Both W and K remain limited to 256 in this
implementation.

An explicit `per_parent_beam_widths` overrides the startup
`constraint_table.candidate_width`, subject to backend `max_top_k` capacity.
Without it, existing behavior and startup configuration restrictions remain.
HTTP clients can pass the schedule in `vllm_xargs` through Completion or Chat;
native non-GR Beam rejects an explicit schedule.

`beam_width` remains the engine capacity in `[1, 256]` and must equal
`beam.max_width`.
The schedule has exactly one integer in `[1, beam_width]` per effective generated
step N, which must equal 3 when the constraint table is enabled. Without
constraints, N can range from 1 through `beam.max_decode_steps`, subject to the
backend's existing capacity limits. Begin/end boundary positions are excluded
from the schedule. With both boundary tokens above, `max_tokens=5` means N=3:
retain 128 SA candidates, then 64 SB candidates, then 32 SC candidates.
The API appends BEGIN before Prefill; do not append it again to the input prompt.
END is handled by the existing boundary/output rules, not an extra scheduled
model step.

Prefill processes the prompt tokens and selects the first-stage winners from
one final prompt position per request. It does not forward 128 Beam model rows
for that first selection. Subsequent Decode stages use the previous retained
width: for B=4, this example forwards 512 and 256 Decode rows. Increasing
schedules such as `[32, 64, 128]` are supported within the same capacity.
All prompts within one call share the same schedule and result limit.

`result_width` is a positive integer no larger than the final retained width.
It limits the number returned per request to at most R; a catalog with fewer
valid paths may produce fewer. If omitted, R is the final schedule width, or
`beam_width` when no schedule is supplied. Setting only `result_width` limits
returned results while retaining the fixed search widths.

Use the existing NPU V1 constraints: an enabled canonical NPU Constraint Table,
`async_scheduling=True`, `enable_chunked_prefill=False`, token/text prompts,
`ignore_eos=True`, and `catalog_strategy=None`. Existing TP/PP/DP/DCP/PCP=1,
FP16/BF16 and unsupported-model restrictions still apply. These options do not
add concurrent sessions or per-prompt schedules. Engine processes must use the
same code and schema revision across the frontend, EngineCore, and Workers.
Stop and restart them together when deploying this change. Mixed-revision
operation and rolling upgrades are unsupported for **all** Beam requests,
including fixed-width requests: the appended array-encoded schedule fields are
sent even when their values are `None`, and an older strict decoder rejects them.
Matching request-contract version numbers alone do not establish compatibility.

This path uses a fixed generated-token horizon: `ignore_eos=True` becomes
`ignore_stop_tokens=True` at engine admission. Surviving paths within a request
have the same final length, so truncating to `result_width` before length
normalization preserves score order. Variable-length early stopping is not
supported by this dynamic-width implementation.

For graph execution, construct `GRLLM` with the canonical engine configuration
above, selecting `constraint_table.backend: npu`, and set these Beam fields
before engine initialization:

```yaml
beam:
  execution_mode: v1
  graph_enabled: true
  graph_batch_size: 4
  graph_widths: [64, 128]
  worker_decision: true
  max_width: 128
  max_decode_steps: 3
```

Retain the existing schema version, attention backend, Constraint Table artifact
path/digests and `constraint_table.max_top_k: 128`. `graph_widths` lists Decode
**input** widths, so this schedule needs 128 and 64, not its final output width
32. Use `GRLLM(..., async_scheduling=True, enable_chunked_prefill=False,
enforce_eager=False, vllm_gr_config=config)` with enough sequence and token
capacity for the batch. The engine also captures `max_width`; omitting
`graph_widths` keeps the previous startup capture of that width alone.

When Beam graphs are enabled, at most **8 distinct widths** may be captured,
including the implicitly added `max_width`. For example, 8 explicit widths
that omit `max_width` would require 9 captures and are rejected before engine
startup. Listing `max_width` explicitly counts it once. This conservative
resource policy bounds the number of retained banks and graphs; it does not
guarantee a configuration fits device memory. Disabling Beam graphs does not
apply this capture-count limit.

Beam FULL replay requires an exact captured `(B, input_width)` shape. A shape
miss rejects Beam FULL while retaining native PIECEWISE/NONE execution; fallback
does not necessarily mean eager-only execution. Additional captured widths keep
separate contiguous KV banks and increase startup time and memory use. The
suffix K/V bank storage alone scales with the sum of the captured widths;
capturing widths 1 through 128 would require 64.5 times the suffix storage of
one width-128 bank and is rejected by the count limit. Graphs, scratch buffers,
and activations require additional memory. Capture only the Decode input
widths needed by the schedules you intend to serve; uncaptured widths retain
the native fallback path described above. The
existing padded-Prefill precision limitation remains independent of Decode
widths; set `VLLM_ENABLE_PREFILL_CUDAGRAPH=0` when isolating Decode parity for
padded or mixed prompt lengths.

Narrower intermediate widths intentionally prune the search; later expansion
cannot recover an earlier discarded prefix. Results and low-precision model
scores can change as execution shapes change. Validate application Recall/NDCG
and latency for the selected schedule; catalog validity and implementation
parity alone do not establish unchanged recommendation quality.

### Profile inference

Add `--profile-dir /path/to/traces` to enable the vLLM Worker profiler. Collection
starts after engine initialization and unprofiled warmup, and stops before graph
statistics and Beam output printing. Without this option, no profiler is
configured or started; use `--repeat` for separate latency measurements.

- `--profile-warmup-runs` defaults to 1; use 0 to include the first inference.
- `--profile-runs` defaults to 1 and collects complete batches, each with one
  Prefill and `generated-steps - 1` Decode stages. Only the last result is printed.
- Prefix caching stays enabled, but separate cache salts for every row and run
  prevent warmup or previous runs from skipping Prefill.
- Stack, shape and memory recording are disabled to reduce profiling overhead.
  Traces cover the Worker rather than the frontend, and are not gzip compressed.

For NPU B=2/W=128, run from the repository root:

```bash
set -o pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PROFILE_RUN="$PWD/test-results/bs2_bw128_profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PROFILE_RUN/logs"
python -u examples/offline_inference/beam_search/offline_beam_search_v1.py \
  --model /path/to/OneRec_Qwen3_1.7B \
  --catalog examples/offline_inference/beam_search/interactive_constraint_triples.json \
  --prompt tests/resources/single_one_rec_prompt.txt \
  --constraint-backend npu --execution graph \
  --batch-size 2 --beam-width 128 --input-len 128 \
  --generated-steps 3 --print-beams 128 \
  --profile-dir "$PROFILE_RUN/profile/graph" \
  --profile-warmup-runs 1 --profile-runs 5 \
  2>&1 | tee "$PROFILE_RUN/logs/graph.log"
```

Do not combine this with `msprof` collection. Graph statistics include warmup and
profile runs; their cumulative dispatch count need not be 2. Confirm the trace
contains Worker model operations and actual graph replay, not just frontend
events. Use separate unprofiled repeated runs for latency/throughput measurements.

### NPU per-stage Beam widths from the CLI

The CLI exposes the Python request options above as `--dynamic-beam-widths`
and `--result-width`. They require `--constraint-backend npu`; CUDA examples
keep their fixed-width behavior. There must be exactly `--generated-steps`
comma-separated widths, each between 1 and `--beam-width`. `--beam-width` remains
the engine capacity, not the first-stage width.

`--result-width R` returns at most R valid Beams per request, independently of
`--print-beams`, which only limits how many returned Beams are printed. For
example, R=32 with `--print-beams 5` still returns up to 32 Beams per request and
prints at most five. If R is omitted it defaults to the last stage width, or to
the capacity when no schedule is supplied. Giving R alone retains fixed search
widths. An empty legal continuation can leave fewer valid results than R.

For a schedule with Decode stages, the example configures
`beam.graph_widths = sorted(set(schedule[:-1]))` before starting the engine.
These are previous-stage winner counts, hence Decode input widths; the engine
also captures the capacity width. For `128,64,32`, the required Decode inputs
are 128 and 64. The final 32 is a selection width, not another model-forward
stage. With `--execution graph`, the example checks that all required `(B, W)`
shapes were captured and that Decode replay occurred. For a single generated
step, selection completes in Prefill and no Decode replay is required.

#### Prepare paths and the NPU device

Run the following blocks from the repository root in the same Bash shell.
Replace `MODEL` with the installed model directory and choose the visible NPU
card for the run. The catalog and prompt below are tracked repository resources;
the catalog tokens must match the selected model's tokenizer.

```bash
set -o pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_ENABLE_PREFILL_CUDAGRAPH=0
MODEL=/absolute/path/to/Qwen3-0.6B-OneRecVocab
CATALOG=tools/AB_Tests/constraints/video_constraint_triples.json
PROMPT=tests/resources/single_one_rec_prompt.txt
DYNAMIC_RUN="$PWD/test-results/npu_dynamic_beam_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DYNAMIC_RUN/logs" "$DYNAMIC_RUN/json" "$DYNAMIC_RUN/profile"
```

The directory creation must precede `tee`: the shell opens its log before Python
runs. These commands disable independent Prefill graphs to isolate Decode;
Beam Decode graphs remain enabled by `--execution graph`. Keep the Prefill
setting and all other conditions unchanged when comparing schedules. Enabling
Prefill graphs retains the precision boundary described earlier.

#### Benchmark B=1 and B=4

These commands time 20 complete batches after three untimed warmups. They use
N=3 with the script's existing begin/end boundaries (`max_tokens=5`), capacity
128, the `128,64,32` schedule and R=32. `--input-len 128` includes the BEGIN token
appended by the API. All batch rows use the same prepared prompt with independent
cache salts; each timed call includes Prefill.

B=1:

```bash
python -u examples/offline_inference/beam_search/offline_beam_search_v1.py \
  --model "$MODEL" --catalog "$CATALOG" --prompt "$PROMPT" \
  --constraint-backend npu --execution graph \
  --batch-size 1 --beam-width 128 --input-len 128 --generated-steps 3 \
  --dynamic-beam-widths 128,64,32 --result-width 32 --print-beams 5 \
  --warmup-runs 3 --repeat 20 \
  --output-json "$DYNAMIC_RUN/json/b1_narrow.json" \
  2>&1 | tee "$DYNAMIC_RUN/logs/b1_narrow.log"
```

B=4:

```bash
python -u examples/offline_inference/beam_search/offline_beam_search_v1.py \
  --model "$MODEL" --catalog "$CATALOG" --prompt "$PROMPT" \
  --constraint-backend npu --execution graph \
  --batch-size 4 --beam-width 128 --input-len 128 --generated-steps 3 \
  --dynamic-beam-widths 128,64,32 --result-width 32 --print-beams 5 \
  --warmup-runs 3 --repeat 20 \
  --output-json "$DYNAMIC_RUN/json/b4_narrow.json" \
  2>&1 | tee "$DYNAMIC_RUN/logs/b4_narrow.log"
```

Each invocation starts a separate engine. The B=1 Decode inputs contain 128 then
64 rows; B=4 contains 512 then 256. For a comparison at the same return limit,
change only the schedule to `128,128,128`, keep R=32, and use different output
filenames. This compares two search schedules, not equal search coverage. Narrow
widths can alter the returned set and low-precision scores; evaluate the quality
tradeoff separately. Re-run with `--execution eager` when comparing Decode graph
and eager execution under otherwise identical conditions.

#### Profile B=1 and B=4 separately

Use separate invocations for profiling and latency measurement. `--repeat`
cannot be combined with `--profile-dir`; `--output-json` is a benchmark option
and requires `--repeat`. The following collect five complete batches after one
unprofiled warmup, and print only the last batch's results.

B=1:

```bash
python -u examples/offline_inference/beam_search/offline_beam_search_v1.py \
  --model "$MODEL" --catalog "$CATALOG" --prompt "$PROMPT" \
  --constraint-backend npu --execution graph \
  --batch-size 1 --beam-width 128 --input-len 128 --generated-steps 3 \
  --dynamic-beam-widths 128,64,32 --result-width 32 --print-beams 5 \
  --profile-dir "$DYNAMIC_RUN/profile/b1_narrow" \
  --profile-warmup-runs 1 --profile-runs 5 \
  2>&1 | tee "$DYNAMIC_RUN/logs/b1_narrow_profile.log"
```

B=4:

```bash
python -u examples/offline_inference/beam_search/offline_beam_search_v1.py \
  --model "$MODEL" --catalog "$CATALOG" --prompt "$PROMPT" \
  --constraint-backend npu --execution graph \
  --batch-size 4 --beam-width 128 --input-len 128 --generated-steps 3 \
  --dynamic-beam-widths 128,64,32 --result-width 32 --print-beams 5 \
  --profile-dir "$DYNAMIC_RUN/profile/b4_narrow" \
  --profile-warmup-runs 1 --profile-runs 5 \
  2>&1 | tee "$DYNAMIC_RUN/logs/b4_narrow_profile.log"
```

Worker trace files and their rank/run subdirectories are under the supplied
profile directory. Startup and output printing are outside collection. Do not
interpret profiled timings as benchmark latency or combine this with `msprof`.
Graph counters include both warmup and collected batches.

#### Optional increasing schedule at capacity 256, B=1

This example uses eager execution within the B=1 capacity-256 validation scope;
it does not claim B=4 capacity-256 hardware coverage. Reuse the same path/device
setup:

```bash
python -u examples/offline_inference/beam_search/offline_beam_search_v1.py \
  --model "$MODEL" --catalog "$CATALOG" --prompt "$PROMPT" \
  --constraint-backend npu --execution eager \
  --batch-size 1 --beam-width 256 --input-len 128 --generated-steps 3 \
  --dynamic-beam-widths 32,64,256 --result-width 32 --print-beams 5 \
  --warmup-runs 3 --repeat 20 \
  --output-json "$DYNAMIC_RUN/json/b1_capacity256_wide.json" \
  2>&1 | tee "$DYNAMIC_RUN/logs/b1_capacity256_wide.log"
```

Its Decode input widths are 32 and 64; this eager example does not request Beam
graph capture. The final stage may retain up to 256 candidates, while the API
returns at most 32 per request. Expanding a later width cannot recover SA/SB prefixes pruned by
an earlier stage. A wider final stage is not a guarantee of baseline search
quality or of R legal final paths.

#### Benchmark JSON and returned-result metadata

The `beam_benchmark` JSON log and `--output-json` file use the same benchmark
payload. The new fields are emitted only when explicitly supplied:

| Field | Meaning |
| --- | --- |
| `beam_width` | Engine capacity |
| `dynamic_beam_widths` | Parsed retained widths, e.g. `[128, 64, 32]`; absent for an omitted schedule |
| `result_width` | Explicit R; absent when omitted, in which case derive the default from the final width |
| `batch_size`, `input_len`, `generated_steps` | Input and generation conditions |
| `repeat`, `warmup_runs` | Measured and untimed batch counts |
| `batch_latency_ms` | Mean, P50, P95, min and max batch API latency |
| `runs[].returned_beams` | Actual total valid Beams returned across the whole batch in that run |
| `requests_per_second` | Timed request throughput |
| `returned_beams_per_second` | Timed throughput based on actual returned counts, not assumed B times R |
| `npu_beam_graph` | Graph-run counters, including captured shapes and fallbacks; present for NPU graph benchmarking |

The benchmark JSON contains conditions and timing/count metadata; it is not a
serialized list of result sequences. Standard output separately prints
`request=... returned_beams=...` for each request in the last batch, followed by
up to `--print-beams` JSON records containing `request`, `rank`,
`cumulative_logprob`, `finish_reason`, `stop_reason`, `sid_token_ids`, and
`sid_tokens`. Printing fewer ranks does not reduce the recorded actual return
counts. The SID fields are the script's trailing generated/boundary token slice.

Compare latency only with matching model/catalog/prompt, device, B, capacity,
input length, return limit, Prefill setting and measurement setup. Keep search
quality and the existing NPU legal-set score contract separate from performance
claims.

### Retired scheduling control

`beam.engine_driven`, `vllm_gr_beam_engine_driven`, the Python
`beam_engine_driven` argument, and `--beam-engine-driven` no longer select an
execution path. Omit them. For migration, `true` is accepted with a warning
and discarded; `false` / `--no-beam-engine-driven` is rejected because the
frontend step loop no longer exists. Null remains equivalent to omission.
The accepted legacy boolean strings follow the same rules; other types and
invalid strings fail validation. This option is never serialized to workers
or auto-enabled by platform defaults. `worker_decision` remains a separate
control, and an explicit False value is still preserved.
