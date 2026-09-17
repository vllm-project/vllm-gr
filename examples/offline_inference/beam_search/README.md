# Offline Beam Search Demo

This directory contains a standalone example demonstrating how to run beam search in **offline mode** using the `vllm-gr` framework for generative recommendation models.

It contains two entrypoints:

- `offline_beam_search.py` for the legacy Beam path.
- `offline_beam_search_v1.py` for the EngineCore-owned asynchronous V1 path.

The V1 example accepts a model, Constraint Table triples document, and prompt.
It compiles a temporary CUDA Constraint Table artifact, builds the canonical V1
configuration, calls `GRLLM.beam_search_v1`, and prints the leading Beam results.
Run `python offline_beam_search_v1.py --help` for its arguments.

## Installation

Follow the instructions in  
[`docs/quickstart.md`](../../../docs/quickstart.md)  
to install the required dependencies and set up your environment.

## Legacy example

This example shows how to run offline beam search on the **OneRec‑1.7B** model using `vllm-gr`.

```bash
python offline_beam_search.py \
    --model OpenOneRec/OneRec-1.7B \
    --beam_width 128 \
    --compare latency
```

Key arguments:

- `--model`: Name or path of the model to load (required).
- `--beam_width`: Beam width used during beam search (default: 128).
- `--compare`: Comparison mode against vanilla vLLM. Options: none, latency, output, both (default: none).

### Expected result

The script prints the latency of running the predefined input through vllm-gr, and—if comparison is enabled—also reports the latency of vanilla vLLM.

Example:

```bash
python offline_beam_search.py --model OpenOneRec/OneRec-1.7B --compare latency
...
...
Vanilla vLLM with BW=128 took 4.882346 seconds
vLLM-gr with BW=128 took 0.627150 seconds
```

## Beam Search V1

Beam Search V1 is the EngineCore-owned asynchronous Beam path. The parameter
combinations described here follow the working local profiling and A/B scripts.

### V1 versus the legacy path

| Area | Legacy `beam_search` | V1 `beam_search_v1` |
| --- | --- | --- |
| Engine mode | `beam.execution_mode: legacy` | `beam.execution_mode: v1` |
| Scheduling | The frontend participates in every Beam step | The request is submitted once and EngineCore schedules Prefill and Decode |
| Beam decision | May use the frontend/legacy decision path | Runs on the Worker with persistent GPU Beam state |
| Intermediate data | Per-step results return through the legacy path | Intermediate Beam state stays on the device |
| Final processing | Legacy materialization path | GPU ranks and packs final candidates, then copies one compact payload to the host |
| Public output | Materialized by the legacy frontend | Token/text fields are lazily materialized on first access |
| Constraints | Supports the existing legacy configurations | Requires an enabled canonical CUDA Constraint Table |
| Width | Uses the legacy width behavior | Request width must equal the engine's fixed width |
| Generation budget | Uses the general legacy behavior | Effective generation budget must be 1, 2, or 3 |

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

Execution mode is selected when the engine is initialized. Do not call
`beam_search_v1` on a legacy engine, call `beam_search` on a V1 engine, or mix
ordinary/legacy Beam requests with V1 requests in the same engine.

### Current restrictions

The current V1 implementation requires:

- CUDA and an enabled canonical CUDA Constraint Table.
- `async_scheduling=True`.
- One prompt per call.
- TP, PP, DP, DCP, and PCP sizes equal to 1.
- Request `beam_width == beam.max_width`.
- Effective generation budget `N` in `{1, 2, 3}`.
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
and compiles a temporary CUDA artifact automatically. For a reusable artifact,
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

The standalone example builds the following configuration in memory. The same
mapping can be stored as a YAML file for other applications:

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
| `engine_driven` | Lets EngineCore chain stages without frontend step RPCs | Omit for the CUDA default, or set `true` |
| `max_width` | Fixed runtime Beam width | Same as request `beam_width` |
| `max_decode_steps` | Maximum effective generation budget | `1`, `2`, or `3` |

#### Constraint Table parameters

| Parameter | Meaning |
| --- | --- |
| `enabled` | Must be `true` for V1 |
| `backend` | Must currently be `cuda` |
| `max_top_k` | Candidate capacity; use the configured Beam width |
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
| `max_logprobs` | Must be large enough for the configured Beam width |
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
penalty, and output-count arguments.
