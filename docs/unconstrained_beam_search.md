# Full-vocabulary Beam Search V1

## Semantics and configuration

Canonical `constraint_table.enabled` selects the candidate source for the
engine lifetime. It is not a per-request switch:

| Setting | Candidates | Vocabulary and score normalization |
| --- | --- | --- |
| `true` | Existing canonical Constraint Table | Existing constrained behavior |
| `false` | Full model vocabulary | FP32 log-softmax over the complete vocabulary |

Disabled means **no catalog, no SID-only vocabulary, no a/b/c layer mask**.
It does not mean unlimited generation length. This first implementation keeps
the V1 runtime limit of 1–3 newly generated tokens and supports the official
OpenOneRec recommendation Beam profile: `temperature=0`, `ignore_eos=False`,
`length_penalty=1`. Other parameter combinations fail admission explicitly.

Example engine configuration (pass as `GRLLM(vllm_gr_config=...)`):

```yaml
schema_version: 1
attention_backend: CUSTOM
beam:
  execution_mode: v1
  worker_decision: true
  max_width: 128
  max_decode_steps: 3
  graph_enabled: false
  graph_batch_size: 1
constraint_table:
  enabled: false
```

No table path, digests, `max_top_k`, catalog, or legacy vocabulary-slice options
are needed. An omitted canonical `constraint_table` also means disabled.
Legacy configuration remains on its existing path; use canonical configuration
to opt into this mode. Restart the engine to change the candidate source.

Use the existing offline `GRLLM.beam_search_v1()` or Python async V1 interface.
This change does not add a new OpenAI HTTP endpoint. Supply
`async_scheduling=True` and `enable_chunked_prefill=False` to the engine.
For eager operation also set `enforce_eager=True`. For graph operation set
`graph_enabled: true`, `enforce_eager=False`, and the actual `graph_batch_size`.
NPU graph execution retains the existing XLLM attention requirements.

```python
from vllm.inputs import tokens_input
from vllm_gr.sampling_params import BeamSearchParams

# prompt_token_ids must already include the official <|sid_begin|> token.
# Do not set begin_token or end_token: both are rejected in full-vocab mode.
params = BeamSearchParams(
    beam_width=128,
    max_tokens=3,
    temperature=0.0,
    ignore_eos=False,
    length_penalty=1.0,
)
outputs = llm.beam_search_v1([tokens_input(prompt_token_ids)], params)
```

`max_tokens=3` counts only new tokens, not the SID-begin token in the prompt.
There is no artificial SID-end suffix. Generated EOS is retained in raw token
outputs, matching native offline Beam; only the tokenizer EOS terminates a
Beam, not additional generation-config stop IDs. Text rendering is not the
golden comparison boundary.

Each parent supplies up to `2 * beam_width` candidates. EOS candidates enter
the completed set immediately and never resume generation. Non-EOS candidates
compete for W live slots. Final ordering uses cumulative logprob divided by
`(prompt length + generated length - trailing EOS) ** length_penalty`; returned
scores remain **unnormalized cumulative logprobs**. Full-vocabulary normalization
includes non-SID tokens, even when every winning token happens to be a SID.
Exact token ordering at tied scores is not portable across accelerator top-k
implementations.

## Runtime boundaries

- Beam width remains 1–256, fixed at engine startup. Candidate width is
  independent (up to 512); model Forward, attention and KV rows remain B×W.
- Batch sizes remain 1, 2 or 4. Existing V1 restrictions remain: text decoder,
  FP16/BF16, TP/PP/DP/DCP/PCP=1, no chunked Prefill, speculative decoding,
  quantization, LoRA or microbatching.
- CUDA ModelRunner V1/V2 share the full-vocabulary selection and persistent
  state path. NPU V1 uses device-side active/completed state and request-local
  KV gather. An all-finished NPU member can occupy masked rows until its batch
  reaches the configured budget; it cannot generate further output tokens.
- Forward graph capture remains separate from LM-head/log-softmax/top-k.
  Request state, EOS and validity masks are rebound without changing graph
  input addresses. Model-local projection policy explicitly disables legacy
  slicing for unconstrained engines.
- Table requests retain wire protocol 1. Full-vocabulary requests use protocol
  2 with `candidate_mode="full_vocab"` and `constraint_resource_id=null`.
  Frontend, Scheduler and Workers must be upgraded together. A mode mismatch
  fails closed. Native progress placeholders ignore EOS; only Beam state owns
  stopping and generated output.

## Validation on another CUDA / Ascend machine

CPU tests do not qualify accelerator execution or real-model numerical parity.
Use matching model/tokenizer revisions, dtype and raw prompt IDs in both runs.
Prepare `prompts.json` as a list of 1, 2 or 4 nonempty token-ID lists, each with
the official SID-begin token already appended. No prompt alteration, filtering,
or catalog repair is performed by the test tool.

Run the baseline and GR in **separate processes**. The native process disables
the GR plugin before importing vLLM. Keep the needed platform plugin on Ascend
using `--native-plugins ascend` (use the installed entry-point name).

Keep two native baselines when qualifying official OneRec compatibility:

- Run `--implementation native` with the official model's **vLLM 0.7.3**
  environment to detect differences from the original recommendation runner.
- Run the same command with native **vLLM 0.22.1** to isolate changes introduced
  by GR from upstream version/backend differences. The GR process itself
  requires its matching 0.22.1 environment.

The native command uses the shared `{"prompt_token_ids": [...]}` input schema
without `cache_salt`, and detects `EngineArgs` fields before supplying
`async_scheduling=False`. It does not require newer `tokens_input` APIs in the
0.7.3 environment. Use each environment's Python executable to invoke the same
script; save the two native reports separately. Matching the 0.22.1 baseline
alone is not evidence of matching the official 0.7.3 runner.

```bash
python tests/system/run_full_vocab_beam_alignment.py run \
  --implementation native --model /models/OneRec-1.7B \
  --prompts prompts.json --beam-width 128 --steps 3 --output native.json

python tests/system/run_full_vocab_beam_alignment.py run \
  --implementation gr --model /models/OneRec-1.7B \
  --prompts prompts.json --beam-width 128 --steps 3 --output gr-eager.json

python tests/system/run_full_vocab_beam_alignment.py compare native.json gr-eager.json

python tests/system/run_full_vocab_beam_alignment.py run \
  --implementation gr --model /models/OneRec-1.7B \
  --prompts prompts.json --beam-width 128 --steps 3 \
  --execution graph --output gr-graph.json

python tests/system/run_full_vocab_beam_alignment.py compare native.json gr-graph.json
```

The report preserves ordered raw generated IDs including EOS, cumulative
logprobs, finish reasons, prompt hash, versions, and graph statistics. Repeated
requests exercise owner release/reuse. A graph run with generated Decode tokens
must report an actual replay, not just successful capture. Comparison is strict
on raw IDs/order and uses an explicit score tolerance (`--atol`, default 1e-4).
Investigate numerical or tied-score differences instead of silently filtering
invalid SIDs or loosening tolerances.

Minimum release matrix: CUDA V1 and V2 (`VLLM_USE_V2_MODEL_RUNNER=1`), NPU V1;
eager and graph; W=1/4/128/256; T=1/2/3; B=1/2/4; FP16 and BF16.
Include prompts with early EOS, normal three-token SID results, non-SID winners,
different prompt lengths, and repeated identical prompts. Retain both JSON
reports plus environment/model revision information. Cross-platform performance
and memory profiling remain hardware acceptance work, not established by CPU
tests.

Relevant local tests:

```bash
python -m pytest -q tests/v1/worker/test_full_vocab.py \
  tests/v1/worker/test_torch_beam_state.py tests/v1/worker/test_npu_full_vocab.py \
  tests/test_beam_search_v1_input_processor.py tests/contract/test_beam_batch_contracts.py \
  tests/system/test_full_vocab_beam_alignment.py
```

Run the CUDA kernel cases in `tests/kernels/test_beam_search_runner_cuda.py`
on a CUDA host; these include K=2W, W=256, EOS and parent-history cases.
On Ascend run `tests/kernels/test_full_vocab_beam_npu.py` for full-vocabulary
scores, persistent EOS state and arbitrary-parent KV gather/address checks.
