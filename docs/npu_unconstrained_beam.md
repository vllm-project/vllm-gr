# Unconstrained Beam Search V1 on NPU

NPU `GRLLM.beam_search_v1()` can select candidates from the model's full output
vocabulary without a catalog or a Constraint Table artifact. This is Beam Search,
separate from ordinary `generate()` sampling. It does not constrain generated
tokens to valid recommendation SIDs. An optional SID stage domain restricts the
token families without validating catalog membership; use the existing catalog
path when generated triples must identify known items.

## Configuration and Python entry point

Use the main branch's supported vLLM 0.30.0 / vLLM-Ascend 0.30.0rc2 vendor stack
described in the [Ascend quickstart](quickstart-ascend.md), including its NPU Beam
attention dependencies. Keep the matching CANN, PyTorch, torch-npu, and Triton
versions from that environment. The full-vocabulary selector uses PyTorch
operations and adds no operator compilation step, but the existing Beam attention
and KV operators are still required.

From the repository root, select an available device and initialize the matching
CANN environment before starting Python. For example:

```bash
source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENABLE_PREFILL_CUDAGRAPH=1
```

Save the following as `npu_beam.py`, replace the model path, and run it as a
script. The `main` guard prevents engine construction from repeating in spawned
worker processes. The canonical configuration explicitly disables constraints:

```python
from vllm_gr.entrypoints.gr import GRLLM
from vllm_gr.sampling_params import BeamSearchParams


def main() -> None:
    batch_size = 4  # Use 1 for a single request.
    config = {
        "schema_version": 1,
        "constrained_generation": False,
        "attention_backend": "CUSTOM",
        "beam": {
            "execution_mode": "v1",
            "worker_decision": True,
            "graph_enabled": True,
            "graph_batch_size": batch_size,
            "max_width": 128,
            "max_decode_steps": 3,
        },
        "constraint_table": {
            "enabled": False,
            "backend": "npu",
            "max_top_k": 128,
        },
    }

    with GRLLM(
        model="/absolute/path/to/model",
        vllm_gr_config=config,
        async_scheduling=True,
        enforce_eager=False,
        enable_chunked_prefill=False,
        block_size=128,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=128,
        max_logprobs=128,
    ) as llm:
        tokenizer = llm.get_tokenizer()
        inputs = [
            {"prompt_token_ids": tokenizer.encode("Recommend the next item.")}
            for _ in range(batch_size)
        ]
        outputs = llm.beam_search_v1(
            inputs,
            BeamSearchParams(beam_width=128, max_tokens=3, ignore_eos=True),
            output_mode="full",
        )
        for request_output in outputs:
            for sequence in request_output.sequences[:5]:
                print(sequence.tokens, sequence.cum_logprob)


if __name__ == "__main__":
    main()
```

Run the saved script after the environment setup above:

```bash
python -u npu_beam.py
```

No path, artifact digest, tokenizer digest, vocabulary offset, or vocabulary slice
is needed. An enabled constraint table or a vocabulary slice contradicts
`constrained_generation=False` and remains an error. The standalone
`offline_beam_search_v1.py` catalog-based SID example continues to require its
catalog; the example above uses the public Python API directly.

`output_mode="full"` returns the usual Beam sequence objects. Omitting the flag
uses the existing compact output format. No new output type is introduced.

## Catalog-free SA/SB/SC generation

For SID generation that permits every SA/SB/SC combination, enable the stage
domain inside `main()`, after building `config` and before constructing `GRLLM`:

```python
config["beam"]["sid_stage_vocab"] = True
```

Use the checkpoint's SID prompt, including its SID-begin marker when required.
Prefill selects SA, the first Decode selects SB, and the second Decode selects
SC. The engine reads the exact `<s_a_N>`, `<s_b_N>`, and `<s_c_N>` token IDs from
the tokenizer. It validates the IDs and retains all members of each family;
counts and ID continuity are not hardcoded. For a checkpoint with 8192 tokens
per family, each stage projects to 8192 columns. No catalog artifact or
fully-connected triples table is constructed, and the constrained TopK operator
is not called.

The existing NPU stage projection owns three weight caches and refreshes them
through the existing weight-update lifecycle. The ordinary generation and
startup dummy paths retain the full LM head. This option requires NPU Beam V1,
worker decisions, `constrained_generation=False`, and at most three generated
tokens. It is disabled by default; CUDA and catalog-based behavior are unchanged.

`VLLM_GR_NPU_STAGE_LM_HEAD=0` disables only the projection optimization. The SID
stage domains, token-ID mapping, and score semantics remain enabled, providing
a full-head reference for the optimized projection. Supported models use the
stage projection by default when the environment variable is unset or `1`.

Set this environment variable before starting Python; changing it does not
reconfigure an existing engine. With `sid_stage_vocab=True` in the saved script,
run each reference or optimized case in a new process:

```bash
VLLM_GR_NPU_STAGE_LM_HEAD=0 python -u npu_beam.py
VLLM_GR_NPU_STAGE_LM_HEAD=1 python -u npu_beam.py
```

Setting `VLLM_GR_NPU_STAGE_LM_HEAD=1` alone does not select the SID stage domain
or override model eligibility checks. With `sid_stage_vocab=False`, catalog-free
Beam selection still uses the full model vocabulary.

In this mode, each parent is normalized across the complete current SID family,
not just its TopK candidates. This is equivalent to a full projection followed
by masking out other families **before** softmax. It is not equivalent to the
default full-vocabulary probabilities: changing the denominator can change
cross-parent Beam rankings. Compare sliced and full heads under the same stage
domain when checking correctness. Catalog membership is not guaranteed.

## Supported execution boundaries

- The initial NPU implementation supports one, two, or three generated tokens,
  with `ignore_eos=True`. It is not an arbitrary-length text Beam decoder.
- Existing NPU restrictions remain: a plain text decoder, FP16/BF16, ModelRunner
  V1, asynchronous scheduling, and TP/PP/DP/DCP/PCP all equal to one.
- Logical batch sizes follow the existing NPU Beam contract (1 through 8). The
  configured maximum width must equal the request's `beam_width`.
- The configured `graph_batch_size` and Decode input width identify exact graph
  shapes. Configure `graph_widths` for additional dynamic input widths, or use the
  existing eager fallback when a matching graph is unavailable. A one-token
  session ends in Prefill and needs no Decode graph.
- Dynamic shrinking and expanding widths use the existing parent/KV ownership
  path. For example, with capacity 128, `dynamic_beam_widths=[128, 64, 32]` and
  `result_width=32` are valid. To capture both Decode input widths, add
  `"graph_widths": [64, 128]` to `beam`; the final 32 has no following Decode.
- Leave `constraint_table.candidate_width` unset for dynamic schedules. Independent
  per-parent candidate widths remain CUDA-only. An explicitly configured scalar
  candidate width must match every retained stage width on this NPU path.

Prefill and Decode reuse the existing model graph machinery. Candidate selection
and Beam decisions retain their existing scheduling boundary outside the captured
model forward. Enabling the Prefill flag does not guarantee every input shape can
use a FULL graph; use the execution diagnostics/profile to verify the actual
dispatch mode.

## Candidate scores

In the default full-vocabulary mode, each parent row is normalized in FP32 over its complete finite model vocabulary
before TopK truncation. A temperature of zero retains the established Beam
meaning of unscaled logits (effective temperature one); positive temperatures
scale the logits before normalization. The same token's logprob must not change
merely because the requested candidate count changes.

The initial selector is a native PyTorch baseline that sorts the complete
vocabulary. Its purpose is functional support; no performance improvement over
constrained selection is implied. Large vocabularies and wide Beam batches can
make this selection step substantial.

NaN and infinite logits are excluded. When fewer than K finite tokens are
available, remaining slots have `valid_mask=False` and a score of negative
infinity. A completely invalid row produces no valid candidates. Exact score ties
prefer the smaller token ID. These guards make the candidate contract explicit;
they do not imply bitwise parity across different model kernels or devices.

Run the focused regressions in the complete Ascend environment:

```bash
python -m pytest tests/test_npu_unconstrained_beam.py \
  tests/test_npu_unconstrained_runner_config.py \
  tests/test_npu_sid_stage_vocab.py \
  tests/test_npu_sid_stage_config.py \
  tests/test_npu_sid_stage_loading.py \
  tests/test_npu_stage_logits.py \
  tests/test_beam_search_v1_input_processor.py \
  tests/test_npu_model_runner_patch.py \
  tests/kernels/test_unconstrained_topk_npu.py \
  tests/kernels/test_sid_stage_topk_npu.py
```

## Version and validation scope

This feature is ported to the main branch's vLLM 0.30 API. Earlier functional and
performance measurements on vLLM 0.22.1 / vLLM-Ascend 0.22.1rc1 are historical
results, not validation of this port. Check the PR's verification results for the
actual environment and configurations tested on 0.30; no performance gain is
implied by the API migration.
