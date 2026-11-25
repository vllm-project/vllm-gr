# Unconstrained Beam search on NPU

<a id="unconstrained-beam-search-v1-on-npu"></a>

Use `GRLLM.beam_search()` on NPU to rank candidates from the model's
full output vocabulary. The example uses the Beam API with no catalog artifact.

!!! note "Catalog membership"

    The optional SID stage domain restricts token families. Use a
    [constraint table](../configuration/constraint_tables.md) when a SID must
    correspond to an item in your catalog.

## Configuration and Python entry point

Prepare the vendor runtime and Beam attention dependencies in the
[NPU installation guide](../getting_started/installation/npu.md). Beam
search on NPU uses [ModelRunner V1](../configuration/README.md#modelrunner-selection).
The full-vocabulary selector uses PyTorch and reuses the Beam attention and
KV operators from that setup.

From the repository root, select an available device and initialize the matching
CANN environment before starting Python. For example:

```bash
source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
```

Save the following as `npu_beam.py`, replace the model path, and run it as a
script. The `main` guard prevents engine construction from repeating in spawned
worker processes. The canonical configuration explicitly disables constraints:

```python
from vllm_gr import GRLLM
from vllm_gr.sampling_params import BeamSearchParams


def main() -> None:
    batch_size = 1
    config = {
        "constraint_mode": "none",
        "worker_decision": True,
        "beam_graph_enabled": True,
        "beam_width": [128] * 3,
        "max_decode_steps": 3,
        "candidate_width": [128] * 3,
    }

    with GRLLM(
        model="/absolute/path/to/model",
        gr=config,
        attention_config={"backend": "CUSTOM"},
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
        outputs = llm.beam_search(
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

This example uses the single-request capture from the packaged defaults. For
larger batches, use the [offline Beam example](../../examples/offline_inference/beam_search/README.md#run-the-v1-example)
with `--constraint-backend npu` and `--batch-size`.

Omit the artifact path and digests, vocabulary offset, and vocabulary slice.
The engine rejects an enabled Constraint Table or a vocabulary slice with
`constrained_generation=False`.

### Offline output formats

`output_mode="full"` is the default and returns the usual Beam sequence objects.
Pass `output_mode="compact"` for eager generated-only candidates. For bulk
consumption, pass `output_mode="flat"`. Fixed-length columnar results expose
read-only NumPy arrays at `output.sequences.token_ids` with shape
`(candidates, tokens)` and `output.sequences.cumulative_logprobs` with shape
`(candidates,)`. Other results expose a tuple of compact candidates. This
includes some fixed-length results, such as those with a small Beam width.
Check the sequence type before accessing the arrays.

```python
from vllm_gr.outputs import FlatBeamSearchSequences

sequences = output.sequences
if isinstance(sequences, FlatBeamSearchSequences):
    token_ids = sequences.token_ids
    scores = sequences.cumulative_logprobs
else:
    token_ids = [candidate.token_ids for candidate in sequences]
    scores = [candidate.cumulative_logprob for candidate in sequences]
```

Both representations support iteration, and `output.as_compact()` returns the
compact form in either case.

## Catalog-free SA/SB/SC generation

For SID generation that permits every SA/SB/SC combination, enable the stage
domain inside `main()`, after building `config` and before constructing `GRLLM`:

```python
config["sid_stage_vocab"] = True
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
startup dummy paths retain the full LM head. This option requires ModelRunner V1,
worker decisions, `constrained_generation=False`, and at most three generated
tokens. It is disabled by default; CUDA and catalog-based behavior are unchanged.

`gr.npu_stage_lm_head: false` disables only the projection optimization. The SID
stage domains, token-ID mapping, and score semantics remain enabled, providing
a full-head reference for the optimized projection. Supported models use the
stage projection by default. To run the reference case, set
`config["npu_stage_lm_head"] = False` before constructing the engine.

Setting `gr.npu_stage_lm_head: true` alone does not select the SID stage domain
or override model eligibility checks. With `sid_stage_vocab=False`, catalog-free
Beam selection still uses the full model vocabulary.

In this mode, each parent is normalized across the complete current SID family,
not just its TopK candidates. This is equivalent to a full projection followed
by masking out other families **before** softmax. It is not equivalent to the
default full-vocabulary probabilities: changing the denominator can change
cross-parent Beam rankings. Compare sliced and full heads under the same stage
domain when checking correctness. Catalog membership is not guaranteed.

## Supported execution boundaries

- The experimental full-vocabulary NPU path admits one through six generated
  tokens. Five- and six-token model validation currently covers
  Qwen3-0.6B-OneRecVocab BF16, B=1/W=128, eight KV heads, head dimension 128,
  ModelRunner V1, Eager and FULL Decode Graph. This does not establish arbitrary
  generation horizons or support for all model geometries: native history
  layouts and scratch-buffer capacity still bound further generalization.
  Use this path with
  `ignore_eos=True` and `gr.sid_stage_vocab: false`. Configure
  `gr.max_decode_steps=6` to reserve capacity for requests through six tokens.
  Each request selects its own `max_tokens` within that capacity; the configured
  maximum does not force shorter requests to generate six tokens. Six-token
  generation executes one output-producing Prefill followed by five Decode
  forwards. This remains a bounded Beam decoder, rather than arbitrary-length
  text Beam generation.
- Configured capacities above three also require known positive KV-head
  geometry that fits the native `SelectUnsharedKV` 95 KiB buffer per K/V copy.
  Admission checks the full configured capacity before allocation, even for a
  shorter request. For example, 64 KV heads with head dimension 128 fit five
  tokens but exceed this buffer at six; eight KV heads at that dimension fit
  six. Reduce `gr.max_decode_steps` when this guard rejects a configuration.
- SID stage projection and Constraint Table generation retain their existing
  three-stage boundary. Dynamic width schedules also retain their one-, two-,
  or three-token contract; longer generation uses a fixed Beam width.
- Existing NPU restrictions remain: a plain text decoder, FP16/BF16, ModelRunner
  V1, asynchronous scheduling, and TP/PP/DP/DCP/PCP all equal to one.
- Logical batch sizes follow the existing NPU Beam contract (1 through 8). The
  configured maximum width must equal the request's `beam_width`.
- FULL Decode replay requires fused xllm and a capture matching the logical
  request count and Decode input width. See
  [Decode graph capture](../design/feature/npu_decode_execution.md#decode-graph-capture)
  for replay eligibility and native fallback. A one-token session ends in Prefill.
- Dynamic shrinking and expanding widths use the existing parent/KV ownership
  path. For example, with capacity 128, `dynamic_beam_widths=[128, 64, 32]` and
  `result_width=32` are valid. The Decode input widths are 128 and 64. The
  final 32 has no following Decode.
- Set `gr.candidate_width` to the same per-step list as `gr.beam_width` for
  unconstrained NPU schedules. With an NPU constraint table, later stages can
  set independent candidate widths within the table's `max_top_k`.
  See [Beam settings](../configuration/README.md#beam-settings).

Prefill and Decode reuse the existing model graph machinery. Candidate selection
and Beam decisions retain their existing scheduling boundary outside the captured
model forward. Enabling the Prefill flag does not guarantee every input shape can
use a FULL graph; use the execution diagnostics/profile to verify the actual
dispatch mode.

In-process Beam Search can consume a ready terminal Decode receipt immediately
when it is the only in-flight work and no abort or other request is pending.

## Candidate scores

Unconstrained NPU Beam Search without SID-stage vocabulary restriction defaults
to full-vocabulary scoring. The selector normalizes each parent over its
complete finite model vocabulary and breaks score ties by ascending token ID.
A token's log probability is independent of the requested candidate width.

Set `gr.npu_unconstrained_topk_fast_path: true` to use native TopK and
normalize only the selected logits in FP32. This uses native TopK tie ordering.
The smaller normalization domain can change cumulative scores, cross-parent
ordering, and the final candidate set.

Temperature zero uses unscaled logits. Positive temperatures scale logits before
normalization in either mode. NaN and infinite logits are excluded. When fewer
than K finite tokens are available, the remaining slots are invalid with
negative-infinity scores. A completely invalid row produces no valid candidates.

Run the focused regressions in the complete NPU environment:

```bash
python -m pytest tests/v1/engine/test_npu_unconstrained_beam.py \
  tests/v1/engine/test_npu_unconstrained_runner_config.py \
  tests/ops/npu/test_npu_sid_stage_vocab.py \
  tests/engine/test_npu_sid_stage_config.py \
  tests/v1/worker/test_npu_sid_stage_loading.py \
  tests/ops/test_npu_stage_logits.py \
  tests/entrypoints/beam_search/test_beam_search_input_processor.py \
  tests/v1/worker/test_npu_model_runner_patch.py \
  tests/kernels/test_unconstrained_topk_npu.py \
  tests/kernels/test_sid_stage_topk_npu.py
```

## Version and validation scope

The current runtime uses vLLM 0.30.0 with vLLM-Ascend 0.30.0rc2. Treat earlier
functional and performance results as evidence for their recorded environment
(vLLM 0.22.1 and vLLM-Ascend 0.22.1rc1).

## Optional XAttention output binding

The [caller-owned output path](../design/feature/npu_decode_execution.md#caller-owned-xattention-output)
uses a separate extension around the installed kernel. The build needs the
xllm-ops helper header `test/python_test/pytorch_npu_helper.hpp` and checks
the runtime versions before compiling.

Run these commands inside the prepared NPU environment, from the vLLM-GR
repository root after completing [NPU installation](../getting_started/installation/npu.md).
Set `XLLM_OPS_DIR` to the source checkout used to build the
installed xllm-ops library. Keep the CANN and custom operator environment
loaded in the current shell.

```bash
export XLLM_OPS_DIR=/path/to/the/installed/xllm-ops/source
export XATTENTION_OUT_BUILD="$PWD/build/xattention-out"

# Inspect the build plan. This command does not compile or install anything.
python tools/npu/xattention_out/build.py \
  --xllm-ops-dir "$XLLM_OPS_DIR" \
  --output-dir "$XATTENTION_OUT_BUILD"

# Build the optional binding in a fresh directory.
python tools/npu/xattention_out/build.py \
  --execute \
  --xllm-ops-dir "$XLLM_OPS_DIR" \
  --output-dir "$XATTENTION_OUT_BUILD"

export PYTHONPATH="$XATTENTION_OUT_BUILD/lib:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch_npu, custom_ops_lib, vllm_gr_xattention_out_ext as m; print(m.__file__)'
```

Keep the receipt next to the binary and the output directory on `PYTHONPATH`
for spawned workers. Restart the engine after changing the extension. To use
the allocating route, start the engine without that directory on `PYTHONPATH`.

Validate against the installed allocating kernel in the prepared environment:

```bash
VLLM_GR_TARGET_DEVICE=npu python -m pytest \
  tests/ops/npu/test_npu_xattention_out.py \
  tests/ops/npu/test_npu_xattention_out_binding.py -q
VLLM_GR_TARGET_DEVICE=npu VLLM_GR_TEST_XATTENTION_OUT_REQUIRED=1 \
  python -m pytest tests/kernels/test_x_attention_out_npu.py -q
```

The required-addon device test checks parity and graph replay and fails
when the extension is unavailable. Full-model validation depends on the
intended request geometry and catalog.
