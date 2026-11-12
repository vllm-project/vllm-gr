# Unconstrained Beam Search

`GRLLM.beam_search()` can select candidates from the model's output vocabulary
without a catalog or Constraint Table artifact. CUDA and NPU implement the same
public Beam interface with different execution backends. This is Beam Search,
separate from ordinary `generate()` sampling, and does not guarantee that generated
tokens identify valid recommendation items. Use [constraint tables](constraint-tables.md)
when catalog membership is required.

## Configuration

Start from a working `gr` mapping for the selected backend and install the
corresponding [CUDA](../getting_started/installation/cuda.md) or
[Ascend](../getting_started/installation/ascend.md) runtime. Keep the model,
Attention, capacity and graph settings; change the generation policy before
constructing `GRLLM`:

```python
from copy import deepcopy


def without_constraints(base_gr: dict) -> dict:
    gr = deepcopy(base_gr)
    gr["constraint_mode"] = "none"
    gr["constraint_table"] = None
    gr["sid_stage_vocab"] = False
    return gr
```

Pass the result as `GRLLM(..., gr=without_constraints(base_gr))`.
The native runtime selects CUDA or NPU. Preserve `candidate_width` so disabling
constraints does not change the per-parent candidate count. Removing the table
also removes the artifact identity and vocabulary slice. No catalog file,
artifact digest or tokenizer digest is required.

Keep asynchronous scheduling enabled and Chunked Prefill disabled. The request
width must fit the configured Beam and candidate capacities. For a common
three-token example, reserve at least three generation steps, construct the
engine with the revised configuration, and call:

```python
from vllm_gr.sampling_params import BeamSearchParams

params = BeamSearchParams(beam_width=128, max_tokens=3, ignore_eos=True)
outputs = llm.beam_search(inputs, params, output_mode="full")
```

Here `llm` is the configured engine, `inputs` are its prompt inputs, and the
configured width is 128. `output_mode="full"` returns the usual sequence objects;
omitting it also returns full output. Pass `output_mode="compact"` for eager
generated-only results. For the catalog-free
[offline Beam example](https://github.com/JiusiServe/vllm-gr/blob/main/examples/offline_inference/beam_search/README.md#run-the-v1-example),
pass `--constraint-backend npu` on Ascend.

## Candidate and score semantics

- Each parent is normalized in FP32 over its complete finite vocabulary before
  Top-K truncation. Changing the candidate count must not change a token's
  normalization domain.
- Temperature zero means unscaled logits for Beam scoring; positive temperatures
  scale logits before normalization.
- Non-finite logits do not form valid candidates. Missing candidates have an
  explicit invalid mask and negative-infinity scores; ties prefer smaller token IDs.
- Batch shapes, dtype and backend kernels can change scores and candidate order.
  A shared interface does not promise bitwise output agreement across devices.

CUDA uses its full-vocabulary selection kernels; the NPU implementation uses
native PyTorch selection. Full-vocabulary work can be substantial for large
vocabularies and wide Beams. Functional support is not a performance claim.

## Execution support

The [Beam execution contract](beam-search.md) owns batch admission, cancellation
and retirement. Use a plain text decoder, FP16/BF16, `ignore_eos=True`, and the
supported single-rank parallel configuration. Runner and capacity limits remain
backend-specific:

| Backend | Runner and generation scope | Graph configuration |
| --- | --- | --- |
| CUDA | ModelRunner V1 or opt-in V2; the documented unconstrained configuration covers one to three generated tokens | See [CUDA graph policies](cuda-graphs.md) for capture, batching and fallback |
| NPU | ModelRunner V1; full-vocabulary generation supports a bounded one-to-six-token path, subject to model geometry and scratch capacity | Fixed `gr.decode_graph_batch_size` and Decode input widths identify captured shapes; configure `gr.decode_graph_widths` for additional dynamic widths |

For NPU capacities above three, known positive KV-head geometry must fit the
native `SelectUnsharedKV` 95 KiB buffer per K/V copy. Admission checks the full
configured capacity, even for shorter requests. With head dimension 128,
64 KV heads fit five tokens but not six; eight KV heads fit six. Longer generation
uses fixed Beam width. Dynamic width schedules retain their one-to-three-token
scope; set `gr.candidate_width` to the same per-step list as `gr.beam_width`
for those NPU schedules.
This bounded extension does not establish arbitrary-length text generation or
support for every model geometry.

A one-token request completes in Prefill without Decode. Enabling graph mode
does not establish that every shape replayed a graph; observe replay and fallback
diagnostics when comparing runs.

## Optional SID token-family restriction

`gr.sid_stage_vocab: true` restricts successive stages to SA, SB and SC token
families without validating catalog membership. This option currently requires
the NPU backend, ModelRunner V1, Worker decisions, unconstrained generation and
at most three generated tokens. It is disabled by default and is not a portable
requirement of unconstrained Beam Search.

Use the model's SID prompt and required begin marker. The tokenizer supplies
the actual `<s_a_N>`, `<s_b_N>` and `<s_c_N>` IDs; family sizes and ID continuity
are not hardcoded. Scores normalize over the complete current family, rather
than its retained Top-K. This changes the denominator relative to full-vocabulary
generation and may change cross-parent rankings.

The NPU stage projection maintains weight caches through the existing update
lifecycle. `gr.npu_stage_lm_head: false` disables only the projection optimization;
family restrictions and scoring remain active. Configure this before engine
startup and compare full-head and sliced-head execution under the same family
domain. Enabling stage projection alone does not enable SID families or
override model eligibility checks.

## Validation entrypoints

| Boundary | Entrypoints |
| --- | --- |
| Public CUDA V1/V2 execution, eager and graph | `tests/entrypoints/gr/test_offline_beam_search.py::test_offline_unconstrained_beam_search` |
| Request validation | `tests/entrypoints/beam_search/test_beam_search_input_processor.py` |
| NPU selection and runner configuration | `tests/v1/engine/test_npu_unconstrained_beam.py`, `tests/v1/engine/test_npu_unconstrained_runner_config.py` |
| Device candidate semantics | `tests/kernels/test_unconstrained_topk_npu.py` |
| Optional SID domain and projection | `tests/ops/npu/test_npu_sid_stage_vocab.py`, `tests/engine/test_npu_sid_stage_config.py`, `tests/v1/worker/test_npu_sid_stage_loading.py`, `tests/ops/test_npu_stage_logits.py`, `tests/kernels/test_sid_stage_topk_npu.py` |

Use the [validation guide](../contributing/tools/README.md#validation-levels) to select the
required runtime and hardware. Check actual execution and business metrics on
the target backend; these entrypoints are not results for the current checkout.
