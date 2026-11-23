# Configuration

`GRLLM` and `vllm serve --gr` use the same startup YAML. Put native vLLM
options at the top level and vLLM-GR options under `gr`. The
[packaged defaults](../../vllm_gr/config/default_config.yaml) list the common,
CUDA and Ascend settings. Native engines started without GR configuration use
the upstream vLLM defaults.

## Load and override a configuration

### Example

Save this example as `scenario.yaml`. It retains eight beams at the first
step, four at the second and two at the third, and returns one result.

```yaml
max-model-len: 4096
gr:
  beam_width: [8, 4, 2]
  candidate_width: [8, 4, 2]
  max_decode_steps: 3
  worker_decision: true
  result_width: 1
  constraint_mode: none
```

#### CLI

Start the server and override the number of returned sequences.

```bash
vllm serve /models/example --gr --config scenario.yaml --gr.result_width 2
```

The equivalent JSON override is `--gr-config '{"result_width":2}'`. See
[serve](../cli/serve.md#configuration) for all CLI forms.

#### Python

Use the same file and public GR field names.

```python
from vllm_gr import GRLLM

llm = GRLLM(
    model="/models/example",
    config="scenario.yaml",
    max_model_len=2048,
    gr={"result_width": 2},
)
```

For Beam inference, call `llm.beam_search()` with `BeamSearchParams` as shown
in the [offline example](../../examples/offline_inference/beam_search/README.md).
Its default `output_mode="full"` returns vLLM Beam sequence objects;
`output_mode="compact"` selects compact results.

### Precedence

Settings take precedence in this order, from highest to lowest.

1. Explicit CLI or Python arguments
2. Startup YAML
3. YAML included with `import: ./base.yaml`
4. Packaged defaults

Mappings merge by key. Lists replace the entire inherited list. Explicit
`null`, `false` and empty lists replace inherited values, subject to validation.
Both entry points use native vLLM converters for values such as
`max-model-len: 4k` (4000 tokens).

### Path resolution

The driver resolves artifact and calibration paths relative to the YAML file
that declares them, including imported files. When parsing a public Python
mapping, the driver also resolves symlinks in absolute artifact and calibration
paths. Workers use the published paths even if the link changes after startup.
Check the startup log for the effective GR settings and restart
the engine to apply edits to its startup YAML.

## ModelRunner selection

The vLLM ModelRunner executes model work on the device. With
`VLLM_USE_V2_MODEL_RUNNER` unset, vLLM chooses **ModelRunner V2 on CUDA**
when the configuration supports it and falls back to **V1** otherwise.
GR defaults to **ModelRunner V1 on Ascend NPU**. To override the default,
set the variable before importing vLLM or starting the CLI.

| Runner | Environment setting | Beam deployment |
| --- | --- | --- |
| [ModelRunner V2](../design/module/model_runner_v2.md) | `VLLM_USE_V2_MODEL_RUNNER=1` | Explicit selection on CUDA. Unsupported configurations fail startup. |
| ModelRunner V1 | `VLLM_USE_V2_MODEL_RUNNER=0` | Required on Ascend NPU and available on CUDA |

GR normalizes explicit boolean spellings before native parsing.
An explicitly empty value is equivalent to `0`. Restart the process after
changing the selection.

Some logs and tests call the GR execution path "Beam Search V1". That path
can use either ModelRunner on CUDA within the supported scope. The names
`constraint_table_v1` and `constraint_table_v2` refer to the
[catalog artifact layout](constraint_tables.md#build).

## Beam settings

| Setting | How to choose it |
| --- | --- |
| `gr.beam_width` | Retained beams at each generated step. Set the request's `beam_width` (HTTP `n`) to the maximum of this list, including for a decreasing schedule such as `[8, 4, 2]`. |
| `gr.candidate_width` | Candidates expanded per parent at each step. Supply one entry per retained width. The first entries must match because the first step has one parent. |
| `max-logprobs` | Native vLLM setting. Its effective limit must cover the largest configured candidate width. The packaged default is 128. |
| `gr.constraint_table.max_top_k` | In table mode, this capacity must cover the largest candidate width. CUDA and Ascend table operators support up to 256 candidates per parent. |
| `gr.result_width` | Number of sequences to return, up to the final retained width. The default (`null`) uses the final retained width. |
| `gr.worker_decision` | `true` enables worker-side Beam selection. `null` uses the platform default. |
| `gr.max_decode_steps` | Generation capacity, excluding begin and end markers. It must cover the width lists and defaults to 3. |

Unconstrained generation accepts widths from 1 to 1024. On Ascend,
unconstrained generation requires equal candidate and retained widths at each
step. Independent candidate widths are available with a constraint table.

A dynamic-width request uses a matching prefix of its configured schedule and
supports one to three effective steps. Fixed-width unconstrained generation on
Ascend supports up to six steps, subject to the model's KV reorder capacity.
CUDA has no corresponding six-step limit. A constraint table requires exactly
three effective steps on either device. Ascend SID-stage vocabulary execution
also supports at most three steps. See the [device compatibility guide](../user_guide/feature_compatibility.md)
for related features.

## Asynchronous scheduling

Beam execution requires `async-scheduling: true` and
`enable-chunked-prefill: false`. Both are packaged defaults. In Python the
corresponding arguments are `async_scheduling=True` and
`enable_chunked_prefill=False`. See the [catalog-constrained example](constraint_tables.md#validate-and-troubleshoot)
or the [Ascend quickstart](../getting_started/quickstart.md#ascend).

### Batch sizes and token budgets

- `gr.beam_batch_sizes` selects the prompt counts eligible for one atomic
  multi-prompt HTTP completion. The default is `[1, 2, 4]`. Add `8` to enable
  the maximum size. Separate HTTP requests are not combined to fill a batch.
  Both CUDA model runners use this list for Prefill and Decode graph capture.
  Ascend has separate Decode graph settings.
- `max-num-seqs` must cover the number of prompts in the Beam batch. For
  example, two prompts at width 128 count as two requests for this check.
- `max-num-batched-tokens` must cover the total prompt tokens at Prefill.
  With chunked prefill disabled, an omitted budget grows to cover
  `max-model-len`. An explicit budget goes through native validation.
  On Ascend, a request with a Decode stage must also fit the batch size times
  the peak input beam width for those stages. Two prompts at fixed width 128
  therefore require capacity for at least 256 Decode rows.

## Ascend NPU execution limits

For a first request, use the [Ascend quickstart](../getting_started/quickstart.md#ascend).
For Python integration, use the [configuration example](#python) and
`GRLLM.beam_search()` with `BeamSearchParams`. The shared
[Beam settings](#beam-settings), [runner selection](#modelrunner-selection),
and [token budgets](#batch-sizes-and-token-budgets) apply.

Use a unimodal recommendation decoder with a uniform layer layout. Run in
FP16 or BF16 with asynchronous scheduling and TP/PP/DP/DCP/PCP all equal to
one. The logical Beam batch size is 1 through 8.

### Full-vocabulary generation

Set `gr.constraint_mode: none` and leave `gr.sid_stage_vocab: false` to
select candidates from the full vocabulary. Use `ignore_eos=True` for this
experimental Ascend path. For requests longer than three tokens, use a fixed
Beam width and increase `gr.max_decode_steps` up to 6. A request's `max_tokens`
selects its length within that capacity; raising the capacity does not lengthen
shorter requests.

Capacities above three require known positive KV-head geometry that fits the
native `SelectUnsharedKV` 95 KiB buffer per K/V copy. Startup validates the
configured maximum even if a request is shorter. For example, 64 KV heads with
head dimension 128 fit five tokens but not six; eight KV heads fit six.
Reduce `gr.max_decode_steps` if this check rejects your configuration. See
[Ascend validation scope](../design/feature/npu_decode_execution.md#validation)
for the measured model geometry and
[candidate scores](../design/feature/npu_decode_execution.md#candidate-selection)
for normalization and tie handling.

### SID token families without a catalog

To restrict successive output tokens to SA, SB, and SC families while allowing
every combination of them, add this fragment to your startup YAML:

```yaml
gr:
  constraint_mode: none
  sid_stage_vocab: true
```

Use the checkpoint's SID prompt, including its SID-begin marker when required.
This option is disabled by default and requires Ascend ModelRunner V1,
worker decisions, and `gr.max_decode_steps <= 3`. The tokenizer must define
the `<s_a_N>`, `<s_b_N>`, and `<s_c_N>` token families.

This limits token families; it does not guarantee catalog membership. Use a
[constraint table](constraint_tables.md) when outputs must match catalog items.
Stage-family normalization can change cross-parent rankings relative to
full-vocabulary generation; see [SID stage selection](../design/feature/npu_decode_execution.md#sid-stage-selection).

On eligible models, `gr.npu_stage_lm_head` defaults to `true` and projects only
the selected family. Set it to `false` to use the full LM head as a reference
while preserving the same family mask and scoring semantics. Setting it to
`true` alone does not enable `gr.sid_stage_vocab` or override model eligibility.

## Constraint table

Use `gr.constraint_mode: none` for unconstrained generation. For catalog
serving, a table configuration records the artifact path, backend, format,
candidate capacity and digests. Once you have built and configured a table,
a scenario can inherit it and override just the serving settings.

```yaml
import: ./table.yaml
gr:
  result_width: 4
```

See [Constraint tables](constraint_tables.md) to build a single- or multi-Trie
artifact, write `table.yaml` and select a catalog per prompt.

For driver/worker serialization and resolved capacities, see the
[configuration lifecycle](../design/architecture_overview.md#configuration-lifecycle).

## Migrate from `configs/`

For GR's packaged defaults, omit `--config configs/default_config.yaml`
or the equivalent Python `config=` argument. To keep the YAML values
explicit in a source checkout, pass `vllm_gr/config/default_config.yaml`.

The old default YAML supplied `max-num-batched-tokens: 8192` explicitly.
With chunked Prefill disabled and `max-model-len: 16384`, omitting the file
and the budget lets GR derive a budget of 16384. To preserve an explicit
budget, pass `--max-num-batched-tokens` on the CLI or set
`max_num_batched_tokens` in Python. When chunked Prefill is disabled,
vLLM requires an explicit budget at least as large as the model length.

Old flat settings such as `additional_config["vllm_gr_beam_max_width"]`
raise an error on read and write. Move their intended values to the current
GR settings and remove the flat keys before starting the engine.
