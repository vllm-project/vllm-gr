# Configuration

`GRLLM` and `vllm serve --gr` use the same startup YAML. Put native vLLM
options at the top level and vLLM-GR options under `gr`. The
[packaged defaults](../../vllm_gr/config/default_config.yaml) list the common,
CUDA and Ascend settings. Native engines started without GR configuration use
the upstream vLLM defaults.

Use a [deployment recipe](../../recipes/README.md) for model-specific overrides.

## Load and override a configuration

Settings take precedence in this order, from highest to lowest.

1. Explicit CLI or Python arguments
2. Startup YAML
3. YAML included with `import: ./base.yaml`
4. Packaged defaults

Mappings merge by key. Lists replace the entire inherited list. Explicit
`null`, `false` and empty lists replace inherited values, subject to validation.
Both entry points use native vLLM converters for values such as
`max-model-len: 4k` (4000 tokens).

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

### Path resolution

The driver resolves artifact and calibration paths relative to the YAML file
that declares them, including imported files. Workers receive the resolved
configuration. Check the startup log for the effective GR settings and restart
the engine to apply edits to its startup YAML.

## Migrate from `configs/`

- For Pangu, use [the Pangu recipe](../../recipes/Pangu/pangu1b.yaml) in
  `--config` or Python's `config=` argument in place of `configs/pangu1b.yaml`.
- For GR's packaged defaults, omit `--config configs/default_config.yaml`
  or the equivalent Python `config=` argument. To keep the YAML values
  explicit in a source checkout, pass `vllm_gr/config/default_config.yaml`.

The old default YAML supplied `max-num-batched-tokens: 8192` explicitly.
With chunked Prefill disabled and `max-model-len: 16384`, omitting the file
and the budget lets GR derive a budget of 16384. To preserve an explicit
budget, pass `--max-num-batched-tokens` on the CLI or set
`max_num_batched_tokens` in Python. When chunked Prefill is disabled,
vLLM requires an explicit budget at least as large as the model length.

## ModelRunner selection

The vLLM ModelRunner executes model work on the device. For the library and
general CLI, set `VLLM_USE_V2_MODEL_RUNNER` before importing vLLM or starting
the CLI.

| Setting | CUDA | Ascend |
| --- | --- | --- |
| Unset or `0` | ModelRunner V1, the vLLM-GR default | ModelRunner V1, required for Beam search |
| `1` | Opt-in [ModelRunner V2](../design/module/model_runner_v2.md) for supported Beam workloads | Use `0` for Beam search |

The CUDA direct offline Beam, HTTP serving and OneRec gRPC examples select
ModelRunner V2 when the variable is unset. See the
[example defaults](../../examples/README.md#choose-a-modelrunner) for the
offline thinking path and the gRPC smoke test.

Some logs and tests use the name "Beam Search V1" for the Beam execution path.
The same path can use either ModelRunner on CUDA within the supported scope.
The [constraint table formats](constraint_tables.md#build) describe the catalog
artifact layout and have their own version identifiers.

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
`enable_chunked_prefill=False`. See the [CUDA walkthrough](../user_guide/onerec_quickstart.md)
or the [Ascend Beam example](../user_guide/beam_search_npu.md#configuration-and-python-entry-point).

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
