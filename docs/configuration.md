# Configuration

All startup paths use the structure in [default_config.yaml](../vllm_gr/config/default_config.yaml).
Native vLLM options are top-level keys. Every GR option is inside `gr`.
GR defaults and Beam validation apply to `GRLLM`, the `--gr` dispatcher, or an
engine constructed with explicit GR configuration. Native engines without GR
input keep upstream defaults and validation.

The packaged file supplies the common defaults. You can supply a YAML file
with only the values you want to override. A file can inherit another file
with `import: ./base.yaml`. Dictionaries merge by key, and lists replace
the inherited list as a whole. Explicit YAML `null`, `false`, and empty lists
replace inherited values. Native string values use the upstream vLLM converters
in both Python and CLI entrypoints (`4k` becomes 4000; `auto` becomes -1).
When chunked prefill is disabled, an omitted token budget grows to cover the
model length. An explicitly supplied budget, even 8192, is preserved for native
validation.

```yaml
max-model-len: 4096
gr:
  beam_width: [8, 4, 2]
  candidate_width: [8, 4, 2]
  max_decode_steps: 3
  worker_decision: true
  result_width: 1
  constraint_mode: none
  beam_batch_sizes: [1, 2, 4]
```

Start a server with this file. Explicit command-line values override YAML values.

```bash
vllm serve /models/example --gr --config scenario.yaml \
  --gr.result_width 2
```

`--gr` selects the GR command. Use `--gr-config '{"result_width":2}'` for a
JSON mapping, or `--gr.result_width 2` for a field override. Both populate the
same `gr` settings used by Python.

Use the same file in Python. Explicit Python arguments override YAML values.

```python
from vllm_gr import GRLLM

llm = GRLLM(
    model="/models/example",
    config="scenario.yaml",
    max_model_len=2048,
    gr={"result_width": 2},
)
```

The driver reads the user file during initialization and resolves YAML-declared
artifact and calibration paths relative to their declaring files. It passes
the resolved immutable configuration to workers. Workers reuse this snapshot
without reopening the YAML. Restart the engine to apply file changes.

The startup log prints the complete resolved GR configuration using the same
public key names.
Native settings are resolved once at the engine boundary. Python and CLI retain
which fields were explicitly supplied, including values equal to defaults.

## Beam settings

- `beam_width` is the number of winners retained after each generated token.
- `candidate_width` is the number of candidates expanded per parent at each step.
- Unconstrained `beam_width` and `candidate_width` both support 1..1024.
  The previous shared candidate ceiling of 256 does not apply to unconstrained
  generation. Current CUDA/NPU constraint-table operators still cap candidates
  per parent at 256; the backend loader checks that independent limit.
  NPU independent per-step candidate widths require table mode; unconstrained
  NPU generation requires matching candidate and retained width lists.
- Both settings are lists with matching lengths. Their first entries must match.
- `max_decode_steps` is the engine's generation capacity, excluding begin/end
  tokens. It must cover the configured width lists. A dynamic request uses
  the matching prefix of those lists and supports one to three effective
  generation steps. Capacity 5 with a three-entry default is valid, but a
  four- or five-step dynamic request is rejected; lists are never padded.
  Fixed-width requests may use the full capacity where the backend supports it.
  NPU unconstrained fixed-width generation supports up to **6** steps, with
  capacity above 3 also checked against the model's native KV reorder buffer.
  CUDA unconstrained fixed-width generation has no separate six-step ceiling.
  Constraint-table and NPU SID-stage paths remain limited to 3 steps.
  These are code-level backend limits, not environment variables; configure the
  desired engine capacity with `gr.max_decode_steps` (default **3**).
- `worker_decision` accepts `true`, `false`, or `null`. With `null`, the
  driver selects the platform default at startup.
- `result_width` controls the number of returned sequences. With `null`, the
  request uses its final retained width.

The configuration reference and supported values for every GR option are
in the common YAML. CUDA and Ascend options are grouped in that file.

## Constraint table

Set `gr.constraint_mode: table` and supply `gr.constraint_table` with its
path, backend, format, candidate capacity, and digests. The common YAML contains
the artifact build command and a complete table example. Set
`gr.constraint_mode: none` for unconstrained generation.

`gr.beam_batch_sizes` is a shared serving and graph policy. It selects which
multi-prompt HTTP completions use one atomic Beam batch, and the CUDA ModelRunner
V1/V2 Prefill and Decode capture sizes. Removing `4` both removes that capture
size and sends four-prompt HTTP completions to the original handler. Separate
requests are never combined to fill a batch. Ascend retains its own Decode graph
settings; native `max-num-seqs` and token budgets still bound admission.
The maximum atomic Beam batch is **8**. The default list enables **1, 2, 4**;
include `8` in `gr.beam_batch_sizes` to route eight-prompt HTTP completions
through Beam. The maximum is a code-level admission limit, not an environment
variable.

Configuration types and their invariants live in `config/models.py`; YAML I/O,
relative paths, inheritance and merging live in `config/parser.py`. Workers use
the resolved configuration snapshot and do not reopen deployment files.
