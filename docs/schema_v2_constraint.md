# Schema v2: constraint mode

Schema v2 replaces the overlapping `constraint_backend`,
`constraint_table.enabled`, and `constrained_generation` switches with one
`constraint.mode`. The canonical constraint table remains the only runtime
backend; the legacy Trie frontend is not part of v2.

The schema is loaded through `--vllm-gr-config` or the Python
`vllm_gr_config=` argument. Schema v1 remains readable in this transition
release; new configurations should use v2.

## Constraint table

```yaml
schema_version: 2

constraint:
  mode: table
  table:
    path: /absolute/path/to/constraint_table.safetensors
    backend: auto # auto | cpu | cuda | npu
    max_top_k: 128
    digests:
      artifact: sha256:...
      tokenizer: sha256:...

beam:
  execution_mode: v1
  max_width: 128
  candidate_width: 128 # optional; defaults to max_width
  max_decode_steps: 3
```

`constraint.table.backend: auto` resolves to the active device during engine
configuration. A table that cannot be loaded is a startup error; there is no
silent fallback to an unconstrained path.

## Unconstrained

```yaml
schema_version: 2

constraint:
  mode: none

beam:
  execution_mode: v1
  max_width: 128
```

`mode: none` replaces `constrained_generation: false`. For `beam_search_v1`,
unconstrained execution requires CUDA or NPU, matching the runtime capability
checks.

## Migration from schema v1

| Schema v1 | Schema v2 |
| --- | --- |
| `constraint_table.enabled: true` | `constraint.mode: table` |
| `constraint_table.path` | `constraint.table.path` |
| `constraint_table.backend` | `constraint.table.backend` |
| `constraint_table.max_top_k` | `constraint.table.max_top_k` |
| `constraint_table.artifact_digest` | `constraint.table.digests.artifact` |
| `constraint_table.tokenizer_digest` | `constraint.table.digests.tokenizer` |
| `constraint_table.candidate_width` | `beam.candidate_width` |
| `constrained_generation: false` | `constraint.mode: none` |
| `catalog_path` / `constraint_backend` | removed; build an artifact and use `constraint.mode: table` |

The runtime still requires `max_top_k` and both digests in this transition
step. Artifact-metadata defaults and removal of the legacy internal aliases
are follow-up work.
