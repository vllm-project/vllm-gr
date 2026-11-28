# Constraint tables

A constraint table limits each generated three-token semantic ID (SID) to a
path in a catalog. Use a single Trie when every prompt uses the same catalog.
Use a multi-Trie artifact when prompts need different catalogs, such as two
regions served by the same model and tokenizer.

## Build

From the repository root, use the [catalog example](../../examples/online_serving/catalog/README.md#extract-constraint-triples-from-the-catalog)
to convert each catalog into a triples JSON file. The examples below use a
local model directory. A remote model ID requires `--revision`. Pass
`--trust-remote-code` if the model requires custom code.

!!! important "Match the inference tokenizer"

    Build the table with the same model snapshot and tokenizer used for inference.

### Single Trie

Compile one triples file to produce a `constraint_table_v1` artifact.

```bash
python tools/constraints/build_constraint_table.py --model /models/OneRec \
  --triples video_triples.json --output video.safetensors
```

### Multiple Tries

Save this manifest as `tries.json`, next to the triples files. List the Tries
in ID order, starting at zero with no gaps. Relative paths resolve from the
manifest's directory.

```json
{
  "tries": [
    {"trie_id": 0, "triples_path": "trie0_triples.json"},
    {"trie_id": 1, "triples_path": "trie1_triples.json"}
  ]
}
```

Compile the manifest to produce a `constraint_table_v2` artifact.

```bash
python tools/constraints/build_constraint_table.py --model /models/OneRec \
  --tries tries.json --output regions.safetensors
```

## Configure the engine

For the single-Trie example, save this as `table.yaml` next to
`video.safetensors`. Replace both digest placeholders with the complete
`artifact_digest` and `tokenizer_digest` values printed by the builder,
including the `sha256:` prefix.

```yaml
attention-config:
  backend: CUSTOM
gr:
  worker_decision: true
  beam_width: [128, 128, 128]
  candidate_width: [128, 128, 128]
  constraint_mode: table
  constraint_table:
    path: ./video.safetensors
    backend: auto
    format: constraint_table_v1
    max_top_k: 128
    digests:
      artifact: "<artifact_digest>"
      tokenizer: "<tokenizer_digest>"
```

For the two-Trie example below, use `./regions.safetensors` as the path,
`constraint_table_v2` as the format and the digests from the manifest build.
`backend: auto` uses the canonical table implementation chosen by the GR
platform. CUDA and Ascend ship with device implementations. A vendor plugin
can supply its own table backend. The CPU implementation serves as a
reference. [Platform contract](../design/module/platform_backend.md#engine-plans)

## Requests

With a single Trie, omit `trie_ids`. With multiple Tries, omission also selects
Trie 0 for all prompts. To use other catalogs, supply one non-negative ID per
prompt in input order. IDs are local to the artifact. The caller maps business
identifiers, such as region names, to these IDs.

### Python

With the two-Trie configuration, this call selects Trie 0 for the first prompt
and Trie 1 for the second. The request width matches the engine width of 128.
The token budget of five covers the begin marker, three SID tokens and the end
marker.

```python
from vllm_gr import GRLLM
from vllm_gr.sampling_params import BeamSearchParams

prompts = [{"prompt": "Recommend a video."}, {"prompt": "Recommend a video."}]
params = BeamSearchParams(
    beam_width=128,
    max_tokens=5,
    begin_token="<|sid_begin|>",
    end_token="<|sid_end|>",
    ignore_eos=True,
)
with GRLLM(model="/models/OneRec", config="table.yaml") as llm:
    outputs = llm.beam_search(prompts, params, trie_ids=[0, 1])
```

Both `full` and `compact` return results in prompt order. Keep the input ID
list if the caller needs the catalog for each result. The public output has
no `trie_id` field.

### HTTP completions

Start the engine with the same two-Trie configuration.

```bash
vllm serve /models/OneRec --gr --config table.yaml --served-model-name onerec
```

Send this JSON body to `POST /v1/completions`.

```json
{
  "model": "onerec",
  "prompt": ["Recommend a video.", "Recommend a video."],
  "use_beam_search": true,
  "n": 128,
  "temperature": 0,
  "max_tokens": 5,
  "ignore_eos": true,
  "vllm_xargs": {
    "begin_token": "<|sid_begin|>",
    "end_token": "<|sid_end|>",
    "trie_ids": [0, 1]
  }
}
```

Selectors apply only to Beam completions. If the request is processed one
prompt at a time, each prompt keeps its selector. The engine rejects IDs
outside the loaded artifact. See the [completions API](../serving/completions.md)
for response fields.

## Device behavior and updates

- CUDA and Ascend both select a Trie per prompt. On CUDA, combining catalogs
  can widen the vocabulary window used for scoring and change rankings.
  Ascend normalizes over the legal candidates at the current prefix. Raw
  scores from the two backends can differ.
- CUDA asynchronous Beam serving supports online table replacement. The new
  table must preserve the loaded Trie count and vocabulary projection. On
  Ascend, restart the engine with the replacement table. Follow the
  [catalog update guide](../user_guide/weight_update.md) for the CUDA procedure.

For tensor layout, backend execution and compatibility, see the
[constraint table design](../design/module/constraint_tables.md).
