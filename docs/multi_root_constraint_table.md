# Multi-Trie Constraint Tables

One `constraint_table_v2` safetensors artifact contains multiple independent
three-token SID Tries. Requests select an integer `trie_id`; all Tries use the
same model vocabulary and tokenizer. Region names and other business selectors
are mapped to Trie IDs by the caller. IDs are local to the loaded artifact and
start at zero.

## Build

Build each Trie's triples document with the existing
`tools/build_constraint_triples.py`, then create a manifest:

```json
{
  "tries": [
    {"trie_id": 0, "triples_path": "trie0_triples.json"},
    {"trie_id": 1, "triples_path": "trie1_triples.json"}
  ]
}
```

IDs must be consecutive, starting at zero; paths are relative to the manifest.

```bash
python tools/build_constraint_table.py \
  --tries /path/to/tries.json \
  --model /path/to/model \
  --output /path/to/multi_trie.safetensors
```

Use the emitted artifact/tokenizer digests and set
`constraint_table.format: constraint_table_v2` in the existing GR configuration.
The legacy `--triples` command still writes six-table v1 artifacts.

## Requests

```python
outputs = llm.beam_search_v1(
    prompts=[prompt_a, prompt_b],
    params=BeamSearchParams(beam_width=64, max_tokens=3, ignore_eos=True),
    trie_ids=[0, 1],
)
```

Omitting `trie_ids` selects Trie 0 for every prompt. An explicit list must have
one non-negative integer per prompt; a single prompt selecting Trie 1 uses
`trie_ids=[1]`. Trie selection is separate from `BeamSearchParams`.
It works with `per_parent_beam_widths` and both compact and full offline outputs.
Online completion requests use `vllm_xargs: {"trie_ids": [0, 1]}` in prompt order.
When a request cannot run as one supported Beam batch, the upstream per-prompt
fallback preserves each prompt's Trie selector. Non-Beam requests cannot use selectors.
Engine startup reads the live Trie count from the Workers once. IDs outside that
range are rejected at EngineCore admission before native request or Worker session
allocation; there is no implicit fallback to an unconstrained search.

CUDA LM-head slicing uses one continuous token-ID window covering all three
stages of every Trie in the artifact. Selecting another Trie does not change the
projection or omit its tokens. More Tries can widen this window and reduce the
speedup. Log probabilities are normalized over the computed window, so combining
separate artifacts can change scores and Beam rankings if their windows differ.
Unquantized heads support weight slicing; quantized heads use full projection
followed by output slicing. The canonical Worker disables slicing for TP > 1.
Each logical request keeps its Trie selection across all stages and parent-beam changes.

## Layout and compatibility

The seven tensors retain v1 dtypes (`int32` except `int64` pair keys).
Here `T=trie_count` and `V` is the model vocabulary size:

| Tensor | Shape / meaning |
| --- | --- |
| `first_token_offsets` | `[T+1]`, each Trie's SA interval |
| `first_token_ids` | Concatenated SA lists |
| `prefix1_offsets` | `[T*V+1]`, one continuous CSR indexed by `trie_id*V+SA` |
| `prefix1_values` | Concatenated SB lists |
| `prefix1_pair_keys` | Sorted sparse keys `trie_id*V*V+SA*V+SB` |
| `prefix2_value_offsets` | SC boundaries per real pair |
| `prefix2_values` | Concatenated SC lists |

Offsets are absolute into their concatenated values arrays. Prefixes from
different Tries remain independent even when their token IDs are identical.
The file records `trie_count` in metadata. All Tries must be non-empty.

Existing v1 files remain loadable and implicitly contain Trie 0. The host loader
normalizes their SA boundaries to `[0, len(first_token_ids)]`. For `T=1`, CUDA
uses the original addressing: root buffers are not uploaded, and the Triton
compile-time Trie count removes root-dependent loads and arithmetic.

Multi-Trie table selection is implemented for CPU and CUDA. CUDA supports mixed
Tries in fixed and dynamic-width batches; the existing W/K rules still apply.
NPU accepts only a single Trie and rejects multi-trie artifacts. Participating
processes must use the same request schema; mixed-version IPC is not supported.
The existing 512 MiB artifact limit and kernel candidate-degree limits remain
in force. This change does not add hot reload, arbitrary-depth Tries or NPZ
import.
