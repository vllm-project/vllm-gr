# Constraint table layout and execution

The [configuration guide](../../configuration/constraint_tables.md) covers
building an artifact and choosing a Trie per prompt. The compiler, artifact
loader and device backends share a three-stage table representation. Each
stage restricts the next SID token to legal continuations of the prefix.

## Artifact layout

A single-Trie safetensors file (`constraint_table_v1`) contains six tensors.
A multi-Trie file (`constraint_table_v2`) adds `first_token_offsets` and
records `trie_count` in its metadata. All Tries share a model vocabulary and
tokenizer. Let `T` be the Trie count, `V` the model vocabulary size, and
`a`, `b`, `c` the three SID token IDs.

| Tensor in `constraint_table_v2` | Contents |
| --- | --- |
| `first_token_offsets` | `T + 1` offsets delimiting the first-token list for each Trie. |
| `first_token_ids` | Concatenated lists of legal first tokens (`a`). |
| `prefix1_offsets` | `T * V + 1` offsets. Row `trie_id * V + a` locates the legal second tokens. |
| `prefix1_values` | Concatenated lists of legal second tokens (`b`). |
| `prefix1_pair_keys` | Sorted keys of present pairs, encoded as `trie_id * V * V + a * V + b`. |
| `prefix2_value_offsets` | Boundaries of the third-token list for each stored pair. |
| `prefix2_values` | Concatenated lists of legal third tokens (`c`). |

Pair keys are `int64` and the other tensors are `int32`. Offsets address the
concatenated arrays. The Trie ID distinguishes identical token prefixes in
different catalogs. For a single-Trie file, the loader derives the root interval
`[0, len(first_token_ids)]`. The tensor payload is limited to 512 MiB. See
the [table validation](../../../vllm_gr/constraints/table.py) and
[safetensors reader and writer](../../../vllm_gr/constraints/artifact.py).

## Request routing

At startup the engine obtains the loaded Trie count from the workers.
The input processor attaches one selector to each request item. EngineCore
validates its range before allocating a native request or worker session.
The selection stays with the request across all generation stages and
parent-beam reordering. A mixed-Trie batch can therefore share one model
forward while applying different catalog constraints to its rows. The selector
identifies a Trie within the table. [Online catalog reload](../feature/weight_update.md#shared-update-lifecycle)
can change which table version serves a later lookup.

## Device execution

### CUDA

The constraint lookup addresses the selected root in the combined
tables. With one Trie it uses the single-root addressing path. When LM-head
slicing is active, the projection uses a continuous token-ID window spanning
all stages of all Tries. Legal candidate selection uses the requested Trie,
while log probabilities are normalized over the computed window. Adding a
catalog can widen that window and affect scores. The [CUDA backend](../../../vllm_gr/constraints/cuda_backend.py)
implements selection.

### Ascend

The loader creates a [six-table view per Trie](../../../vllm_gr/constraints/npu_tables.py).
It rebases offsets and pair keys and preserves the model's token IDs. For
stage vocabulary slicing, each SID stage has a shared token map built from
the union of the Tries. The worker expands request selectors in request-major
beam order. The [NPU backend](../../../vllm_gr/constraints/npu_backend.py)
groups contiguous rows with the same selector and invokes `RecConstrainedTopK`
on their table. This selection runs outside the captured model forward.
Scores are normalized over legal candidates at the current prefix.

### CPU reference

The [CPU backend](../../../vllm_gr/constraints/cpu_backend.py) provides a
reference implementation for table selection. The GR Beam inference path
admits CUDA and Ascend devices.
