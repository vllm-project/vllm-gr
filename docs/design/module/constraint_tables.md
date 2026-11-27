# Constraint table layout and execution

The [configuration guide](../../configuration/constraint_tables.md) covers
building an artifact and choosing a Trie per prompt. The compiler, artifact
loader and device backends share a three-stage table representation. Each stage
restricts the next SID token to legal continuations of the prefix.

## Artifact layout

A single-Trie safetensors file (`constraint_table_v1`) contains six tensors. A
multi-Trie file (`constraint_table_v2`) adds `first_token_offsets` and records
`trie_count` in its metadata. All Tries share a model vocabulary and tokenizer.
Let `T` be the Trie count, `V` the model vocabulary size, and `a`, `b`, `c` the
three SID token IDs.

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
`[0, len(first_token_ids)]`. The tensor payload is limited to 512 MiB, and the
built-in device operators reject `max_top_k > 256`. See the
[table validation](../../../vllm_gr/constraints/table.py) and
[safetensors reader and writer](../../../vllm_gr/constraints/artifact.py).

## Request routing

At startup the engine obtains the loaded Trie count from the workers. The input
processor attaches one selector to each request item. EngineCore validates its
range before allocating a native request or worker session. The selection stays
with the request across all generation stages and parent-beam reordering. A
mixed-Trie batch can therefore share one model forward while applying different
catalog constraints to its rows. The selector identifies a Trie within the
table. [Online catalog reload](../feature/weight_update.md#update-lifecycle)
can change which table version serves a later lookup.

## Device execution

### CUDA

The constraint lookup addresses the selected root in the combined tables. With
one Trie it uses the single-root addressing path. When LM-head slicing is
active, the projection uses a continuous token-ID window spanning all stages of
all Tries. Slicing is not applied when tensor parallelism exceeds 1 and is
dropped when the computed window equals the full vocabulary. Legal candidate
selection uses the requested Trie, while log probabilities are normalized over
the computed window. Adding a catalog can widen that window and affect scores;
this is documented intent rather than a tested invariant. The
[CUDA backend](../../../vllm_gr/constraints/cuda_backend.py) implements
selection.

### Ascend

The loader creates a six-table view per Trie. A multi-Trie table is rebased;
a single-Trie table is returned unchanged. When stage vocabulary slicing is
enabled, the operator receives a second, compacted six-table set whose token
values are remapped through `searchsorted`, so it does not see model token IDs
on that path. The loader builds two shared prefix maps, for the first and
second SID positions, plus three stage token-ID tables, all from the union of
the Tries.

Stage vocabulary is optional. `build_npu_stage_vocab` returns `None` when it
cannot shrink the vocabulary, and the backend falls back to full-vocabulary
selection when the active logits width does not match the stage vocabulary.
The [NPU backend](../../../vllm_gr/constraints/npu_backend.py) groups
contiguous rows with the same selector and invokes `rec_constrained_topk` on
their table. The worker path uses `select_for_rows` with one host Trie ID per
physical row, which avoids a device readback and keeps the call capture-safe.
Selection runs outside the captured model forward. Scores are normalized over
legal candidates at the current prefix.

### CPU

The [CPU backend](../../../vllm_gr/constraints/cpu_backend.py) is a host-table
backend that follows the shared full-window scoring contract. It is not a
reference for NPU log probabilities, because the fused NPU operator normalizes
over the legal candidate set. Parity tests compare it with the torch fallback
routed through a CPU-resident operator.

## Reload and retention

A reload rejects a changed Trie count or a changed vocabulary projection.
Publishers must keep each version's artifact immutable. The management
endpoint returns success immediately for the committed version. A worker
loading a candidate checks the artifact digest against a same-named table
only while that table is retained. Old versions expire lazily on a later
table operation after `gr.trie_retention_seconds` (default 60). See the
[versioned table](../../../vllm_gr/constraints/versioned_table.py).

## Validation

| Test | Coverage |
| --- | --- |
| [Table tests](../../../tests/constraints/test_table.py) | Tensor layout, offsets, pair keys and payload limits. |
| [Artifact tests](../../../tests/constraints/test_artifact.py) | Safetensors round trips and multi-Trie metadata. |
| [Backend parity tests](../../../tests/constraints/test_backend_parity.py) | CPU and torch fallback against the shared scoring contract. |
| [NPU routing tests](../../../tests/constraints/test_npu_multi_trie_routing.py) | Per-row selector expansion and grouped dispatch. |

## Source map

| File | Responsibility |
| --- | --- |
| [Table validation](../../../vllm_gr/constraints/table.py) | Tensor shapes, offsets, pair keys and limits. |
| [Artifact I/O](../../../vllm_gr/constraints/artifact.py) | Reads and writes single- and multi-Trie safetensors. |
| [CUDA backend](../../../vllm_gr/constraints/cuda_backend.py) | CUDA lookup, slicing and selection. |
| [NPU tables](../../../vllm_gr/constraints/npu_tables.py) | Six-table views, rebasing and stage vocabulary. |
| [NPU backend](../../../vllm_gr/constraints/npu_backend.py) | Per-row routing and `rec_constrained_topk`. |
| [CPU backend](../../../vllm_gr/constraints/cpu_backend.py) | Host-table selection and the shared scoring contract. |
