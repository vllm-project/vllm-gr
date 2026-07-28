# Beam Search Decision Helper Contract

## Purpose

`vllm_gr/beam_search_decision_utils.py` is the shared decision boundary for
Offline and OpenAI/Online Beam Search. It owns the candidate transformations
that must remain identical across the two entry points:

1. candidate masking;
2. sampled-token deduplication;
3. EOS discovery and completion;
4. Top-K index selection;
5. active child Beam construction and grouped-request fork mapping.

The module uses stateless, module-level functions. There is intentionally no
stateful helper class or intermediate Beam domain model.

The helpers preserve the existing list and NumPy data flow. This boundary is
for sharing decisions, not transport or request orchestration.

## Ownership

### Shared decision module

`vllm_gr/beam_search_decision_utils.py` owns:

- candidate score masking without candidate removal;
- candidate segment extraction and sampled-token deduplication;
- EOS index discovery;
- EOS-completed Beam construction and EOS score masking;
- legacy NumPy Top-K selection;
- selected child Beam construction;
- `(parent_index, token_id)` fork mapping.

### Offline entry point

`vllm_gr/entrypoints/gr.py` owns:

- Offline engine submission and output collection;
- ordinary and CUSTOM grouped execution;
- Catalog lookup timing;
- session IDs and grouped-request cleanup;
- NPU grouped Beam geometry;
- final logprob reconstruction, sorting, and output construction.

### OpenAI serving entry point

`vllm_gr/entrypoints/openai/serving_engine.py` owns:

- asynchronous `EngineClient` interaction;
- ordinary and grouped `BeamRequest` transport;
- streaming and error `RequestOutput` behavior;
- abort and cleanup timing and payloads;
- data-parallel rank selection and propagation;
- request IDs, child request IDs, priority, and trace headers;
- metrics and final response formatting.

The shared module must not import or own these orchestration concerns.

## Public Functions

### `mask_candidate_logprobs`

```python
def mask_candidate_logprobs(
    token_ids: Sequence[int],
    logprobs: Sequence[float],
    allowed_token_ids: Collection[int] | None,
) -> list[float]:
    ...
```

Contract:

- returns a new list;
- preserves candidate order;
- does not remove candidates;
- replaces a disallowed candidate score with negative infinity;
- returns a list copy when `allowed_token_ids` is `None`;
- preserves the historical `zip(token_ids, logprobs)` length semantics.

Catalog lookup remains the caller's responsibility.

### `extract_and_dedup_segment`

```python
def extract_and_dedup_segment(
    raw_token_ids: Sequence[int],
    raw_logprobs: Sequence[float],
    raw_ranks: Sequence[int | None] | None,
    raw_decoded: Sequence[str | None] | None,
    start_offset: int,
    end_offset: int,
    logprobs_num: int,
) -> tuple[
    Sequence[int],
    Sequence[float],
    Sequence[int | None] | None,
    Sequence[str | None] | None,
]:
    ...
```

Flat candidate blocks use the layout:

```text
[sampled token, top-1, ..., top-K]
```

When the block contains more than `logprobs_num` elements:

- if the sampled token appears in the following Top-K entries, the sampled
  position is dropped;
- otherwise the first `logprobs_num` entries are retained.

The same slice is applied to token IDs, scores, ranks, and decoded tokens.
`None` ranks or decoded-token metadata remain `None`.

Grouped callers calculate offsets with their existing floor-stride logic.
Any remainder outside the supplied offsets remains ignored; the helper does
not reinterpret the grouped layout.

### `find_eos_indices`

```python
def find_eos_indices(
    token_ids: np.ndarray,
    eos_token_id: int,
) -> np.ndarray:
    ...
```

The result is exactly the legacy NumPy expression:

```python
np.where(token_ids == eos_token_id)[0]
```

Its ordering and dtype are preserved.

### `complete_eos_candidates`

```python
def complete_eos_candidates(
    *,
    parents: Sequence[BeamSearchSequence],
    token_ids: np.ndarray,
    scores: np.ndarray,
    beam_flat_cache: Sequence[BeamCandidateCache | None],
    candidates_per_parent: int,
    eos_token_id: int,
    initial_logprobs: FlatLogprobs,
    include_eos_token: bool,
    inherit_lora_request: bool,
) -> tuple[list[BeamSearchSequence], np.ndarray]:
    ...
```

Contract:

- materializes every EOS candidate before active Beam selection;
- keeps EOS candidates in the candidate arrays;
- returns a copy of `scores` with every EOS score set to negative infinity;
- leaves the input score array unchanged;
- maps a candidate to its parent with
  `candidate_index // candidates_per_parent`;
- preserves EOS discovery order;
- sets `finish_reason="stop"` and `stop_reason=eos_token_id`;
- assigns `_lp_parent` and the parent's `_lp_step_data` candidate cache;
- preserves `int(token_id)` and `float(score)` conversion;
- uses the supplied `initial_logprobs` reference;
- raises `AssertionError` when the selected parent cache is `None`.

The two entry points intentionally pass different output policy flags:

| Entry point | `include_eos_token` | `inherit_lora_request` |
| --- | --- | --- |
| Offline | `True` | `True` |
| OpenAI | `include_stop_str_in_output` | `False` |

These differences are compatibility requirements and must not be unified as
part of a decision-sharing refactor.

When `ignore_eos=True`, callers skip this helper completely.

### `select_top_indices`

```python
def select_top_indices(
    scores: np.ndarray,
    beam_width: int,
) -> np.ndarray:
    ...
```

The helper preserves the legacy selection expression:

```python
if scores.size > beam_width:
    return np.argpartition(np.negative(scores), beam_width)[:beam_width]
```

Otherwise it returns NumPy's integer index range from zero to
`scores.size`.

Callers and future changes must not:

- sort the returned indices;
- replace partitioning with a full sort or heap;
- filter negative-infinity candidates;
- change tie behavior;
- change the NumPy dtype.

This includes the historical fallback behavior when every score is negative
infinity.

### `materialize_selected_beams`

```python
def materialize_selected_beams(
    *,
    parents: Sequence[BeamSearchSequence],
    selected_indices: Sequence[int],
    token_ids: np.ndarray,
    scores: np.ndarray,
    beam_flat_cache: Sequence[BeamCandidateCache | None],
    candidates_per_parent: int,
    initial_logprobs: FlatLogprobs,
) -> tuple[list[BeamSearchSequence], list[tuple[int, int]]]:
    ...
```

Contract:

- constructs active child Beams in `selected_indices` order;
- maps candidates to parents with
  `candidate_index // candidates_per_parent`;
- inherits the parent's `orig_prompt` and `lora_request`;
- assigns `_lp_parent` and `_lp_step_data`;
- preserves `int(token_id)` and `float(score)` conversion;
- uses the supplied `initial_logprobs` reference;
- returns fork entries in the same order as the child Beams;
- returns each fork entry as `(parent_index, token_id)`;
- raises `AssertionError` when the selected parent cache is `None`.

The returned fork mapping is consumed only by grouped request transport.
Ordinary execution may ignore it, but both paths use the same child Beam
construction.

## Preserved Decision Flow

Each entry point retains its existing orchestration while applying the shared
functions in this order:

```text
engine output
    |
    v
extract_and_dedup_segment
    |
    v
mask_candidate_logprobs
    |
    v
add parent cumulative scores
    |
    v
complete_eos_candidates  (unless ignore_eos)
    |
    v
select_top_indices
    |
    v
materialize_selected_beams
```

`find_eos_indices` is used internally by `complete_eos_candidates`.

Final completed-Beam sorting is not part of the shared helper boundary.

## Compatibility Invariants

Changes to the shared module must preserve:

- candidate count and order;
- Catalog masking without removal or reordering;
- candidate-to-parent geometry;
- completed EOS Beam order and score;
- EOS masking before Top-K;
- active child Beam order;
- parent lineage and candidate cache identity;
- grouped `fork_info`;
- NumPy `argpartition` ordering, ties, and all-negative-infinity behavior;
- optional ranks and decoded-token metadata;
- Offline EOS LoRA inheritance;
- OpenAI EOS output and LoRA policy;
- active child LoRA inheritance.

The shared decision boundary does not authorize changes to:

- Beam algorithms or performance policy;
- EngineCore wire or codec formats;
- Scheduler, ModelRunner, Attention, GPU, or NPU execution;
- Catalog lookup or completion semantics;
- request, response, streaming, abort, cleanup, DP, or metrics behavior;
- external APIs.

## Verification

The contract is covered by:

- `tests/test_beam_search_decision_utils.py` for focused helper behavior;
- `tests/test_offline_beam_parity.py` for Offline compatibility;
- `tests/test_beam_parity.py` for production-helper Online/Offline parity;
- `tests/test_serving_beam_lifecycle.py` for OpenAI transport and lifecycle
  compatibility.
