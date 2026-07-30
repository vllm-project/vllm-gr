# Beam Search Decision Helper Contract

## Purpose

`vllm_gr/beam_search_decision_utils.py` provides the decision boundary shared
by Offline and OpenAI Beam Search after each entry point has generated and
filtered its candidates.

Given aligned candidate arrays, the shared boundary owns:

1. stop-token completion and score masking;
2. Top-K index selection;
3. active child Beam construction;
4. `(parent_index, token_id)` fork mapping.

The module uses the existing NumPy arrays and `BeamSearchSequence` objects. It
does not own engine transport, request lifecycle, or Catalog policy.

## Catalog Boundary

Catalog lookup and candidate modification remain entry-point responsibilities:

- Offline uses `mask_candidate_logprobs()`, preserving candidate IDs, count,
  and order while masking invalid scores.
- OpenAI uses `Catalog.apply_catalog_filter()`, whose `fill` strategy may
  replace candidates and whose `null` strategy may leave fewer candidates.

These behaviors are intentionally not unified by the decision helpers.
Changing their equivalence or fill semantics is a separate correctness change.

The handoff to the shared boundary is:

```text
entry-point candidate extraction and Catalog filtering
    |
    v
token_ids + scores + beam_idx + beam_flat_cache
    |
    v
complete_eos_candidates
    |
    v
select_top_indices
    |
    v
materialize_selected_beams
```

`beam_idx[candidate_index]` is the authoritative parent mapping. Candidate
blocks are not required to have a fixed width.

## Candidate Utilities

### `mask_candidate_logprobs`

```python
def mask_candidate_logprobs(
    token_ids: Sequence[int],
    logprobs: Sequence[float],
    allowed_token_ids: Collection[int] | None,
) -> list[float]:
    ...
```

This Offline utility:

- returns a new list;
- preserves candidate order and token IDs;
- replaces disallowed scores with negative infinity;
- retains the historical `zip(token_ids, logprobs)` length behavior.

Catalog lookup remains the caller's responsibility.

### `extract_and_dedup_segment`

The helper extracts one candidate block and applies the historical sampled
token rule to token IDs, scores, ranks, and decoded tokens together:

- if the sampled token is duplicated in Top-K, drop the sampled position;
- otherwise retain the first `logprobs_num` entries;
- preserve `None` optional metadata;
- preserve the caller's grouped stride boundaries.

## Shared Decisions

### `complete_eos_candidates`

```python
def complete_eos_candidates(
    *,
    parents: Sequence[BeamSearchSequence],
    token_ids: np.ndarray,
    scores: np.ndarray,
    beam_idx: np.ndarray,
    beam_flat_cache: Sequence[BeamCandidateCache | None],
    stop_token_ids: Collection[int],
    initial_logprobs: FlatLogprobs,
    include_eos_token: bool,
    inherit_lora_request: bool,
) -> tuple[list[BeamSearchSequence], np.ndarray]:
    ...
```

Contract:

- complete every candidate whose token ID is in `stop_token_ids`;
- preserve candidate-index order when producing completed Beams;
- resolve the parent with `int(beam_idx[candidate_index])`;
- use the candidate's actual token ID as `stop_reason`;
- set `finish_reason="stop"`;
- assign `_lp_parent` and the matching parent candidate cache;
- return a copy of `scores` with all stop candidates masked to negative
  infinity;
- leave the input score array unchanged.

The entry points intentionally use different output policies:

| Entry point | Stop tokens | Include stop token | Inherit stop Beam LoRA |
| --- | --- | --- | --- |
| Offline | tokenizer EOS | `True` | `True` |
| OpenAI | tokenizer EOS and optional SID End | `include_stop_str_in_output` | `False` |

When `ignore_eos=True`, callers skip completion and masking.

### `select_top_indices`

```python
def select_top_indices(
    scores: np.ndarray,
    beam_width: int,
    catalog_strategy: str = "fill",
) -> np.ndarray:
    ...
```

Contract:

- always return an integer `np.ndarray`;
- do not sort selected indices;
- preserve NumPy `argpartition` ordering and tie behavior.

For `fill`:

- retain the legacy `np.argpartition(np.negative(scores), beam_width)` path;
- retain negative-infinity fallback behavior;
- return all indices when the candidate count is at most `beam_width`.

For `null`:

- select only finite scores;
- return every finite index when there are at most `beam_width`;
- return `np.empty(0, dtype=np.intp)` when no finite candidate exists.

This helper does not replace, remove, or mask Catalog candidates.

### `materialize_selected_beams`

```python
def materialize_selected_beams(
    *,
    parents: Sequence[BeamSearchSequence],
    selected_indices: Sequence[int],
    token_ids: np.ndarray,
    scores: np.ndarray,
    beam_idx: np.ndarray,
    beam_flat_cache: Sequence[BeamCandidateCache | None],
    initial_logprobs: FlatLogprobs,
) -> tuple[list[BeamSearchSequence], list[tuple[int, int]]]:
    ...
```

Contract:

- construct child Beams in `selected_indices` order;
- resolve every parent with `int(beam_idx[candidate_index])`;
- inherit the parent's prompt and active-Beam LoRA request;
- assign `_lp_parent` and the matching parent candidate cache;
- convert the selected token and score to Python `int` and `float`;
- return one `(parent_index, token_id)` fork entry per child in the same
  order.

The fork mapping is consumed by grouped BeamRequest transport. Ordinary and
grouped serving paths still use the same materialization decision.

## Entry-Point Ownership

Offline continues to own synchronous engine execution, CUSTOM grouped
execution, Catalog lookup and masking, session cleanup, and final output
construction.

OpenAI continues to own `EngineClient` calls, Catalog filtering, request IDs,
BeamRequest lifecycle and cleanup, streaming/error output, data-parallel rank,
metrics, and final response formatting.

The shared module must not import or own those orchestration concerns.

## Compatibility Requirements

Changes at this boundary must preserve:

- candidate-index ordering;
- explicit candidate-to-parent geometry;
- completed Beam ordering and scores;
- stop masking before Top-K;
- fill `argpartition` behavior;
- null finite-only behavior;
- active child ordering and lineage;
- grouped `fork_info`;
- Offline and OpenAI output-policy differences;
- existing Catalog behavior in each entry point.
