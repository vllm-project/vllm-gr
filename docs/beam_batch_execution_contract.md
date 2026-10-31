# Beam Batch Execution Contract

## Production boundary

Beam Search V1 submits a `BeamBatchRequest` through the atomic
`SUBMIT_GR_SESSION` EngineCore message and returns a `BeamBatchResult` through
`EngineCoreOutput.kv_transfer_params["gr_batch_result"]`. These are the two
transport contracts. Native `SchedulerOutput` carries scheduling data; there
is no separately serialized Plan message.

```text
GRLLM.beam_search_v1 / OpenAI Beam V1
  -> input processing and request validation
  -> SUBMIT_GR_SESSION(BeamBatchRequest)
  -> EngineCore admission and native SchedulerOutput
  -> Worker Prefill / Decode stages and resource retirement
  -> BeamBatchResult
  -> frontend result validation and public outputs
```

Contracts live in `vllm_gr/v1/beam/batch_contracts.py`. Production request
transport uses the typed decoder in `vllm_gr/v1/engine/codec.py`, and final
result validation lives in `gr_session_client._terminal_result`.
The NumPy CPU reference is an independent selection oracle, not the production
scheduler or Worker loop.

## Vocabulary

- `batch`: one business Beam batch / GR session;
- `item`: one native request in that batch;
- `beam`: one candidate competing within an item;
- `row`: one physical tensor or buffer row;
- `sequence`: one returned candidate;
- `parent_beam_index`: the logical Beam producing a candidate.

Request and Result carry `protocol_version == 1` and the same `batch_id`.
Native `EngineCoreRequest.request_id` identifies each item. A retry uses a new
batch identity; execution ownership and dispatch generations remain runtime
state, not additional public request fields.

### Deployment compatibility

All frontend, EngineCore, and Worker processes participating in Beam execution
must use the same code and schema revision. Stop and restart these processes as
one deployment; mixed-revision operation and rolling upgrades are unsupported
for both fixed-width and dynamic-width Beam requests. A matching
`protocol_version == 1` alone does not establish schema compatibility.

`BeamBatchParams` uses an array-like MsgPack struct with strict unknown-field
validation. Its appended `dynamic_beam_widths`, `candidate_width`, and
`per_parent_beam_widths` fields have defaults so a new decoder can read an older,
shorter array. However, array encoding still emits trailing fields when their
values are `None`; an old strict decoder rejects that new array even for a
fixed-width request. Supporting rolling upgrades would
require capability negotiation and compatible encoding, not just a protocol
number change or relaxing validation.

### Request

```python
class BeamBatchItem:
    item_index: int
    request: EngineCoreRequest


class BeamBatchParams:
    beam_width: int
    max_tokens: int
    temperature: float
    stop_token_ids: tuple[int, ...]
    ignore_stop_tokens: bool
    constraint_resource_id: str = "default"
    length_penalty: float = 1.0
    catalog_strategy: Literal["fill", "null"] | None = None
    dynamic_beam_widths: tuple[int, ...] | None = None
    candidate_width: int | None = None
    per_parent_beam_widths: tuple[int, ...] | None = None


class BeamOutputOptions:
    num_return_sequences: int
    include_stop_token: bool
    include_cumulative_logprob: bool
    include_parent_beam_indices: bool


class BeamBatchRequest:
    protocol_version: int
    batch_id: str
    items: tuple[BeamBatchItem, ...]
    beam_params: BeamBatchParams
    output_options: BeamOutputOptions
```

`EngineCoreRequest.request_id` is the single request identity. The contract does
not duplicate it in `BeamBatchItem`.

`BeamBatchParams` is authoritative for the Worker Beam loop. Native
`EngineCoreRequest.sampling_params` is per-forward adapter state, so its
`max_tokens` does not need to equal Beam `max_tokens`.

Beam `max_tokens` counts generated tokens, including the token selected from
Prefill logits:

```text
1 Prefill candidate selection + N - 1 Decode forwards = N generated tokens
```

Protocol version 1 admits only `constraint_resource_id="default"`. It does not
transport a catalog path, digest, Trie, table payload, backend adapter,
`constraint_step`, or `beam_suffix`.

### Result

```python
class BeamOutputSequence:
    token_ids: tuple[int, ...]
    cumulative_logprob: float | None
    parent_beam_indices: tuple[int, ...]
    finish_reason: Literal["stop", "length"]
    stop_token_id: int | None


class BeamRequestOutput:
    item_index: int
    request_id: str
    sequences: tuple[BeamOutputSequence, ...]


class BeamBatchOutput:
    items: tuple[BeamRequestOutput, ...]


class BeamBatchFailure:
    code: str
    message: str
    item_index: int | None


class BeamBatchResult:
    protocol_version: int
    batch_id: str
    output: BeamBatchOutput | None
    failure: BeamBatchFailure | None
```

Exactly one of `output` and `failure` is present. A failure never carries a
partial output.

Internal Decode state contains `B * W` Beam rows. Final output contains up to
`B * R` compact sequences, where:

```text
R = num_return_sequences
1 <= R <= W
```

Only valid sequences are returned, without padding validity bits. The generic
contract and CPU oracle can represent empty items and stop completion. The
production V1 boundary is stricter: it requires at least one sequence per
successful item, `finish_reason="length"`, the effective generation budget,
and the score/parent fields selected by the request. A Worker with no valid
final candidates returns `no_valid_beams` instead of a successful empty item.

Failure codes are non-empty strings, such as `admission_rejected`,
`no_valid_beams`, and `worker_result_unavailable`. A failure may identify any
item in the batch: `item_index` is either `None` or in `[0, len(request.items))`.
The frontend validates nested field types and identities on both in-process
and serialized result paths. An invalid transport payload is not a success.

The CPU oracle also defines stop-token exclusion: the excluded stop still
contributes to cumulative score and parent history, and `stop_token_id`
remains populated. This does not enable early stopping in the production V1
path, which currently requires `ignore_eos=True`.

## Scheduling and resource ownership

Native input processing and scheduling own prompt accounting, KV blocks,
slot mappings, and scheduled token counts. GR stage metadata carries session,
member, dispatch, and generation identities for the Worker execution view.
It must agree with the native scheduling data.

A complete Prefill may reuse prefix-cache tokens. For every item:

```text
num_computed_tokens + num_scheduled_tokens == num_prompt_tokens
```

The scheduled count need not equal the full prompt length on a cache hit.
Chunked/partial Prefill and recompute after generation are rejected. Decode
advances native accounting by one logical token while Worker execution uses
physical Beam rows. NPU stage widths and CUDA fixed-width state have their
own adapter checks.

The runtime currently admits one execution owner with two dispatch slots.
Persistent buffers are rebound/reset under matching ownership; physical KV
zeroing is not a wire-protocol requirement. Completion, cancellation, and
failure must drain outstanding work before resources are reused. Terminal
results are delivered only after matching resource-release confirmation.
See [asynchronous scheduling](async_scheduling.md) for receipts and retirement.

## Validation boundaries

| Boundary | Responsibility |
| --- | --- |
| `batch_contracts` / typed codec | Structural types, protocol, identities, parameters, output shape |
| V1 `input_processor` | Render/tokenize once, create native requests, validate device/config/request |
| EngineCore `validate_gr_request` | Repeat admission checks at the process boundary before scheduling |
| GR scheduler and Worker execution view | Capacity, complete remaining Prefill, native counts, member order and dispatch ownership |
| `_terminal_result` | Normalize in-process/serialized data, match batch/items, validate failure index and requested result fields |

Structural checks include contiguous item indices starting at zero, unique
non-empty request IDs, positive widths and generation budgets,
`num_return_sequences <= beam_width`, finite non-negative temperature, valid
stop IDs, sampling rather than pooling, and the supported resource identity.
Booleans are not accepted in place of integers.

`BeamBatchLimits` is a non-wire capacity snapshot. Runtime admission also
checks model length, token budget, native request capacity, Beam row capacity,
and supported device features. Logical B=1..8 is supported by admission; the
online batch/capture policy selects which exact sizes are enabled. Do not
confuse this policy with native `max_num_seqs` capacity.

Constraint Table generation supports up to three effective output steps and a
per-row candidate width (`K`) of at most 256. The Beam width (`W`, rows per
request) is a separate dimension with its own operator ceiling of 1024. The ceiling is
raised together with the scratch-tiling fix that makes the wide-`W` selection
path correct; it also gates the raw Beam operators, so the constant and the
kernels move as one. Widths above 256 take the hierarchical path; the fused
`O(W**2)` merge kernels stay gated on `K == W` and `W in {64, 128, 256}`.

On CUDA the serving path consumes `K` end to end: the sampler is asked for `K`
logprobs per row, the postprocess workspace is keyed by `(W, K)`, and every
stage's selection takes `K` candidates per parent. A `beam_max_width` above 256
therefore needs an explicit `candidate_width`; left unset, `K` follows `W` and
exceeds the 256 candidate ceiling, which is rejected at parse time. With a
dynamic schedule, a configured `K` must be at least every stage width and is the
top-k of every stage. An unset `K` is not written onto the request: each stage
keeps taking its own output width as top-k, exactly as before `K` existed, and
the sampler, workspace and `BeamSearchState` are sized from `W`. NPU still
requires `K == W` and raises for any other combination.

CUDA V1 requests can additionally configure each stage's per-parent `K` with
`per_parent_beam_widths` through the dynamic execution path; such a request
schedule overrides the configured `K` (see the candidate boundary below).

Unconstrained V1 generation has its own configuration and budget checks; it
does not require a catalog artifact. NPU dynamic widths must satisfy their
stage-specific capacity/parent bounds. These deployment checks are implemented
in `vllm_gr/v1/engine/beam_search_v1.py`, not in a parallel Plan schema.

## CPU candidate boundary

The CPU reference starts after candidate generation. It does not query the
Constraint Table, run logits TopK, parse physical Beam rows, or manipulate KV.

```python
class BeamItemCandidates:
    item_index: int
    parent_beam_indices: np.ndarray  # [P]
    token_ids: np.ndarray  # [P, K]
    logprobs: np.ndarray  # [P, K]
    valid_mask: np.ndarray  # [P, K]
```

Bootstrap uses `P=1`. Decode uses `P=num_active_beams`, with
`P <= W`. The reference never fabricates W Prefill parent rows.

`K` is modelled as an independent dimension of `W` throughout this contract,
with `1 <= K <= MAX_CANDIDATE_WIDTH`. Without `per_parent_beam_widths`,
candidate generation uses the configured `K`, or `K=W_out` when it is unset.
CUDA requests may configure a per-stage candidate schedule: bootstrap requires
`K_1=W_1`, and later stages require `W_i <= W_(i-1)*K_i`. Every `K_i` is
limited to 256, so a request schedule needs `W_1 <= 256`.
An explicit request schedule uses dynamic selection, which flattens the
`P*K` candidate pool and selects W winners.

An explicit `per_parent_beam_widths` schedule overrides the configured
`constraint_table.candidate_width` stage by stage, including when W varies by
stage, and is validated on its own (`input_width * K >= output_width`). Backend
admission checks the actual request K schedule against `max_top_k`. Without a
request K schedule, a configured `K` is the top-k of every stage and must be at
least every dynamic stage width; an unset `K` keeps each stage's output width.

The constrained Top-K, the postprocess workspace, and `BeamSearchState`
capacity are all sized from `K`; rows stay `W`. `K != W` takes the hierarchical
selection path.

Both candidate adapters provide FP32 local log probabilities. The selection
core does not compute softmax.

### Sampler adapter

Native Sampler input is `[sampled, top1, ..., topK]`. If the sampled token is in
a valid Top-K slot, slot zero is removed. Otherwise the first K slots are kept,
including sampled.

### Constraint adapter

Constraint TopK has no sampled slot and does not deduplicate duplicate token
IDs. Its adapter supplies an explicit validity mask. The finite `-1e20` padding
sentinel is interpreted by the adapter; the core never uses `isfinite()` to
infer Constraint validity.

## CPU Beam semantics

The public stages are:

```text
select_initial_beams:   one Prefill parent -> W Beam state
select_next_beams:      active Beam state -> next W Beam state
select_final_sequences: completed + final candidates -> compact Top-R output
```

At every generation, cumulative log probability is calculated in FP32:

```text
np.float32(parent_logprob) + np.float32(local_logprob)
```

Within one round, selection order is:

```text
cumulative_logprob descending
then lower parent-major logical flat index
```

For parent Beam `p`, candidate rank `r`, and candidate width K:

```text
tie_break_index = p * K + r
```

Score order decides which candidates survive. The selected active candidates
are then stably grouped by parent Beam to define execution Beam order for the
next Beam KV reorder. Selection order and execution order are distinct.

Terminal candidates enter the completed pool before active Top-W selection.
The final stage merges historical completed Beams with every valid candidate
from the final round, selects Top-R, and does not construct another active
state or perform a KV reorder.

For equal final scores from different generations, the earlier generation wins;
the per-generation logical flat index breaks the remaining tie. There is no
length penalty or final token-sequence deduplication in protocol version 1.

The production NPU V1 dynamic-width path has a narrower, fixed-horizon scope:
the frontend requires `ignore_eos=True`, and engine admission requires
`ignore_stop_tokens=True`. Every surviving path within a request generates the
same number of tokens (N=1, 2, or 3). Consequently final length normalization
applies one common denominator, and selecting Top-R by cumulative log
probability before normalization preserves ranking. This does not extend to
variable-length early completion, which needs separate ranking semantics; the
generic CPU reference above continues to cover stop completion independently.

Raw-logit ties inside a per-parent Constraint TopK operation are outside this
change. The CPU reference assumes each parent row already has a deterministic
candidate order and freezes only global Beam selection semantics.

## Package boundary

`vllm_gr.v1.beam` lazily exposes existing production runtime primitives only.
New contracts, codec, and CPU reference APIs are imported explicitly from
their submodules. Importing the Beam package or `cpu_reference` does not import
Torch through the CPU reference path.
