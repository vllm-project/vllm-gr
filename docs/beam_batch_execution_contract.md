# Beam Batch Execution Contract

## Status and scope

This document defines protocol version 1 of the prospective Worker-owned Beam
batch boundary and its pure NumPy CPU reference implementation. This change
defines transport-safe data, validation, and the correct Beam selection result.
It does not connect the contracts to production execution.

This change includes:

- Beam batch Request, Plan, and Result contracts;
- a typed, fail-closed MsgPack codec for all three contracts;
- structural, deployment-limit, Plan-completeness, and Result validation;
- a pure NumPy CPU reference that starts after candidate generation;
- hand-computed golden, legacy differential, and OneRec parity tests; and
- lazy imports for existing Beam runtime primitives.

It does not add an EngineCore wire enum, change `process_input_sockets()`,
implement Scheduler FCFS or atomic reservation, allocate Prompt KV, call a
ModelRunner, load Constraint resources, implement Worker run-to-completion,
restore the removed legacy GR step-update protocol, or modify the production
persistent Constraint Table path. Chunked Prefill, mixed ordinary Requests, and Decode
micro-batching are also out of scope.

## Vocabulary

- `batch`: one business Beam batch;
- `item`: one request in the batch;
- `beam`: one candidate that is still competing;
- `row`: one physical tensor or buffer row;
- `sequence`: one final result returned to the caller;
- `parent_beam_index`: the logical Beam that produced a candidate.

The public API uses `Params`, `Options`, `Limits`, `Output`, `Result`, and
`Failure`. It does not introduce Child, Lane, Lease, RunSlot, Session, Step,
Epoch, Fragment, or Continuation protocol types. Types have no `V1` suffix.

## Three top-level contracts

The protocol has exactly three top-level transport-safe contracts:

```text
BeamBatchRequest
        |
        v
BeamBatchPlan + native SchedulerOutput
        |
        v
BeamBatchResult
```

All three carry `protocol_version == 1` and the same `batch_id`. A batch ID is
unique across in-flight work and retained results. A retry creates a new batch
ID, so protocol version 1 has no `execution_id` or `run_id`.

The codec supports typed round trips for all three values. This specifies their
serialization; it does not claim that a production transport channel exists in
this change.

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

The Request keeps `EngineCoreRequest` input generic. It does not derive prompt
length from `prompt_token_ids` and does not reject modality by inspecting
Request fields. Future EngineCore runtime admission must reject unsupported
modalities before scheduling. The Worker must not be the first component to
discover a predictable unsupported-input failure.

### Plan

```python
class BeamPrefillData:
    item_index: int
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int
    num_scheduled_tokens: int


class BeamBatchPlan:
    protocol_version: int
    batch_id: str
    items: tuple[BeamPrefillData, ...]
    beam_params: BeamBatchParams
    output_options: BeamOutputOptions
```

`BeamBatchPlan` is a validated scalar snapshot. The native
`SchedulerOutput` remains the execution authority for request data, scheduled
tokens, block tables, and slot mappings:

```text
Native SchedulerOutput             BeamBatchPlan
|-- request data                   |-- batch/item identity
|-- scheduled token counts         |-- Beam parameters
|-- block tables                   |-- validated Prefill counts
`-- slot mappings                  `-- output options
              \                     /
               future BeamWorkerInput
```

The integration boundary must cross-check Plan counts and identities against
the native `SchedulerOutput` before invoking the Executor. The Plan does not
copy Prompt KV bindings, block tables, or slot mappings.

Full Prefill is expressed by one equation per item:

```text
num_computed_tokens + num_scheduled_tokens == num_prompt_tokens
```

This rejects chunked Prefill without adding chunk, barrier, fragment, or
accumulator state. Native InputProcessor and Scheduler accounting provide the
authoritative prompt token counts.

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

Only valid sequences are returned. An item may return fewer than R sequences,
including zero, without failing the batch. There is no padding `valid` bit or
redundant missing-sequence count.

When a stop token is excluded from `token_ids`, its log probability still
contributes to `cumulative_logprob`, its generation still appears in
`parent_beam_indices`, and `stop_token_id` remains populated. A `length`
sequence contains exactly `max_tokens` token IDs.

Failure codes are open, non-empty strings. Expected future integration codes
include `UNSUPPORTED_INPUT`, `ADMISSION_REJECTED`, `PLAN_MISMATCH`,
`MODEL_FAILURE`, and `INTERNAL_ERROR`; this list is descriptive, not an enum.

## Fixed Beam KV Buffer

Protocol version 1 assumes one active Beam batch and one fixed Worker-owned
Beam KV Buffer. There are no multiple selectable buffers, reservations, leases,
generations, or buffer identities in the contract.

Every batch starts from a reset logical state. For the first runtime
integration, the Worker must physically zero the fixed Beam KV Buffer before a
new batch so stale rows cannot affect correctness. Ordinary writes may then
overwrite the rows used by the batch. Buffer reset is runtime behavior and is
not reported in `BeamBatchResult`.

The deployment limit `BeamBatchLimits.max_num_beam_rows` admits or rejects
`B * W`; the Plan does not report that a slot was acquired.

## Validation layers

Validation is deliberately staged so predictable failures occur before Worker
execution:

```text
Codec and contract validation
        |
EngineCore runtime admission
        |  reject unsupported modality/resource combination
Native InputProcessor accounting
        |  authoritative prompt token count
Scheduler and Plan validation
        |  full Prefill, limits, native SchedulerOutput cross-check
Executor boundary
        |
Worker defensive assertions
```

The pure Contract layer validates backend-neutral shape and identity. Future
runtime admission owns modality support. Plan validation owns facts that only
exist after native prompt processing and scheduling. Worker checks are a final
defense, not the primary admission mechanism.

### Structural validation

- non-empty batch and contiguous `item_index` values from zero;
- unique, non-empty `EngineCoreRequest.request_id` values;
- positive Beam width, `max_tokens`, and return width;
- `num_return_sequences <= beam_width`;
- finite, non-negative temperature;
- unique, non-negative stop token IDs;
- `sampling_params` present and `pooling_params` absent;
- protocol version 1 resource ID; and
- full Prefill equation for every Plan item.

Python booleans are not accepted as integers.

### Deployment limits

`BeamBatchLimits` is a non-wire deployment snapshot:

```python
class BeamBatchLimits:
    max_batch_size: int
    max_beam_width: int
    max_tokens: int
    max_num_beam_rows: int
    max_num_batched_tokens: int
    max_model_len: int
    supported_constraint_resource_ids: tuple[str, ...]
```

Admission checks:

- `B <= max_batch_size`;
- `B <= max_num_seqs`;
- `W <= max_beam_width`;
- `max_tokens <= limits.max_tokens`;
- `B * W <= max_num_beam_rows`;
- `sum(num_scheduled_tokens) <= max_num_batched_tokens`; and
- `num_prompt_tokens + max_tokens <= max_model_len` for each item.

The scheduled-token budget intentionally uses `num_scheduled_tokens`, while
model-length feasibility intentionally uses the complete authoritative prompt
length. The two values must not be conflated.

The current Constraint Table deployment uses `max_beam_width=256`,
`max_tokens=3`, and `supported_constraint_resource_ids=("default",)`. Future
capacity changes update Limits, not the protocol ABI.

## CPU candidate boundary

The CPU reference starts after candidate generation. It does not query the
Constraint Table, run logits TopK, parse physical Beam rows, or manipulate KV.

```python
class BeamItemCandidates:
    item_index: int
    parent_beam_indices: np.ndarray  # [P]
    token_ids: np.ndarray            # [P, K]
    logprobs: np.ndarray             # [P, K]
    valid_mask: np.ndarray           # [P, K]
```

Bootstrap uses `P=1`. Decode uses `P=num_active_beams`, with
`P <= W`. The reference never fabricates W Prefill parent rows. Production
Constraint execution uses `K=W`; the CPU reference permits any positive K to
make semantics testable independently of deployment capacity.

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

Raw-logit ties inside a per-parent Constraint TopK operation are outside this
change. The CPU reference assumes each parent row already has a deterministic
candidate order and freezes only global Beam selection semantics.

## Package boundary

`vllm_gr.v1.beam` lazily exposes existing production runtime primitives only.
New contracts, codec, and CPU reference APIs are imported explicitly from
their submodules. Importing the Beam package or `cpu_reference` does not import
Torch through the CPU reference path.
