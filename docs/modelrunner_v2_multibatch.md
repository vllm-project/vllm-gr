# ModelRunner V2 multi-batch execution

ModelRunner V2 supports one Beam Search V1 call containing B=1 through 8 logical
requests. Each member has its own prompt, candidates, scores and KV history.
Beam width W is fixed by the engine configuration; Decode processes B * W
physical rows. B is not the beam width or the asynchronous dispatch depth.

## Reuse of native vLLM

The implementation keeps native request state, Prefix Cache, KV block tables
and Prefill execution. For an admitted all-new GR Prefill batch, registration
skips unused native sampler initialization and refresh, while preserving
request removal, model state, block IDs and per-slot prompt-logprob resets.
Every member must pass admission before any slot changes. Ordinary, mixed,
new-plus-cached and unsupported batches retain native registration.
`gpu_prefill_requests_v2` owns this registration and its staged-write ABI;
the V2 stage hooks own execution and lifecycle transitions.

Fresh batches on an empty native request set can flush token, total-length,
computed-token and single-group block-table writes through one packed transfer
and one Triton kernel. This includes multi-request batches with unequal prompt
lengths and reused native slots. The specialization requires complete prompts,
zero computed tokens and no previously queued writes; other registrations keep
native flushing. Queue contents are cleared only after successful submission,
and the subsequent native block-table flush still publishes block counts.

For supported pure Prefill batches, one Triton kernel prepares input tokens,
positions, sequence lengths, query offsets, logits indices and sampling row
indices. Multi-request batches and single requests with request padding transfer
one small packed descriptor containing native slot indices and query offsets.
The existing single-request path without request padding needs no descriptor
transfer. GPU computed-token state determines tokens and positions, while CPU
upper bounds remain available to native attention code.
Request padding has monotonic query offsets and zero sequence lengths. Sampling
indices have per-dispatch storage for asynchronous consumers. Block-table gather,
slot mapping and attention metadata construction remain native operations.

Native preparation may reorder requests. The V2 adapter selects the last-token logit rows in the
original logical member order before Beam sampling; it does not reorder
native request or KV state.

After Prefill, both runners use the existing GR stage runner, Beam state,
Constraint Table and Decode graph primitives. V2 alone uses a worker-owned
input allocation with room for 8 * W rows; each batch uses a view of that
storage. V2 startup captures the configured Decode shapes, defaulting to
B=1/2/4, bounded by `max_num_seqs`. Switching B updates buffer
contents and selects an existing graph without request-time capture. V2 can
fall back to eager execution for an individually unsupported capture after
checking cleanup and device health; other captured shapes remain usable.
Legal sizes outside the configured capture policy use eager Decode. A missing
graph for a successfully captured configured size remains an error.

V2 retirement passes all logical member IDs back through native request cleanup.
V2 model replacement and shutdown invalidate both Prefill and Decode captures
before the native lifecycle operation, including when loading fails. Prefill
cleanup releases graph/output bindings and clears readiness, capture results
and counters, then restores an uncaptured native manager. The next capture
installs a fresh GR plan and allocates output storage for the new model.
Decode replay also checks the captured model, context and input storage before
launching.

V2 constrained selection branches on the current device SID stage. Each branch
calls the shared CUDA Top-K kernel with that stage's complete maximum candidate
degree. Full-window score normalization and exact score/tie ordering remain
shared. This avoids processing first-stage padding in later stages without
predicting progress on the Host. V1 retains table-wide capacity for
device-driven selection. The `CudaStageHooks` contract requires an explicit
`specialize_constraint_steps` policy: `False` for V1 and `True` for V2.

V2 warms the stage-specialized kernel before publishing the initial constraint
backend or a replacement Trie. Warmup covers every supported power-of-two K
capacity for the model and configured output-head dtypes, including the table's
current stage degrees and Trie count. A capacity change therefore compiles
during table loading rather than the first request. Failed warmup releases the
candidate backend before it can become active.

### V1 compatibility boundary

ModelRunner V1 retains the behavior of the rebased parent `7ee1d95`: the GR
Prefill grid and per-request capacity clamp, exact native-input admission for
async Prefill, B=1 startup Decode capture, and lazy capture against stage-owned
buffers when a new batch needs a graph. Its Backend row bound, retirement and
shutdown semantics remain unchanged. Upstream V1/NPU fixes in that parent,
including B=2 Prefill capture, are retained.

Set `beam.adapter_cache_enabled: true` in the canonical service configuration
to opt into cross-batch adapter caching. It defaults to false, preserving the
previous latest-adapter reuse and lazy capture policy. Disabling this cache
does not disable Decode CUDA Graphs or same-batch adapter reuse. The legacy
configuration key is `vllm_gr_beam_adapter_cache_enabled`.

When this option and Decode graphs are enabled, the shared runner retains one drained
adapter per logical batch size while the context, workspace and search geometry
remain compatible. Returning to an already-warmed B reuses that adapter's fixed
input/attention addresses and its Decode graphs instead of clearing and
recapturing them. Each session still rebinds its own inputs and selectors.
Incompatible resources or geometry invalidate the bounded cache; V2 runtime
cleanup removes it before model replacement or shutdown.
The switch is read from service configuration, not from individual requests;
it does not add startup pre-capture or change schedule-miss behavior.

`CudaStageHooksV1` and `CudaStageHooksV2` subclass the `CudaStageHooks` abstract
base class in `gpu_stage_hooks.py`. The ABC rejects missing operations and policy
flags at construction. The class-side `CudaStageHooksType` Protocol additionally
checks constructor compatibility at binding sites; mypy checks operation signatures
and policy types. Hooks are constructed before session admission or graph-cache
invalidation, so construction failure leaves those resources available for retry.
The installed CUDA entrypoints explicitly bind
hooks for both execution and startup capture; the shared stage core has no V1
default. A Worker cannot switch its concrete hooks type after releasing a stage
owner, including during repeated startup: `validate_stage_hooks_type` compares
the bound class with the requested class. NPU dispatch retains its separate
adapter and does not import either CUDA implementation.

V1 retains stage-owned storage and lazy graph capture. V2 selects fixed Worker
storage, configured startup graph capture, batched Backend admission and
full-member retirement; its existing dynamic-shape recapture and recoverable
failure handling remain in the V2 implementation. The V2 installer alone supplies
the extended Backend capacity, kernel preparation and runtime cleanup callbacks.
Shared graph construction has an optional graph class; V2 uses a subclass for
storage/model validation.
There is no new environment switch or duplicated Beam state machine.

The contracts document eleven operations and three policy flags,
including Prefill row ordering/counter updates, the V1 Host-input fence, input
storage lifetime and capture recovery. They do not move remaining native field
cleanup out of the core or unify graph policies. The dedicated mypy group checks
the ABC, class-side Protocol, both implementations, core and binding sites together;
see [CUDA stage hooks contract](PRE_COMMIT.md#cuda-stage-hooks-contract).

Both runners retain upstream's opt-in vocabulary-sliced LM head. Decode hooks
may return `None` after staging hidden states for sliced sampling; those states
must remain valid when a later graph replay overwrites its output buffer.

## Configuration and scope

### vLLM 0.30 integration

V2 remains experimental and requires explicit opt-in; the default runner is V1.
The adapter consumes the native `BatchReqState`, preserves the native request
order and padding fields, and advances Prefill counts through
`postprocess_num_computed_tokens`. GR consumes and clears the pending native
forward state before producing its own Beam output.

The Prefill manager uses the 0.30 `capture() -> None` contract and keeps its
captured graphs in the native registry. Capture capacity follows the native
dummy batch's evenly distributed token counts. Ordinary requests retain the
native input preparation and sampling paths.

Graph memory profiling uses the same GR Prefill plan as startup and includes
Decode graphs and their fixed state. GR's token/request grid has nonuniform
capture costs, so its complete plan is measured in the native temporary pool
instead of extrapolating two FULL samples. This adds startup capture work.
The adapter retires temporary Decode graphs, inputs, owners and newly created
Beam context before real initialization. It preserves the enclosing capture
window and does not freeze the serving heap during profiling.

The worker applies this budget when
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`. This does not add a profiling pass
when upstream skips it (for example, native `cudagraph_mode=NONE` or an explicit
KV cache byte budget); those configurations need their own memory allowance.
Memory estimates remain configuration dependent and are not a performance
baseline.

Functional tests use `shutdown_timeout=10` and assert EngineCore exits with
code zero; inference success alone does not establish clean shutdown.

### Supported configuration

Use the Beam configuration and Constraint Table artifact described in the
[asynchronous quickstart](onerec_v1_async_quickstart.md). Set
`VLLM_USE_V2_MODEL_RUNNER=1` before importing vLLM, and submit the complete
batch through `GRLLM.beam_search_v1()`. `beam.execution_mode="v1"` selects the
Beam protocol; it does not select ModelRunner V1.

`beam.online_batch_sizes` defaults to `[1, 2, 4]`. For example, `[1, 3, 7]`
enables fixed HTTP batches of 3 and 7 and selects those startup Decode capture
sizes. It also selects Prefill request counts unless the Prefill-specific
environment override is set. Increase `max_num_seqs` explicitly to at least 7;
the policy never enlarges engine capacity automatically.

For B=4, set `max_num_seqs` to at least 4 and make
`max_num_batched_tokens` large enough for the sum of all remaining prompt
tokens. `max_model_len` remains a per-member limit. The native request capacity
does not need to equal B * W; GR allocates the expanded Decode rows separately.
Keep request beam width equal to `beam.max_width`, with sufficient Constraint
Table top-k capacity.

The implemented scope is CUDA, one executing logical batch, fixed beam width,
N=1/2/3 generation steps, asynchronous scheduling, and TP/PP/DP/DCP/PCP=1.
Keep `ignore_eos=True` and Chunked Prefill disabled. This scope does not include
unbounded B, concurrent batch owners, distributed execution, LoRA, multimodal
inputs or speculative decoding. The validation below exercises offline calls;
it does not establish automatic batching through an HTTP service.

V2 consumes the native Prefill token capture list; V1 retains its GR grid.
Configure V2's list explicitly for large prompts and compare actual
`(B, total_tokens)` shapes when evaluating the runners.
See [GPU Prefill capture configuration](prefill_graph_capture.md) for capacity
bounds, Prefix Cache attention bounds and legacy environment overrides.

## Regression coverage

Keep tests at the layer that owns the contract. The fast tests and CUDA tests
are complementary: mocks check admission and error handling, while native
state and actual CUDA replay check request order and storage reuse.

| Contract | Tests |
| --- | --- |
| V1/V2 startup selection, lazy capture, per-B fallback and cleanup failure | `tests/test_gr_startup.py` |
| Native token budget and `(B,T)` admission, V1 grid isolation | `tests/test_gpu_prefill_graph_runner_v2.py` |
| Logical member order, retirement and Backend capacity policy | `tests/test_v2_multibatch.py` |
| Native row/slot reuse and unchanged request state | `tests/kernels/test_gpu_beam_stage_v2_state.py` |
| Batched registration and fused input parity, cached offsets, token-block boundaries, padding and atomic fallback | `tests/kernels/test_gr_prefill_v2.py` |
| SID-stage capacity, exact outputs and changing device step under replay | `tests/kernels/test_stage_topk_v2.py` |
| B-switch replay, fixed storage, configured B=3/7 and 6/8, per-shape fallback | `tests/kernels/test_v2_multibatch_graphs.py` |
| Captured model/storage binding; reload/shutdown cleanup opt-in | `tests/test_beam_decode_graph.py`, `tests/v1/worker/test_constraint_lifecycle.py` |
| Prefill reload/shutdown invalidation, recapture with new model/output storage | `tests/test_gpu_prefill_graph_runner_v2.py`, `tests/kernels/test_prefill_graph_manager_v2.py` |
| Existing public protocol, generation steps and scheduler integration | `tests/st/test_onerec_v1_scheduler_st.py`, `tests/st/test_modelrunner_v2_stage_parity_st.py` |
| Unconstrained B=1/4, V1/V2 Eager, default Graph and explicit FULL Prefill | `tests/test_offline_beam_search.py::test_offline_unconstrained_beam_search_v1` |
| Real-model multi-batch execution and V1 baseline comparison | `tests/st/model_runner_policy_isolation.py` (manual CLI) |
| Real video business accuracy and cross-B comparison | `tests/st/onerec_multibatch_accuracy.py` (manual CLI) |

Partial-prompt cases exercise low-level input-helper parity with native vLLM;
they do not extend the public GR scope to Chunked Prefill.

The two manual CLIs are explicit experiments, not pytest-collected tests.
Profiling, synthetic latency sweeps and intermediate-tensor diagnostics are
not maintained as additional regression entrypoints.

## Reproduce the execution regression

Set `VLLM_GR_TEST_MODEL` to the local OneRec model and run each combination in
a fresh process and output directory. For example:

```bash
CUDA_VISIBLE_DEVICES=0 USE_TF=0 \
python -m tests.st.model_runner_policy_isolation \
  --runner v2 --mode graph --output /tmp/v2-multibatch-graph
```

Repeat with `--runner v2 --mode eager` and `--runner v1 --mode graph`.
The probe uses distinct prompts with unsorted lengths. It checks the
B=1→2→4→1→4→2 sequence with cold/warm Prefix Cache, an N=1 B=4 case and a
mixed-cache B=4 case. Assertions cover actual Decode replay/eager calls,
V2 FULL Prefill replay, fixed input addresses, absence of V2 request-time
capture, candidate structure and resource release. Repeated matching
shapes must produce identical outputs after batch switches.

To verify V1 baseline behavior, copy this single probe into a checkout of
`7ee1d95`, run `--runner v1 --mode graph`, then run the current checkout with
`--reference /path/to/baseline/results.json`. This requires exact ordered
candidates and scores for every matching case. `--expected-source` can
additionally assert the Worker imports the intended checkout.

Cross-B or Graph/Eager equality is a separate numerical investigation; it is
not a required identity in this probe. Different shapes can change BF16
scores and candidate order. Use the real-data evaluator below to enforce the
benchmark's business thresholds and record cross-B differences separately.

### Unconstrained generation

The existing offline test also runs without a Constraint Table, using the
configured `max_top_k` capacity. Run its six isolated engine configurations:

```bash
CUDA_VISIBLE_DEVICES=0 USE_TF=0 \
python -m pytest --run-slow -q \
  tests/test_offline_beam_search.py::test_offline_unconstrained_beam_search_v1
```

The matrix covers V1/V2 with Eager, default Graph and an explicit FULL Prefill
grid. Each engine processes B=1 and B=4 with W=128. The FULL cases require
actual Prefill replay at 1024 tokens and at 4102 tokens padded to 4224;
enabling graph mode alone does not establish FULL Prefill coverage. The test
uses V1's GR Prefill grid and V2's native token capture list respectively.
It checks runner identity, Decode execution, candidate structure and scores,
request retirement and clean EngineCore shutdown. V2 graph cases also reject
request-time capture. Results are saved in each pytest temporary directory's
`unconstrained.json`. This is a functional check, with no cross-mode numerical
identity or business accuracy claim.

## Reproduce the real-data accuracy run

Set `VLLM_GR_TEST_MODEL`, `DATASET_PATH` and `CATALOG_PATH` as in the system
benchmark. Prepare catalog triples using the quickstart. Load the benchmark
policy's `PASS_AT_1_THRESHOLD`, `PASS_AT_32_THRESHOLD` and
`RECALL_AT_32_THRESHOLD` environment variables; the script does not lower
thresholds when changing runners. Point `ONEREC_TRIPLES` to the triples file
and `ONEREC_ARTIFACT` to a reusable Constraint Table artifact path.

Run each runner in a fresh process, with a new output directory:

```bash
CUDA_VISIBLE_DEVICES=0 USE_TF=0 \
python -m tests.st.onerec_multibatch_accuracy \
  --runner v2 --batches 1 2 4 --num-prompts 100 \
  --triples "$ONEREC_TRIPLES" --artifact "$ONEREC_ARTIFACT" \
  --output /tmp/onerec-v2-multibatch
```

Repeat with `--runner v1` and a different output directory for the controlled
comparison. The script preserves original prompts, uses the system benchmark
loader and evaluator, and records source fingerprints, configuration, raw
candidates, per-sample metrics and graph counters. Exit code zero means that
the structural checks and business thresholds passed; inspect the recorded
cross-batch comparisons separately.

## Online functional validation

The shared `tests/st/test_onerec_v1_scheduler_st.py` suite explicitly
parameterizes ModelRunner V1 and V2. The V1 in its filename and public
`beam_search_v1` API names denotes the GR protocol, not a runner override.
Select `test_onerec_v1_completion_batch_online` with `-k v2` to exercise
independent offline and real HTTP engines at B=1/2/4 and B=1/3/7.
The paired engines use the same fixed KV capacity.

V2 probes observe native input preparation and completed Prefill FULL replays;
they also reject request-time capture, Decode eager fallback and residual
native requests or pending sampling output. The HTTP suite covers ordered
candidates and scores, concurrent sessions, waiting/active TCP disconnects,
recovery and EngineCore shutdown. These fixed-input checks do not replace
business accuracy or performance acceptance.
