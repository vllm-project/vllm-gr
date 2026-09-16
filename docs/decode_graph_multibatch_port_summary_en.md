# decode_graph Multi-Batch and Async Beam Search Port Summary

## 1. Branch and objective

- Branch: `decode-graph-multibatch-port`
- Base: `0efef35 feat(beam): rebase async stage scheduling on merged A2 (#384)`
- Objective: port Beam Search multi-batch support onto `decode_graph` and integrate resident-Request asynchronous scheduling without breaking the synchronous path.

The work is organized into four implementation commits:

| Commit | Topic | Purpose |
|---|---|---|
| `9415424` | Beam multi-batch capacity plumbing | Adds configuration, validation, and fixed resource capacity for B greater than 1 |
| `459beb2` | Synchronized NPU multi-batch | Implements the complete synchronous path for B requests with W beams per request |
| `c626c12` | Async resident multi-batch | Adds resident-Request asynchronous scheduling and extends it to B greater than 1 |
| `abc6ab4` | Configurable NPU async test driver | Provides isolated-process NPU validation with variable batch size, repetitions, and profiling |

## 2. `9415424`: Beam multi-batch capacity plumbing

### Changes

The commit introduces `beam_max_batch_size`, which declares how many logical Beam Sessions a Worker can host concurrently:

- `GRConfig.beam_max_batch_size`, defaulting to 1;
- additional-config key `vllm_gr_beam_max_batch_size`;
- CLI option `--beam-max-batch-size`;
- schema-v1 `beam.max_batch_size` parsing, serialization, and round-trip support;
- validation range `1..128`.

The value is propagated into ModelRunner, the Beam attention context, NPU workspaces, and graph capture. Resource geometry is extended from W-only sizing to:

```text
logical requests = B
physical decode rows = B * W
```

B is bounded by `beam_max_batch_size`, while W is bounded by `beam_max_width`.

### Design intent

This commit establishes capacity and configuration only; it does not yet implement cross-request scheduling. The default remains 1, preserving existing single-batch behavior. Multi-batch fixed buffers are allocated only when a larger capacity is explicitly configured.

### Supporting updates

- Added configuration parsing, serialization, CLI, and boundary tests;
- updated the runtime patch inventory;
- adjusted GPU/NPU ModelRunner initialization to size fixed workspaces along the B dimension.

## 3. `459beb2`: Synchronized NPU Beam multi-batch

### Changes

This commit implements the complete B-greater-than-1 path under synchronous scheduling.

#### 3.1 Batched step protocol

It adds the `BEAM_STEP_BATCH` EngineCore request type and corresponding codec, SyncMPClient, and InprocClient support. The Frontend can submit B `BeamRequestStepUpdate` objects for the same decode step in one message.

#### 3.2 Cohort barrier

EngineCore tracks expected and pending members for each cohort. A partial cohort is held until all requests for the same step arrive, preventing early execution from mixing request slots, prefix KV, or suffix KV.

#### 3.3 `BeamExecutionBatch`

The new `vllm_gr/v1/beam/execution_batch.py` exposes B independent Beam Sessions as a zero-copy execution view:

- the Scheduler retains B independent request IDs;
- attention and beam decision see contiguous `B * W` physical rows;
- slots must be contiguous in `0..B-1` order;
- W, decode step, and terminal state must match across the cohort;
- winner-parent mappings may not cross request boundaries.

#### 3.4 Worker, attention, and result splitting

- Remaps positions, sampling metadata, logits indices, and related state from B logical rows to `B * W` physical rows;
- exposes unshared KV from multiple sessions as a contiguous layout for xllm attention;
- runs one NPU beam decision over the complete cohort;
- uses `split_batch_decision()` to recover B request-local results and convert global parent indices back to each request's `0..W-1` range.

### Safety checks

The implementation validates batch order, contiguous slots, shared context, equal width, equal step, and equal terminal state before entering kernels. Invalid geometry fails explicitly instead of silently producing incorrect output.

### Compatibility work

The commit also updates EngineCore protocol contracts, InputBatchProxy coverage, NPU runner tests, and upstream private-method contract tests so that the multi-batch patch continues to satisfy existing single-batch contracts.

## 4. `c626c12`: Async resident Beam multi-batch

### Changes

This commit adds the Stage 1 resident-Request asynchronous path on top of synchronous multi-batch support.

#### 4.1 Configuration and API

- Adds `beam_async` / `vllm_gr_beam_async`;
- adds `GRLLM.async_beam_search()`;
- keeps vLLM async scheduling enabled when `beam_async=True`;
- keeps the existing CUSTOM `GRLLM.beam_search()` path on synchronous Stage 0 and rejects accidental use with async scheduling.

The asynchronous API currently requires:

- `beam_async=True`;
- `async_scheduling=True`;
- CUSTOM attention;
- `ignore_eos=True`;
- no Trie constraints;
- `concurrency_limit <= beam_max_batch_size`.

#### 4.2 Single resident Request lifecycle

Each input submits one prefill Request to EngineCore:

1. The Frontend sends `BEAM_SESSION_TEMPLATE` first.
2. It submits B prefill Requests in one `ADD_BATCH`.
3. The prefill sampler produces the first top-W tokens and initializes device Beam state.
4. The same Request stays in the AsyncScheduler running set.
5. Later decode steps no longer require Frontend re-registration.
6. Full sequences and scores are returned only at the terminal step.

#### 4.3 Resident NPU fast path

- Expands B logical requests into `B * W` physical decode rows;
- `_resident_beam_overwrite_input_ids_npu()` reads winner tokens directly from device sequence buffers and overwrites the next-step input IDs;
- x_attention reads both shared prefix KV and per-beam unshared suffix KV;
- suffix KV is reordered immediately according to winner parents after beam decision;
- non-terminal steps publish only lightweight routing stubs, avoiding full per-step Beam D2H transfers;
- the terminal step performs compact D2H and splits final results by request.

#### 4.4 Async B greater than 1

Resident execution reuses `BeamExecutionBatch`, allowing B long-lived Requests to execute together at each decode step while preserving independent request IDs, slots, sequences, scores, and result channels.

#### 4.5 Integration fixes

The port and targeted regression work also fix several compatibility issues:

- patch markers now use explicit Boolean semantics, avoiding accidental truthiness from `MagicMock` or proxy objects;
- sampling-metadata remapping operates on an independent copy instead of mutating the caller-owned object;
- sampling metadata is warmed and cached during prepare to avoid repeated gathers;
- upstream ModelRunner private-method signature changes remain contract-compatible;
- missing runtime imports in the asynchronous output path are restored;
- guards and diagnostics for synchronous batch context versus async resident deployment are tightened.

## 5. `abc6ab4`: Configurable NPU async test driver

The commit adds:

```text
examples/offline_inference/beam_search/offline_async_beam_search.py
```

The previous two test-script commits have been squashed into this single commit.

### Capabilities

- Each process tests one batch geometry, preventing Engine, graph, and NPU runtime state from being reused between B=1 and B=2.
- `--batch-size` supports 1 through 128, including B=1, 2, 4, 8, 16, 32, and 64.
- `--engine-batch-capacity` separates active batch size from fixed engine capacity.
- `--execution graph|eager` compares graph and eager execution.
- `--runs` defaults to five rounds, with profiling enabled only for the final round.
- Model, prompt, beam width, max tokens, memory utilization, and scheduling capacity are configurable.
- JSON output stores every beam token sequence, recommendation tokens, score, SHA-256 token hash, and per-round latency.
- Startup validation checks resource conditions such as `max_num_seqs` and `max_num_batched_tokens >= B * W`.
- The model can be supplied through `GR_B2_MODEL` or `VLLM_GR_TEST_MODEL`.

## 6. Validation status

### Verified

- Targeted tests for multi-batch protocols, cohorts, codecs, InputBatchProxy, and BeamExecutionBatch pass.
- Configuration, ModelRunner, async scheduler, and upstream-contract regression tests pass.
- Synchronous B=1 and B=2 NPU paths execute end to end.
- Async resident B=1 and B=2 execute end to end and produce JSON, logs, and profiles.
- The driver supports graph/eager switching and larger capacities such as B=4, 8, and 16.

### Open issues

1. **Numerical and ranking equivalence for B greater than 1 still requires further validation.** A successful run does not prove equivalence to two isolated B=1 executions.
2. Divergence on multi-token generation has also been observed in eager mode. Graph capture alone is therefore not a sufficient root-cause explanation; resident token overwrite, x_attention suffix-KV slots, winner parents, and same-step KV reorder remain the primary investigation points.
3. Noticeable Frontend time remains after NPU kernels complete, mainly from token assembly, sorting, `batch_decode()`, and JSON/log materialization for all beams. This work has not yet optimized that phase.
4. The async resident path currently terminates by a fixed decode-step count and does not support Trie constraints or per-beam early EOS.

## 7. Final commit structure

```text
abc6ab4 Add configurable NPU async beam example
c626c12 Port async resident beam scheduling to multi-batch
459beb2 Port synchronized NPU beam multi-batch
9415424 Add beam multi-batch capacity plumbing
0efef35 feat(beam): rebase async stage scheduling on merged A2 (#384)
```

Because the test-script commits were squashed and the previous documentation commit was removed, use the following command if this branch has already been pushed:

```bash
git push --force-with-lease origin decode-graph-multibatch-port
```

Do not use an unprotected `--force` push.
