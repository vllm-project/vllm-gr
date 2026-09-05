# Issue #54 — Step 5 review and evidence

Step: 5 — GPU ModelRunner V1 ASYNC

Commit: this Step 5 commit, with parent
`7ab07246b39b65eba9266d7a0b304050270ac46b` on `scheduler_async`.
All hardware artifacts below were produced from the pre-commit dirty snapshot;
their manifests record runtime content hashes as well as that parent SHA.

Exit gate: **PASS under the explicitly recorded team-revised Step 5 acceptance
contract**. Original literal legacy-hit parity and `issued-completed <= 1`
are **not PASS**; see the separate [strategy review](issue54_step5_strategy_review.md).
This is not full Issue/Phase-A acceptance or permission to claim those original
conditions were satisfied. Remote PR publication remains separate delivery work.
NPU hardware is unavailable and its eventual acceptance remains outstanding.

### Current acceptance boundary (2026-09-05)

The sections below are chronological evidence, not all current blockers.
The source-frozen current smoke comparison, bounded reference checks and
consolidated trace/counter/resource audit now support the revised Step 5
implementation gate. A fresh serial GPU-0 matrix is recorded under
`/tmp/step5_current_matrix_20260905`; a started case is not acceptance evidence.
The initial invocation completed the disabled pair and stopped at
`compile_miss` because the profiler observed zero prefix-query metric delta.
After a real-GPU diagnostic proved Offline stats delivery loss, the four
cache-enabled cases were rerun with a labelled direct-query observer in
`/tmp/step5_current_matrix_native_cache_20260905`. All six current cases now
pass the hardened pair comparator, including every call's compact ABI.
The failed invocation remains preserved and is not relabelled as PASS.
Its runner is `/tmp/step5_current_matrix_run.py`, with commands and source
hashes in `manifest.json`. It compares compiled Beam-eager and FULL Beam Graph
for disabled/miss/hit, B=1, W=128, S=3, L=1024, BF16. Fully uncompiled eager
is a separate numerical execution axis and is not relabelled as this pair.

The existing real D→D schedule/kernel overlap establishes an ASYNC overlap
observation; the latest P→D capture has no overlap and remains a performance
limitation. The complete shape/dtype/executor matrix belongs to Step 8, while
long-duration soak, P99/goodput and controlled E2E/gap improvement belong to
Step 9. These remain mandatory for the overall issue, but are not silently
redefined as prerequisites for every Step-5 local change. Step-5 short-session
memory stability and its actual smoke/trace gates remain required.

The team's issued/completed interpretation below differs from the original
literal `issued - completed <= 1`: that literal assertion is **not PASS**.
The accepted depth-two accounting uses outstanding <= 2 and lookahead <= 1.
Likewise, the bounded candidate reference is not strict legacy-hit parity or
an independent model/KV reference. All deviations remain visible for user
review; none authorizes ModelRunner V2, bundled execution, or a NPU PASS.

Before the latest active-shutdown fix, 563 focused CPU tests passed. That runtime revision
passed 16 real GPU 0 tests; the current revision additionally passes four
full-model eager and four verified CUDA Graph abort/reuse cases on GPU 0.
This includes terminal abort
release-error handling and an Offline abort wait-loop fix. Remaining lifecycle
cases, complete reference parity, soak and performance acceptance are still
outstanding. Detailed commands and scope limits appear in the final sections.
The latest active-shutdown revision additionally passes real GPU 0 eager
final-Prefill and verified Graph Decode shutdown probes, plus 26 focused CPU
shutdown/frontend cases. Expanded regression first reported 598 passes and one
new late-output red test failure. After the terminal transport ownership fix,
the expanded suite passes **601 tests** and the real GPU 0 transport/backend
regression passes **16 tests**. This does not close the overall Step 5 gate.
After removing the redundant notification-stream context, the expanded CPU
suite passes **655 tests**, the real GPU regression again passes **16**, and
same-configuration three-session Graph traces preserve final compact digests
and transfer counts. Measured notification-range time is lower; overall E2E
improvement is not established by two traced samples.
Real-model Python Worker exception checks now also pass for eager first-Decode
prepare and Graph first-Decode Beam advance, with 14 terminal gates per case.
These are two specific UniProc cases, not the complete Worker-error matrix.

## Strategy decisions for review

| Original requirement / baseline | Implemented interpretation or change | Reason / verification obligation |
| --- | --- | --- |
| One executing stage plus one prepared stage, alongside `issued - completed <= 1` | Count both dispatched stages as issued: `outstanding = issued - completed <= 2`; `lookahead = max(outstanding - 1, 0) <= 1` | The two original statements conflict for a depth-two BatchQueue. This interpretation was accepted in the design review. Actual overlap still requires device trace evidence. |
| One logical Native Request, W physical Beam rows | Keep SchedulerOutput logical counts at one; GPU V1 exposes W only inside the execution frame and restores counts in `finally` | Native request bookkeeping and Beam attention geometry have distinct meanings. Typed sidecar declares the physical width. |
| Persistent Beam state | `session.context.decision` owns sequences and cumulative scores; backend workspace adds raw parent/local-logprob columns, two fixed control slots and final backtrace scratch | Avoid a second independently advancing Beam history. Final backtrace materializes lineage once. |
| Prefix binding persists; no Decode lookup | Bind a device view of the native prefix block table once, and keep Native logical output blocks out of Beam attention | GPU V1 uses `MultiGroupBlockTable.block_tables`; only one native KV group is currently supported. |
| GPU/NPU prepare differences live in adapters | GPU publishes device tokens after upstream `_prepare_inputs()`; the NPU contract requires publication before it | GPU's original method would overwrite an earlier D2D write. GPU attention metadata is constructed after preparation. |
| Legacy vs ASYNC parity | Keep the old self-advance path as a performance baseline; use corrected precision commit `89f4178` only where its output fields are valid | Old self-advance loses Prefill cumulative score and double-counts the first Decode local score. An adversarial bootstrap/advance test guards the correct semantics. The corrected legacy path itself has a selected-logprob lineage defect, so it cannot validate parent/logprob semantics. |
| Independent reference for broken legacy fields | Use independent CPU candidate replay for Beam selection, FP32 score accumulation, lineage, selected local logprobs, next Forward token/position and compact final encoding | Team audit accepts this bounded substitution, not a claim of strict legacy-hit parity. Actual sampler candidates are inputs to the oracle: candidate completeness, softmax/constraints, model Forward, KV contents and consumer block-table mapping require separate evidence. Bootstrap is not an independent full-vocabulary TopK check. |
| Prefix-cache smoke evidence from Offline Prometheus counters | A profile-only observer records actual persistent `get_computed_blocks()` calls and returned hit lengths, while preserving raw Prometheus readings separately | Real GPU evidence shows query stats on empty BatchQueue outputs are not recorded by upstream Offline logging. This is a measurement-source change, not permission to waive a miss/hit gate or alter ordinary requests. Four current cache-enabled cases pass using direct observed counts; their timings are explicitly not performance evidence. |
| Cache-disabled correctness oracle | Do not use legacy cache-disabled output as the correctness reference; retain mandatory cache-disabled validation for the persistent path | Legacy incremental requests reconstruct Prefix ownership through cache hits without a cache-disabled guard. This separate lifetime defect is not a blanket explanation or waiver for cache-enabled hit/miss numerical differences. |
| eager vs CUDA Graph parity | Compare compiled Beam-eager against Beam CUDA Graph under identical compilation configuration; also report `enforce_eager=True` separately | The model compilation axis changes outputs on this workload. A Beam Graph comparison must isolate Graph replay. This does not waive investigation of numerical differences or the broader correctness matrix. |
| Unsupported combinations | Fail fast for mixed persistent Decode batches, Native preempt/resume, forced prefix reset during a session, unsupported KV groups, PP, EC/KV transfer and speculative decode | Persistent ownership cannot safely resume through Native request reconstruction. No silent fallback to bundled execution is allowed. |
| Six trace markers | Use upstream optional profiling scopes, supporting NVTX process-tree capture. Final packing is `gr_final_output`; its copy/wait is separately `gr_final_d2h` | Worker-only traces cannot establish EngineCore overlap. Marker presence and Graph dispatch counts alone are insufficient. |
| Static Decode metadata upload | Generate request-to-row indices and fixed attention lengths through `BeamDecodeGeometry(request_widths, prefix_lengths)` into caller-owned buffers | The generator is request-group based, not an all-zero/B=1 protocol. Runtime admission and backend/prefix-binding gates still require B=1. Multi-group layout tests do not authorize B>1 execution. |
| Request-keyed metadata cache | Cache only the geometry currently resident in the one set of fixed buffers; clear old views/initialized markers on layout changes or generic Beam overwrite | A map of all previously initialized request IDs was not a valid cache of one physical buffer. A→B→A could reuse B's lengths, and Python references grew with session count. This corrects both persistent and legacy Beam metadata caching without changing standard-request metadata construction. |
| Upstream input-preparation Event wait | For a validated persistent GPU Decode execution frame, query the original Event and skip its redundant synchronize only when already complete; otherwise delegate the complete upstream guard | Preserves ownership of every reused Host DMA source and the final Event record. This is not a substitute for completion notification or a waiver of the strict no-intermediate-sync gate. Prefill, dummy and ordinary requests retain the original guard. |
| Native completion build dependency | CUDA-target installation builds a small C++17/pthread `_completion` extension; CPU/NPU installation does not build it. GPU V1 persistent ASYNC now requires native ABI 2 at initialization | No NVCC, CUDA headers, pybind11 or Worker-time compilation. ABI 2 adds an unbounded condition-variable wait, preserving the upstream output interface's lack of a new deadline; old binaries fail version validation and must be rebuilt. |
| Two fixed control slots across request lifetimes | A CUDA-private runner-owned lease pool is now wired into V1 sampling/output construction. Each session selects its initial available physical slot once and keeps a fixed parity offset | Logical GR state release is not permission to overwrite an in-flight control. Device completion and detached Host consumption/discard are both required. Monotonic session epochs and frozen output leases survive state-cache reuse. Producer/reader guards prevent shutdown races; failures quarantine storage until a successful terminal drain. |
| Upstream asynchronous output constructor | Preserve the original class identity and wrap its constructor/get_output only within a validated persistent sampling scope derived from native ExecuteModelState | Ordinary/partial-Prefill output construction delegates upstream. GR copies the real control into two fixed pinned buffers, records real Events, notifies on a separate CUDA stream, and returns detached Host values only after native completion. No additional Future queue is introduced. |
| Persistent GR abort release | Wait for the existing release RPC Future on the EngineCore owner thread; retire only the affected Engine/Executor instance if submission or reply fails | Terminal abort now observes errors instead of discarding the Future. This is an RPC-response barrier, not a guarantee of completed GPU DMA or consumed control slots. No healthy scheduling/dispatch wrapper, new queue, thread, or deadline is added. |
| Offline persistent final-only collection after abort | Match the current step's final output first; otherwise check the target internal ID in the native OutputProcessor registry and raise a termination error if it is gone | Prevent an infinite empty-step loop without inventing a final Beam result. Other active requests cannot keep a terminated target's wait alive; ordinary collection is unchanged. |
| Active persistent request shutdown | Mark the native frontend RequestState immediately after registration; finish persistent Native Requests through Scheduler.finish_requests only after Worker shutdown; then abort marked frontend records through the local OutputProcessor API | The marker records ownership, not Beam state. No ID-prefix heuristic, second registry, post-shutdown RPC, or ordinary-request cleanup is added. Worker drain failure preserves later layers; a separate core-closed flag permits frontend cleanup retry without repeating core teardown. |
| Late output after terminal CUDA drain | Track successful terminal drain separately from closed admission; serialize quarantine ownership with one module-level short Host lock | A late output still raises instead of reading freed control storage, but cannot re-register a drained transport as a globally retained failure owner. Failed drain still retains resources. The lock is not used on the healthy step path and is not held across GPU drain or pool locks. |
| CUDA completion notification stream context | Call `notify.wait_event(ready)` directly, then register completion on the same explicit stream pointer | Local PyTorch `Stream.wait_event` passes the stream into `Event.wait`; CUDAEvent applies the stream's device guard. Callback registration was already outside the context. The copy-stream context and all device consistency checks remain; no device query or synchronization is added. Trace and real CUDA regression must verify the change. |

No GPU ModelRunner V2 implementation is introduced. No bundled or recursive
execute path is introduced.

## Changed files and behavior

- `vllm_gr/v1/worker/gpu_model_runner_patch.py`: CUDA persistent backend,
  physical execution geometry, fixed input/output storage, final compact D2H,
  prefix extraction, GPU memory accounting and Graph descriptor safeguards.
- `vllm_gr/v1/worker/model_runner_common.py`: control-only intermediate sampler
  output, persistent session ownership, scoped failure cleanup and trace range.
- `vllm_gr/v1/worker/model_runner_patch.py`: select backend once and install
  platform memory diagnostics.
- `vllm_gr/v1/beam/beam_decision.py`: fixed persistent workspace, split selection
  from KV reorder, backend-driven progression and shared trace ranges.
- `vllm_gr/v1/attention/backends/beam_attn_metadata.py`: consume the persistent
  prefix binding.
- `vllm_gr/v1/core/sched/async_scheduler.py`: optional GR scheduling trace;
  ordinary requests directly delegate to upstream.
- `vllm_gr/v1/engine/engine_core_patch.py`: observe terminal persistent-GR
  release RPC errors and reject reuse of a failed Engine/Executor, while
  leaving its shutdown path reachable.
- Related unit tests cover physical/logical restoration, state reuse, prefix
  binding, control slots, score seeding, final backtrace and Graph gates.

## Tests run

Earlier native-completion CPU regression (superseded by the later results
below; two pytest workers, per the updated user limit):

```bash
CUDA_VISIBLE_DEVICES= USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m pytest -p no:rerunfailures -n 2 -q \
  tests/test_beam_decode_geometry.py tests/test_beam_attention_metadata.py \
  tests/test_model_runner_common.py tests/test_gpu_model_runner_patch.py \
  tests/test_model_runner_patch.py tests/test_gr_async_scheduler.py \
  tests/test_beam_decision.py tests/profiling/test_gr_candidate_oracle.py \
  tests/profiling/test_gr_prefill_diagnostic.py tests/profiling/test_gr_session_trace.py \
  tests/test_cuda_completion.py tests/test_gr_completion_build.py \
  tests/packaging/test_install_metadata.py
```

Result: **326 passed**, log `/tmp/step5_native_completion_combined_cpu.log`.
The preceding ready-event/trace revision passed 287 tests in
`/tmp/step5_input_ready_cpu_xdist2_fixed.log`.
The initial two-worker run had 286 passes and one failure
(`/tmp/step5_input_ready_cpu_xdist2.log`): a capture test replaced the entire
`torch.cuda` module, breaking delayed imports when test order changed. The
isolated case reproduced the failure. Its mock now intercepts only Stream
allocation and used stream/synchronize calls while retaining CUDA/accelerator
modules and Stream/Event types. No product code or test skip was added for
this fix. The earlier serial 264-test run remains recorded in
`/tmp/step5_input_ready_cpu_regression.log`.
Warnings include the unavailable CUDA driver interface inside the CPU sandbox;
this run is not hardware acceptance. Physical GPU evidence is recorded
separately with its source snapshot and limitations below.

Initial focused CPU command (historical):

```bash
USE_TF=0 python -m pytest -p no:rerunfailures -q \
  tests/test_gr_async_scheduler.py tests/test_gpu_model_runner_patch.py \
  tests/test_beam_decision.py tests/test_model_runner_common.py
```

Result: 163 passed. `pytest-rerunfailures` was disabled because its startup socket
is prohibited in the test sandbox. CUDA initialization warnings in this CPU run
are not GPU acceptance evidence. Ruff and `git diff --check` passed.

Independent CUDA backend lineage test on physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m pytest -p no:rerunfailures -q tests/kernels/test_persistent_gr_cuda.py
```

Result: **6 passed, no skips**. Log: `/tmp/step5_gpu1_lineage_test.log`.
The test enumerates all candidates in Python using exact binary FP32 values,
stable tie breaking and parent grouping. It compares all four final ABI fields
against real CUDA bootstrap, selection, state progression, backtrace and final
copy. It covers W=64/128, result width 1/full, and checked/deferred selection.
It verifies backend lineage, not model-forward numerical parity.

The first continuous compiled Beam-eager/disabled-cache probe also completed:
`/tmp/step5_gpu1_compile_disabled_acceptance.json`. One warmup plus three measured
sessions returned identical compact ABI digest
`a68f9dd4456eff61b20b5af33d6f764458fefbb0d888556b84fbc8fdf92c3feb`.
Every session ended with active state zero; final created/freed counts were 4/4.
This probe predates the expanded memory/score-sum instrumentation and is not the
complete soak or memory exit gate.

Fresh expanded cache-disabled E2E comparison:

- `/tmp/step5_gpu1_accept_compile_disabled.json`
- `/tmp/step5_gpu1_accept_graph_disabled.json`

Both completed one warmup and three measured sessions. Public output digests,
full compact ABI and each sequence/parent/logprob/score slice matched exactly.
The compact ABI digest is the same `a68f9dd4…c3feb` reported above. FULL dispatch
deltas were 0 for compiled Beam-eager and 6 for Beam Graph. State counts remained
balanced with zero active state after every session. Measured Native KV, Beam
KV, workspace, Graph-designated pool and allocator allocated/reserved bytes were
constant within each process. Per-token logprob sums matched cumulative scores.
This short repeatability check does not replace the long Step 9 soak.

Reproduction uses `/tmp/run_step5_gpu1_acceptance.sh`; it runs each engine
serially on GPU 1 and limits compilation threads to one. The runtime diff is
preserved as `/tmp/issue54_step5_gpu1_runtime_snapshot.patch`, SHA256
`8467069667134a76d598ff58c1f6f02d1dd41022a38f7b9bc02f64708130de59`.
The frozen acceptance helper is `/tmp/step5_acceptance_common.py`, SHA256
`eb5724e4f925f4107b2d91be6d745a5f52a077327d02a842c4a1061f9d35bf34`.

The hit pair also passed the same strict comparison:
`/tmp/step5_gpu1_accept_compile_hit.json` and
`/tmp/step5_gpu1_accept_graph_hit.json`. Prefix hit rate is 1008/1024 in both.
One priming session, one warmup and three measurements give created/freed 5/5
with zero active state at each boundary. The compact ABI digest is
`29a0b1cc039ca4994cb6a028b9df2bb96fab9215b88ead50c6d31cbb6e3f3314`;
the parent slice digest is
`1e69e212a8b13cf303793a216683f83a9657a873388fe7da0957f0beae8a79f6`.
Measured FULL dispatch delta is again 0/6 for Beam-eager/Graph respectively.
All 18 sessions across the four runs report the hard counters listed below.
The full command and final comparator PASS are in
`/tmp/step5_gpu1_acceptance_matrix.log`.

## Prior GPU evidence to retain

Physical GPU 1: L20, UUID `GPU-231d8177-3abc-1ecc-1b7a-763683e5a393`.
OneRec-1.7B, B=1, W=128, input length 1024, BF16, three generation stages:
one final-prefill bootstrap plus two Decode dispatches.

| Evidence | Artifact | Scope |
| --- | --- | --- |
| Persistent no-compile eager miss | `/tmp/step5_gpu1_persistent_eager_logprobs.json` | Token/score exact against corrected `89f4178` miss; selected logprob token alignment is 0 mismatches out of 384 positions. |
| Corrected legacy eager miss | `/tmp/step5_gpu1_precision_89f_eager_logprobs.json` | Token/score oracle only; selected logprob token alignment has 372 mismatches out of 384 positions. |
| Compiled Beam-eager miss | `/tmp/step5_gpu1_persistent_compile_eager_raw.json` | Public sequence/score/logprob/full-output digests exactly match the next row. |
| Compiled Beam Graph miss | `/tmp/step5_gpu1_persistent_graph_fixed_raw.json` | Exact width 128 captured, two FULL dispatches. |
| No-compile eager, cache disabled | `/tmp/step5_gpu1_eager_disabled_fixed.json` | Public digests match no-compile miss. |
| Persistent no-compile eager hit | `/tmp/step5_gpu1_persistent_eager_hit_raw.json` | 1008/1024 prefix tokens hit; strict corrected-legacy hit parity remains unresolved. |
| Corrected legacy eager hit | `/tmp/step5_gpu1_precision_89f_eager_hit.json` | 125/128 sequences shared with persistent hit; three differ. This is not a parity PASS. |

These earlier artifacts describe their recorded worktree, not automatically the
latest diff. New trace instrumentation requires fresh hardware evidence.

Hard counters recorded in the earlier persistent GPU smoke artifacts:

```text
native_request_created_per_session = 1
scheduler_add_request_per_session = 1
gr_state_created_per_session = 1
gr_state_freed_per_session = 1
decode_request_objects_created = 0
beam_request_step_update_calls = 0
decode_prefix_cache_queries = 0
intermediate_beam_state_d2h_bytes = 0
decode_scheduler_calls = 2
decode_worker_dispatches = 2
final_result_d2h_count = 1
final_result_d2h_bytes = 5120
```

The 5120-byte counter measures the compact Beam payload. CUDA staging additionally
copies its status word in the same transfer; trace byte accounting must include
that word and report control transfers separately. Instrumented counters alone
do not prove absence of all unexpected transfers.

## Remaining exit-gate work

Fresh joint CPU/CUDA trace was captured serially on GPU 1 with one compilation
thread. Artifacts: `/tmp/step5_gpu1_async_graph_trace.nsys-rep`,
`/tmp/step5_gpu1_async_graph_trace.sqlite`, and
`/tmp/step5_gpu1_trace_analysis.json`. Reproduction command is preserved in
`/tmp/run_step5_gpu1_trace.sh`; read-only analysis is in
`/tmp/analyze_step5_nsys.py`. It joins CUDA launch correlation IDs by process
and intersects scheduling ranges with actual kernel intervals, not their
bounding envelope. These preliminary measurements do not constitute gate PASS.
Historical accounting below stops at `gr_final_output.end`. It excludes
post-materialization control delivery and final Scheduler reconciliation;
see the later full-session accounting correction before using these totals.

- Each of the three sessions (one warmup, two measurements) has three
  output-bearing scheduling stages, one bootstrap, two Decode preparations,
  two Beam decisions, two intermediate progress markers and one final output.
  A trailing empty housekeeping schedule is excluded from generation stages.
- In the two measured sessions, Decode-to-Decode scheduling overlaps prior
  stage kernels by 105.085 us and 96.498 us respectively. Prefill-to-Decode
  scheduling has zero measured kernel overlap. All next-stage scheduling
  ranges finish before the previous stage's reconciliation begins.
- Each measured interval contains two 4-byte D2H transfers and one 5124-byte D2H
  transfer: the compact final payload plus status word. No intermediate
  Beam-sized copy appears in this capture.
- Each measured interval also contains four CUDA Event synchronize calls and six CUDA
  Stream synchronize calls outside the final-output range, plus one Event
  synchronize inside final output. Their call sites, waits and relationship
  to intermediate execution still require attribution. They must not be
  silently omitted or treated as satisfying the no-intermediate-sync gate.

Trace profiling timings are not used as E2E performance improvement estimates.
Further GPU validation remains strictly single-instance; no parallel pytest
workers, with `OMP_NUM_THREADS=1` and `TORCHINDUCTOR_COMPILE_THREADS=1`.

### Follow-up: static metadata upload attribution

The next serial trace, `/tmp/step5_gpu1_static_metadata_trace.nsys-rep` and
`.sqlite`, corresponds to `/tmp/issue54_step5_static_metadata_snapshot.patch`
(SHA256 `6d1e8b765a934913fa5006766713bfd7eea060a636b023e812b897142a700fae`).
Command: `/tmp/run_step5_gpu1_trace_static.sh`; analysis:
`/tmp/step5_gpu1_static_metadata_analysis.json`. This snapshot predates the
request-group geometry extraction described below; do not treat it as an
automatic E2E validation of later edits.

- All three sessions' public output and complete compact ABI match the prior
  joint trace exactly. Parent bounds, score/logprob sums and state release
  checks passed.
- Each pre-final interval's Stream synchronize count fell from six to four. No Stream
  synchronize remains between the first Decode schedule and final-output
  entry. The four remaining calls are in Prefill preparation. Event boundaries
  in upstream input reuse/control collection and final-copy wait remain.
- For the last measured session, Decode H2D counts changed from five to four
  1024-byte transfers, and the sole 528-byte transfer disappeared; other H2D
  size/count entries match. This isolates one sampling-map upload and one
  initial seqlens upload, not all upstream input transfers.
- Measured P→D schedule/kernel overlaps were 159.956 us and 138.988 us;
  D→D overlaps were 143.876 us and 141.541 us. This proves overlap in this
  capture, not a stable E2E speedup or a causal P→D improvement from the two
  Decode-only changes. Warmup still had no overlap.

On review, the B=1 special-case generation was extracted into a shared static
`BeamDecodeGeometry`. It holds only immutable request widths/prefix lengths,
owns no tensors or lifecycle, and writes into existing buffers during prepare,
outside forward Graph capture. Two-group geometry `(2,3)` with prefixes `(7,11)`
generates request indices `[0,0,1,1,1]`, prefix lengths `[7,11]`, prefix query
offsets `[0,2,5]`, and suffix query offsets `[0,1,2,3,4,5]`. This avoids making
the B=1 optimization a protocol assumption. Mixed/noncontiguous native slots
would require a separate explicit mapping; current B>1 execution still fails
fast. Fresh CPU regression including generator and admission guards:
**208 passed**, including integer overflow and rejection of Tensor inputs to
Host-only index validation. The earlier 191-pass run tested the pre-extraction
version. The first geometry/lineage CUDA run passed **8 tests without skips**,
including capture of a consumer followed by preparation of new request geometry
and replay. That two-group consumer test verifies a reusable layout, not B>1
session execution. Log: `/tmp/step5_gpu1_geometry_lineage_test.log`.

The unchanged limitation of this local optimization is explicit: proxy index
storage is still allocated on first use. The pre-existing request-keyed metadata
cache was subsequently reviewed and corrected below; long soak acceptance is
still outstanding. No new scheduler, admission, session concurrency, or backend
capability is enabled by the geometry descriptor.

The extracted runtime, including the new untracked geometry module, is preserved
in `/tmp/issue54_step5_group_geometry_snapshot.patch`, SHA256
`8a309d144a1eb602098ae87fb92e2d3c6e6a30316827d961b81b12588363898f`.
This is separate from the two earlier Nsight snapshots; a new model-level
trace/parity run is still needed for the extracted version.
The final CUDA rerun after the Host-input/dtype guards also passed **8 tests,
no skips**, on GPU 1: `/tmp/step5_gpu1_geometry_lineage_final_test.log`.

### Follow-up: resident metadata cache correctness and bounded retention

The pre-fix minimal reproduction is preserved in
`/tmp/step5_metadata_cache_before.log`: all five new cases failed. In four
cases, A→B→A or A→uncached-write→A returned prefix length `[12]` instead of
A's `[8]`; the fifth detected a second request-keyed view entry. Tests exercise
both legacy and persistent metadata construction without a model or GPU.

`_select_geometry_cache()` now retains at most one resident key/view set/ready
marker. Key changes invalidate old entries, both direct upload and view lookup
select the resident key, and generic Beam seqlens writes invalidate before
overwriting storage. A same-key view-size change rebuilds both views and static
contents. Standard metadata and block-table updates do not clear unrelated
Beam lengths. Underlying device storage and Graph addresses are unchanged.

The focused metadata suite passed **29 tests** after the first fix
(`/tmp/step5_metadata_cache_after.log`). Expanded regression passed **217 tests**
(`/tmp/step5_metadata_cache_regression.log`), including 30 successive request
keys with bounded entries, standard-request isolation, initialization failure
recovery, and same-key incompatible view extents. These results establish the
cache's reference bound, not long-run allocator/Graph-pool memory acceptance.

Physical GPU 1 regression passed **10 tests without skips**, including two
actual-CUDA A→B→A consumer-Graph cases (cached and uncached overwrites), the
request-group geometry refresh cases, and the independent Beam lineage cases.
Log: `/tmp/step5_gpu1_resident_cache_test.log`. The two cache cases assert both
restored values and unchanged captured input addresses. The model-level E2E
and joint trace for this latest cache revision remain outstanding.

Latest model-level trace for the resident-cache revision is now captured:
`/tmp/step5_gpu1_resident_cache_trace.nsys-rep`, `.sqlite`, and `.json`.
The exact runtime diff is `/tmp/issue54_step5_resident_cache_snapshot.patch`,
SHA256 `4441b71542474425529ccac3fab824590bcf6dd12a3fdf4d370d615eb084fe88`.
Reproduce with `bash /tmp/run_step5_gpu1_trace_static.sh resident_cache`.
Public output and all compact ABI fields across its three sessions match the
preceding static-metadata trace exactly; parent bounds, score/logprob sums and
state-release gates passed. Analysis:
`/tmp/step5_gpu1_resident_cache_analysis.json`.

Measured P→D scheduler/kernel overlaps were 131.753 us and 120.827 us; D→D
overlaps were 152.795 us and 140.212 us. Every pre-final interval still had four Stream
synchronize calls outside final output, four Event synchronize calls outside
final output and one final-copy Event synchronize. These findings supersede the
pending-trace status for this snapshot, not the remaining strict no-sync gate.
The warmup showed no overlap; no performance percentage is inferred.

### Independent oracle on real model candidates (diagnostic, not performance)

`tests/profiling/gr_candidate_oracle.py` implements a CPU-only selection oracle
without importing production Beam code. It independently advances survivors
from captured constrained candidates, preserving candidate order, valid masks,
FP32 stepwise accumulation, global top-W tie order, stable parent grouping and
final score order. It then checks actual next-Forward input rows and all final
compact ABI fields. Production choices are comparison targets, never the next
oracle state. Selected local logprob error is reported separately from the
transport ABI's independently calculated FP32 cumulative-score difference.

The capture is isolated in `/tmp/step5_candidate_capture.py`, installed by
`/tmp/step5_candidate_wrapper.py`; no diagnostic hook is added to product code.
Per-stage tensors are copied D2D immediately, and one post-request diagnostic
RPC packs and returns them after final output. This adds a diagnostic D2H and
device storage, explicitly outside product transport counters. The JSON marks
`performance_evidence_valid=false`; these timings are never benchmark evidence.

Cache-disabled OneRec-1.7B/BF16/B=1/W=128/steps=3 on GPU 1 passed:
`/tmp/step5_gpu1_candidate_disabled.json`, log
`/tmp/step5_gpu1_candidate_disabled.log`. The oracle enumerated **12,943 valid
candidates**; actual next-input row ordering, stage decisions and final
sequence/parent/score/logprob fields matched. Maximum error against the original
selected local logprobs was **0.0**. One extra **415,236-byte diagnostic D2H**
occurred after final output. Replay the frozen result without a GPU:

```bash
python tests/profiling/gr_candidate_oracle.py /tmp/step5_gpu1_candidate_disabled.json
```

Prefix-hit replay also passed on the same runtime snapshot:
`/tmp/step5_gpu1_candidate_hit.json`, log
`/tmp/step5_gpu1_candidate_hit.log`. The priming request used full Prefill;
the second request reused **1008/1024** Prefix tokens and enumerated **13,018
valid candidates**. Both requests matched all checked stage, next-input and
final ABI fields, with selected raw local-logprob error **0.0**. The hit request
added one **407,172-byte diagnostic D2H** after final output. Neither run is
performance/transport evidence.

The CPU oracle now rejects optimized Python (`-O`, which removes assertions),
invalid result width/final shape, an extra production batch, nonbinary masks,
malformed counts, empty candidate rows and malformed tensor descriptors. The
empty-row rejection matters even when other rows supply W winners: this
diagnostic covers successful sessions, and the deferred CUDA path marks such
a request invalid. Audit reproductions became negative regression tests.
The expanded oracle-only suite passed **18 tests** with one pre-existing pytest
configuration warning (`timeout` is unknown). It imports no production Beam
selection code. Frozen GPU captures still pass under these stricter checks.

The same-capture first/last comparison now verifies the visible prompt suffix
and independently validates both requests before comparing Prefill candidates:

```bash
python tests/profiling/gr_candidate_oracle.py \
  /tmp/step5_gpu1_candidate_hit.json --compare-prefill-first-last
```

Report: `/tmp/step5_gpu1_candidate_hit_prefill_comparison.json`. The full-Prefill
and hit requests share **126/128** initial candidate token IDs, but all 126
shared scores differ (maximum absolute difference **0.12950897216796875**).
The reference-only tokens are `158329,159295`; hit-only tokens are
`158047,158999`. Thus the difference is already observable at the Prefill
sampler output, before persistent Decode selection/backtrace. This localizes
the next investigation to Prefill/model/cache computation; it does **not**
establish a numerical-only explanation, validate cached KV, or resolve the
separate corrected-legacy-hit comparison. The comparison reports an observation,
not a parity PASS, and makes no performance claim.

This proves model-produced-candidate selection/lineage, not independent logits,
softmax, constraint-table mathematics, KV contents, or corrected-legacy hit
parity. The case requires full-width active bootstrap and at least W valid
Decode candidates. Cache-hit support checks the actual native computed-prefix
extent; it does not infer that extent from the observed positions. Chunked
Prefill remains outside this diagnostic's coverage.

### Prefill attention-input and cached-Prefix diagnostic

A fresh, single-instance GPU 1 probe completed on the unchanged runtime diff:

```bash
bash /tmp/run_step5_prefill_probe.sh
python -m tests.profiling.gr_prefill_diagnostic \
  /tmp/step5_gpu1_prefill_probe.json --layers 28
```

Capture/log: `/tmp/step5_gpu1_prefill_probe.json`,
`/tmp/step5_gpu1_prefill_probe.log`. Analysis:
`/tmp/step5_gpu1_prefill_probe_analysis.json`. The isolated hooks are in
`/tmp/step5_prefill_capture.py` and `/tmp/step5_prefill_wrapper.py`. No hook was
added to the product. Capture SHA256:
`d6ba00b13ce93857bb84b14bdf9ba59d95f729289dd1f5c1d938f60dbbe958a9`;
capture-hook SHA256:
`61d955b41bc55d7720c088d6639104993ae3b6e92c4f943fc7f5bdfb8d25afb0`.
The first/full-Prefill and second/hit requests both
preserve exactly the final compact ABI of the previous candidate diagnostic.
Their extra post-request diagnostic transfers were respectively 11,827,028 and
11,706,516 bytes; these are **not** product D2H counters or benchmark runs.

At all 28 layers, the hook snapshots the last 16 actual Q/K/V rows immediately
before attention and output rows immediately after attention. After priming,
it gathers and independently clones the reusable Prefix from the native block
table. Before each hit attention writes K/V, it gathers that request's physical
Prefix blocks and compares on the GPU against the saved priming contents.
Operations are ordered on the same stream; snapshots never replace model input,
KV cache or output. The retained priming Prefix snapshots are dropped per layer
after comparison. This diagnostic temporarily adds storage and cannot establish
product memory/soak behavior.

The earliest difference is already at `model.layers.0.self_attn.attn` input:

| Last-16-row field | Elements | Different elements | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| Q | 32768 | 14 | 0.0078125 |
| K | 16384 | 3 | 0.00390625 |
| V | 16384 | 9 | 0.000244140625 |
| Attention output | 32768 | 96 | 0.0009765625 |

All 28 captured native Prefix K/V comparisons have zero different elements and
zero maximum error before hit writes. This is **elementwise numerical equality**,
not a bitwise claim (`+0/-0` are not distinguished). The capture uses common
metadata/block tables, whereas standard FA consumes `std_*` metadata. Therefore
it does not yet exclude an additional std-FA mapping/metadata problem.

The first-layer V difference is particularly useful: installed Qwen3 splits V
directly from QKV projection and applies Q/K normalization and RoPE only to Q/K.
Thus at least one source of divergence precedes both attention's cache read and
persistent Decode. Next investigation is first-layer projection/input
normalization and shape-dependent execution, **not** an assumed GEMM root cause
or a blanket waiver of Prefix parity. The original final candidate differences
remain reproducible; no numerical tolerance has been relaxed.

The CPU-only analyzer and negative checks are in
`tests/profiling/gr_prefill_diagnostic.py` and
`tests/profiling/test_gr_prefill_diagnostic.py`. Combined with the candidate
oracle suite, **25 tests passed** (single pytest process, one existing unknown
`timeout` configuration warning). Missing layers, mismatched shapes/extents,
nonfinite/ragged snapshots and an actually changed Prefix are covered. This is
diagnostic validation, not the GPU correctness gate.

### Isolated first-layer projection reproduction

**Separate legacy correctness issue identified by the user:** legacy
`_apply_beam_step()` rebuilds a Native Request per Decode with Prompt tokens
plus only the newest token per Beam, and explicitly relies on
`get_computed_blocks()` recovering the Prefill Prefix. Its scheduler wrapper
pads/cache-binds the full Prefill blocks for this purpose. There is no
Prefix-Caching-disabled admission guard in these inspected paths.
`_segment_requests()` does describe recomputing a missing Prefix with standard
attention, but an active incremental Beam session in `BeamAttentionImpl.forward`
enters the suffix-pool branch and returns before the standard
`reshape_and_cache` path. Metadata for recomputation alone therefore does not
establish that the new native Prefix blocks have been populated. This is a KV
ownership/write-order correctness issue, separate from the projection-shape
numerics reproduced below; old cache-disabled outputs must not be treated as
a trustworthy correctness oracle without a dedicated reproduction/fix.

Persistent ASYNC has a different ownership path: `GRAsyncScheduler` is explicitly
excluded from those legacy per-step cache patches, and CUDA bootstrap binds the
existing request's device block-table Prefix row. Decode retains that request
instead of reconstructing it and looking up Prefix Cache again. Consequently
the old-path issue does not justify disabling Prefix Cache support in ASYNC;
the new path's cache-disabled correctness remains a mandatory gate. Existing
candidate/lineage checks are not a full independent KV-content proof.

The first-layer V difference is now reproduced without any Scheduler, Beam,
attention or KV-cache operation. The probe loads the **same cached input-norm
Triton kernel** used by the real first compiled partition, loads checkpoint
embedding/norm/QKV weights, and calls the partition's `extern_kernels.mm` with
the same transposed-weight strides. Input shapes are M=1024 and M=16. It also
replays M=16 using the exact normalized tail from the M=1024 computation, so
independent normalization cannot explain that comparison.

The GPU 1 result is `/tmp/step5_gpu1_projection_probe_v2.json`, with full weight
content SHA256 values, module/capture hashes and actual matmul precision flags.
Torch is `2.11.0+cu130`; CUDA is `13.0`. BF16 reduced-precision reduction is
enabled in this environment. Both the initial probe and provenance-enriched
repeat gave identical difference metrics:

| Comparison | Different elements | Maximum absolute difference |
| --- | ---: | ---: |
| Embedding tail, M1024 vs M16 | 0 / 32768 | 0 |
| Normalized input tail, M1024 vs M16 | 0 / 32768 | 0 |
| QKV tail, M1024 vs M16 | 29 / 65536 | 0.0009765625 |
| QKV tail, fixed normalized input, M1024 vs M16 | 29 / 65536 | 0.0009765625 |
| Independently normalized vs fixed-input M16 QKV | 0 / 65536 | 0 |
| M1024 V vs actual full-Prefill capture | 0 / 16384 | 0 |
| M16 V vs actual hit capture | 0 / 16384 | 0 |
| M1024 V vs M16 V | 9 / 16384 | 0.000244140625 |

This establishes that **projection M-shape changes suffice to reproduce the
observed first-layer V difference in this environment**. It does not establish
that every final candidate/score difference has this sole cause, nor validate
Q/K postprocessing or actual FA `std_*` metadata. FP64-rounded projection values
are included as a local numerical reference only; no acceptance tolerance or
ordinary-request GEMM configuration is changed.

The frozen-case reproducer is now `tests/profiling/gr_projection_probe.py`.
It accepts explicit model, capture and trusted compiled-module paths and checks
the generated module's SHA256 before executing it. This diagnostic deliberately
requires the captured OneRec 1024/16 case; it is not runtime B=1 admission logic.

```bash
CUDA_VISIBLE_DEVICES=1 USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m tests.profiling.gr_projection_probe \
  --model /home/z00469465/OneRec/OneRec-1.7B \
  --capture /tmp/step5_gpu1_prefill_probe.json \
  --compiled-module /root/.cache/vllm/torch_compile_cache/torch_aot_compile/306c4972ad560224c717460b4d6c235fcb79ebe48a99671fc2d2f7e4781af4bc/inductor_cache/tw/ctwouxyqsg34xyzaea6mfwswsajj2h6kgocmt5nq36k7svkqywrs.py
```

### Completion-notification feasibility, not product integration

An isolated, single-process GPU 1 experiment
(`/tmp/step5_completion_probe.py`, result
`/tmp/step5_gpu1_completion_probe.json`) returned 12 real device-written controls
in order through two reusable D2H slots. A CUDA callback on a separate
notification stream signals Host completion; there are no intermediate
`Event.synchronize()` calls. There **is still a Host wait**, and the experiment
does not prove model overlap, E2E improvement or strict no-wait acceptance.
Raw enqueue/callback/wait times and device gaps are saved, not promoted to a
performance claim. This script is a success-path experiment, not a production
implementation or an error-injection harness.

The API choice has important limitations. NVIDIA describes
`cudaStreamAddCallback` as planned for eventual deprecation/removal; it reports
device errors and blocks subsequent work on its stream until the callback
returns. Its alternative `cudaLaunchHostFunc` does not guarantee callback
delivery on context error. Neither callback may call CUDA APIs.
[CUDA Runtime stream API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html).

Internal review therefore does **not** approve connecting the Python callback
prototype directly to V1. Before integration it needs a native notification
trampoline, lifetime retention through callback return, separate delivery
cancellation vs device completion, generation-tagged slot ownership, and safe
error/abort/timeout/shutdown draining. Control-output completion also does not
protect upstream input staging: the reused pinned H2D sources require their own
completion lease. Removing the existing prepare Event without that lease is
unsafe. The next implementation experiment must establish these resource
lifetimes while continuing to use the original EngineCore BatchQueue/Future;
no second engine queue and no ordinary-request behavior change are authorized.

### Native completion lifetime experiment

The Python-callback prototype has now been followed by a native-only callback
primitive in `tests/profiling/native/gr_completion_ticket.cc`, exercised by
`tests/profiling/gr_native_completion_probe.py`. Neither is imported by product
code. The native callback only stores status and signals a condition variable;
it does not acquire the Python GIL, call CUDA, or own torch/device tensors.

Ownership is explicit: a new ticket has an owner reference; single-shot
registration reserves a callback reference before invoking the runtime;
each Python wait/poll/register borrows its own reference under the owner lock.
Owner close is idempotent. Callback release occurs only after unlocking and
notifying. A failed registration releases its reserved reference but never
marks preceding DMA complete. Cancelling delivery is separate from device
completion; a consumed delivery result is frozen against a later cancel.

CPU sanitizer verification:

```bash
c++ -std=c++17 -pthread -O1 -g -fsanitize=address,undefined \
  -DGR_COMPLETION_TEST_MAIN tests/profiling/native/gr_completion_ticket.cc \
  -o /tmp/step5_completion_ticket_test
/tmp/step5_completion_ticket_test
```

This passed cancellation, timeout, registration failure, immediate callback,
duplicate-registration rejection, consumed-result stability, deterministic
close-during-registration, and 1000 owner/waiter/callback races, ending with
zero live tickets and no sanitizer report. The first sandboxed invocation
failed because LeakSanitizer could not inspect `/proc` thread information; the
subsequent approved unsandboxed runs retained ASan/UBSan/LeakSanitizer checks and
exited zero. These are simulated CPU callback errors, **not** CUDA fault tests.

Physical GPU 1 validation, one process and one visible GPU:

```bash
c++ -std=c++17 -pthread -O2 -fPIC -shared \
  tests/profiling/native/gr_completion_ticket.cc -o /tmp/libstep5_completion_ticket.so
CUDA_VISIBLE_DEVICES=1 USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m tests.profiling.gr_native_completion_probe \
  --library /tmp/libstep5_completion_ticket.so \
  --runtime /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcudart.so.13
```

Final result/log: `/tmp/step5_gpu1_native_completion_final.json`,
`/tmp/step5_gpu1_native_completion_final.log`. Eight device-written controls
matched stage/generation values through two slots. One real nonready timeout
was observed; delivery cancellation did not grant slot reuse. All eight device
completions arrived; seven deliveries were consumed and one remained cancelled.
Six attempts to reuse an occupied slot were held until the old owner was
drained. Native live-ticket count returned to zero. The recorded zero wrong
generation updates covers these observed transitions, not arbitrary delayed
callback fault injection or a production slot manager.

Frozen source SHA256:
`015f1429872bdd31a5f154cfd127aedfba297fb76edbc1d7a210c2399dd1a3c3`;
shared library SHA256:
`d6b28db0710ebdbfe893713ffce3595ad545063ea30cea3efb68282af68cb62b`;
result SHA256:
`94f147c915904495a6ccec9986ea845be473d3d048a6c789838c678c87004eef`.

This establishes a narrower notification/lifetime building block. Host waits
remain, and the experiment uses a final device drain while retaining all
libraries, tickets and tensor owners. Real context failure, shutdown racing
with new admission, missing callbacks, H2D input-staging leases and V1 adapter
integration are still unverified. In particular a failing final CUDA drain
cannot be assumed to authorize resource release. Step 5 remains FAIL; no
Event protection has been removed from the ModelRunner based on this experiment.

### Packaged native completion boundary (not enabled in ModelRunner)

`vllm_gr/ops/cuda/native/gr_completion_ticket.cc` is now the sole native Ticket
implementation. `tests/profiling/native/gr_completion_ticket.cc` includes it
for the existing standalone/sanitizer tests. A minimal CPython module exposes
`API_VERSION=1`; the C ABI independently exposes `gr_ticket_api_version()==1`.
Ticket construction catches allocation/condition-variable construction failure
and returns a null pointer rather than allowing a C++ exception through ctypes.

`setup.py` builds this host C++ extension **only** for the CUDA install target.
It needs C++17 and pthread, not CUDA headers, NVCC or pybind11. CPU/NPU routing
tests require no extension. An actual isolated `build_ext` and Python import
passed at
`/tmp/pytest-of-root/pytest-313/gr_completion_build0/lib/vllm_gr/ops/cuda/_completion.cpython-312-x86_64-linux-gnu.so`.
This does not yet verify a published wheel/sdist or replace the installed
environment's extension.

`vllm_gr/v1/worker/cuda_completion.py` provides an explicitly invoked, cached
loader and strongly referenced Ticket handles. Importing it does not import
Torch or start device work. The loader checks both ABI versions and resolves
CUDA callback symbols through the `torch._C` dependency handle instead of
searching arbitrary runtime paths. A CPU-only symbol lookup with visible GPUs
disabled confirmed this handle exports `cudaStreamAddCallback` on the installed
PyTorch build; it did not register a real CUDA callback. Missing artifacts or
capabilities raise actionable errors with no compilation/synchronous fallback.

The Python wrapper validates generation, pointer and timeout integer ranges
before ctypes can truncate them. `wait_device()` returns true only after
successful actual completion; timeout returns false, while registration/device
errors raise. Cancellation is a separate delivery outcome, not DMA completion.
Closing an owner leaves the native callback/waiter references intact. Native
`delivery_claimed` means the caller claimed a completed result, **not** that a
pinned Host payload was read; the future storage lease needs a separate ack.

Evidence:

- Packaging/build/routing plus existing packaging tests: **15 passed**.
- Real compiled Ticket ABI with CPU-only fake registration: **24 passed**,
  log `/tmp/step5_cuda_completion_cpu_tests.log`. Cases cover immediate/late
  notification, cancellation, timeout, registration failure, device-error
  status, owner-close/waiter/callback races, overflow and lazy-loader gates.
  There is no CUDA registration or Tensor resource lease in these tests.
- Rebuilt standalone ASan/UBSan/LSan check: 1000 reference races passed,
  `live=0`, at `/tmp/gr-completion-sanitizer-3RN0b7/gr_completion_ticket_test`.
  The approved unsandboxed rerun retained LeakSanitizer after the sandbox's
  `/proc` restrictions prevented its thread inspection.
- Native source SHA256:
  `2a6974ef63e90b64d5bb34e743760f6e30b245f1888641f432c330ac64a32fa6`;
  isolated extension SHA256:
  `8630d806f04c5b7410a4cf139be331e61d24dd53ce7cb18e52ae07b1fc33796b`.

The adapter is **not enabled**: current ModelRunner copies/Events are unchanged
by this packaging work. Integration must intercept the original async-output
constructor before allocation/D2H, retain runner-owned two-slot leases across
final state release, and account for every Host consumer. Unsupported Host
output-history consumers cannot be given a fake Event. Registration/drain
failure must quarantine resources and prevent subsequent reuse; throwing from
`backend.release()` alone is insufficient because shared state release still
calls `session.end_session()`. Continue using the existing EngineCore Future /
BatchQueue, never a second queue. Actual CUDA errors, shutdown, adapter wiring
and performance remain unverified; Step 5 remains FAIL.

### Input-preparation ready-event fast path

An independent audit of the local upstream V1 preparation region identified
nine groups of reused pinned H2D sources: six runner `CpuGpuBuffer` instances
(`input_ids`, `query_start_loc`, `discard_request_mask`, `req_indices`,
`query_pos`, `num_scheduled_tokens`), native computed-token counts, the native
block table, and `_kv_block_zeroer._ids_pinned`. The last source can be written
inside `_update_states()` even during Decode when new logical blocks are
allocated. A guard limited to `_prepare_inputs()` or token IDs is insufficient.
Conditional paths can use further sources; this is not an exhaustive inventory
for every upstream feature.

The implemented optimization therefore **does not switch Host buffers or
introduce a double-buffer lease**. It preserves the upstream guard around the
whole preparation region, including `_update_states()` and metadata/preprocess.
Only the GPU adapter knows about Event query/record. A true query permits
entry without calling synchronize; a false query delegates to the original
context manager. Query errors propagate before Host writes. The same Event is
recorded at the original boundary even if the preparation body raises.

The fast path is scoped by `_gr_cuda_logical_geometry`, which exists only
inside the validated persistent Decode execute frame and is removed in
`finally`. Stale `_current_beam_info` / `_current_scheduler_output` cannot
activate it after that frame. No B=1 constants are added to the synchronization
protocol; existing persistent admission restrictions are unchanged. Ordinary,
Prefill, dummy and disabled-async/None-Event calls fully delegate upstream.
The existing single-threaded/non-reentrant Worker assumption is retained.

Runtime diagnostics separately expose `input_prep.ready_event_skips` and
`input_prep.not_ready_upstream_waits`. A nonready wait is a real synchronization,
not success under the strict Step 5 gate. Native completion remains an isolated
experiment, not a newly introduced build dependency or a runtime fallback.
These counters count branch entry, not successful preparation/wait completion.
The patch is installed on a fresh Worker; hot-patching an already initialized
and previously marked runner is not supported.

The runtime diff for this experiment is saved in
`/tmp/issue54_step5_input_ready_runtime.patch`, SHA256
`fdfdf1e9597d4c37288e77bf9bc292ce20f5119f6c6accc0f3c17a566aa04ded`.
It is relative to the Step 4 HEAD and includes tracked `vllm_gr` changes only;
the additional untracked runtime file `vllm_gr/v1/beam/decode_geometry.py` has
SHA256 `fb9f7453a01b88050ad0ddf5f36ad3df803a0689487ce34c1b777c19fbf8e2fd`.

Validation commands and results:

```bash
CUDA_VISIBLE_DEVICES=1 USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m pytest -p no:rerunfailures -q tests/kernels/test_persistent_gr_cuda.py
bash /tmp/run_step5_gpu1_trace_static.sh input_ready
python /tmp/analyze_step5_nsys.py /tmp/step5_gpu1_input_ready_trace.sqlite
```

- GPU tests: **12 passed, zero skips**, log
  `/tmp/step5_gpu1_input_ready_cuda_tests.log`. The new real-H2D tests use the
  upstream V1 guard and one pinned source. With a delayed device read, the
  nonready branch waits before overwriting Host data; the first copy retains
  its old values and the second receives the new values. Ready and pending
  branches are explicitly asserted; a missed timing setup fails the test.
- Fresh GPU 1 OneRec/BF16/B=1/W=128/S=3/CUDA Graph/cache-disabled trace:
  `/tmp/step5_gpu1_input_ready_trace.{json,log,sqlite,nsys-rep}`. One warmup plus
  two measured sessions; real Worker counters end at **6 ready skips, 0
  nonready waits**. Public output fingerprints and all three complete compact
  sequence/score/parent/logprob records exactly match the resident-cache trace.
  All three short session release checks pass. This is not a soak test or the
  full hardware matrix.
- Original schedule-overlap analysis:
  `/tmp/step5_gpu1_input_ready_analysis.json`. Measured scheduler/kernel overlap
  is 138.025 / 124.214 us for P→D and 140.614 / 99.653 us for D→D. Warmup has
  no observed overlap. These figures do not establish preparation overlap or
  a performance percentage.
- **Trace accounting caveat:** the original `/tmp/analyze_step5_nsys.py`
  interval ends at `gr_final_output.end`, not final Scheduler reconciliation.
  Its `[4, 4, 5124]` D2H byte list and synchronization counts describe only
  that interval. They must not be presented as full-session totals; output
  retrieval/control-copy work can remain after final Beam materialization.
- Traced E2E samples are 92.340091 / 96.412782 ms, versus the preceding
  resident-cache capture's 90.014424 / 95.512727 ms. Two instrumented samples
  are insufficient to establish improvement/regression or a noise bound.
  No speedup claim is made; full-session transfer/synchronization accounting
  and the strict Step 5 gate remain outstanding.

### Full-session trace accounting correction

`tests/profiling/gr_session_trace.py` now extends each window through the
third output-bearing `AsyncOutputFuture.result` and its Scheduler reconcile.
It requires an explicit generation-stage count, FIFO output markers, matching
same-thread EngineCore scopes, and PID isolation; missing or ambiguous evidence
is rejected. This is a serial UniProc topology reader, **not** a request-ID-aware
concurrent/multiprocess analyzer. Two later housekeeping reconciles without an
output-bearing Future are recorded separately and excluded from the window.

```bash
python -m pytest -p no:rerunfailures -q tests/profiling/test_gr_session_trace.py
python tests/profiling/gr_session_trace.py \
  /tmp/step5_gpu1_input_ready_trace.sqlite --generation-stages 3
python tests/profiling/gr_session_trace.py \
  /tmp/step5_gpu1_resident_cache_trace.sqlite --generation-stages 3
```

CPU tests: **23 passed**, log `/tmp/step5_session_trace_cpu_tests.log`. Cases
include post-final copies/waits, missing final reconciliation, extra Futures,
cross-thread ambiguity, independent PIDs, successive sessions, one-stage
generation, and boundary-crossing transfers. The two real-capture reports are
`/tmp/step5_gpu1_input_ready_full_session.json` and
`/tmp/step5_gpu1_resident_cache_full_session.json`.

| Full-window metric, each of three sessions | Resident-cache revision | Ready-event revision |
| --- | ---: | ---: |
| D2H copies | 4 | 4 |
| D2H bytes | 5136 | 5136 |
| Event synchronize calls, including final | 8 | 6 |
| Stream synchronize calls | 4 | 4 |

The old cutoff omitted one 4-byte D2H and three Event synchronizations per
session: two inside `AsyncOutputFuture.result` and one in a trailing empty
dispatch's preparation. This changes the accounting, not the underlying
runtime. The new optimization removes exactly two already-ready preparation
Event waits per session in these captures. Remaining calls are not waived.
Copy size/timing alone does not prove payload identity, and Host API durations
must not be equated with device stall time. **Step 5 remains FAIL.**

### Remaining Step 5 acceptance work

The user's updated validation allocation permits physical GPU 0 and two CPU
test workers. GPU 0 initially appeared idle, but another workload allocated
memory during engine startup. The first compiled Beam-eager/prefix-hit attempt
failed before model loading: 27.65 GiB free versus the requested 31.07 GiB
(`gpu_memory_utilization=0.7`). Log:
`/tmp/step5_gpu0_input_ready_compile_hit.log`.

An explicitly labelled functional-only retry used `gpu_memory_utilization=0.5`,
without altering the model, dtype, Beam width, sequence lengths or execution
semantics. It also failed before model loading when external use grew: 1.34 GiB
free versus the requested 22.2 GiB. Log:
`/tmp/step5_gpu0_input_ready_compile_hit_mem50.log`. Both processes terminated
with exit code 1; neither attempt is GPU 0 correctness or performance evidence.
No unrelated GPU processes were stopped. Do not reduce the functional case
further or claim a persistent-state memory-budget result from these admission
failures; validation awaits available GPU capacity.

The second attempted command (remove the memory override and use the first
log/output names to reproduce the initial attempt):

```bash
CUDA_VISIBLE_DEVICES=0 USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
python /tmp/step5_profile_wrapper.py \
  --model /home/z00469465/OneRec/OneRec-1.7B \
  --catalog /home/z00469465/OneRec/generated_data/video_constraint_triples.json \
  --prompt tests/resources/single_one_rec_prompt.txt \
  --output-json /tmp/step5_gpu0_input_ready_compile_hit_mem50.json \
  --backend gpu --execution graph --beam-graph-off --prefix-cache hit \
  --gpu-memory-utilization 0.5 \
  --implementation step5-persistent-input-ready-compile-hit-shared-gpu-functional \
  --base-sha 7818e255f1173b689730caeda8e7cbace4cccf06 \
  --pr-324-sha 8f6724dd3443ddae9de0bb71b6ac5ee6082ff127 \
  --runs 2 --warmups 1 --async-scheduling
```

- Compiled Beam-eager vs Graph hit/disabled comparisons, including compact
  parent/logprob ABI and short consecutive-session resource checks, have passed.
- Resolve strict corrected-legacy hit differences or establish a valid
  independently checked oracle for the same numerical execution path.
- Extend the passing synthetic backend oracle to establish a valid
  model-level parent/logprob reference for strict legacy parity; mode parity
  by itself cannot validate those fields independently.
- Measure uncovered gaps and variability across captures; both P→D and D→D
  overlap are observed in the latest measured joint-trace sessions.
- Attribute intermediate synchronization calls and verify retained-memory
  behavior beyond the short checks already completed.
- Preserve runnable commands, artifacts and limitations before Step 5 commit.

No performance percentage is claimed. Full E2E/P99/goodput, lifecycle and
cross-backend acceptance remain subject to the later issue gates.

### GPU 0 resumed validation and control-slot ownership (2026-09-05)

The user reconfirmed physical GPU 0 availability. This supersedes the earlier
capacity blocker, not the outstanding Step 5 integration/acceptance gates.
GPU 0 UUID is `GPU-78990023-5606-bb80-49bf-8ddfe8683461` (L20). GPU work ran
serially; CPU regression used two pytest workers with one thread per worker.

The first hardware-test attempt inside the restricted sandbox produced CUDA
initialization error 304 and **12 skips**. Log:
`/tmp/step5_gpu0_resume_cuda_tests.log`. These skips are not acceptance.
The authorized outside-sandbox rerun produced **12 passed, zero skips** in
21.93 seconds, log `/tmp/step5_gpu0_resume_cuda_tests_escalated.log`:

```bash
CUDA_VISIBLE_DEVICES=0 USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
python -m pytest -p no:rerunfailures -q tests/kernels/test_persistent_gr_cuda.py
```

The existing 326-test CPU regression also passed again with no failures or
skips in 29.23 seconds, log `/tmp/step5_gpu0_resume_cpu.log`. It used the earlier
combined CPU test list with `CUDA_VISIBLE_DEVICES=` and `-n 2`.

#### Model-level prefix-hit run

```bash
CUDA_VISIBLE_DEVICES=0 USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
python /tmp/step5_profile_wrapper.py \
  --model /home/z00469465/OneRec/OneRec-1.7B \
  --catalog /home/z00469465/OneRec/generated_data/video_constraint_triples.json \
  --prompt tests/resources/single_one_rec_prompt.txt \
  --output-json /tmp/step5_gpu0_resume_compile_hit.json \
  --backend gpu --execution graph --beam-graph-off --prefix-cache hit \
  --gpu-memory-utilization 0.7 \
  --implementation step5-persistent-input-ready-compile-hit-gpu0-resume \
  --base-sha 7818e255f1173b689730caeda8e7cbace4cccf06 \
  --pr-324-sha 8f6724dd3443ddae9de0bb71b6ac5ee6082ff127 \
  --runs 2 --warmups 1 --async-scheduling
```

This is compiled-model **Beam-eager**, not the no-compilation eager reference.
B=1, Beam=128, generation stages=3, prompt=1024, BF16. There were four sessions:
cache prime, warmup and two measured requests. The harness's compact-ABI,
parent-bounds, score/local-logprob and state-release checks passed. The strict
single-case checker in `/tmp/step5_compare_acceptance.py` also passed, including
the expected 1008/1024 prefix hit rate and stable measured allocator/attributed
memory. This is not a new legacy comparison or an eager/Graph pair comparison.

- Both measured compact ABI digests:
  `29a0b1cc039ca4994cb6a028b9df2bb96fab9215b88ead50c6d31cbb6e3f3314`.
- Each session ended with active state zero; cumulative created/freed counters
  were 1/1, 2/2, 3/3 and 4/4.
- Allocated/reserved memory was 33,078,535,168 / 33,273,413,632 bytes at all four
  snapshots. Four sessions do not establish a long-run soak result.
- Raw measured E2E samples: 147.020063 and 145.456965 ms. No performance gain,
  regression bound or statistically meaningful P99 claim follows from two
  samples. This run did not capture a joint Scheduler/device trace.
- The JSON's overall evidence gate remains FAIL; individual functional checks
  do not override trace, full memory attribution or Step 5 requirements.

Artifacts: `/tmp/step5_gpu0_resume_compile_hit.{json,log}`. JSON SHA256:
`de5bd6d2a98eb25a510647ab89273526573a915707502d6d8471917abf9d35d6`.
Tracked runtime/build diff against HEAD `7ab07246b39b65eba9266d7a0b304050270ac46b`:
`/tmp/issue54_step5_gpu0_resume_runtime.patch`, SHA256
`b4a01aaae882f7efab12f4c955f044d20d5304016126d9ede5ff837c2545ad1b`.
That patch excludes untracked sources; the earlier geometry/native-source
digests and the component digests below identify those separately. The new
control pool was not imported by this model run.

#### Two-slot control ownership component

`cuda_control_output.py` adds an explicit lease identity (request ID, session
epoch, device step, slot, monotonic allocation generation). It holds storage
independently of logical GR state release, never allocates a third slot and
never waits for a Host consumer during admission. An occupied slot is rejected
before the adapter may fill its device tensor. Late/forged/foreign leases
cannot discard or acknowledge a later owner.

The producer guard spans device-use start through completion registration or
explicit submission failure. A Host-reader guard spans native notification
wait and actual Host read. Both block shutdown; discard alone never releases
pending DMA. Notification/registration/enqueue failures quarantine the pool.
Only a successful backend-supplied terminal drain permits shutdown release;
failed drains retain storage, and successful shutdown is idempotent.

Independent audit found and the implementation corrected three interleavings:

- shutdown before a begun producer actually enqueues its copy;
- another slot's device error masking an existing Host-reader exception;
- quarantine between device-use start and registration stranding the active
  producer marker and preventing a later terminal drain.

```bash
CUDA_VISIBLE_DEVICES= USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
python -m pytest -p no:rerunfailures -n 2 -q \
  tests/test_cuda_control_output.py tests/test_cuda_completion.py \
  tests/test_gr_completion_build.py

CUDA_VISIBLE_DEVICES=0 USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
python -m pytest -p no:rerunfailures -q tests/kernels/test_cuda_control_output.py
```

Final revision: **54 CPU passed** in 12.38 seconds (22 new ownership cases plus
the 32 Ticket/build cases); **2 real CUDA passed** in 6.94 seconds, no skips.
Logs: `/tmp/step5_control_pool_combined_cpu.log` and
`/tmp/step5_gpu0_control_pool_cuda.log`. Ruff and `git diff --check` passed.
The CUDA tests perform actual fixed pinned-buffer D2H and native notification
on a separate stream, read eight device-written controls, preserve both slot
addresses and reject reuse after cancellation until late DMA finishes. Their
explicit setup/terminal/late-completion test drains are not runtime no-sync
evidence; no real CUDA context fault is injected.

Source SHA256:

- `vllm_gr/v1/worker/cuda_control_output.py`:
  `90e2b3b48e6f44eb639467a41aa58c81d24e8ca1eaa2651465f905698afd563a`.
- `tests/test_cuda_control_output.py`:
  `5acf2f3bbbce0122ab439dd1d5d46ad2419e77ad8c8e3c6dd87c121696e9379a`.
- `tests/kernels/test_cuda_control_output.py`:
  `33c8331bca894aa31aefb93feb1ef4f6382f1767e601d1854c3ac8e2c13fc02d`.

Next mandatory integration work: preserve upstream output class identity and
normal-request delegation; acquire before control fill; gate the native
InputBatch's second Host-history consumer; validate non-overlapping physical
Host/device slot storage; return detached Host controls; handle constructor/
attach exceptions without losing DMA owners; keep final logical-state release
separate from transport release; and quarantine all affected runner resources
on failed drain. A busy-slot error must not become an ordinary abort regression.
These components are **not yet connected to ModelRunner**. Neither their CPU
nor CUDA tests demonstrate EngineCore/Future lifecycle, final output transport,
or elimination of the remaining intermediate synchronizations.

Step: 5 (in progress)
Commit: none; exit gate not met
Changed files: new CUDA control ownership component and its CPU/CUDA tests;
this review report
Behavior change: no ModelRunner runtime change in this increment
Tests run: commands and logs above
Test result: component/functional checks pass within the stated scope
Hard counters: model state created=freed=4, active=0; component slots<=2;
full Step 5 transport counters still require runtime integration and trace
Trace/benchmark: two raw model samples; no new joint trace or speedup claim
Known limitations: output adapter remains unconnected; real NPU deferred
Exit gate: FAIL

### Native output adapter connected to GPU V1 (current revision)

The earlier standalone-component sections are historical. The current runtime
uses `cuda_async_output.py` through the GPU V1 backend and sampling wrapper.
It is no longer an unconnected prototype, but Step 5's overall gate is still
FAIL pending full trace/lifecycle/parity/performance acceptance.

Implemented boundaries:

- Acquire a lease before `_control_output()` writes the device tensor; there
  is no unscoped runtime fallback. A session's phase/epoch is reset at its one
  Prefill bootstrap, not inferred from the identity of a reusable GR state.
- Preserve the native AsyncGPUModelRunnerOutput class, real copy-ready Event,
  original Future and BatchQueue. Scope selection uses the current ephemeral
  ExecuteModelState, not a stale scheduler-output attribute.
- Reject the native InputBatch's independent Host output-history consumer and
  routed-expert output before original sampling, then recheck at construction.
- Preallocate two non-overlapping pinned Host slots (8 bytes total for B=1),
  verify device/stream affinity, and freeze the device used for terminal drain.
- Successful final-state release does not cancel the independently owned final
  control. Host consumption returns detached values before acknowledging reuse.
- Prepare/forward/sampling failures quarantine the transport and retained
  workspace; later execute is rejected before entering the original method.
  GPU shutdown drains transport before common state/session release, and does
  not enter common release if drain fails.

An initial implementation imposed a 30-second completion timeout. Review
identified this as an unintended change to the upstream contract, and it was
removed before the current revision. Native ABI **2** accepts an explicit
unbounded wait selected by Python `timeout_ms=None`; negative numeric public
timeouts remain invalid. This does not add a request deadline or polling loop.
CPU tests cover real native completion/cancellation of this wait. An old test
that still classified None as invalid blocked the first boundary-test run;
only that run's verified pytest processes were terminated, its outdated case
was updated, and the full run was repeated. The interrupted log is not PASS.

Build the changed native artifact before launching a Worker:

```bash
VLLM_GR_TARGET_DEVICE=cuda python setup.py build_ext --inplace
```

Build log: `/tmp/step5_control_adapter_native_build_v2.log`. The local artifact
`vllm_gr/ops/cuda/_completion.cpython-312-x86_64-linux-gnu.so` has SHA256
`5490b6298a5395498da7789e6360d8332d9fa4bb8949e040778382f0d9b7f054`.
This is a local build, not published-wheel/sdist validation.

#### Functional and ownership verification

The first model-level connected run used the earlier ABI-1 bounded-wait
revision, with the same physical GPU 0 compiled Beam-eager/prefix-hit command
as the preceding run, changing only these arguments:

```text
--output-json /tmp/step5_gpu0_control_adapter_compile_hit.json
--implementation step5-persistent-native-control-adapter-compile-hit
```

It completed four sessions. The same-case checker and strict public/compact
digest comparison against `/tmp/step5_gpu0_resume_compile_hit.json` passed.
Both measured compact ABI digests remained
`29a0b1cc039ca4994cb6a028b9df2bb96fab9215b88ead50c6d31cbb6e3f3314`.
This establishes unchanged outputs for that case, not complete legacy parity.

Per session: three actual control copies and consumptions, 12 control bytes,
zero occupied slots at completion, two-slot high-watermark, 8 fixed pinned
bytes. Across four sessions, cumulative copies=consumed=12 and state
created=freed=4, active=0. JSON/log:
`/tmp/step5_gpu0_control_adapter_compile_hit.{json,log}`. JSON SHA256:
`152b24f552c3cc72f9ec13cb4c4abf5c46b1a8e784447a52b7228c61f1a8f8af`.

Current ABI-2 revision:

- **376 CPU passed**, 35 warnings, 30.69 seconds, no skips. Log:
  `/tmp/step5_control_adapter_full_cpu_v2.log`. This is the earlier frozen
  13-file focused suite plus `tests/test_cuda_control_output.py` and
  `tests/test_cuda_async_output.py`, using two workers, CUDA hidden and one
  thread per worker. It is not the full repository CI suite.
- The adapter's 22 CPU cases include the real upstream AsyncOutputFuture in
  single-value/list modes, ordinary delegation, frozen old-session leases,
  initial alternate-slot selection, final logical-state release before output
  construction, actual detached Host consumption, and device/stream gates.
- **14 real GPU tests passed**, no skips, 24.75 seconds. Command:
  `CUDA_VISIBLE_DEVICES=0 USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python -m pytest -p no:rerunfailures -q tests/kernels/test_persistent_gr_cuda.py tests/kernels/test_cuda_control_output.py`.
  Log: `/tmp/step5_gpu0_control_adapter_cuda_v2_explicit.log`.
- The Beam lineage cases now construct and consume real V1 async outputs for
  each stage while retaining all sequence/score/parent/local-logprob oracle
  assertions. The preceding run had six failures because its simulated runner
  used the unresolved device alias `cuda` while tensors reported `cuda:0`.
  It was corrected to the actual runner's explicit device-index contract;
  the device gate was not removed. The earlier failure log remains
  `/tmp/step5_gpu0_control_adapter_cuda_v2.log`.

Tracked runtime/build diff: `/tmp/issue54_step5_native_control_v2_tracked.patch`,
SHA256 `f4441115e842fbcfd47ccdfdd9d55d6c1f2d60475a9c6866d15e492624620426`.
Untracked runtime source hashes:

- `cuda_async_output.py`: `ed1fe6d4cfc33e09ad7c73b5f7837904d448f3611a7d8bbb027629bd798e88a1`.
- `cuda_control_output.py`: `7308e8b7f9ef41627b09aa960992cd0d315b8abef23707b5d767f7ccbb4de5c1`.
- `cuda_completion.py`: `cccfbc203d3b4213169b5c0e71fa78f76f3849f894b801d3e83d0d2ed4c99daa`.
- Native C++: `60d40c110f64093e0c85301e7cbb2c05bcdaf71eb6cf7ff71e573d6dfe1e85a0`.

This supersedes the preceding "no ModelRunner runtime change" progress report.
No Step 5 commit has been made. Abort/timeout/shutdown with real pending
EngineCore output, complete numerical/reference parity, long-session soak and
the strict full-session no-sync/overlap gates remain mandatory. No performance
percentage is claimed, and NPU remains deferred to real hardware.

#### ABI-2 full-session GPU 0 trace

```bash
CUDA_VISIBLE_DEVICES=0 USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
VLLM_NVTX_SCOPES_FOR_PROFILING=1 VLLM_CUSTOM_SCOPES_FOR_PROFILING=0 \
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --export=sqlite \
  --python-functions-trace=/tmp/step5_nsys_annotations.json \
  --output=/tmp/step5_gpu0_native_control_v2_trace \
  python /tmp/step5_profile_wrapper.py \
  --model /home/z00469465/OneRec/OneRec-1.7B \
  --catalog /home/z00469465/OneRec/generated_data/video_constraint_triples.json \
  --prompt tests/resources/single_one_rec_prompt.txt \
  --output-json /tmp/step5_gpu0_native_control_v2_trace.json \
  --backend gpu --execution graph --prefix-cache disabled \
  --implementation step5-persistent-native-control-v2-nsys \
  --base-sha 7818e255f1173b689730caeda8e7cbace4cccf06 \
  --pr-324-sha 8f6724dd3443ddae9de0bb71b6ac5ee6082ff127 \
  --runs 2 --warmups 1 --async-scheduling

python tests/profiling/gr_session_trace.py \
  /tmp/step5_gpu0_native_control_v2_trace.sqlite --generation-stages 3 \
  > /tmp/step5_gpu0_native_control_v2_full_session.json
python /tmp/analyze_step5_nsys.py /tmp/step5_gpu0_native_control_v2_trace.sqlite \
  > /tmp/step5_gpu0_native_control_v2_overlap.json
```

The model and Nsight capture completed successfully. All three sessions
(warmup plus two measured) passed the functional checks, with the full compact
ABI digest `a68f9dd4456eff61b20b5af33d6f764458fefbb0d888556b84fbc8fdf92c3feb`.
All public fingerprints and all three compact results exactly match the prior
GPU 1 input-ready/Graph/cache-disabled capture. This is numerical evidence,
**not a same-device before/after performance comparison**.

The conservative full-session reader includes all three native Future
consumptions and Scheduler reconciliations, including the final control copy.
For every session:

| Metric | Observed |
| --- | ---: |
| Generation stages / Decode dispatches | 3 / 2 |
| Control D2H count / bytes | 3 / 12 |
| Final Beam/status D2H count / bytes | 1 / 5124 |
| Total D2H count / bytes | 4 / 5136 |
| Control Future Event synchronizations | 0 |
| Full-session Event synchronizations | 3 |
| Full-session Stream synchronizations | 4 |
| Control slots high-watermark / occupied at finish | 2 / 0 |
| Fixed control pinned storage | 8 bytes |

The 5124-byte final copy contains 5120 bytes of Beam ABI plus a 4-byte status
word. Its launch is inside `gr_final_d2h`; each 4-byte control launch is inside
the V1 async output-construction scope. No intermediate Beam-sized D2H occurs.

The three remaining Event waits are initial Prefill preprocessing, final
Beam/status return, and a trailing housekeeping execute before the last
Future reconciliation. All four Stream waits occur within the same initial
Prefill execute. There are no synchronize API calls in the two Decode execute
frames or any control-output Future in this capture. None of the remaining
full-session calls is silently excluded or treated as zero cost.

Measured D→D Scheduler/actual-kernel overlap is **170.583 / 144.722 us**.
Measured P→D overlap is **0 / 0 us**; warmup overlap is zero for both transitions.
The previous GPU 1 capture had P→D overlap, but device/clock variability and
changed CPU output work prevent attributing this difference from these two
captures alone. The next investigation is the exposed Prefill handoff and
notification-registration cost, not a claim that all gaps have been removed.
The legacy `/tmp` analyzer is used only for launch-correlated kernel overlap;
its pre-final-only transfer/synchronization summaries are not acceptance data.

Traced raw E2E samples are **94.9922 / 98.291496 ms**. Two instrumented samples
cannot establish an E2E improvement/regression, noise bound or meaningful P99.
Request counters report one create/add/state create/free, zero Decode request
rebuild/Prefix lookup, two Decode dispatches and one final Beam result copy.
After three sessions state created=freed=3 and active=0. This is still a short
resource check, not the required soak or all-lifecycle proof.

Artifacts share `/tmp/step5_gpu0_native_control_v2_trace` with suffixes
`.nsys-rep`, `.sqlite`, `.json` and `.log`. SHA256:

- Raw result JSON: `d523e26e3444856804c4e92020f4dbf8e57de332bc1576c048920f74d7d0178c`.
- SQLite: `79b2d1d5b7dff8343825b74643f2931b21bb9ef48e79652d1f517e0111bef1da`.
- Full-session report: `35ae321a2a4a53f2401fefad42e50b1ec5be515ebce86f42c5b2c897f587c0d7`.
- Overlap report: `4e2dd8c94e08a3a6ab402270a6a639513d7880fc40c927872fda939af965a309`.

#### Abort/queue inspection (not lifecycle test acceptance)

Read-only inspection of installed vLLM 0.22.1 shows that normal
EngineCoreProc/UniProc abort does not remove the queued Future: abort finishes
Scheduler requests; `step_with_batch_queue()` consumes `future.result()` before
stale reconciliation; `has_work()` includes a nonempty BatchQueue. Thus zero
remaining requests does not prevent the process loop from draining controls.

InprocClient has no background loop. If the caller aborts and stops stepping,
queued controls can remain owned until a later step/new request/shutdown.
Do not claim abort-return itself implies zero occupied transport slots.
Exception/shutdown safety depends on quarantine and successful device drain,
not necessarily individual Future consumption. The existing abort release RPC
also discards its nonblocking Future, so injected release errors can be hidden;
this needs an explicit error-observation test/fix before lifecycle acceptance.

#### Real BatchQueue/control adapter composition checks

Five additional CPU cases now call the installed vLLM 0.22.1
`EngineCore.step_with_batch_queue()` and `AsyncOutputFuture` directly, with
the real control adapter and simulated CUDA storage/completion:

- Abort before draining, and an abort processed after Future consumption:
  Host control acknowledgement precedes stale-result reconciliation.
- Immediate next-session admission: the upstream queue schedules the new
  output before consuming the old one; the two occupied slots remain distinct,
  and the detached old/new controls are 101/202 without overwrite.
- In-process caller stops stepping: a completed DMA does not acknowledge Host
  delivery; the slot remains owned until explicit successful terminal drain.
- Injected completion failure: upstream Future records the original exception,
  no Scheduler reconciliation occurs, and quarantine retains the slot/ticket
  until successful shutdown drain.

These do **not** run an actual EngineCore process, GPU DMA, or frontend abort
RPC. They prove queue/adapter composition, not complete GPU lifecycle acceptance.
The Scheduler reconciliation endpoint is a test double; existing Scheduler
tests are included separately in the regression command.

The adapter file has 27 passing cases. The expanded focused regression has
**395 passed, 35 warnings, 31.18 s**, with two CPU pytest workers and no GPU:
the preceding 376-case command, plus the five new adapter cases and
`tests/test_engine_core_gr_abort.py tests/test_gr_model_runner_state.py`.
Log: `/tmp/step5_control_batch_queue_lifecycle_cpu.log`.
Ruff for the changed test file and `git diff --check` both pass.
Runtime code is unchanged by this test addition; the previous GPU capture's
runtime fingerprint remains applicable. The release-RPC error-observation
gap above remains open and is not covered by the passing composition tests.

#### Prefill handoff attribution from the same capture

Independent trace review decomposed the two measured last-Prefill-kernel to
next-`gr_async_schedule` intervals. All values below are microseconds from the
instrumented GPU 0 capture, not uninstrumented benchmark latency:

| Nonoverlapping interval | Measured session 1 | Measured session 2 |
| --- | ---: | ---: |
| Last Prefill kernel end to bootstrap scope end | 50.065 | 49.000 |
| Bootstrap end to async output constructor start | 178.411 | 175.206 |
| Async output constructor | 340.007 | 325.830 |
| Constructor end to sample_tokens end | 65.892 | 67.831 |
| sample_tokens end to EngineCore step end | 7.609 | 6.482 |
| Between EngineCore steps | 17.501 | 17.089 |
| Next step start to schedule start | 18.376 | 17.541 |
| Total exposed P→D handoff | 677.861 | 658.979 |

The Prefill control's `cudaStreamAddCallback_v5000` Host API durations are
14.012/15.562 us, included inside the constructor row. They cannot explain
the full handoff. Across all nine controls, callback registration has a
14.581-us median; the first cold call is 140.107 us and the other eight range
from 11.018 to 22.794 us. No callback API replacement is justified by this
capture. The next measurement should split output-constructor work and the
bootstrap-to-constructor interval, while checking instrumentation overhead.

Attribution artifact: `/tmp/step5_gpu0_native_control_v2_handoff.json`, SHA256
`6440ae10c7e634f2ac7f09ca4c35751cf3d53ab5affe43f6585f5728487461c8`.

#### Terminal release error observation (implemented after the capture)

The audited recommendation is now implemented: observe
`release_future.result()` on the existing executor FIFO, only for terminal
persistent-GR abort, instead of dropping the Future. A callback alone cannot
drive the installed Multiproc lazy Future. Submission/reply failures retain
the original exception identity and add the affected request IDs as a note.

Only on failure, retire that Engine instance's `add_request`, `step`,
`step_with_batch_queue`, and already-bound `step_fn`, plus its Executor's
`collective_rpc` entry. This also rejects dummy/utility RPCs and prevents an
in-process caller from catching the error and reusing a possibly misaligned
Multiproc response FIFO. Other healthy Engines are untouched. Native shutdown
remains callable; the verified UniProc/Multiproc shutdown implementations do
not need this RPC entry. Retirement does not clear BatchQueue, cancel old
Futures, or free device/control storage. It cannot undo previously submitted
device work.

This changes terminal abort latency: it waits for a release RPC reply,
not a universal GPU/control-slot drain. In UniProc, pending control leases still
belong to their old Future or terminal transport shutdown. No new timeout is
proposed: upstream FutureWrapper rejects `result(timeout=...)`, and an RPC
deadline does not bound all preceding FIFO waits. Device/response-thread hangs
therefore remain a separate lifecycle risk, not a reason to free live storage.

Eleven CPU cases cover actual UniProc Future success/error, direct submission
failure, actual Multiproc lazy FutureWrapper FIFO success/error, detached prior
outputs, ordinary/repeated/unknown aborts, first-error retention, pre-bound
`step_fn` replacement, healthy-engine isolation, and actual UniProc/Multiproc
shutdown entrypoints with fake Workers/queues. No model or actual Multiproc
IPC is exercised by these cases.

Two new real-CUDA cases exercise the actual V1 async output class, native
completion transport, UniProc release RPC, and GPU shutdown adapter. Successful
release leaves the old Host-unconsumed slot owned; an immediate new session
uses the other slot and both controls remain 7/11. On injected release error,
Engine/RPC reuse fails while the old slot remains owned; actual UniProc shutdown
reaches the GPU adapter, drains before common cleanup, and leaves zero slots
and native tickets. Worker request state and common cleanup are test doubles:
these cases do not prove complete model/Beam KV/session lifecycle acceptance.

Card 0 results: the four-case transport file passes, then the complete focused
CUDA suite passes **16 tests, 18 warnings, 26.05 s**, zero skips:

```bash
CUDA_VISIBLE_DEVICES=0 USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m pytest -p no:rerunfailures -q \
  tests/kernels/test_persistent_gr_cuda.py tests/kernels/test_cuda_control_output.py
```

Logs: `/tmp/step5_gpu0_abort_control_boundary_real.log` and
`/tmp/step5_gpu0_abort_release_cuda_regression.log`. An initial sandbox invocation
hit CUDA initialization error 304 and skipped all four cases; that run
(`/tmp/step5_gpu0_abort_control_boundary.log`) is not acceptance evidence.
The subsequent authorized card-0 runs above executed the tests.

Runtime delta above the previous capture is isolated in
`/tmp/issue54_step5_abort_release_runtime.patch`, SHA256
`c3138bd5baf9db4d68eee08533e81e9b15cad6cb4ff3cfe954ddec8ebd58c43f`.
The complete current `engine_core_patch.py` SHA256 is
`bf65a9368175120710dc221440120eb8d8bac7bdc2a1449d50dd22c9456aebc9`.
No new full-model trace or performance claim is attached to this lifecycle
change. The earlier handoff/output trace remains evidence for its recorded
revision, not a new-model abort test of the current revision.

The expanded current CPU regression completes with **559 passed, 35 warnings,
120.16 s**, zero skips. This includes ordinary-request/legacy self-advance,
protocol and frontend lifecycle regression tests in addition to the transport
and abort cases. It is a focused regression, not the complete repository CI.

```bash
CUDA_VISIBLE_DEVICES= USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -m pytest -p no:rerunfailures -n 2 -q \
  tests/test_beam_decode_geometry.py tests/test_beam_attention_metadata.py \
  tests/test_model_runner_common.py tests/test_gpu_model_runner_patch.py \
  tests/test_model_runner_patch.py tests/test_gr_async_scheduler.py \
  tests/test_beam_decision.py tests/profiling/test_gr_candidate_oracle.py \
  tests/profiling/test_gr_prefill_diagnostic.py tests/profiling/test_gr_session_trace.py \
  tests/test_cuda_completion.py tests/test_gr_completion_build.py \
  tests/packaging/test_install_metadata.py tests/test_cuda_control_output.py \
  tests/test_cuda_async_output.py tests/test_engine_core_gr_abort.py \
  tests/test_gr_model_runner_state.py tests/test_beam_self_advance.py \
  tests/test_beam_request_unit.py tests/contract/test_engine_core_protocol_contracts.py \
  tests/test_serving_beam_lifecycle.py
```

Log: `/tmp/step5_abort_release_cpu_regression.log`, SHA256
`ef4ebf1e5522e13db342cd1ccdbe2271796d4ecba249d2e013c3f8a52e7ac577`.
The 16-case CUDA log SHA256 is
`c37843a6a32e76acdfacfb66c413ddb70ff45b80435eab87ce67a00f51f933e8`.
Ruff for the runtime/abort/kernel changes and `git diff --check` pass.
Independent implementation and test audits found no additional blocking issue
within the stated boundary; full-model GPU abort and actual Multiproc IPC
lifecycle acceptance are still outstanding.

#### Full OneRec eager abort/reuse and Offline wait-loop fix

Preparing the full-model probe exposed a real frontend bug: after a successful
abort, FINAL_ONLY produces no final Beam output, while the Offline collector
previously kept stepping forever. A bounded red test failed with
`empty step loop exceeded its bound` (1 failed / 3 passed). The product fix is
limited to persistent collection: after checking for this step's final output,
look up the target internal ID in `output_processor.request_states`; absence
raises `persistent GR request terminated before final output`, including both
internal and external IDs. No synthetic output, global unfinished-request
heuristic, or ordinary-path change is used.

Four new cases pass, and the complete serving lifecycle file passes 13 cases.
Red/green logs: `/tmp/step5_persistent_abort_empty_step_red.log`,
`/tmp/step5_persistent_abort_empty_step_green.log`, and
`/tmp/step5_serving_lifecycle_full_green.log`. Repeating the complete focused
CPU command above with these changes yields **563 passed, 35 warnings,
119.19 s**, zero skips. Log `/tmp/step5_offline_abort_cpu_regression.log`, SHA256
`4601d132ff3349277264c08f9863f948f7715fff6f28fbdc39583927f90031fd`.

`tests/profiling/gr_lifecycle_probe.py` now loads the actual OneRec model,
GPU V1 runner and UniProc executor in-process. It calls normal
`GRLLM.beam_search`, then injects the real
`llm_engine.abort_request([internal_id], internal=True)` after a complete
non-final frontend step. The product's new termination check exits collection;
the probe does not throw a sentinel, delete request/state dictionaries, or
construct fake results. Successful final calls explicitly drain any trailing
zero-token BatchQueue entry before clean-state comparisons.

Actual GPU 0 / OneRec-1.7B / B=1 / W=128 / steps=3 / prompt=1024 / BF16 /
enforce-eager / Prefix Cache disabled results:

| Abort boundary | Recovery | Result |
| --- | --- | --- |
| Prefill issued=1, completed=0, device_step=1 | Explicit old-queue drain, then new session | PASS |
| Same Prefill boundary | Immediate new session, no pre-drain | PASS |
| Decode issued=2, completed=1, device_step=2 | Explicit old-queue drain, then new session | PASS |
| Same Decode boundary | Immediate new session, no pre-drain | PASS |

The JSON records, independently checked after capture, contain the target's
pending Future both before and after every abort and one occupied control slot
after abort. Thus immediate reuse actually crosses an unconsumed control, not
an already-empty queue. Neither pending Future nor device_step proves a GPU
kernel was still running at the instant of the abort; these are post-dispatch,
pre-reconciliation boundaries, not hardware execution-time fault injection.

Across one reference success, four aborts and four recovery successes:

- State created=freed=9, active=0 at finish.
- Every abort restores the Native free-block count to 15956, releases the one
  Beam session slot, and leaves zero Scheduler/Frontend requests.
- One control remains owned immediately after abort, as intended. After drain
  or recovery success, queue and control slots are empty.
- Final control copies=consumed=21, 84 bytes total; fixed pinned storage stays
  8 bytes and slot high-watermark is 2.
- All five successful outputs have identical public token/score/logprob/full
  ABI digests and identical complete compact Beam digest
  `784494c6b702c026b10fc70a084d794b6d49f3f8668e1c480136aedf1e6465eb`.
  This is same-engine recovery parity, not an independent Beam oracle or a
  legacy/eager/Graph numerical comparison.
- Clean successful snapshots retain allocated=33094524928 and
  reserved=33227276288 bytes. This short repeatability check is not a soak.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
python -m tests.profiling.gr_lifecycle_probe \
  --model /home/z00469465/OneRec/OneRec-1.7B \
  --catalog /home/z00469465/OneRec/generated_data/video_constraint_triples.json \
  --prompt tests/resources/single_one_rec_prompt.txt \
  --output-json /tmp/step5_gpu0_full_model_lifecycle_eager.json --execution eager
```

The artifact includes configuration, dependency/device information, runtime
source-file digests, prompt/catalog digests and per-boundary snapshots.
JSON SHA256: `a7ecc9df6ad885d891b29987b808ac924e6c9d3081a96fa2ed13df170bf843ce`.
Log: `/tmp/step5_gpu0_full_model_lifecycle_eager.log`.
At this point Graph lifecycle was still a separate gate: a graph configuration
alone does not count as Graph acceptance without measured capture/dispatch
evidence. The strengthened run below supplies that evidence. Partial-Prefill
cancellation, timeout, Worker error, shutdown with an active full-model request
and actual Multiproc IPC still remain separate gates.

#### Verified CUDA Graph full-model abort/reuse

The first graph-configured run also passed functional recovery, but
`/tmp/step5_gpu0_full_model_lifecycle_graph.json` did not record Graph dispatch
counts. It is not used as proof of Graph execution. After that process ended,
the probe was strengthened to reject disabled Python assertions and require:

- Exact issued/completed/device cursors at each injection.
- The target's pending Future before and after abort, with one still-owned
  control slot afterward.
- Captured width 128 and exact FULL Graph dispatch deltas: 2 per successful
  session, 0 for a Prefill abort and 1 for a Decode abort.

The strengthened card-0 run passes all four full-model Graph cases using the
same command as the eager probe, replacing the final arguments with:

```bash
--output-json /tmp/step5_gpu0_full_model_lifecycle_graph_verified.json --execution graph
```

The actual runner is `GPUModelRunner`, captured widths are `[128]`, and final
FULL dispatch count is **12**: five successful sessions × 2 plus two Decode
aborts × 1. Each aborted session retains exactly one pending control until
normal queue consumption or the immediately following new session. Final
state created=freed=9, active=0; control copied=consumed=21, occupied=0;
Native free blocks return to this mode's baseline of 15949; Beam free slots=1,
active slots=0; Scheduler/Frontend request counts and BatchQueue are empty.
All clean recovery snapshots have allocated=33087054848 and
reserved=33319550976 bytes, a short repeatability observation only.

All five Graph-mode successful outputs exactly match their own reference,
including full compact digest
`a68f9dd4456eff61b20b5af33d6f764458fefbb0d888556b84fbc8fdf92c3feb`.
This differs from the enforce-eager reference above: the compilation axis
differs, so these runs prove recovery parity **within** each execution mode,
not cross-mode numerical parity or independent-oracle correctness.

Artifacts and SHA256:

- `/tmp/step5_gpu0_full_model_lifecycle_graph_verified.json`:
  `81e63a4b6a948568f7f29cd139c752cf707bc6bb59b90459f00bcada4650b9a3`.
- `/tmp/step5_gpu0_full_model_lifecycle_graph_verified.log`:
  `12266949ad42d6794b92965c5261fead37067d62ea7f49bde6f37cf0e3acfa89`.
- Probe revision: `a24f5684d6cef376e0afe7f0f0449e76618d4ccc490141ed24eb0d8388774d11`.
- Current `gr.py`: `c28fdcd666f121c43ef9c4de2bc2eb11e73f7f6b495e3f6e8fa15f1ab8475c6e`.

Twenty-two new CPU probe-gate tests reject cursor mismatches, completed or
wrong-target Futures, premature control release, missing Graph capture,
missing/excess dispatches and resource-accounting leaks. Together with the
13 serving-lifecycle tests, the focused command
`pytest -p no:rerunfailures -n 2 -q tests/profiling/test_gr_lifecycle_probe.py tests/test_serving_beam_lifecycle.py`
passes **35 tests, 7 warnings, 23.56 s** with CUDA hidden. This overlaps the
563-case runtime regression; the counts are not added as disjoint suites.
Log `/tmp/step5_lifecycle_probe_gates_cpu.log`, SHA256
`ac73f419b7179a8490d92d95a38b251c402fce805faf1e72225b71ce4f891017`.

#### Active-shutdown finding (historical, before the fix below)

Code inspection identifies the next lifecycle gap: current GRLLM shutdown
reaches EngineCore executor shutdown and Scheduler shutdown, but the latter
only closes its publisher/connector and does not finish Native Requests.
OutputProcessor request registries are likewise not explicitly cleared.
Worker GR state/session release alone therefore cannot prove zero retained
Scheduler/Frontend ownership when shutdown starts with an active request.
This needs a controlled full-model reproduction and a GR-scoped cleanup fix;
the successful abort/reuse probes above started shutdown only after clean
completion and do not establish the active-shutdown gate.

#### Active final-Prefill shutdown: real GPU 0 RED → GREEN

The real-model probe `tests/profiling/gr_terminal_probe.py` suspends only its own
caller after a real stage, without sending abort. It retains strong references
to the Engine, EngineCore, ModelRunner and Native Request while calling the real
`GRLLM.shutdown()`. No registry clearing, synthetic final result or state release
is performed by the probe. Configuration: GPU 0 L20, OneRec-1.7B, V1/in-process,
B=1, W=128, three generation stages, L=1024, BF16, eager, Prefix Cache disabled.
At the boundary, issued/device step are 1 and completed is 0; the target request
has an unconsumed BatchQueue Future and one owned control slot. This is a
post-dispatch/pre-reconcile boundary, not proof of a kernel still executing.

RED JSON `/tmp/step5_gpu0_active_shutdown_red.json`, SHA256
`42341fb30c610edf17205016bd9bb2df2be5233cdffdbb4dc9c5c8e16b1fa2dc`:
Worker state/session/control storage released, but Scheduler and Frontend each
retained one request, the Native Request was not finished, and 64 Native blocks
were not returned (15892 free versus the initial 15956).

GREEN JSON `/tmp/step5_gpu0_active_shutdown_green.json`, SHA256
`b809ddba4ecf49841699dad5197f4a67879146dd55dced1e889823c58dd298ae`:
all nine recorded resource/terminal gates pass. Native status is
`FINISHED_ABORTED`; Scheduler/Frontend request counts and external-ID mappings
are empty; all 15956 Native blocks are free; GR state created/freed are 1/1;
Beam sessions and occupied Beam/control slots are zero. Repeated shutdown is
a no-op. The command exited 0. Each JSON embeds source fingerprints, complete
configuration, dependency/device details and input/catalog digests.

Reproduction (replace `green` with `red` only when using the recorded old source):

```bash
env CUDA_VISIBLE_DEVICES=0 USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
python -m tests.profiling.gr_terminal_probe \
  --model /home/z00469465/OneRec/OneRec-1.7B \
  --catalog /home/z00469465/OneRec/generated_data/video_constraint_triples.json \
  --prompt tests/resources/single_one_rec_prompt.txt \
  --output-json /tmp/step5_gpu0_active_shutdown_green.json \
  --execution eager --boundary prefill \
  > /tmp/step5_gpu0_active_shutdown_green.log 2>&1
```

The retained closed EngineCore still contains an unconsumed BatchQueue Future;
the probe does not claim queue emptiness or consume that result after shutdown.
Transport shutdown has drained and released its control storage. Graph Decode
shutdown, Worker error, timeout and actual multiprocess teardown remain separate
acceptance cases. No OpenAI shutdown acceptance is inferred from Offline.

The follow-up Graph Decode shutdown probe also passes:
`/tmp/step5_gpu0_active_shutdown_graph_decode.json`, SHA256
`cf8d3c79a5c3c37507a5e8cdf46ebcf01b51e6ad43eec6682f855fc61321b414`.
Use the command above with `--execution graph --boundary decode` and the
corresponding JSON/log paths. The strengthened probe checks actual capture at
width 128 and FULL dispatch delta exactly 1 (0 before request, 1 before
shutdown), not merely enabled configuration. The suspended boundary is
issued=2/completed=1/device=2 with one target pending Future and one control
slot. All nine terminal gates pass: state 1/1, zero active sessions/slots and
request registries, Native free blocks restored to 15949. Control copies are
2 × 4 bytes, consumed count 1; successful terminal drain releases the remaining
slot without delivering a synthetic output. The command exited 0.

CPU `pytest -p no:rerunfailures -n 2 -q tests/test_gr_terminal_shutdown.py
tests/test_serving_beam_lifecycle.py tests/test_pangu_llm_passthrough.py` with
CUDA hidden passes **26 tests, 35 warnings, 25.60 s**. Log:
`/tmp/step5_gr_terminal_shutdown_green.log`. Composition tests invoke the actual
upstream `EngineCore.shutdown()` with a fake executor and check exact order:
Worker drain → native finish → Scheduler shutdown → local frontend abort.
Drain failure must preserve Native/frontend ownership; frontend abort failure
must retry without repeating core shutdown. These CPU fakes do not constitute
real Worker failure or multiprocess hardware acceptance.

Expanded two-worker regression (the earlier 563-case command plus terminal
shutdown, Pangu passthrough and lifecycle-probe tests) completed with
**598 passed, 1 failed, 35 warnings, 119.44 s**. The single failure is
`test_late_output_after_successful_shutdown_does_not_restore_quarantine`:
the audit's new red test was collected before it was moved from
`tests/test_cuda_async_output.py` into its standalone final location
`tests/test_cuda_terminal_late_output.py`. Log:
`/tmp/step5_terminal_shutdown_cpu_regression.log`, SHA256
`6f685b24502a45af45ac9f1253c401ed89304c9db26a250b9b4b718922e52101`.
This is recorded as a failing regression, not a skip or waiver.

The red test proves that a retained output consumed after successful shutdown
is rejected before reading Host memory, but its error handler re-adds the
already drained transport to the global quarantine list. This is terminal
reference retention, not observed DMA use-after-free. A minimal fix must track
successful drain separately from `_closed` (which is also set before a failed
drain), and serialize terminal/quarantine ownership changes without holding a
Host lock while waiting for device drain. It must not manufacture output,
consume old Futures just to empty a queue, or relax failed-drain retention.

That fix is now implemented in `cuda_async_output.py` (SHA256
`cb13ab5e659a4ecfe893fc667b8c499d686815b3cf1986c894acd45f2218f7f1`).
The shared lock guards terminal success and quarantine list updates across all
transports; pool operations and device drain execute outside it. New CPU tests
cover late output, failed drain/retry, and a blocked drain on transport A while
transport B enters quarantine. A's successful removal must not remove B.

Focused CPU command, CUDA hidden:
`pytest -p no:rerunfailures -q tests/test_cuda_terminal_late_output.py tests/test_cuda_async_output.py`.
Result: **30 passed, 3 warnings, 17.10 s**, no skips. Log
`/tmp/step5_terminal_quarantine_green.log`, SHA256
`3804deddd803daa24d5aa904fa15fdfdbbf5721c1e7f85a01649b040a0277df7`.

Latest real GPU 0 regression:

```bash
env CUDA_VISIBLE_DEVICES=0 USE_TF=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
python -m pytest -p no:rerunfailures -q \
  tests/kernels/test_cuda_control_output.py tests/kernels/test_persistent_gr_cuda.py \
  > /tmp/step5_gpu0_terminal_quarantine_cuda.log 2>&1
```

Result: **16 passed, 18 warnings, 25.42 s**, no skips. Log SHA256
`957eb978332e6c94993cc4a1516db08cf2539d1b3ad6546c203f9ad5c7cea971`.
These are real CUDA transport/backend and native UniProc boundary tests with
fake model Workers, not full-model Worker-failure acceptance. The full-model
shutdown probes above precede the late-output ownership fix; their per-file
fingerprints remain the authoritative revision scope.

The expanded CPU rerun now passes **601 tests, 35 warnings, 77.08 s**, no skips:
`/tmp/step5_terminal_shutdown_cpu_regression_green.log`, SHA256
`fcd7adfabd9c5a4bc35f6b70670d68fa874ee297f9e5f1fd72e3b4aa4a73f94d`.
Run the earlier 563-case command with the same hidden CUDA/OMP=1 settings and
two pytest workers, additionally including `tests/test_gr_terminal_shutdown.py`,
`tests/test_pangu_llm_passthrough.py`,
`tests/profiling/test_gr_lifecycle_probe.py`, and
`tests/test_cuda_terminal_late_output.py`. This rerun includes the previously
failing late-output check in its standalone file and two further failed-drain
and cross-transport tests; no failing check was removed or marked skip/xfail.

The real CUDA test was subsequently strengthened as well: after an injected
UniProc release RPC error, retain the real `AsyncOutputFuture`, complete actual
CUDA/Worker shutdown, then call that old Future's `result()`. It must raise
`control pool is closed`, leave consumed count zero and native live tickets
zero, and not restore global quarantine ownership. The same two-file GPU command
with log `/tmp/step5_gpu0_terminal_late_output_cuda.log` passes **16 tests,
18 warnings, 25.44 s**, no skips; SHA256
`c279c2b9c191cd3c375f3d59b1e8bcb857e9388b3ce66e7aacd38b516c838891`.
This supersedes the preceding 16-case GPU run for late-output coverage, while
remaining a fake-model Worker/real CUDA and native executor boundary test.

#### Real-model Worker exception probe: two validated boundaries

`tests/profiling/gr_worker_fault.py` installs one-shot instance hooks only after
real model initialization. `prepare` runs the original first-Decode
`_prepare_inputs` completely before throwing the probe's unique
`InjectedWorkerError`; `advance` runs the actual CUDA backend Beam decision and
control-slot preparation before throwing, before GR State accepts the result
or reorders KV. Both enforce persistent metadata, generation index 1, device
step 1 and the logical/physical sidecar geometry. Ordinary/Prefill calls are
not injection targets. The original instance method/override is restored.

`gr_terminal_probe.py --boundary decode --fault prepare|advance` requires the
exact injected exception object to reach the real frontend; unrelated runtime
errors fail the probe. It then calls real shutdown without manual abort/free,
retaining Engine/Worker/Request references. Failed transport ownership and,
for advance, a device-started but unregistered control lease must remain until
terminal drain. Logical state/session release during exception unwinding is
not permission to reuse these retained device resources.

Two harness failures are retained rather than relabelled product failures:

- `/tmp/step5_gpu0_worker_prepare_error_eager.json` (SHA256
  `6cba5dff5639b3c0e61de634a1419d01e41b0042e04a72b4fdb221cdbae4f6fe`)
  failed before any request: eager creates Beam context lazily, unlike Graph
  initialization. Initial capture now records that absence, without allocating
  a session or inserting a warmup request.
- `/tmp/step5_gpu0_worker_prepare_error_eager_v2.json` (SHA256
  `c7efe21d7fe3b5960c2b5290f0fba5af68d3e56efbddeec7c9b83cf088954669`)
  reached the exact Worker exception and passed 13 of 14 terminal checks, but
  its slot-capacity check incorrectly compared free slots with the live-slot
  dictionary length. The latter becomes zero on release. The corrected check
  uses `registry.max_batch`, validates the exact free-ID set without duplicate
  entries, and requires both live maps empty. This run exited 1, not PASS.

CPU injection-helper and shutdown/frontend tests: **56 passed, 35 warnings,
26.42 s**, using two workers and CUDA hidden. Log
`/tmp/step5_worker_fault_probe_cpu.log`, SHA256
`5292d2675321a3e6a86876a946785f751a11c902674b594568fcae61919bd5ec`.
Command: `pytest -p no:rerunfailures -n 2 -q
tests/profiling/test_gr_worker_fault.py tests/test_gr_terminal_shutdown.py
tests/test_serving_beam_lifecycle.py tests/test_pangu_llm_passthrough.py`.
Separate slot-gate tests use the real `BeamSessionRegistry` at capacities 1
and 3: **13 passed, 3 warnings, 6.87 s**, no GPU. Log
`/tmp/step5_terminal_probe_gate_cpu.log`, SHA256
`30ea18b86a6e7c20e4fb9c0e61b0192f6cf2f19c8cc98a0728c8cd592aa8f067`.

The `control_pool_pinned_ownership_released` gate means transport/pool ownership
is gone, not that every pinned allocation is destroyed: retained old output
objects can still reference Host tensors. No queue-empty or full object-GC
claim is made. Python Worker leaf errors in UniProc do not prove CUDA context
fault, notification failure, actual multiprocess, timeout or OpenAI handling.

Corrected prepare run: **PASS**, `/tmp/step5_gpu0_worker_prepare_error_eager_v3.json`,
SHA256 `30bcd02733a37cdf3e80f87ddcd44bf445d4cc49abe4a44fa0290f742abbf92c`.
All 14 terminal gates pass and the command exits 0. Initial Beam context is
absent, as expected for eager. After injection, state created/freed is 1/1 and
active state is zero, but one transport-owned workspace and the original
Prefill control lease remain quarantined. Its registered, device-started
control has not been consumed (one 4-byte D2H). Native/Frontend requests are
still live and 65 Native blocks are owned (15891 free versus baseline 15956).
Normal shutdown restores all 15956 Native blocks, finishes the one request,
empties both registries, frees all Beam slots, drains control ownership and
leaves zero native completion tickets. This proves the tested Python prepare
failure boundary, not a CUDA device fault or next-engine reuse.

```bash
env CUDA_VISIBLE_DEVICES=0 USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
python -m tests.profiling.gr_terminal_probe \
  --model /home/z00469465/OneRec/OneRec-1.7B \
  --catalog /home/z00469465/OneRec/generated_data/video_constraint_triples.json \
  --prompt tests/resources/single_one_rec_prompt.txt \
  --output-json /tmp/step5_gpu0_worker_prepare_error_eager_v3.json \
  --execution eager --boundary decode --fault prepare \
  > /tmp/step5_gpu0_worker_prepare_error_eager_v3.log 2>&1
```

Combined harness/shutdown CPU regression adds
`tests/profiling/test_gr_terminal_probe.py` to the 56-case command above:
**69 passed, 35 warnings, 26.40 s**, no skips. Log
`/tmp/step5_worker_fault_and_shutdown_cpu.log`, SHA256
`b0d2a4aa9600fa5d6a76cabce0940d497d58fb15139f7dac33da41c032815332`.
These overlap earlier suites and are not added as disjoint product-test totals.

The first Graph advance run remains **FAIL** in
`/tmp/step5_gpu0_worker_advance_error_graph.json`: the real backend returns step 2
and reserves the second control slot, then the exact injected error is stored
in the Decode sampling Future. Native BatchQueue first consumes the older
Prefill output. Its poisoned pool rejects that delivery with
`RuntimeError('control pool quarantined; device drain is required')`, explicitly
caused by the same injected error. The initial harness accepted only direct
delivery and therefore stopped before recording terminal gates. It did call
shutdown in cleanup, but no hardware PASS is inferred from that cleanup alone.

The revised harness additionally accepts only this specific advance route:
exact RuntimeError type and args, direct `__cause__ is expected`, never an
arbitrary chain or implicit context. It also requires that the already-done
Decode Future remaining in BatchQueue holds the identical injected exception;
reading `.exception()` is allowed only after verifying `.done()`. No pending
control is consumed, no result is fabricated and no runtime protection changes.
CPU delivery and capacity-gate contracts pass **53 tests, 3 warnings, 7.21 s**
in `/tmp/step5_worker_fault_delivery_cpu.log` (two workers, no GPU).

The revised Graph advance probe passes all 14 terminal gates and exits 0:
`/tmp/step5_gpu0_worker_advance_error_graph_v2.json`, SHA256
`6836c0d3762b14ddfbe3205c06f09dae2e4071a4327d915ea4ec758fa44714c7`.
Use the complete prepare command above with `--execution graph --fault advance`
and JSON/log paths `/tmp/step5_gpu0_worker_advance_error_graph_v2.json` and
`/tmp/step5_gpu0_worker_advance_error_graph_v2.log` respectively. Each case uses
a new engine/process; a failed engine is not reused.

The original-error identity is preserved in the done Decode Future. Delivery
route is `quarantined_previous_control`, with the precise direct-cause check
described above. Real captured width is 128; FULL dispatch count advances from
0 to 1. The backend returns device step 2 before injection, but GR State remains
at step 1 and KV reorder for step 2 has not been called. Thus no completed
step-2 SelectKV or successful generation result is claimed.

After exception unwinding, state created/freed is 1/1, while one complete
workspace and two device-started control leases remain quarantined. The step-1
lease is registered; the step-2 lease is unregistered and no longer has an
active producer. Only the Prefill 4-byte D2H occurred; neither control was
consumed. Native/Frontend requests remain live before shutdown. Real shutdown
restores Native free blocks from 15884 to the initial 15949, finishes the one
request, empties both registries and Beam ownership, drains all control slots
and clears all native tickets. Repeated successful shutdown is a no-op.

No product runtime changes were needed for these two fault cases. Harness
corrections are preserved with their failed runs rather than replacing the
raw evidence. Timeout, partial Prefill, actual multiprocess/OpenAI, device
faults and the full lifecycle matrix remain unverified by these cases.

#### Constructor attribution: process-local profiling, not a runtime change

`/tmp/step5_ctor_profile_wrapper.py` adds exactly three NVTX scopes to the
persistent async-output constructor by transforming the installer AST inside
the diagnostic process: `gr_control_ctor_preflight`,
`gr_control_copy_enqueue`, and `gr_control_notify_register`. Stripping only
those scopes recovers the original AST exactly; ordinary delegation, statement
order, class identity and the registration `try/finally` are preserved. No
production source file, CUDA Event, wait, tensor read or D2H operation is added.
Wrapper, original installer, transformed AST and runtime-source hashes are
embedded in the profile JSON. These markers can themselves perturb timing;
their durations are attribution data, not an uninstrumented speedup estimate.

The first trace `/tmp/step5_gpu0_ctor_segments_trace.{json,sqlite,nsys-rep}`
completed three model calls and passed its functional/state/compact-output
checks. Its strict full-session reader, however, rejected
`output/Future/reconcile FIFO order mismatch`. The old session's empty BatchQueue
frame can be reconciled after the next session's Prefill starts, before that
new session's first output-bearing Future. The first trace lacks explicit
empty-frame evidence, so those extra reconciles must not be dropped by timing
heuristics or counted as another generation stage. It is not a full-session
transport/attribution PASS.

The follow-up `/tmp/step5_ctor_tagged_profile_wrapper.py` retains the three
constructor scopes and tags an empty upstream reconciliation *inside* the
existing GR reconcile range. The predicate requires a GR Scheduler and empty
scheduled-token map, GR sidecars, Beam metadata, new requests and cached request
IDs. The wrapper also rejects `python -O`. Neither scheduling decisions nor
queue operations are changed. The tagged trace is captured with the same
GPU 0, V1, UniProc, Graph, cache-disabled, B=1/W=128/S=3/L=1024/BF16 workload,
one warmup and two measured calls, under `/tmp/step5_gpu0_ctor_tagged_trace`.

The read-only `/tmp/analyze_step5_ctor_trace.py` must reject missing, duplicate,
cross-thread or out-of-order markers. It reports API sum *and* interval union,
checks unique PID/correlation ownership and exactly one 4-byte D2H for each
control constructor, and keeps asynchronous copies even if completion is
outside their constructor's Host interval. API-uncovered wall time remains
unclassified, not automatically Python execution time. Only explicit empty
reconcile evidence permits exclusion before the original FIFO validation.

The tagged trace now passes constructor and complete-session analysis. Its JSON
is `/tmp/step5_gpu0_ctor_tagged_trace.json` (SHA256
`ecb1c62cfadaea6722bf4a375d9bc04a89d9a678a216f89017ff572a0a3cc7ce`),
constructor report `/tmp/step5_gpu0_ctor_tagged_segments.json` (SHA256
`2dfd0d1c82655e728ea7974e4dd0fed23359bf28a4ea7a1a6068da85f3547964`),
full-session report `/tmp/step5_gpu0_ctor_tagged_full_session.json` (SHA256
`c2e9f697d30aedcb73abaec2036705d7eb133b959dbad2db6b148a8ba0ab3639`).
Two explicitly tagged empty reconciles are excluded before the original FIFO
checks; all three generation reconciles remain per session. Each full session
has 4 D2H transfers / 5136 bytes (three 4-byte controls and one 5124-byte compact
final buffer including status). Each individual control constructor has one
4-byte D2H, with measured device duration 0.992 us in the warm Prefill cases.

Warm Prefill constructor attribution, microseconds including profiling overhead:

| Range | Sample 1 | Sample 2 |
|---|---:|---:|
| Entire constructor | 361.381 | 337.511 |
| Preflight checks / field setup | 37.638 | 35.849 |
| Copy enqueue including copy-stream context | 189.371 | 176.582 |
| Notification wait / registration including notification-stream context | 97.921 | 91.437 |
| Residual outside three child scopes | 36.451 | 33.643 |
| CUDA runtime API union inside constructor | 56.044 | 50.736 |

The API intervals do not overlap in these two cases. Uncovered wall time is
not automatically pure Python time. D→D actual kernel/schedule overlap remains
visible, whereas P→D overlap is absent; the older overlap script's
`issued_before_previous_reconcile` field must not be used on this tagged trace
because that older script still pairs the unfiltered housekeeping reconcile.

The first small runtime change removes only the redundant notification stream
context. Its new CPU regression was red before the edit (unexpected
`enter_stream notification`) and green after. The preserved adapter-only rerun
has **28 passed, 7 warnings, 21.53 s** in `/tmp/step5_notify_context_green.log`,
SHA256 `3c07d174a9d4185345023ec9d74a5270a307dc9e7e53fe6448d2366d8f4e3052`.
It overwrote an earlier same-named 61-case log; that earlier run is not used as
a durable evidence artifact. The expanded run below independently revalidates
all four relevant adapter/control/terminal files. The original failure remains
in `/tmp/step5_notify_context_red.log`.
Actual GPU 0 two-file CUDA regression again passes **16 tests, 18 warnings,
25.42 s**, no skips, log `/tmp/step5_gpu0_direct_notify_cuda.log`.
No E2E or percentage improvement is inferred from this code simplification.

Expanded regression with CUDA hidden, OMP/compile threads 1, pytest workers 2:
**655 passed, 35 warnings, 118.55 s**, no skips. Use the earlier 601-case command
and additionally include `tests/profiling/test_gr_worker_fault.py` and
`tests/profiling/test_gr_terminal_probe.py`; the adapter file includes the new
notify-context test. Log `/tmp/step5_direct_notify_cpu_regression.log`, SHA256
`4298aa3b70d901018ead9c1e1d4d29ba8af17b12a0dff2af04697fd732d2a13d`.

The after trace `/tmp/step5_gpu0_direct_notify_trace.json` (SHA256
`0f1171dc5cc81e0b19736915a29ee159b899fb1a099be8481bdbf0d4266e8aa1`)
uses `/tmp/step5_ctor_direct_notify_profile_wrapper.py`, adapted to the direct
wait statement while retaining the same three scopes and semantic empty-frame
marker. Original-AST recovery is still required. The same tagged analysis
passes into `/tmp/step5_gpu0_direct_notify_segments.json` and
`/tmp/step5_gpu0_direct_notify_full_session.json`.

| Warm Prefill range (us) | Before 1 / 2 | After 1 / 2 |
|---|---:|---:|
| Notification wait / registration | 97.921 / 91.437 | 54.739 / 54.680 |
| Entire constructor | 361.381 / 337.511 | 329.904 / 333.433 |
| Copy enqueue | 189.371 / 176.582 | 199.928 / 192.899 |

Notification context removal has a measurable local effect in these samples,
but copy/residual noise absorbs much of the parent-range difference. This is
not a P99, goodput or E2E speedup claim. Before and after, each of three sessions
has the identical final compact digest
`a68f9dd4456eff61b20b5af33d6f764458fefbb0d888556b84fbc8fdf92c3feb`,
3 generation reconciles and 4 D2H transfers totalling 5136 bytes. Functional
state/parent/logprob checks pass. Full-session EventSync/StreamSync counts
remain 3/4, with the same initial-Prefill, final-copy and post-final ownership
boundaries; no new synchronization is introduced by the change.

Residual copy enqueue still creates/destroys one temporary CUDA Event per
constructor via PyTorch `Stream.wait_stream()` (`stream.record_event()` with
no reusable Event). That is a concrete next candidate to audit for bounded
per-slot reuse, not an already implemented change or a promised gain.

### Current GPU-0 smoke refresh and cache-metric evidence failure

Run command: `env CUDA_VISIBLE_DEVICES=0 USE_TF=0 python /tmp/step5_current_matrix_run.py`.
The runner serializes model processes, requires the approved L20 UUID, records
each expanded command, and verifies source hashes before and after each case.
Its frozen manifest includes runtime Python sources, installed native
extension binaries, the profiler and the acceptance wrappers. The original
source glob did not include `.cc`; those C++ sources are captured by the git
commit, whereas the actually executed `_completion.so` is manifest-hashed.
No runtime source change was made during these three cases.

`compile_disabled` and `graph_disabled` completed, each with one warmup plus
three measured sessions. All four compact outputs per case have digest
`a68f9dd4456eff61b20b5af33d6f764458fefbb0d888556b84fbc8fdf92c3feb`.
Measured public/compact ABI pair comparison passes, state created/freed is
1/1 through 4/4 with zero active, and measured attributed/allocator memory
plateaus. The Graph case captures W=128 and accumulates eight FULL Decode
dispatches over four sessions; 12 controls are copied/consumed, 48 control
bytes total, zero occupied slots at completion and high-watermark two.
There are zero persistent Decode input-prep not-ready upstream waits. This
does not mean initial Prefill or final return has no synchronization.

Result SHA256 values:

- `compile_disabled.json`: `a364fc14ef5cc1d9f14760bdacac73f6337b76459714b6d98ff30ac160714ab8`.
- `graph_disabled.json`: `5f3342c76035692d448efd97ef8dd50b31b402aa7d907081a45ba53bb79ada95`.
- failed `compile_miss.json`: `0363ec0f85b45e7b631ec1c6b4577454ddf351d41c1c626392afe7f8ed2b98a2`.
- stopped `manifest.json`: `c72526643a864cb237d6490a0e1d3d6a4db27bdd1a2f7f40340887e891c91fa2`.

`compile_miss` exited 1 at `_validate_prefix_evidence`: `queries=0.0,
hits=0.0`. The runner stopped instead of continuing the matrix or relaxing
the miss gate. Remaining cases have no evidence from this invocation.
The diagnostic `/tmp/step5_miss_metrics_probe.py` independently observes native
cache queries, actual Scheduler reconciliation stats and logger delivery,
without GPU reads, cache/result replacement or timing claims. Its initial
sandbox invocation could not access CUDA and is environment-unvalidated;
the real-GPU retry uses separate `_hardware` output/log names. A suspected
empty-EngineCore-output stats delivery loss must be proven before changing
the profiling evidence source. Neither a zero metric nor a planned diagnostic
is a cache-miss PASS.

The real-GPU retry has now confirmed that delivery loss. In
`/tmp/step5_gpu0_miss_metrics_events.json`, warmup records query=1024/hit=0
and delivers those stats to the logger. Each of the following three measured
requests also makes one actual native query with `gr_issued_steps=0`,
query=1024 and returned hit=0. An old empty scheduled frame then reconciles
with no request outputs and carries those 1024 queries; no logger record
follows it. Actual Prefill reconciliation subsequently carries zero cache
stats. Thus measured native queries total 3072, whereas the Prometheus delta
is zero. The original strict profiler still exits 1, as it should for its
chosen evidence source. This is not counter reset and does not demonstrate
a missed cache lookup or output error. Source confirmation: upstream
`Scheduler.update_from_output` attaches `make_stats()` to empty outputs,
`KVCacheManager.make_prefix_cache_stats` takes and clears the accumulator,
and `LLMEngine.step` only calls `logger_manager.record` when
`len(outputs.outputs) > 0`.

The bounded remediation is a profiling-only actual-query observer plus
retained Prometheus snapshots. It must not synthesize hits from configuration,
flush BatchQueue early, insert a synchronization, modify ordinary logger
semantics, or call a zero-query run PASS. Runtime native counters printed by
GR are not substituted for directly observed call counts: several are
declarations or reconciliation counts rather than independent instrumentation.

Diagnostic provenance: events SHA256
`2ae38b6e089dc3bfb310398942c54935b28c925d1dfaea0ee5aaaf50ae7fd37c`,
hardware failure-result SHA256
`e8d6fb348328bf11e76b90661844387efad6b47c522baf380c82df0cf970b2cc`,
probe source SHA256
`1c218df63287f3db5e952f1c8727c6aea89542cbe34101ff06328b009cb9ca1a`.

The fresh matrix validator additionally rejects missing declared counters,
missing final ABI fields and incorrect per-call control/Graph deltas. Both
current disabled outputs pass the hardened pair check. Combined CPU tests
for this validator and the pending actual-consumer metadata capture pass
**46 tests in 4.35 s**. Command:

```bash
env CUDA_VISIBLE_DEVICES= USE_TF=0 OMP_NUM_THREADS=1 \
  python -m pytest -p no:rerunfailures -q \
  /tmp/test_step5_current_matrix_compare.py \
  /tmp/test_step5_consumer_metadata.py
```

Durable log: `/tmp/step5_matrix_consumer_tools_cpu_root.log`, SHA256
`065e9ebe4bec2e670d2ba07f33a1c8d26e36981b8539ab8e74c61e4bc3dc6f66`.
The actual-consumer metadata diagnostic has CPU negative checks only so far;
it has not been run on a model and is not hardware acceptance evidence.

### Six current smoke cases pass with explicit cache evidence

The direct-query observer passed **25 CPU tests in 0.83 s** independently at
the root before GPU use. Log `/tmp/step5_native_cache_observer_cpu_root.log`,
SHA256 `d68d462a9257c4f557bb12ac67753bf961e400f3f663c136ab1f6b9374b1a14f`.
Observer SHA256:
`fa4023e0ee4521e3c4610e1e7c78cd539fa3cc835f0d8d0c7de912fae95fc89c`.
It binds the actual CUDA UniProc KVCacheManager before requests, preserves
native return objects/exceptions, checks unique persistent request IDs and
one method call per request, and records effective queries separately from
disabled/skip-read calls. It requires zero computed/issued steps on measured
queries. Missing observations cannot pass as zero-query cache-disabled.

Command: `env CUDA_VISIBLE_DEVICES=0 USE_TF=0 python /tmp/step5_current_matrix_resume.py`.
The resumed runner verifies the entire original source snapshot is unchanged
and freezes the additional observer/wrapper. All four cache-enabled cases
exit 0 with unchanged source hashes. JSON/logs, expanded commands and the
manifest are under `/tmp/step5_current_matrix_native_cache_20260905`.

| Cache | Compiled Beam-eager / FULL Beam Graph | Actual measured queries / hits per mode | Current compact result |
| --- | --- | --- | --- |
| disabled | PASS / PASS | 0 / 0 (cache disabled) | `a68f9dd4…` |
| miss | PASS / PASS | 3072 / 0 | `a68f9dd4…` |
| hit | PASS / PASS | 3072 / 3024 | `29a0b1cc…`; priming request is `a68f9dd4…` |

Full hit digest:
`29a0b1cc039ca4994cb6a028b9df2bb96fab9215b88ead50c6d31cbb6e3f3314`.
All cache-enabled modes retain measured Prometheus 0/0 alongside the direct
counts and explicitly mark the mismatch. New JSON sets
`performance_evidence_valid=false`; raw timing samples remain diagnostic,
not before/after performance evidence.

The six-case comparator checks the complete public dictionary and every
call's complete compact ABI, including priming/warmup; state increments and
release; three control copies/consumptions and twelve control bytes per
session; zero occupied control slots after each session; two FULL Beam
dispatches per Graph session and none for Beam-eager; and measured attributed
and allocated/reserved device-memory plateaus. There are 26 sessions across
six model instances (four each for disabled/miss, five each for hit).
No unexpected skip is used. This is short-session evidence, not long soak.
Disabled vs miss is exact in the reported fields; hit vs miss/disabled still
differs and is explicitly reported, not waived as a cross-cache parity PASS.

Comparison command:

```bash
python /tmp/step5_current_matrix_compare.py \
  --compile-disabled /tmp/step5_current_matrix_20260905/compile_disabled.json \
  --graph-disabled /tmp/step5_current_matrix_20260905/graph_disabled.json \
  --compile-miss /tmp/step5_current_matrix_native_cache_20260905/compile_miss.json \
  --graph-miss /tmp/step5_current_matrix_native_cache_20260905/graph_miss.json \
  --compile-hit /tmp/step5_current_matrix_native_cache_20260905/compile_hit.json \
  --graph-hit /tmp/step5_current_matrix_native_cache_20260905/graph_hit.json \
  --output-json /tmp/step5_current_matrix_comparison.json
```

Result **PASS**. Comparison SHA256:
`4e9b9c23f5c6b630800a9303d70301de366d81733c00842111f2a0eed9326f35`.
Resumed manifest SHA256:
`0227a82ba4051aaff3867f0ee5862d04cfd4cfec42384cf1f4a32efb8ccd7baa`.
The four result SHA256 values, in compile-miss, graph-miss, compile-hit,
graph-hit order:

```text
0e0f15d345d31d5dd025b9e4b325488c8f38dfc0773fe53e579e22a515d87c51
6e4811390ec3313f78667d1eef2b9413862e9bdacd72765f4e4d6cbe0b5a7ff3
74e8af63a814671cc9610294919a155305f80d68aa10072f7b305afecf6a14f4
e590dcc6182f41125ecc4309911a4ee467cddbdc54fe1fee38f5e75cf7a2c031
```

The separate successful-Native-lifecycle observer hooks the actual
`Request.__init__` (not a factory-return proxy), Scheduler add/free, KV free
and UniProc dispatch. Constructor epochs prevent object-ID reuse from merging
sessions; no Request strong reference is retained. The observer's own Host
event storage is diagnostic retention, not Host-RSS soak evidence.
Root regression: **9 passed, 16 warnings, 20.96 s**, log
`/tmp/step5_native_lifecycle_observer_cpu_root.log`, SHA256
`20f124abcfdf58b925f2e157523ce1c29c77f195f282248ab7650d238092e5a6`.
Hardware lifecycle/candidate/consumer diagnostics are separate from this
six-case comparator; their completed checks are recorded next.

### Current independent reference and Native lifecycle checks

Command: `env CUDA_VISIBLE_DEVICES=0 USE_TF=0 python /tmp/step5_current_reference_run.py`.
The serial driver checks the same original runtime snapshot, additionally
hashes the diagnostic hooks and records exact commands in
`/tmp/step5_current_reference_20260905/manifest.json`. Both model processes
exit 0 with unchanged source. The compiled Beam-eager and truly uncompiled
eager processes each execute one full-Prefill priming request and one hit.
These captures have an extra post-request diagnostic D2H and retained Host
events; none is product-transfer, latency or Host-RSS-soak evidence.

Independent analysis commands:

```bash
python tests/profiling/gr_candidate_oracle.py \
  /tmp/step5_current_reference_20260905/candidate_compile_hit.json
python tests/profiling/gr_candidate_oracle.py \
  /tmp/step5_current_reference_20260905/consumer_eager_hit.json
env PYTHONPATH=/home/z00469465/PROJECT/vllm-GR-Scheduler/vllm-gr \
  python /tmp/step5_consumer_metadata_validate.py \
  /tmp/step5_current_reference_20260905/consumer_eager_hit.json \
  --layers 28 \
  --output-json /tmp/step5_current_reference_20260905/consumer_metadata_analysis.json
```

All four candidate replays **PASS**, with zero selected raw-local-logprob
error. Full/hit computed-prefix lengths are 0/1008. Enumerated valid candidate
counts are 12943/13018 for compiled execution and 13047/12998 for uncompiled
eager. The independent replay validates selection, cumulative score, parent
lineage, selected logprobs, actual next Forward token/position and final ABI;
it does not independently validate model/sampler/KV mathematics.

Both captures' actual Native lifecycle event reports pass all eleven gates.
For every one of their four sessions: successful `Request.__init__`=1,
Scheduler.add_request=1, Scheduler._free_request=1, KVCacheManager.free=1,
actual Executor generation-stage dispatches={0:1, 1:1, 2:1}, Decode dispatches=2,
no active native session or observer anomaly. These are direct call observations,
not GR counter fields prefilled with constants.

The eager actual-consumer metadata analysis also **PASS**: 28 layers share the
observed returned metadata; actual standard-attention tables/indices/lengths
and Prefill K/V slot mappings agree with the committed native row; the hit
references the primed 63 reusable blocks; first persistent Decode's actual
prefix table agrees with persistent binding/native row, with correct Prefix
and suffix extents and fixed-buffer address. It covers full/hit Prefill and
first Decode **at the eager metadata return boundary**, not Graph replay,
later Decode, attention-output mathematics or an independent suffix-KV oracle.

Artifact SHA256 values under that directory:

| Artifact | SHA256 |
| --- | --- |
| candidate_compile_hit.json | `fb8fb3c264815cdccc2ed562f8297147af090dbe8d003a18812f4112cebc47ae` |
| consumer_eager_hit.json | `3b6eb5fd4e0060315eff23a84c4388787295a8fe66ebd3e76f94c15fadb6c3c3` |
| candidate_compile_oracle.log | `90c76a1ae7abe74b90309048fe9c6ab311c8136c4c8e894f9a75bd07233267f5` |
| candidate_eager_oracle.log | `1ac3edc9354bedab2d8012fe7950198b4cf0979f3c78bc588ca15f96b0d779b5` |
| consumer_metadata_analysis.json | `0a4871c494a9fbafda74ccaa2ef88a7767082f7b17b6e576ca3197f92a8611c6` |
| manifest.json | `a7417069bcc82993c73f96ca5351d22b0deb61356e24e17332a8c8d5edeb3ffb` |

The independent audit additionally rehashed the 140 original manifest entries,
including `_completion.so`, and confirmed equality across current cases and
the working runtime. The six-case pair results, actual per-call control/Graph
increments, and eighteen measured memory snapshots were independently checked.
Together with the previously completed 655 focused CPU/16 real GPU tests,
complete-session transport/overlap trace, abort/reuse, active shutdown and
Worker-error probes, this closes the **revised Step 5** implementation gate.
It does not close the original literal reference conditions, Step 6 or Step 8/9.

### Preserved evidence bundles and remaining ranking difference

The current acceptance drivers/hooks, raw six-case data, four reference
sessions, failed miss evidence, analyzers and tool-test logs are archived at
`/tmp/issue54-step5-evidence.iNKZtw/current-acceptance.tar.gz` (SHA256
`dda8983b54611ea43d005a5ae916933dcbbd706d93be43730fb1e4f859532dd6`).
Current/preceding constructor Nsight data and analyzers, 655 CPU/16 GPU logs,
and selected shutdown/Worker-error results are archived at
`/tmp/issue54-step5-evidence.iNKZtw/trace-and-regressions.tar.gz` (SHA256
`603fe3da4ae4c833000c0f2df28c98031210909188ea6ff62640dac241610e84`).
These local bundles preserve reproducer dependencies; they have not been
uploaded to a remote PR. Historical baseline paths outside these bundles
remain separately listed above. The local `scheduler_async` head has no
matching PR in either configured GitHub repository at the read-only check;
no unrelated candidate PR was modified and no remote branch was pushed.

An additional bounded, CPU-only analysis aligns Decode parents by actual
token prefix, not Beam row position, and independently validates each record
before comparing. `/tmp/step5_candidate_ranking_comparison.json` has SHA256
`76d4755c9517ed6aa26ad4b294560fb946345900dad614ecb532d2f1d085195f`.
The analyzer is `/tmp/analyze_step5_candidate_ranking.py`; its three negative/
permutation tests independently pass at root (0.73 s), with log
`/tmp/step5_candidate_ranking_cpu_root.log`.

For compiled full/hit, Prefill candidates share 126/128 tokens; every shared
score differs and 89 shared-token ranks move. Final sequences share 124/128.
For uncompiled eager full/hit, Prefill candidates share 123/128 and final
sequences share 123/128. The first observed divergence is thus already in
Prefill sampler candidates/ranking, not first introduced by Decode parent
reordering. Compiled vs uncompiled full-Prefill itself differs too (118/128
Prefill candidates and 117/128 final sequences shared). These observations
must not be collapsed into one cache/GEMM root cause or strict legacy parity.
Missing candidates' scores are not inferred. The sampler/model-level cause
and desired numerical consistency remain a separate unresolved item.

Step: 5
Commit: this Step 5 commit (parent 7ab07246b39b65eba9266d7a0b304050270ac46b)
Changed files: GPU V1 output adapter, native ABI-2 wait, backend/sampling/
shutdown wiring, EngineCore terminal release handling, ownership/adapter/kernel
tests, review report
Behavior change: real fixed-slot control D2H and native completion now replace
the GR branch of upstream async output materialization; terminal GR abort
observes release failure and retires the damaged instance; healthy ordinary
branch retained
Tests run: current expanded focused 655 CPU and 16 real GPU boundary cases;
prior revision four full-model eager and four verified Graph abort/reuse cases,
eager final-Prefill and verified Graph Decode active-shutdown probes,
real-model eager prepare and Graph Beam advance Python Worker errors,
prefix-hit comparison and Graph/cache-disabled complete-session Nsight capture;
26 current smoke sessions, four current independent reference/lifecycle sessions,
46 matrix/consumer-tool + 25 cache-observer + 9 lifecycle-observer CPU checks
Test result: PASS under revised Step 5 contract; original literal legacy-hit parity
and issued-completed<=1 NOT PASS (explicit replacements, not silent waivers)
Hard counters: actual construct/add/Scheduler-free/KV-free each 1 in observed
reference sessions; stage 0/1/2 each 1; actual Decode dispatches 2;
state create/free balanced; no intermediate Beam payload D2H in product traces;
3 x 4-byte controls and one 5124-byte final result/status copy per traced session
Trace/benchmark: real D→D overlap; P→D overlap absent in this capture;
two raw traced E2E samples, no percentage or speedup claim
Known limitations: hit/miss and legacy-hit differences remain; candidate reference
is bounded, eager definition differs from fully uncompiled mode; full CI,
Step 8/9 matrix/soak/performance and NPU hardware acceptance remain outstanding;
no whole-Issue PASS, no remote PR publication claim
Exit gate: PASS (explicit team-revised Step 5 contract only)
