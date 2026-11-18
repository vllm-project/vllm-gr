# Hot-update tests

[Chinese version](hot_update_tests.zh-CN.md)

This record maps test items to their cases, grouped into new unit/component
coverage, hot-update end-to-end coverage, and existing-functionality regression.
Counts are parametrized pytest cases; standalone service scripts are separate.

Earlier broad regression (2026-11-09, before the fixes below): 78 new unit/component cases passed (77 Sidecar cases
plus the new scheduler case). The complete CI pytest selection was covered by
7,803 passing cases before a final fixture assertion failure, followed by all
7 remaining cases passing after correction: **7,810 passed, 93 skipped**, plus
166 passing subtests. This is a combined result, not one uninterrupted green run.
Three existing test fixtures/assertions were adapted to the versioned wrapper;
no production code changed during this regression. System gates are tracked below.

Focused fix regression (2026-11-10), rebased onto upstream `main@46c82f2`: **636 passed** across two
selections (282 + 354), including all 86 Sidecar cases, sliced projection, model-load lifecycle, V2 prefill
and NPU-side lifecycle fixtures (no NPU kernels), GPU constraints, V1/V2 CUDA
Graph runners, and async scheduling. The new coverage includes
partial native reload failure, source withdrawal after Sidecar restart, concurrent
reload/resume rejection, cancellation, and CPU/GPU tied-head cache refresh.

Production-service fix checks also passed for both
`tests/e2e/function/run_checkpoint_reload.py --graph --head-cache sliced --tied-weights`
(Llama) and `--graph --head-cache transposed --tied-weights` (Qwen3). Each
performed three successful native reloads, retained one cache address with stable
live GPU allocations, rejected resume/reload after failure, and matched a fresh
service after manual restart. Each recorded five prefill and ten decode graph
replays. Static pre-commit hooks passed; the full accuracy/performance/HDFS gates
were not rerun for these fixes.

Async-pause response fix (2026-11-10): **473 passed** across `tests/sidecar`,
`tests/test_grpc_entry.py`, and `tests/test_gr_async_scheduler.py`. The production
service case below also passed. This is focused coverage for the HTTP response
contract, not a new compile/Graph or performance acceptance run.

Output-head identification fix: the original `lm_head` cases passed before the
fix, while all eight `output`/alias combinations reproduced stale caches. After
type-based detection, **75 selected CPU cases passed** in
`tests/sidecar/test_worker_update.py` and `tests/test_sliced_logits.py`
(`-k "not cuda"`; 34 cases deselected). Fixtures use native `ParallelLMHead`
with TP disabled and the native layerwise reload API; no full-model or new
Graph/performance run was added.

Checkpoint-retention fix: **38 CPU cases passed** in `tests/sidecar/test_retention.py`
and `tests/sidecar/test_runtime.py`. Sixteen new cases cover the configured limit,
local publication ordering, live references, interrupted updates and cleanup errors.
`UpdateTask(max_local_checkpoints=2)` keeps at most two completed checkpoint copies
when safe. Polls and update completion remove the oldest unused directories. The
selected update and versions referenced by any configured instance are protected;
staging directories and symlinks are excluded. Unknown/invalid service state skips
cleanup, and deletion errors only log warnings. Protected copies may temporarily
exceed the limit and are reconsidered on later polls. Each Task cache has one
Sidecar owner; stop or relocate instances before removing them from configuration.
This limit does not alter Trie file retention.

Combined affected-scope regression after all three fixes: **720 passed** in
`tests/sidecar`, `tests/test_grpc_entry.py`, `tests/test_gr_async_scheduler.py`,
`tests/test_sliced_logits.py`, the V1/V2 ModelRunner patch suites, patch
initialization, and constraint lifecycle. The real in-flight Sidecar drain case
also passed again. This is targeted regression, not a new full CI, accuracy or
performance gate result.

## 1. New logic: unit and component tests

Case names below are relative to `tests/sidecar/` unless otherwise stated.
Local HTTP cases exercise actual listeners or ASGI applications; HDFS unit cases
use a fake CLI subprocess. They do not substitute for real HDFS/GPU validation.

| Feature | Case type | Test items | Corresponding cases |
| --- | --- | --- | --- |
| Checkpoint disk cache | Valid + invalid | 1. Default to two copies and reject invalid limits<br>2. Delete by local publication age even after restart and when remote mtimes differ<br>3. Clean history without a new publication. | `test_retention.py::test_checkpoint_retention_requires_a_positive_integer`; `test_checkpoint_retention_uses_publication_age_after_restart`; `test_retention_runs_without_a_new_publication` |
| Checkpoint cache reference protection | Valid + recovery | 1. Keep the old path until a single-copy update finishes<br>2. Preserve references from non-target instances<br>3. Retry deferred cleanup on a no-change poll<br>4. Retain both versions during partial rollout and reconcile after restart. | `test_retention.py::test_retention_preserves_current_path_until_single_copy_update_finishes`; `test_non_target_reference_keeps_oldest_until_a_no_change_poll`; `test_partial_rollout_retains_both_versions_until_restart_reconciles` |
| Checkpoint cleanup failure and isolation | Error + valid | 1. Skip cleanup when an instance cannot be queried or has invalid weights<br>2. Leave staging directories and symlinks untouched<br>3. Bound rejected checkpoint history<br>4. Preserve successful updates when deletion fails<br>5. Leave Trie files unchanged. | `test_retention.py::test_unreachable_instance_skips_cleanup_without_failing_update`; `test_invalid_checkpoint_state_keeps_uncommitted_downloads`; `test_cleanup_leaves_staging_and_symlinks_untouched`; `test_rejected_checkpoints_do_not_accumulate`; `test_cleanup_error_does_not_undo_a_successful_update`; `test_checkpoint_retention_does_not_apply_to_trie_tasks` |
| Default checkpoint checker | Valid + invalid | 1. Allow weight changes with identical file names/sizes<br>2. Reject changed file lists<br>3. Reject changed file sizes. | `test_checkers.py::test_checkpoint_same_files_and_sizes_allow_new_weights`; `test_checkpoint_rejects_file_list_or_size_changes` |
| Default Trie checker | Valid + invalid | 1. Allow tensor size changes with identical names/count<br>2. Reject changed tensor names or count. | `test_checkers.py::test_trie_compares_names_without_requiring_same_tensor_sizes` |
| HDFS version discovery and download | Valid + invalid | 1. Select the latest version with a completion marker<br>2. Skip unfinished versions<br>3. Download the pinned version even if a newer one appears<br>4. Reject an absent selected version without fallback<br>5. Reject an existing local destination. | `test_hdfs.py::test_pinned_download_and_ready_discovery`; `test_missing_selected_version_never_falls_back` |
| HDFS clients and version formats | Valid + invalid/error | 1. Generate client-specific hdfs get or hdfs dfs -get commands and download options<br>2. Parse timestamp/date version names<br>3. Ignore non-version names<br>4. Ignore the known wrapper warning when identifying missing markers<br>5. Still propagate permission errors. | `test_hdfs.py::test_novi_client_dialects`; `test_novi_version_names`; `test_wrapper_notice_does_not_turn_missing_marker_into_failure` |
| HDFS errors and timeouts | Error | 1. Propagate metadata errors instead of reporting no update<br>2. Terminate timed-out CLI processes. | `test_hdfs.py::test_metadata_errors_are_not_no_update`; `test_timeout_terminates_cli` |
| Rolling and synchronous updates | Valid + invalid | 1. Respect rolling concurrency and pause/reload/resume order<br>2. Perform only reload for synchronous online updates<br>3. Reject incomplete instance groups before service mutation. | `test_runtime.py::test_rolling_groups_and_reload_ordering`; `test_sync_online_only_reloads`; `test_invalid_mode_is_rejected_before_mutation` |
| Stateless version reconciliation | Valid + recovery | 1. Update only mismatched instances after restart<br>2. Detect live version drift<br>3. Use the latest version without replaying old progress<br>4. Reconcile after cancellation<br>5. Accept instance-list changes without local state migration. | `test_runtime.py::test_restart_updates_only_mismatched_instances`; `test_instance_version_drift_is_detected_without_sidecar_restart`; `test_restart_uses_latest_version_without_replaying_old_progress`; `test_cancelled_sidecar_restarts_without_persistent_progress`; `test_changed_instance_list_needs_no_local_state_migration` |
| Stateless recovery after loading | Recovery | 1. Recover from a lost reload reply using the live version<br>2. Only resume when the target version is already loaded and the service is paused. | `test_runtime.py::test_lost_reload_reply_resumes_from_live_version_without_reloading`; `test_http.py::test_loaded_paused_checkpoint_only_resumes` |
| Cross-Task instance exclusion | Valid + invalid state | 1. Share instance locks and recheck status under the lock<br>2. Block online Trie updates during a paused checkpoint update<br>3. Reject online updates to a paused service. | `test_runtime.py::test_tasks_share_instance_lock_and_recheck_status`; `test_paused_service_blocks_online_task_until_checkpoint_resumes`; `test_online_update_rejects_paused_service` |
| Update preconditions | Valid no-op + invalid/error | 1. Do not call the service after an invalid download<br>2. Do not pause/reload after checker failure<br>3. Do not mutate instances without a published version<br>4. Reject default checks without an old resource path. | `test_runtime.py::test_invalid_download_never_calls_server`; `test_custom_checker_failure_never_pauses_or_reloads`; `test_no_publication_does_not_touch_instances`; `test_default_checker_requires_an_old_resource_path` |
| Python configuration and custom checkers | Valid | 1. Load Python configuration<br>2. Pass old/new paths to synchronous and asynchronous checkers<br>3. Avoid importing inference runtime when importing Sidecar. | `test_runtime.py::test_python_config_export_and_lightweight_import`; `test_task_custom_checker_receives_old_and_new_paths`; `test_http.py::test_sidecar_import_does_not_import_inference_runtime` |
| HTTP control path | Valid + invalid/error | 1. Execute paused and online updates over local HTTP<br>2. Invoke HTTP control after a simulated HDFS download<br>3. Exercise the production AdminServer<br>4. Verify argument forwarding and error response mapping. | `test_http.py::test_paused_process_real_http`; `test_online_process_real_http`; `test_hdfs_to_http_task_end_to_end_without_progress_files`; `test_management.py::test_sidecar_calls_production_admin_listener`; `test_production_routes_forward_arguments_and_map_errors` |
| Service status | Valid + invalid | 1. Accept no resource/version query parameters<br>2. Return service state and both resource versions<br>3. Restore internal enums after HTTP/IPC decoding<br>4. Reject invalid state strings. | `test_management.py::test_status_has_one_enum_state_and_both_resource_versions`; `test_status_enum_restored_at_engine_core_ipc_boundary`; `test_production_routes_forward_arguments_and_map_errors`; `test_http.py::test_client_rejects_invalid_service_state` |
| Native checkpoint reload and version publication | Valid + error | 1. Call upstream native reload directly<br>2. Report the old version during loading<br>3. Publish the new version only after success<br>4. Do not record a successful version after failure. | `test_management.py::test_weights_endpoint_calls_original_rpc_without_core_reload_wrapper`; `test_failed_native_load_does_not_record_success`; `test_status_publishes_checkpoint_only_after_successful_reload` |
| Pause and request draining | Valid + error | 1. Return and await the upstream drain Future without additional polling<br>2. Accept synchronous and asynchronous pause success through the real AdminServer and ControlClient<br>3. Do not reload or resume before drain completion<br>4. Propagate drain failure without reloading or resuming. | `test_management.py::test_pause_returns_upstream_drain_future_without_custom_polling`; `test_sidecar_calls_production_admin_listener` (`idle`, `pending`, `failed`) |
| Trie loading and version switching | Valid + invalid/error | 1. Invoke a single Worker reload operation<br>2. Switch only after successful loading<br>3. Retain the same runner wrapper<br>4. Do not publish a new version after Worker failure<br>5. Reject unsupported execution modes before Worker RPC. | `test_management.py::test_trie_has_only_one_worker_reload_operation`; `test_trie_worker_failure_does_not_report_a_new_version`; `test_versioned_table_reload_switches_without_prepare_or_publish`; `test_runner_keeps_the_same_wrapper_across_reload`; `test_trie_rejects_unsupported_execution_before_worker_rpc` |
| Trie access and expiry | Valid + invalid | 1. Default to the latest table<br>2. Verify old/new candidate sets during retention<br>3. Release expired tables on API calls without a timer thread<br>4. Reject access to expired versions<br>5. Reject different contents under the same version ID. | `test_management.py::test_versioned_table_lazy_expiry_and_default_access`; `test_expired_table_is_released_by_api_without_session_notifications`; `test_worker_update.py::test_reload_trie_switches_then_releases_old_table_on_expiry` |
| Sliced cache and native weight reload | Valid | 1. Refresh sliced weights through native post-load processing<br>2. Reload three times for bias/no bias and inference/ordinary cache creation<br>3. Verify aligned logits<br>4. Preserve cache addresses<br>5. Exclude derived caches from checkpoint parameters. | `test_worker_update.py::test_head_refreshes_its_own_slice_during_native_weight_processing`; `test_sliced_cache_survives_native_layerwise_reload` (lm_head/output/alias × bias/no bias × inference/ordinary cache creation, three reloads each) |
| Sliced cache window changes | Valid | 1. Reuse the proxy for repeated projections<br>2. Do not rewrap the reload hook after window changes<br>3. Refresh the current window on reload<br>4. Preserve cache addresses with and without bias. | `test_worker_update.py::test_sliced_reload_hook_tracks_window_changes_without_rewrapping` (bias/no bias; 2 cases) |
| Prefix-cache clearing after drain | Valid | 1. Allow forced prefix-cache clearing after drain<br>2. Preserve unstarted queued GR requests. | `tests/test_gr_async_scheduler.py::test_prefix_reset_after_drain_preserves_waiting_gr_sessions` |
| Reload safety and manual recovery | Invalid state + cancellation | 1. Reproduce partial failure with the native layerwise loader<br>2. Keep old version but mark weights invalid<br>3. Reject resume and concurrent reload<br>4. A fresh Sidecar cannot resume invalid weights after version withdrawal<br>5. Reject checkpoint reload while running. | `test_management.py::test_partial_native_reload_remains_invalid_after_sidecar_restart_and_withdrawal`; `test_reload_in_progress_rejects_resume_and_concurrent_reload`; `test_running_service_cannot_reload_checkpoint` |
| Tied LM-head cache refresh | Valid | 1. Load through embedding or head using the native loader<br>2. Refresh sliced and GPU transposed caches across three reloads<br>3. Preserve cache storage and tied weights<br>4. Register hooks once at model load. | `test_worker_update.py::test_tied_head_caches_follow_native_layerwise_reload` |

## 2. Hot-update end-to-end tests

Recorded runs used H20, tiny random Llama/Qwen3 checkpoints, production gRPC,
a separate HTTP control listener, and multiprocess Engine Core. Requests used
batch 4, 64 prompt tokens, beam width 4, and 3 decode steps, with prefix caching
on. Checkpoints were single-file Hugging Face safetensors directories with
identical architecture/configuration and different weights.

Earlier service-script runs on 2026-11-09 passed against the status API at that
time: four checkpoint variants, eager/graph Trie reload,
eager/graph memory loops, and the real HDFS/two-service scenario.

| Test item | Corresponding script / scenario | Recorded result |
| --- | --- | --- |
| Complete a Sidecar checkpoint update while a real GR request drains; preserve the in-flight request output and verify drain → reload → resume ordering | `tests/e2e/function/run_pause_drain.py --output <directory>` | Reproduced the null-response failure before the fix; passed after the fix on H20 with multiprocess Core and tiny FP16 Llama, B4 × 64 tokens, 3 steps. Uses the actual ControlClient/PausedUpdateProcess; test-only stage delays and event probes ensure a pending native drain Future. |
| Switch checkpoint A→B; queue a request while paused during B→A; verify it waits and reproduces A | `tests/e2e/function/run_checkpoint_reload.py`: `A -> B`, `B -> A` | Passed in eager and `--graph` modes |
| Reject a missing checkpoint, retain the last committed version, reject resume/reload, and recover only after model-service restart | `tests/e2e/function/run_checkpoint_reload.py`: `failed reload stays paused and invalid until manual service restart` | Updated contract verified with tied Llama/Qwen3 and `--graph` on 2026-11-10 |
| Match hot-reloaded B against a separately started B service, including tokens and beam offsets | `tests/e2e/function/run_checkpoint_reload.py`: `reloaded B matches fresh B` | Passed in eager and graph modes |
| Refresh both GR head caches while retaining storage addresses and correct cached values | `tests/e2e/function/run_checkpoint_reload.py --graph --head-cache sliced` and `--head-cache transposed`: all four checkpoint scenarios | Passed; sliced Llama window `[0, 16)`, transposed Qwen3; allocations stable at 226.560 / 226.571 MiB |
| Switch Trie A→B→A without pause/resume; verify candidate membership and restored ordering, including after retention | `tests/e2e/function/run_trie_reload.py`: `A -> B`, `after retention expires`, `B -> A` | Passed in eager and graph modes |
| Reject an invalid Trie artifact while preserving the current version and output | `tests/e2e/function/run_trie_reload.py`: `invalid artifact rejected` | Passed in eager and graph modes |
| Detect checkpoint memory growth over repeated native reloads | `tests/e2e/perf/run_reload_memory.py --cycles 10`, with/without `--graph`: checkpoint cycles | Passed; live allocation stable at 98.360 / 226.554 MiB; reload peak 98.391 / 226.585 MiB |
| Verify old Trie memory is actually released and repeated reloads do not accumulate live allocations | `tests/e2e/perf/run_reload_memory.py --cycles 10`, with/without `--graph`: Trie cycles | Passed; old/new tables 106.861 / 235.055 MiB, after expiry 102.609 / 230.803 MiB; each release frees 4.252 MiB |
| Discover completed HDFS versions, download once, roll two services with concurrency 1, and skip unchanged versions | `tests/e2e/function/run_sidecar_hdfs.py`: `01-ready-version-and-rolling`, `02-no-change`, `10-final-no-change` | Passed using real HDFS and two graph-enabled services |
| Reject a changed checkpoint file size before pausing services | `tests/e2e/function/run_sidecar_hdfs.py`: `03-default-checker-rejection` | Passed; existing generation remained valid |
| Recover a partial rollout after a reload failure and a new Sidecar process | `tests/e2e/function/run_sidecar_hdfs.py`: `04-partial-rollout-failure`, `05-restart-reconcile` | Passed; only the remaining instance reloaded |
| Recover a resume failure without reloading the already committed checkpoint | `tests/e2e/function/run_sidecar_hdfs.py`: `06-resume-failure`, `07-restart-resume-only` | Passed; committed paused instance only resumed |
| Download and update Trie concurrently without pause/resume, then restore the previous contents | `tests/e2e/function/run_sidecar_hdfs.py`: `08-online-trie`, `09-trie-back-to-a` | Passed; candidate membership remained correct after expiry |

Graph runs enabled Inductor (`compilation.mode=3`), native FULL/PIECEWISE graphs,
and GR prefill/decode graphs. Memory runs observed 12 prefill and 24 decode graph
replays per service. Each memory test used one warm-up plus ten measured reloads;
GPU sampling was synchronized. Reserved allocator pools remained cached even
when live `allocated` memory fell, so release assertions use live allocation.

Checkpoint A differed in ranks 3/4 for one input between eager and graph modes;
checkpoint B and both Trie outputs matched across modes. Within graph mode,
A→B→A restored the original ordering. This is separate from reload correctness.

The HDFS run used two services sharing one H20. Fault injection (reload/resume
HTTP 503) and tracing lived only in the test launcher. All ten invocations completed with their expected outcomes. The no-change
invocation required one bounded retry after an HDFS connection reset; both
attempt logs were retained. Services were stopped afterward;
uploaded fixtures were retained for inspection.

Reproduce with the project's configured GPU environment. Each script requires
`--output <directory>`; HDFS also requires `--hdfs-root <writable-test-root>`.
Scripts save `result.json` and service logs; HDFS also saves its remote URI,
control timeline, and per-invocation logs.

## 3. Existing-functionality regression and CI checks

Results are recorded by run; overlapping suites must not be added together.
The current branch has not completed all CI/performance/accuracy gates.

| Regression item / scenario | Corresponding suite / command | Recorded result and scope |
| --- | --- | --- |
| Native/GR scheduling, admission, dispatch/receipts, cancellation, cleanup, pause and prefix-cache rules | `tests/test_gr_async_scheduler.py` | Passed in the complete CI selection; includes the new queued-session reset case |
| Sliced projection, cropping/alignment, bias, TP/gather, quantized fallback and native dtype behavior | `tests/test_sliced_logits.py`; `tests/sidecar/test_worker_update.py` | Passed in the complete CI selection: 89 existing projection cases and 8 Worker update cases |
| Constraint loading/publication, failure/shutdown cleanup, memory accounting and projection windows | `tests/v1/worker/test_constraint_lifecycle.py` | Passed in the complete CI selection; fixtures follow the wrapper contract |
| Real-model V1/V2 parity, completion batching, async scheduling and offline/serving lifecycle | `tests/e2e/function/test_modelrunner_v2_stage_parity_st.py`; `tests/e2e/function/test_onerec_v1_scheduler_st.py`; `tests/test_offline_beam_search.py`; `tests/test_serving_beam_search.py` | Passed with slow tests enabled; covers eager/graph, custom batches, unconstrained generation, fallback and actual graph replay |
| Existing GPU and simulated NPU lifecycle fixtures adapted to VersionedConstraintTable | `tests/kernels/test_multi_trie_beam.py`; `tests/kernels/test_v2_multibatch_graphs.py`; `tests/v1/worker/test_npu_stage_projection_lifecycle.py` | GPU files: 19 passed; CPU-based NPU lifecycle file: 11 passed. Only test setup/identity assertions changed; no NPU hardware validation |
| Formatting and static checks: SPDX, isort, Ruff, typos, Markdown, workflow/shell and mypy 3.11/3.12/3.13 | `pre-commit run --all-files --show-diff-on-failure` | All applicable static hooks passed; run-tests executed separately. External local configuration selects system Node |
| Wheel build/install smoke: Python 3.11/3.12/3.13, CPU and CUDA | `tools/run_wheel_smoke_test.sh` | All six variants passed in this run |
| Runtime version/source isolation and CUDA device mapping | `tools/ci/resolve_runtime.py verify`; `tools/ci/check_cuda_device_mapping.py` | Passed in a dedicated local environment importing this checkout; shared runtime unchanged |
| Complete repository test selection, including slow tests | `bash tools/pre_commit/run_tests.sh` (`python3 -m pytest -x --run-slow --durations=10 -s tests`) | Combined coverage: 7,810 passed, 93 skipped, 166 passing subtests. Final 7 cases were resumed after an assertion fix; not one uninterrupted green invocation |
| OneRec accuracy and Recall gates | `tests/system/run_benchmark_and_verify.sh`; `tests/system/run_beam_v1_recall.sh` | Blocked by missing sid2pid.json/sid2iid.json (HTTP 401). The regular gate exited during evaluation; Recall could not start |
| Pangu offline accuracy | `tests/system/run_pangu1b_acc_and_verify.sh` | Passed: accuracy 1.0, threshold 1.0 |
| Pangu online API and accuracy | `tests/system/run_pangu1b_online_and_verify.sh` | Startup failed: FlashAttention scheduler_metadata shape mismatch. Reproduced with the same environment on baseline 55b9f42; not unique to this branch |
| Local offline ABBA performance comparison using CI workload/thresholds | `tests/system/run_beam_v1_performance.sh`; `benchmarks/beam_search/v1_performance_ab.py` | Passed: B=1/2/4 mean changes +0.04% / -0.07% / -0.16%; all mean/p50/p90 changes below +1%. Model identity normalized in derived comparison inputs; not a GitHub event-job result |

The local performance run used the pinned OneRec revision, 1K inputs, beam 128,
3 decode steps, and B=1/2/4 in ABBA order, with 10 warm-ups and 20 measurements
per pass. The initial comparison rejected the local snapshot path as a model ID.
After verifying the pinned checkpoint blob SHA-256, only model ID/revision in
derived setup copies were canonicalized for the existing comparator. Raw setup,
timings, prompts and artifacts were retained unchanged; thresholds were not relaxed.
This run does not establish complete CI success: OneRec data gates remain blocked,
and Pangu online has a startup failure also reproduced on the baseline.

NPU, eight physical GPUs, sustained serving traffic, sharded checkpoints, and
forced termination during downloads/weight writes were not covered by the
service runs. Input serialization comparisons are outside this feature's scope.
