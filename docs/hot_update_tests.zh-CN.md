# 热更测试

[English version](hot_update_tests.md)

本文按测试项列出对应 case，分为新增逻辑 UT/组件测试、热更端到端测试、已有功能回归三部分。
pytest 数量包含参数化用例；独立服务测试脚本单独记录。

此前的完整回归（2026-11-09，下述修复之前）：新增 UT/组件测试 78 项通过（77 项 Sidecar，加 1 项调度器用例）。
完整 CI pytest 选择先执行到 7,803 项通过，在最后一组 fixture 断言处失败；
修正后，剩余 7 项全部通过。合并结果为 **7,810 项通过、93 项跳过**，另有 166 个 subtest 通过。
这是分段合并结果，不是一次无中断全绿运行。
本轮仅适配了三个已有测试 fixture/断言的版本化 wrapper 契约，没有修改生产代码。
系统门禁结果见第三部分。

2026-11-10 修复专项回归，基于主仓 `main@46c82f2`：两组共 **636 项通过**（282 + 354），
包含全部 86 项 Sidecar 用例，以及切片投影、模型加载生命周期、V2 prefill 和 NPU 侧生命周期夹具（不执行 NPU 内核），以及 GPU 约束、V1/V2 CUDA Graph runner 和异步调度器。
新增覆盖原生 reload 部分失败、Sidecar 重启后源版本撤回、并发 reload/resume 拒绝、取消和 CPU/GPU tied-head 缓存刷新。

真实服务修复验证也已通过：
`tests/st/run_checkpoint_reload.py --graph --head-cache sliced --tied-weights`（Llama）
和 `--graph --head-cache transposed --tied-weights`（Qwen3）。
每组完成三次原生 reload，缓存地址不变、GPU 活跃分配稳定；失败后拒绝 resume/reload，
人工重启服务后输出与新加载的服务一致。每组记录到 5 次 prefill 和 10 次 decode graph replay。
静态 pre-commit 检查通过；本轮未重跑完整精度、性能及 HDFS 门禁。

2026-11-10 异步 pause 响应修复：`tests/sidecar`、`tests/test_grpc_entry.py` 和
`tests/test_gr_async_scheduler.py` 共 **473 项通过**，下表中的真实服务用例也通过。
本轮只验证 HTTP 响应契约，不作为新增 compile/Graph 或性能门禁的验收。

输出头识别修复：修改前原 `lm_head` 用例通过，`output`/别名的八种组合均复现旧缓存问题。
改为按类型识别后，`tests/sidecar/test_worker_update.py` 和 `tests/test_sliced_logits.py`
的 **75 项 CPU 用例通过**（`-k "not cuda"`，34 项未选择）。夹具使用关闭 TP 的原生
`ParallelLMHead` 和原生逐层 reload API；未新增完整模型、Graph 或性能回归。

checkpoint 本地保留修复：`tests/sidecar/test_retention.py` 和 `tests/sidecar/test_runtime.py`
共 **38 项 CPU 用例通过**，其中新增 16 项。`UpdateTask(max_local_checkpoints=2)` 默认保留两份，
按本地下载完成时间优先删除最旧的闲置目录，轮询和更新结束时执行清理。保护本次更新版本及配置中
所有实例仍引用的版本，不删除临时下载目录和符号链接；状态未知或权重无效时跳过清理。删除失败只记录警告，
不回滚已经成功的更新。受保护版本可能暂时超限，后续轮询重试。每个缓存目录只能由一个 Sidecar 管理，
实例从配置中移除前需先停服或迁出引用。本配置不改变 Trie 的磁盘文件保留行为。

三个修复完成后的影响范围回归：**720 项通过**，覆盖 `tests/sidecar`、gRPC、GR 调度、
切片 logits、V1/V2 ModelRunner patch、patch 初始化及约束表生命周期；真实请求执行中的
Sidecar 排空用例也再次通过。本次是定向回归，不代表重跑了完整 CI、精度或性能门禁。

## 1. 新增逻辑：UT 和组件测试

除特别注明外，case 路径相对于 `tests/sidecar/`。
本地 HTTP 用例使用真实监听器或 ASGI 应用；HDFS UT 使用模拟 CLI 子进程，
不能替代真实 HDFS/GPU 验证。

| 功能点 | 用例类型 | 具体测试项 | 对应 case |
| --- | --- | --- | --- |
| checkpoint 磁盘缓存 | 合法 + 非法 | 1. 默认两份，拒绝非法上限<br>2. 按本地下载完成时间删除，重启后仍有效，不受远端目录时间影响<br>3. 没有新版本时也清理历史。 | `test_retention.py::test_checkpoint_retention_requires_a_positive_integer`; `test_checkpoint_retention_uses_publication_age_after_restart`; `test_retention_runs_without_a_new_publication` |
| checkpoint 引用保护 | 合法 + 恢复 | 1. 单份保留时，更新完成前保留旧路径<br>2. 保护非本次更新目标的实例引用<br>3. 无变化轮询重试延期清理<br>4. 部分滚动更新失败时保留新旧版本，重启恢复后再清理。 | `test_retention.py::test_retention_preserves_current_path_until_single_copy_update_finishes`; `test_non_target_reference_keeps_oldest_until_a_no_change_poll`; `test_partial_rollout_retains_both_versions_until_restart_reconciles` |
| checkpoint 清理异常与隔离 | 异常 + 合法 | 1. 实例不可访问或权重无效时跳过清理<br>2. 不删除临时下载目录和符号链接<br>3. 被 checker 拒绝的历史版本不持续积累<br>4. 删除失败不撤销成功更新<br>5. 不清理 Trie 文件。 | `test_retention.py::test_unreachable_instance_skips_cleanup_without_failing_update`; `test_invalid_checkpoint_state_keeps_uncommitted_downloads`; `test_cleanup_leaves_staging_and_symlinks_untouched`; `test_rejected_checkpoints_do_not_accumulate`; `test_cleanup_error_does_not_undo_a_successful_update`; `test_checkpoint_retention_does_not_apply_to_trie_tasks` |
| 默认 checker：checkpoint | 合法 + 非法 | 1. 文件名、大小一致时允许权重内容变化<br>2. 文件列表变化时拒绝<br>3. 文件大小变化时拒绝。 | `test_checkers.py::test_checkpoint_same_files_and_sizes_allow_new_weights`; `test_checkpoint_rejects_file_list_or_size_changes` |
| 默认 checker：Trie | 合法 + 非法 | 1. tensor 名称和数量一致时允许大小变化<br>2. tensor 名称或数量不一致时拒绝。 | `test_checkers.py::test_trie_compares_names_without_requiring_same_tensor_sizes` |
| HDFS：版本发现和下载 | 合法 + 非法 | 1. 选择带完成标记的最新版本<br>2. 跳过未完成版本<br>3. 下载锁定的版本，不随新发布版本变化<br>4. 指定版本不存在时拒绝，不回退到其他版本<br>5. 本地目标已存在时拒绝覆盖。 | `test_hdfs.py::test_pinned_download_and_ready_discovery`; `test_missing_selected_version_never_falls_back` |
| HDFS：客户端和版本格式 | 合法 + 非法/异常 | 1. 按客户端生成 hdfs get 或 hdfs dfs -get 命令及下载参数<br>2. 解析时间戳和日期格式的版本名<br>3. 忽略非版本名称<br>4. 固定 wrapper warning 不影响缺失完成标记的判断<br>5. 权限错误仍抛出异常。 | `test_hdfs.py::test_novi_client_dialects`; `test_novi_version_names`; `test_wrapper_notice_does_not_turn_missing_marker_into_failure` |
| HDFS：错误和超时 | 异常 | 1. 元数据访问失败时抛出错误，不当成没有更新<br>2. 命令超时后终止子进程。 | `test_hdfs.py::test_metadata_errors_are_not_no_update`; `test_timeout_terminates_cli` |
| 更新方式：Rolling / Sync | 合法 + 非法 | 1. 滚动更新遵守配置的并发数量和 pause/reload/resume 顺序<br>2. 同步在线更新只执行 reload<br>3. 遗漏目标实例的分组在修改服务前被拒绝。 | `test_runtime.py::test_rolling_groups_and_reload_ordering`; `test_sync_online_only_reloads`; `test_invalid_mode_is_rejected_before_mutation` |
| 无状态恢复：版本对齐 | 合法 + 异常恢复 | 1. 重启后只更新版本不一致的实例<br>2. 发现运行期间的实例版本漂移<br>3. 重启后使用最新版本，不重放旧进度<br>4. 任务取消后可重新对齐<br>5. 实例列表变化不需要迁移本地状态。 | `test_runtime.py::test_restart_updates_only_mismatched_instances`; `test_instance_version_drift_is_detected_without_sidecar_restart`; `test_restart_uses_latest_version_without_replaying_old_progress`; `test_cancelled_sidecar_restarts_without_persistent_progress`; `test_changed_instance_list_needs_no_local_state_migration` |
| 无状态恢复：已加载但未恢复 | 异常恢复 | 1. reload 响应丢失后依据服务实际版本恢复<br>2. 目标版本已加载且服务暂停时只执行 resume，不重复 reload。 | `test_runtime.py::test_lost_reload_reply_resumes_from_live_version_without_reloading`; `test_http.py::test_loaded_paused_checkpoint_only_resumes` |
| 并发控制：跨 Task 互斥 | 合法 + 非法状态 | 1. 多个 Task 共享实例锁，取得锁后重新检查状态<br>2. checkpoint 更新暂停期间阻止在线 Trie 更新<br>3. 在线更新遇到暂停服务时拒绝执行。 | `test_runtime.py::test_tasks_share_instance_lock_and_recheck_status`; `test_paused_service_blocks_online_task_until_checkpoint_resumes`; `test_online_update_rejects_paused_service` |
| 更新前置检查 | 合法无更新 + 非法/异常 | 1. 下载无效时不调用服务<br>2. 自定义 checker 失败时不 pause/reload<br>3. 没有已发布版本时不修改实例<br>4. 缺少旧资源路径时默认 checker 拒绝更新。 | `test_runtime.py::test_invalid_download_never_calls_server`; `test_custom_checker_failure_never_pauses_or_reloads`; `test_no_publication_does_not_touch_instances`; `test_default_checker_requires_an_old_resource_path` |
| Python 配置和 checker 扩展 | 合法 | 1. 加载 Python 配置<br>2. 同步和异步 checker 均收到新旧资源路径<br>3. Sidecar 导入不加载推理运行时。 | `test_runtime.py::test_python_config_export_and_lightweight_import`; `test_task_custom_checker_receives_old_and_new_paths`; `test_http.py::test_sidecar_import_does_not_import_inference_runtime` |
| 管控 HTTP 链路 | 合法 + 非法/异常 | 1. 通过本地 HTTP 执行暂停式和在线更新<br>2. 模拟 HDFS 下载后调用 HTTP 管控接口<br>3. 通过正式 AdminServer 完成调用<br>4. 验证路由参数转发和错误响应映射。 | `test_http.py::test_paused_process_real_http`; `test_online_process_real_http`; `test_hdfs_to_http_task_end_to_end_without_progress_files`; `test_management.py::test_sidecar_calls_production_admin_listener`; `test_production_routes_forward_arguments_and_map_errors` |
| 服务状态查询 | 合法 + 非法 | 1. status 无资源或版本查询参数<br>2. 一次返回服务状态及两个资源版本<br>3. HTTP/IPC 接收后恢复内部枚举类型<br>4. 拒绝非法状态字符串。 | `test_management.py::test_status_has_one_enum_state_and_both_resource_versions`; `test_status_enum_restored_at_engine_core_ipc_boundary`; `test_production_routes_forward_arguments_and_map_errors`; `test_http.py::test_client_rejects_invalid_service_state` |
| checkpoint：原生 reload 和版本发布 | 合法 + 异常 | 1. 直接调用上游原生 reload RPC<br>2. 加载期间仍返回旧版本<br>3. 成功后才发布新版本<br>4. 加载失败不记录成功版本。 | `test_management.py::test_weights_endpoint_calls_original_rpc_without_core_reload_wrapper`; `test_failed_native_load_does_not_record_success`; `test_status_publishes_checkpoint_only_after_successful_reload` |
| pause：请求排空 | 合法 + 错误 | 1. 返回并等待上游原生 drain Future，不额外轮询<br>2. 真实 AdminServer 和 ControlClient 均接受同步、异步成功响应<br>3. 排空完成前不调用 reload/resume<br>4. 排空失败正常报错，不执行 reload/resume。 | `test_management.py::test_pause_returns_upstream_drain_future_without_custom_polling`；`test_sidecar_calls_production_admin_listener`（`idle`、`pending`、`failed`） |
| Trie：加载和版本切换 | 合法 + 非法/异常 | 1. 只调用一次 Worker reload<br>2. 新表加载成功后切换版本<br>3. runner 始终持有同一个 wrapper<br>4. Worker 失败后不发布新版本<br>5. 不支持的执行方式在 Worker RPC 前被拒绝。 | `test_management.py::test_trie_has_only_one_worker_reload_operation`; `test_trie_worker_failure_does_not_report_a_new_version`; `test_versioned_table_reload_switches_without_prepare_or_publish`; `test_runner_keeps_the_same_wrapper_across_reload`; `test_trie_rejects_unsupported_execution_before_worker_rpc` |
| Trie：版本访问和过期释放 | 合法 + 非法 | 1. 默认访问最新表<br>2. 保留期内新旧表各自返回预期候选<br>3. API 调用触发过期旧表释放，无计时线程<br>4. 过期版本不可再访问<br>5. 同一版本号对应不同内容时拒绝更新。 | `test_management.py::test_versioned_table_lazy_expiry_and_default_access`; `test_expired_table_is_released_by_api_without_session_notifications`; `test_worker_update.py::test_reload_trie_switches_then_releases_old_table_on_expiry` |
| 切片缓存：原生权重 reload | 合法 | 1. 原生 post-load 刷新切片权重<br>2. bias/无 bias、inference/普通模式四种组合各 reload 三次<br>3. 对齐后的 logits 正确<br>4. 缓存地址不变<br>5. 派生缓存不进入 checkpoint 参数树。 | `test_worker_update.py::test_head_refreshes_its_own_slice_during_native_weight_processing`; `test_sliced_cache_survives_native_layerwise_reload` (lm_head/output/alias × bias/no bias × inference/ordinary cache creation, three reloads each) |
| 切片缓存：窗口切换 | 合法 | 1. 重复计算复用 proxy<br>2. 切换窗口后不重复注册 reload hook<br>3. reload 刷新当前窗口<br>4. 有/无 bias 两种情况均保持缓存地址。 | `test_worker_update.py::test_sliced_reload_hook_tracks_window_changes_without_rewrapping` (bias/no bias; 2 cases) |
| prefix cache：排空后的清理 | 合法 | 1. 排空后允许强制清理 prefix cache<br>2. 尚未执行的排队 GR 请求保留。 | `tests/test_gr_async_scheduler.py::test_prefix_reset_after_drain_preserves_waiting_gr_sessions` |
| reload 安全与人工恢复 | 非法状态 + 取消 | 1. 原生逐层加载器复现部分加载失败<br>2. 保留旧版本但标记权重无效<br>3. 拒绝 resume 和并发 reload<br>4. 新 Sidecar 在版本撤回后也不能恢复无效权重<br>5. running 时拒绝 checkpoint reload。 | `test_management.py::test_partial_native_reload_remains_invalid_after_sidecar_restart_and_withdrawal`; `test_reload_in_progress_rejects_resume_and_concurrent_reload`; `test_running_service_cannot_reload_checkpoint` |
| tied LM-head 缓存刷新 | 合法 | 1. 原生加载器通过 embedding 或 head 加载<br>2. 连续三次更新切片及 GPU 转置缓存<br>3. 缓存地址和共享权重保持不变<br>4. 模型加载后仅注册一次钩子。 | `test_worker_update.py::test_tied_head_caches_follow_native_layerwise_reload` |

## 2. 热更端到端测试

已记录的测试使用 H20、随机初始化的小型 Llama/Qwen3 checkpoint、正式 gRPC 服务、
独立 HTTP 管控监听器和多进程 Engine Core。请求为 batch 4、64 个输入 token、
beam width 4、decode 3 步，开启 prefix caching。
checkpoint 为单个权重文件的 Hugging Face safetensors 目录，架构和配置一致，仅权重不同。

本节所有服务脚本均已于 2026-11-09 按当前 status API 重跑并通过：
四种 checkpoint 组合、eager/graph Trie reload、eager/graph 显存循环，
以及真实 HDFS/双服务场景。

| 测试项 | 对应脚本 / 场景 | 已记录结果 |
| --- | --- | --- |
| 请求正在执行时，通过真实 Sidecar 客户端完成 checkpoint 更新；原请求保持旧权重输出，并验证排空 → reload → resume 顺序 | `tests/st/run_pause_drain.py --output <directory>` | 修复前复现 null 响应错误，修复后通过。H20、multiprocess Core、tiny FP16 Llama，B4 × 64 tokens、3 步生成。使用真实 ControlClient/PausedUpdateProcess；仅在测试启动器中延迟 stage 并记录事件，确认原生 drain Future 尚未完成。 |
| checkpoint A→B 后输出变化；B→A 时暂停期间提交的请求等待 resume，之后复现 A 的输出 | `tests/st/run_checkpoint_reload.py`: `A -> B`, `B -> A` | eager 和 `--graph` 模式均通过 |
| checkpoint 路径错误时保持暂停及无效，不发布失败版本；拒绝 resume 和再次 reload，重启模型服务后恢复 | `tests/st/run_checkpoint_reload.py`: `failed reload stays paused and invalid until manual service restart` | 新约定于 2026-11-10 在 tied Llama/Qwen3 + `--graph` 下验证 |
| 热更到 B 的服务与直接加载 B 启动的独立服务，输出 token 和 beam offsets 一致 | `tests/st/run_checkpoint_reload.py`: `reloaded B matches fresh B` | eager 和 graph 模式均通过 |
| sliced / transposed 两种 GR head 缓存优化下，reload 后缓存值正确、地址不变 | `tests/st/run_checkpoint_reload.py --graph --head-cache sliced` 和 `--head-cache transposed`：全部四个 checkpoint 场景 | 通过；sliced Llama 窗口 `[0, 16)`，transposed Qwen3；显存分配稳定在 226.560 / 226.571 MiB |
| Trie 不经过 pause/resume 完成 A→B→A；候选属于当前 Trie，恢复 A 后顺序一致，旧表过期后仍正常 | `tests/st/run_trie_reload.py`: `A -> B`, `after retention expires`, `B -> A` | eager 和 graph 模式均通过 |
| 非法 Trie 文件更新失败，当前版本和输出保持不变 | `tests/st/run_trie_reload.py`: `invalid artifact rejected` | eager 和 graph 模式均通过 |
| 连续多次原生 checkpoint reload 后，GPU 活跃显存不持续增长 | `tests/st/run_reload_memory.py --cycles 10`, 启用/不启用 `--graph`：checkpoint 循环 | 通过；活跃分配稳定在 98.360 / 226.554 MiB；reload 峰值为 98.391 / 226.585 MiB |
| 旧 Trie 过期后释放对应 GPU 显存，多次 reload 不累积活跃分配 | `tests/st/run_reload_memory.py --cycles 10`, 启用/不启用 `--graph`：Trie 循环 | 通过；新旧表共存时为 106.861 / 235.055 MiB，过期后为 102.609 / 230.803 MiB；每次释放 4.252 MiB |
| 发现 HDFS 已完成版本、下载一次、两个服务按并发 1 滚动更新；版本不变时跳过更新 | `tests/st/run_sidecar_hdfs.py`: `01-ready-version-and-rolling`, `02-no-change`, `10-final-no-change` | 真实 HDFS 和两个启用 graph 的服务验证通过 |
| checkpoint 文件大小变化时，在 pause 前拒绝更新 | `tests/st/run_sidecar_hdfs.py`: `03-default-checker-rejection` | 通过；原有推理仍正常 |
| 部分实例 reload 失败后，启动新 Sidecar 继续滚动更新 | `tests/st/run_sidecar_hdfs.py`: `04-partial-rollout-failure`, `05-restart-reconcile` | 通过；仅剩余未完成实例执行 reload |
| resume 失败后恢复，不重复 reload 已加载的目标 checkpoint | `tests/st/run_sidecar_hdfs.py`: `06-resume-failure`, `07-restart-resume-only` | 通过；目标版本已提交且暂停的实例仅执行 resume |
| 下载并同时在线更新 Trie，不调用 pause/resume，再恢复之前的内容 | `tests/st/run_sidecar_hdfs.py`: `08-online-trie`, `09-trie-back-to-a` | 通过；旧表过期后候选仍属于当前 Trie |

graph 测试开启 Inductor（`compilation.mode=3`）、原生 FULL/PIECEWISE graph，
以及 GR prefill/decode graph。显存测试观察到每个服务 12 次 prefill 和 24 次 decode graph replay。
每项显存测试包含一次预热更新和十次测量更新，采样前同步 GPU。
释放检查使用活跃 `allocated` 显存；即使活跃显存降低，allocator 的 reserved 缓存池仍可能保留。

跨 eager/graph 比较时，checkpoint A 有一个输入的第 3/4 名顺序不同；
checkpoint B 和两套 Trie 输出跨模式一致。graph 模式内部 A→B→A 恢复原始顺序。
跨模式排名差异与 reload 正确性分开记录。

HDFS 测试由两个服务共享一张 H20。reload/resume HTTP 503 故障注入和调用跟踪仅存在于测试启动器。
十轮执行均符合预期；“无变化”场景因 HDFS connection reset 进行了一次有界重试，
两次执行日志均已保留。测试结束后已停止服务，上传的 fixture 保留供检查。

复现时使用项目已配置的 GPU 环境。每个脚本都要求 `--output <directory>`；
HDFS 脚本还要求 `--hdfs-root <writable-test-root>`。
脚本保存 `result.json` 和服务日志；HDFS 脚本还记录远端 URI、管控时间线和每轮执行日志。

## 3. 已有功能回归和 CI 检查

按执行轮次记录结果，重叠测试集的数量不能相加。
当前分支尚未完成全部 CI、性能和准确率门禁。

| 回归项 / 场景 | 对应测试集 / 命令 | 已记录结果及范围 |
| --- | --- | --- |
| Native/GR 调度、准入、派发与回执、取消、清理、pause 和 prefix cache 规则 | `tests/test_gr_async_scheduler.py` | 完整 CI 选择中通过，包含新增排队请求 reset 用例 |
| 切片投影、裁剪与对齐、bias、TP/gather、量化回退和原生 dtype 行为 | `tests/test_sliced_logits.py`; `tests/sidecar/test_worker_update.py` | 完整 CI 选择中通过：89 项已有投影用例和 8 项 Worker 更新用例 |
| constraint 加载和发布、失败及退出清理、显存统计和投影窗口 | `tests/v1/worker/test_constraint_lifecycle.py` | 完整 CI 选择中通过，fixture 遵循 wrapper 契约 |
| 真模型 V1/V2 对齐、completion batch、异步调度及离线/服务生命周期 | `tests/st/test_modelrunner_v2_stage_parity_st.py`; `tests/st/test_onerec_v1_scheduler_st.py`; `tests/test_offline_beam_search.py`; `tests/test_serving_beam_search.py` | 开启 slow tests 后通过；覆盖 eager/graph、自定义 batch、无约束生成、回退和实际 graph replay |
| 已有 GPU 和模拟 NPU 生命周期 fixture 适配 VersionedConstraintTable | `tests/kernels/test_multi_trie_beam.py`; `tests/kernels/test_v2_multibatch_graphs.py`; `tests/v1/worker/test_npu_stage_projection_lifecycle.py` | 两个 GPU 文件 19 项通过；CPU 模拟 NPU 生命周期文件 11 项通过。仅修改测试构造或对象身份断言，不代表 NPU 硬件验证 |
| 格式与静态检查：SPDX、isort、Ruff、typos、Markdown、workflow/shell 和 mypy 3.11/3.12/3.13 | `pre-commit run --all-files --show-diff-on-failure` | 全部适用静态 hook 通过；run-tests 单独执行。仓库外的本地配置选择系统 Node |
| Python 3.11/3.12/3.13 × CPU/CUDA 的 wheel 构建和安装 | `tools/run_wheel_smoke_test.sh` | 本轮六种组合全部通过 |
| 运行时版本、源码隔离和 CUDA 设备映射 | `tools/ci/resolve_runtime.py verify`; `tools/ci/check_cuda_device_mapping.py` | 专用本地环境验证通过，导入本 checkout；共享运行时未修改 |
| 仓库完整测试选择，包括 slow tests | `bash tools/pre_commit/run_tests.sh` (`python3 -m pytest -x --run-slow --durations=10 -s tests`) | 合并覆盖：7,810 项通过、93 项跳过、166 个 subtest 通过。最后 7 项在断言修正后补跑，不是一次无中断全绿执行 |
| OneRec 准确率和 Recall 门禁 | `tests/system/run_benchmark_and_verify.sh`; `tests/system/run_beam_v1_recall.sh` | 因缺少 sid2pid.json/sid2iid.json（HTTP 401）受阻。普通门禁在评估阶段退出，Recall 无法启动 |
| Pangu 离线准确率 | `tests/system/run_pangu1b_acc_and_verify.sh` | 通过：准确率 1.0，阈值 1.0 |
| Pangu 在线 API 和准确率 | `tests/system/run_pangu1b_online_and_verify.sh` | 启动失败：FlashAttention scheduler_metadata 形状不匹配。同环境下基线 55b9f42 复现相同错误，并非本分支独有 |
| 按 CI 负载和阈值执行的本地离线 ABBA 性能对比 | `tests/system/run_beam_v1_performance.sh`; `benchmarks/beam_search/v1_performance_ab.py` | 通过：B=1/2/4 均值变化 +0.04% / -0.07% / -0.16%；mean/p50/p90 均未超过 +1%。仅在派生对照输入中规范化模型身份，不是 GitHub event job 的执行结果 |

性能负载为固定 OneRec revision、1K 输入、beam 128、decode 3 步、B=1/2/4，
ABBA 四轮每轮预热 10 次、测量 20 次。首次汇总因模型参数是本地 snapshot 路径而被身份检查拒绝；
核验固定 revision 的 checkpoint blob SHA-256 后，仅在派生 setup 副本中规范化模型名和 revision，
再使用原有比较器。原始 setup、时延、prompt、artifact 均未修改，阈值未放宽。
因此本轮不能声明完整 CI 全绿：OneRec 数据门禁受阻，Pangu online 存在基线同样复现的失败。

服务测试未覆盖 NPU、八张物理 GPU、持续服务流量、分片 checkpoint，
以及下载或权重写入期间的强制终止。输入序列化性能对比不属于本功能的测试范围。
