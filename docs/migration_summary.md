# vllm-gr → vllm-GR 迁移完成总结

> 日期: 2026-05-28
> 版本: v1.0

## 一、迁移概述

将 vllm-gr（小写，支持 OpenOneRec 模型）的核心功能迁移到 vllm-GR（大写，支持 HSTU/FuXi-alpha 模型）主仓库。迁移完成后，vllm-GR 同时支持两种模型的推理：

| 模型类型 | 启动命令 | 说明 |
|----------|----------|------|
| HSTU/FuXi-alpha | `vllm-gr serve <model> ...` | 自定义 API 服务器，走 GRAsyncLLM + adapter 路径 |
| OpenOneRec | `vllm serve <model> --trust-remote-code ...` | vLLM 原生 API 服务器，beam search patches 通过 plugin 自动加载 |

---

## 二、迁移功能模块总览

| 功能模块 | 说明 | 迁移方式 |
|----------|------|----------|
| Beam Search 引擎 | ADD_BATCH/BEAM_FORK 协议，beam search 调度 | 直接复制 |
| BeamAttentionBackend | 共享前缀 cascade attention 后端 | 直接复制 |
| BeamForkRequest 类型 | beam fork 请求类型定义 | 直接复制 |
| EngineCore 通信 | beam fork/add_batch 的 core 层通信 | 直接复制 |
| Scheduler 扩展 | beam_prefix_groups 调度支持 | 直接复制 |
| GPUModelRunner 扩展 | beam search 的 attention metadata 构建 | 直接复制 |
| FlatLogprobs 优化 | logprobs 扁平化性能优化 | 直接复制 |
| BeamSearchParams | beam search 采样参数扩展 | 直接复制 |
| Catalog (Trie) | 合法 token 约束系统 | 直接复制 |
| Offline Beam Search | LLM.beam_search 离线推理 | 直接复制 |
| Online Beam Search | OpenAI serving beam search 在线推理 | 直接复制 |
| LMCache 适配器 | LMCache connector 兼容补丁 | 直接复制 |
| C++ MinHeap 扩展 | 高效 top-K 选择 (pybind11) | 直接复制 |
| CLI 参数扩展 | --catalog-path 等 beam search 参数 | 直接复制 |
| OpenOneRec Benchmark | 完整 benchmark 套件（6 种任务类型） | 部分适配 |
| Examples (离线/在线) | beam search 推理示例 | 部分适配 |
| 单元/E2E 测试 | 9 个测试文件 + kernel 测试 | 部分适配 |
| 文档 | 12 个文档文件 | 迁移 + 更新 |

---

## 三、文件迁移详细列表

### 3.1 直接复制（字节级一致，0 修改）

以下文件从 vllm-gr 复制到 vllm-GR 后，内容完全一致，无需任何修改：

#### 核心引擎模块 (v1/)

| # | 文件路径 | 说明 |
|---|----------|------|
| 1 | `vllm_gr/v1/engine/types.py` | BeamForkRequest 类型定义 |
| 2 | `vllm_gr/v1/engine/core.py` | EngineCore ADD_BATCH/BEAM_FORK 处理，`_handle_beam_fork` |
| 3 | `vllm_gr/v1/engine/core_client.py` | AsyncMPClient 新增方法 (add_requests_async/beam_fork_async) |
| 4 | `vllm_gr/v1/engine/core_client_patch.py` | apply_batch_fork_patches |
| 5 | `vllm_gr/v1/engine/engine_core_patch.py` | 子进程 patch: BeamForkRequest 注入、枚举扩展、Scheduler/GPUModelRunner patch |
| 6 | `vllm_gr/v1/engine/async_llm.py` | AsyncLLM beam search 方法 |
| 7 | `vllm_gr/v1/engine/__init__.py` | 空文件 |
| 8 | `vllm_gr/v1/__init__.py` | 空文件 |

#### Attention 后端

| # | 文件路径 | 说明 |
|---|----------|------|
| 9 | `vllm_gr/v1/attention/backends/beam_attn.py` | BeamAttentionBackend 完整实现 |
| 10 | `vllm_gr/v1/attention/backends/__init__.py` | BeamAttentionBackend 导出 |
| 11 | `vllm_gr/v1/attention/__init__.py` | 空文件 |

#### Metrics 扩展

| # | 文件路径 | 说明 |
|---|----------|------|
| 12 | `vllm_gr/v1/metrics/stats.py` | RequestStateStats 扩展字段 |
| 13 | `vllm_gr/v1/metrics/__init__.py` | 空文件 |

#### 独立功能模块

| # | 文件路径 | 说明 |
|---|----------|------|
| 14 | `vllm_gr/logprobs.py` | FlatLogprobs 优化实现 |
| 15 | `vllm_gr/logprobs_patch.py` | patch_flat_logprobs |
| 16 | `vllm_gr/sampling_params.py` | BeamSearchParams 扩展 |
| 17 | `vllm_gr/csrc/minheap.cpp` | C++ MinHeap (pybind11) |

#### Entrypoints 模块

| # | 文件路径 | 说明 |
|---|----------|------|
| 18 | `vllm_gr/entrypoints/llm.py` | 离线 beam search (LLM.beam_search) |
| 19 | `vllm_gr/entrypoints/beam_search_patch.py` | 离线 beam search patch |
| 20 | `vllm_gr/entrypoints/openai/beam_search_patch.py` | Online beam search + BEAM_FORK patch |
| 21 | `vllm_gr/entrypoints/openai/protocol.py` | to_beam_search_params |
| 22 | `vllm_gr/entrypoints/openai/serving_engine.py` | Online beam search 实现 |
| 23 | `vllm_gr/entrypoints/openai/serving_models.py` | Catalog trie + init patch |

#### 3rd Party

| # | 文件路径 | 说明 |
|---|----------|------|
| 24 | `vllm_gr/3rd_party/lmcache/` (整个目录) | LMCache 适配器补丁 |

#### Benchmark (直接复制的部分)

| # | 文件路径 | 说明 |
|---|----------|------|
| 25 | `benchmarks/open_one_rec/__init__.py` | 包初始化 |
| 26 | `benchmarks/open_one_rec/base_evaluator.py` | 评估器基类 |
| 27 | `benchmarks/open_one_rec/one_rec_patch.py` | vLLM benchmark monkey patch |
| 28 | `benchmarks/open_one_rec/open_one_rec_dataset.py` | OneRecDataset 实现 |
| 29 | `benchmarks/open_one_rec/open_one_rec_loader.py` | 数据加载器 |
| 30 | `benchmarks/open_one_rec/qwen3_soft_switch.jinja2` | Jinja2 模板 |
| 31 | `benchmarks/open_one_rec/tools/convert_catalog.py` | Catalog 转换工具 |
| 32 | `benchmarks/open_one_rec/tools/create_task_catalog.py` | Catalog 创建工具 |
| 33 | `benchmarks/open_one_rec/tasks/item_understand/` (4 文件) | item_understand 任务 |
| 34 | `benchmarks/open_one_rec/tasks/label_pred/` (4 文件) | label_pred 任务 |
| 35 | `benchmarks/open_one_rec/tasks/rec_reason/` (4 文件) | rec_reason 任务 |
| 36 | `benchmarks/open_one_rec/tasks/recommendation/` (5 文件) | recommendation 任务 |

#### Examples (直接复制的部分)

| # | 文件路径 | 说明 |
|---|----------|------|
| 37 | `examples/online_serving/beam_search/client.sh` | 在线 beam search 客户端脚本 |
| 38 | `examples/online_serving/beam_search/README.md` | 在线示例文档 |
| 39 | `examples/online_serving/catalog/verify_catalog.py` | Catalog 验证脚本 |
| 40 | `examples/online_serving/catalog/README.md` | Catalog 示例文档 |
| 41 | `examples/offline_inference/beam_search/README.md` | 离线示例文档 |

#### Tests (直接复制的部分)

| # | 文件路径 | 说明 |
|---|----------|------|
| 42 | `tests/test_catalog.py` | Catalog 单元测试 |
| 43 | `tests/test_dp_beam_search_unit.py` | DP beam search 单元测试 |
| 44 | `tests/test_task_specific_catalog_program.py` | Catalog 程序测试 |
| 45 | `tests/kernels/attention/test_cascade_beam_flash_attn.py` | Beam attention kernel 测试 |
| 46 | `tests/resources/single_one_rec_prompt.txt` | 测试资源文件 |

**小计: 46+ 个文件，字节级完全一致。**

---

### 3.2 适配迁移（有少量修改）

以下文件从 vllm-gr 复制到 vllm-GR 后，做了 API 适配修改：

| # | 文件路径 | 修改内容 | 修改行数 |
|---|----------|----------|----------|
| 1 | `benchmarks/open_one_rec/one_rec_main.py` | `from vllm_gr.patch import run_patch` → `re_apply_patches` | 2 行 |
| 2 | `benchmarks/open_one_rec/one_rec_acc_test.py` | `from vllm_gr.patch import run_patch` → `re_apply_patches` | 2 行 |
| 3 | `examples/offline_inference/beam_search/offline_beam_search.py` | `import vllm_gr.init` → `from vllm_gr.patch import re_apply_patches; re_apply_patches()` | 3 行 |
| 4 | `tests/test_batch_and_fork_unit.py` | mock patch 路径更新 + `run_patch()` → `_apply_gr_beam_patches(force=True)` | 5 行 |
| 5 | `tests/test_serving_beam_search.py` | 适配 vllm-GR 测试框架 | 少量 |
| 6 | `tests/test_offline_beam_search.py` | 适配 vllm-GR 测试框架 | 少量 |
| 7 | `tests/test_serving_catalog.py` | 适配 vllm-GR 测试框架 | 少量 |

**适配模式统一为**: 将 vllm-gr 的 `run_patch()` / `import vllm_gr.init` 替换为 vllm-GR 的 `re_apply_patches()`，统一由 plugin 入口管理。

---

### 3.3 重命名迁移

| # | 源文件 (vllm-gr) | 目标文件 (vllm-GR) | 说明 |
|---|---|---|---|
| 1 | `vllm_gr/engine/arg_utils.py` | `vllm_gr/engine/arg_utils_gr.py` | 与 vllm-GR 的 arg_utils.py 功能不冲突，重命名避免覆盖 |

---

### 3.4 新建文件

| # | 文件路径 | 说明 |
|---|----------|------|
| 1 | `vllm_gr/entrypoints/openai/__init__.py` | 空文件，包初始化 |
| 2 | `tests/conftest.py` | 测试配置 (`--loops`, `--run-slow`) |

---

### 3.5 修改的 vllm-GR 现有文件

| # | 文件路径 | 修改内容 |
|---|----------|----------|
| 1 | `vllm_gr/patch.py` | 新增 `_apply_gr_beam_patches()` 函数，在 `_apply_patches()` 末尾调用，合并 7+1 个 beam search patches |
| 2 | `vllm_gr/__init__.py` | 新增 `BeamAttentionBackend` 延迟导入 |
| 3 | `vllm_gr/entrypoints/cli/main.py` | 新增 `_maybe_launch_beam_search_serve()` 处理 beam search 模式 |
| 4 | `setup.py` | 新增 C++ minheap_cpp 扩展构建配置 (pybind11) |
| 5 | `pyproject.toml` | 新增 pybind11/numpy 构建依赖，pandas>=3.0.0，scikit-learn，pyarrow，lmcache 可选依赖 |

---

### 3.6 文档迁移

从 vllm-gr 迁移到 vllm-GR 的文档：

| # | 文件路径 | 迁移方式 |
|---|----------|----------|
| 1 | `docs/DESIGN.md` | 直接复制 |
| 2 | `docs/PRD.md` | 直接复制 |
| 3 | `docs/PRE_COMMIT.md` | 直接复制 |
| 4 | `docs/assets/vLLM-GR.png` | 直接复制 |
| 5 | `docs/benchmarks/README.md` | 复制 + 更新（移除 `--gr`） |
| 6 | `docs/benchmarks/accuracy.md` | 复制 + 更新（移除 `--gr`） |
| 7 | `docs/benchmarks/throughput.md` | 复制 + 更新（移除 `--gr`、移除 `--gr` 参数表行） |
| 8 | `docs/data_parallelism.md` | 复制 + 更新（移除 `--gr`） |
| 9 | `docs/lmcache.md` | 复制 + 更新（移除 `--gr`） |
| 10 | `docs/quickstart-ascend.md` | 直接复制 |
| 11 | `docs/quickstart.md` | 复制 + 更新（移除 `--gr`，更新引用） |
| 12 | `docs/tools/README.md` | 直接复制 |

新增文档：

| # | 文件路径 | 说明 |
|---|----------|------|
| 13 | `docs/migration_plan.md` | 迁移方案（中文版） |
| 14 | `docs/migration_plan_en.md` | 迁移方案（英文版） |
| 15 | `docs/migration_summary.md` | 本文件：迁移完成总结 |

---

### 3.7 不迁移的文件

以下文件为占位符、已合并或非核心文件，未迁移：

| 文件 | 原因 |
|------|------|
| `vllm_gr/core/scheduler.py` | 占位符，实际 patch 在 engine_core_patch.py |
| `vllm_gr/core/kv_cache_manager.py` | 纯占位符 |
| `vllm_gr/core/__init__.py` | 仅导出占位符类 |
| `vllm_gr/worker/__init__.py` | 纯占位符 |
| `vllm_gr/model_executor/__init__.py` | 纯占位符 |
| `vllm_gr/model_executor/models/__init__.py` | 纯占位符 |
| `vllm_gr/sample/__init__.py` | 纯占位符 |
| `vllm_gr/utils/__init__.py` | 纯占位符 |
| `vllm_gr/init.py` | 自动 patch 机制，改用 plugin 统一管理 |
| `vllm_gr/patch.py` (vllm-gr版) | 功能已合并到 vllm-GR 的 patch.py |
| `vllm_gr/__init__.py` (vllm-gr版) | 功能已合并到 vllm-GR 的 __init__.py |
| `vllm_gr/entrypoints/__init__.py` (vllm-gr版) | 空文件，vllm-GR 已有自己的 |
| `vllm_gr/entrypoints/cli/main.py` (vllm-gr版) | 功能合并到 vllm-GR 的 CLI |
| `collect_env.py` | 环境收集，非核心 |
| `.github/` | CI 配置，后续单独处理 |

---

## 四、Patch 机制合并说明

### 迁移前（两套独立 patch）

| | vllm-GR | vllm-gr |
|---|---------|---------|
| 激活方式 | `vllm_gr.plugin:register` (general_plugins) | `__init__.py` 导入时自动 patch + `VLLM_GR_AUTOPATCH` 环境变量 |
| 入口函数 | `_apply_patches()` / `re_apply_patches()` | `run_patch()` |
| patch 对象 | 核心数据类型 (EngineCoreRequest, Request, NewRequestData) | 核心行为方法 (Scheduler.schedule, GPUModelRunner, EngineCoreProc) |

### 迁移后（统一入口）

所有 patch 通过 `vllm_gr.plugin:register` 自动加载，无需 `--gr` 标志或环境变量：

```
_apply_patches(force):
    _apply_gr_core_patches(force)        # vllm-GR 原有: 核心类型替换
    _apply_yuanrong_npu_patch(force)     # vllm-GR 原有: NPU 补丁
    _apply_kv_merge_patch(force)         # vllm-GR 原有: KV merge 补丁
    _apply_gr_beam_patches(force)        # [新增] vllm-gr: beam search 全部 patches
        ├─ patch_flat_logprobs()
        ├─ patch_beam_search()
        ├─ patch_sampling()
        ├─ patch_add_cli_args()
        ├─ patch_OpenAIServingModels_init()
        ├─ patch_offline_beam_search()
        ├─ patch_batch_and_fork()
        └─ _patch_lmcache_adapter() [可选]
```

**关键兼容性**: 两套 patch 完全不重叠（数据类型 vs 行为方法），可安全共存。子进程级 patch 通过 `run_engine_core()` 包装器在 EngineCore 子进程启动时自动应用。

---

## 五、测试验证结果

### 5.1 单元测试

| 测试文件 | 测试内容 | 结果 |
|----------|----------|------|
| `tests/test_catalog.py` | Catalog Trie 构建与查询 | 通过 |
| `tests/test_batch_and_fork_unit.py` | ADD_BATCH/BEAM_FORK 协议单元测试 | 通过 |
| `tests/test_dp_beam_search_unit.py` | DP beam search 单元测试 | 通过 |
| `tests/test_task_specific_catalog_program.py` | Catalog 程序化构建测试 | 通过 |

### 5.2 Kernel 测试

| 测试文件 | 测试内容 | 结果 |
|----------|----------|------|
| `tests/kernels/attention/test_cascade_beam_flash_attn.py` | Beam cascade attention kernel 正确性 | 通过 (990 个测试用例) |

### 5.3 GPU 端到端测试

| 测试文件 | 测试内容 | 结果 |
|----------|----------|------|
| `tests/test_serving_beam_search.py` | Online beam search API 服务端测试 | 通过 (2 个场景) |
| `tests/test_offline_beam_search.py` | Offline beam search 推理测试 | 通过 (2 个场景) |
| `tests/test_serving_catalog.py` | Catalog 集成测试 (需 GPU) | 通过 (1 个场景) |

### 5.4 示例验证

| 示例 | 测试内容 | 结果 |
|------|----------|------|
| `examples/offline_inference/beam_search/` | 离线 beam search 推理 | 通过 (Vanilla vLLM: 1.04s, Patched vllm-GR: 0.54s) |
| `examples/online_serving/beam_search/` | 在线 beam search API | 通过 (128 beam results via API) |

### 5.5 Benchmark 验证

| 测试内容 | 结果 |
|----------|------|
| `label_pred` 任务离线推理 | 通过 |
| `product` 任务离线推理 | 通过 |
| `label_pred` 任务在线推理 | 通过 |
| `product` 任务在线推理 | 通过 |

### 5.6 HSTU/FuXi 回归测试

| 测试内容 | 结果 |
|----------|------|
| vllm-GR 原有 358 个单元测试 | 全部通过 |
| GPU smoke test | 通过 |

### 5.7 测试总计

```
总测试数: 1354
通过: 1354
失败: 0
```

---

## 六、迁移统计

| 分类 | 数量 |
|------|------|
| 字节级直接复制文件 | 46+ |
| 适配迁移文件 | 7 |
| 重命名文件 | 1 |
| 新建文件 | 2 |
| 修改的现有文件 | 5 |
| 迁移文档 | 12 |
| 新增文档 | 3 |
| 不迁移文件 | 15 |
| **测试通过** | **1354 / 1354** |
