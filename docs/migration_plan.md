# vllm-gr → vllm-GR 迁移适配方案

> 版本: v2.0 | 日期: 2026-05-28
> v2.0 更新: 迁移完成后回溯更新，反映实际代码状态，补充遗漏的 API 适配说明和在线服务使用方式

## 一、背景

- **vllm-GR** (大写, `/home/t00464839/merge/vllm-GR`): 基于 vLLM v0.14.1 的生成式推荐推理引擎，支持 HSTU/FuXi-alpha 模型
- **vllm-gr** (小写, `/home/t00464839/merge/vllm-gr`): 基于 vLLM v0.14.* 的推荐推理框架，支持 OpenOneRec (Qwen 架构) 模型

**目标**: 将 vllm-gr 的核心功能迁移到 vllm-GR 主仓，使合并后的代码仓同时支持 HSTU/FuXi-alpha 和 OpenOneRec 模型推理。

## 二、整体迁移策略

**原则**: 以 vllm-GR（大写）为主仓，在其 `vllm_gr` 包中增量添加 vllm-gr（小写）的核心功能。两套 monkey-patch 机制合并为统一的 plugin+patch 入口。独立功能（如 beam search、logprobs 优化）尽量保持原有代码逻辑和命名不变。

**核心冲突点**: 两个仓库都用了 `vllm_gr` 作为包名，都依赖 `vllm==0.14.*`，但 patch 机制完全不同：
- vllm-GR：plugin.py (vLLM general_plugins) + patch.py (核心类型替换)
- vllm-gr：init.py 自动 patch + patch.py (beam search/catalog/logprobs)

## 三、两个代码仓架构对比

### 3.1 vllm-GR 架构特点

| 特性 | 说明 |
|------|------|
| 架构模式 | 适配器模式 + Monkey-Patch + 子类化 |
| 核心类型替换 | EngineCoreRequest→GREngineCoreRequest, Request→GRRequest, NewRequestData→GRNewRequestData |
| 模型支持 | HSTU (DLRM v3) + FuXi-alpha |
| 推理路径 | 评分/Pooling 路径，不生成 token，使用合成 prompt_token_ids |
| 注意力后端 | HSTUAttentionBackend (prefix/delta/full 三模式) |
| 数据流 | GRPrompt → msgpack → gr_features → 旁路传输 → 特征解码 → 模型前向 |
| 激活机制 | vLLM general_plugins 入口点 (`vllm_gr.plugin:register`) |
| CLI 入口 | 独立 CLI 解析器，注册为 `vllm-gr` 命令 |
| Worker | GRWorker (继承 GPUWorker) + GRModelRunner |
| NPU 支持 | 支持 GPU + NPU 双平台，Yuanrong KV Pool 后端 |

### 3.2 vllm-gr 架构特点

| 特性 | 说明 |
|------|------|
| 架构模式 | Monkey-Patch 插件 |
| 核心功能 | BeamAttentionBackend + ADD_BATCH/BEAM_FORK 协议 |
| 模型支持 | OpenOneRec/OneRec-1.7B (Qwen 架构)，无自定义模型实现 |
| 推理路径 | Beam Search (在线 + 离线)，token 生成路径 |
| 注意力后端 | BeamAttentionBackend (共享前缀 cascade attention) |
| 数据流 | 标准 vLLM prompt → tokenizer → beam search → Catalog 约束过滤 |
| 激活机制 | `__init__.py` 导入时自动 patch + `VLLM_GR_AUTOPATCH` 环境变量 |
| CLI 入口 | 包装原生 vLLM CLI，通过 `--gr` 标志触发 patch |
| 在线服务启动方式 | 迁移后 OpenOneRec 模型使用 `vllm serve`（不需要 `--gr`），所有 patch 通过 `vllm.general_plugins` 自动加载 |
| Catalog | Trie 树合法 token 约束，确保只生成合法推荐结果 |
| C++ 扩展 | minheap_cpp (pybind11)，高效 top-K 选择 |

### 3.3 两套 Patch 机制对比

#### vllm-GR patch.py

激活方式: 通过 vLLM 插件入口点 `vllm_gr.plugin:register` (在 pyproject.toml 中注册)

核心函数: `_apply_patches()` / `re_apply_patches()`

patch 内容 (3 类):
1. `_apply_gr_core_patches()` — 替换 EngineCoreRequest/Request/NewRequestData 为 GR 变体（通过 sys.modules 全局替换）
2. `_apply_yuanrong_npu_patch()` — NPU 平台上的 Yuanrong 后端注入
3. `_apply_kv_merge_patch()` — K+V 合并 KV cache 支持

幂等性: 使用 `_PATCHES_APPLIED` / `force` 参数保证幂等

验证机制: `verify_patches()` 检查关键消费者模块是否已正确 patch

#### vllm-gr patch.py

激活方式: 在 `__init__.py` 中通过 `init.py` 自动调用 `run_patch()` (带 `VLLM_GR_AUTOPATCH` 环境变量控制)

核心函数: `run_patch()`

patch 内容 (7+1 类):
1. `patch_flat_logprobs()` — FlatLogprobs 性能优化
2. `patch_beam_search()` — OpenAI serving beam search 替换
3. `patch_sampling()` — ChatCompletionRequest.to_beam_search_params 替换
4. `patch_add_cli_args()` — EngineArgs CLI 参数扩展（catalog_path）
5. `patch_OpenAIServingModels_init()` — OpenAIServingModels 初始化扩展（Catalog trie）
6. `patch_offline_beam_search()` — LLM.beam_search 替换
7. `patch_batch_and_fork()` — ADD_BATCH + BEAM_FORK 引擎管线补丁
8. 可选: `_patch_lmcache_adapter()` — LMCache 适配器补丁

**结论: 两套 patch 机制完全不重叠，可以安全合并。**

**实际合并结果**: vllm-gr 的 `run_patch()` 不直接迁移，其功能被分解到 `_apply_gr_beam_patches()` 中统一调用。vllm-gr 的 `init.py` 自动 patch 机制不迁移，改由 vllm-GR 的 `plugin.py` 统一管理。所有 patch 通过 `vllm.general_plugins` 入口点自动加载，无需手动调用或环境变量控制。

## 四、v1/ 目录兼容性分析（关键问题）

### 4.1 问题背景

vllm-GR 本身就基于 vLLM v1 引擎开发，其核心 EngineCore 基于 vLLM 的 v1 版本。vllm-gr 的 `vllm_gr/v1/` 目录也包含对 vLLM v1 引擎组件的 monkey-patch。核心问题是：**两者都操作 vLLM v1 引擎，vllm-gr 的 v1/ patch 是否还能直接复制到 vllm-GR 中？**

### 4.2 两者的 Patch 层面分析

vllm-GR 和 vllm-gr 虽然都基于 vLLM v1 引擎，但它们 patch 的层面**完全不重叠**：

| 维度 | vllm-GR | vllm-gr |
|------|---------|---------|
| patch 方式 | `sys.modules` 全局替换核心**数据类型** | monkey-patch 替换核心**行为方法** |
| 替换对象 | `EngineCoreRequest→GREngineCoreRequest`<br>`Request→GRRequest`<br>`NewRequestData→GRNewRequestData` | `Scheduler.schedule`<br>`GPUModelRunner.execute_model`<br>`EngineCoreProc.run_engine_core`<br>`EngineCoreProc.process_input_sockets` |
| 目的 | 让 GR 特征 (gr_features) 通过 vLLM 管线传输 | 让 beam search 的 BEAM_FORK/ADD_BATCH 通过引擎管线传输 |

**核心结论：数据类型替换和行为方法替换互不干扰，两套 patch 可以安全共存。**

### 4.3 v1/ 逐文件兼容性分析

#### 4.3.1 `v1/engine/engine_core_patch.py` — 需要适配（不能直接复制）

这是最关键的文件，包含 6 个 patch 函数，逐一分析：

| patch 函数 | 冲突分析 | 结论 |
|------------|----------|------|
| `apply_engine_core_child_patches()` 注入 `BeamForkRequest` | 注入全新类型，不与 vllm-GR 的 `GREngineCoreRequest`/`GRRequest` 冲突 | 可直接使用 |
| `_add_enum_member()` 注入 `ADD_BATCH`/`BEAM_FORK` | 扩展枚举，vllm-GR 替换的是数据类不是枚举 | 可直接使用 |
| `make_patched_engine_core_init()` 包装 `EngineCore.__init__` | vllm-GR 不 patch `EngineCore.__init__` | 可直接使用 |
| `run_engine_core()` 替换 `EngineCoreProc.run_engine_core` | **关键**: vllm-GR 当前没有子进程 patch 机制，需要集成到 plugin 初始化流程 | 需要适配 |
| `apply_scheduler_patch()` 替换 `Scheduler.schedule` | vllm-GR 不 patch Scheduler，其 `GRNewRequestData` 通过全局类型替换自动生效 | 可直接使用 |
| `apply_worker_patches()` 替换 `GPUModelRunner` 方法 | **关键**: vllm-GR 有 `GRModelRunner(GPUModelRunner)` 子类 | 见下方详细分析 |

**关于 `apply_worker_patches()` 与 `GRModelRunner` 的兼容性：**

vllm-GR 的 `GRModelRunner` 继承自 `GPUModelRunner`，重写了 `_update_states`, `_prepare_inputs`, `_preprocess`, `_pool` 等方法，但**没有重写** `execute_model` 和 `_build_attention_metadata`。vllm-gr 的 `apply_worker_patches()` 将 beam search 相关逻辑 monkey-patch 到 `GPUModelRunner` 上。由于 Python MRO（方法解析顺序），`GRModelRunner` 实例在调用 `execute_model` 和 `_build_attention_metadata` 时会使用 patch 到父类上的版本。**这实际上是兼容的** — beam_prefix_groups 的逻辑会在 GRModelRunner 的继承链中正确执行。

**关于 `run_engine_core()` 子进程入口：**

**实际实现**: `patch_batch_and_fork()` 在主进程中将 `EngineCoreProc.run_engine_core` 替换为 vllm-gr 的包装器。该包装器在子进程启动时调用 `apply_engine_core_child_patches()`，应用 Scheduler/GPUModelRunner 等子进程级 patch。子进程中的 patch 分两层：
1. **主进程 patch**（通过 `vllm.general_plugins` 在所有进程自动加载）: `OpenAIServing.beam_search`, `AsyncMPClient` 新方法, `EngineCoreProc.run_engine_core` 包装等
2. **子进程级 patch**（由 `run_engine_core` 包装器在 EngineCore 子进程启动时调用 `apply_engine_core_child_patches()` 应用）: `ADD_BATCH/BEAM_FORK` 枚举、`Scheduler.schedule` beam_prefix_groups、`GPUModelRunner._build_attention_metadata` 等

**文件本身的核心逻辑不需要修改**，实际验证已确认此机制在 vllm-GR 中正常工作。

#### 4.3.2 `v1/engine/core.py` — 可以直接复制

定义了 `_cache_beam_request`, `_handle_beam_fork`, `process_input_sockets` 三个函数。

**兼容性亮点**: `_handle_beam_fork` 中有 `from vllm.v1.request import Request`。在 vllm-GR 的子进程中，`patch.py` 的全局替换已经将 `Request` 替换为 `GRRequest`。因此 beam fork 创建的子请求会自动是 `GRRequest` 实例，带有 `gr_features` 字段。**这不会引起问题，反而是正确的行为** — beam search 的子请求也能携带 GR 特征。

#### 4.3.3 其他 v1/ 文件 — 全部可以直接复制

| 文件 | 兼容性分析 |
|------|-----------|
| `v1/engine/types.py` | 纯新增类型 `BeamForkRequest`，不与任何现有类型冲突 |
| `v1/engine/core_client.py` | 纯新增方法 `add_requests_async`/`beam_fork_async`，不修改已有行为 |
| `v1/engine/core_client_patch.py` | 注入到 `AsyncMPClient`。vllm-GR 的 `GRAsyncLLM(AsyncLLM)` 继承自 `AsyncLLM`，monkey-patch 到 `AsyncLLM` 上的方法对子类可见 |
| `v1/engine/async_llm.py` | 定义 beam search 方法被绑定到 `AsyncLLM`，与 `GREngineCoreRequest` 兼容（子类关系） |
| `v1/attention/backends/beam_attn.py` | 完整的独立 attention backend，注册为 CUSTOM 类型，与 HSTUAttentionBackend 互不干扰 |
| `v1/metrics/stats.py` | 纯新增字段，不修改已有字段 |

### 4.4 v1/ 迁移结论（仅针对 v1/ 目录下的文件）

| 结论 | 文件数 | 说明 |
|------|--------|------|
| **可以直接复制** | 7 个 | types.py, core.py, core_client.py, core_client_patch.py, async_llm.py, beam_attn.py, stats.py |
| **需要适配集成** | 1 个 | engine_core_patch.py — 核心逻辑不改，需确保在 plugin 初始化流程中正确调用 |

## 五、Monkey-patch 机制合并方案

### 5.1 合并策略

1. **保留 vllm-GR 的 plugin.py 作为统一入口**（通过 vLLM general_plugins），在其 `initialize_runtime()` 中同时调用两套 patch
2. **修改 vllm-GR 的 patch.py**，在现有 `_apply_patches()` 末尾追加调用 vllm-gr 的 patch 函数
3. **不迁移 vllm-gr 的 init.py**（自动 import 时 patch 的机制），改用 vllm-GR 的 plugin 统一管理
4. **不迁移 vllm-gr 的 patch.py**（功能已合并到 vllm-GR 的 patch.py 中）

### 5.2 修改后的 patch.py 流程

```
_apply_patches(force):
    _apply_gr_core_patches(force)        # vllm-GR: 核心类型替换
    _apply_yuanrong_npu_patch(force)     # vllm-GR: NPU Yuanrong 补丁
    _apply_kv_merge_patch(force)         # vllm-GR: KV merge 补丁
    _apply_gr_beam_patches(force)        # [新增] vllm-gr: beam search 相关 patches
```

其中 `_apply_gr_beam_patches()` 内部依次调用:
```
patch_flat_logprobs()                   # logprobs 优化
patch_beam_search()                     # online beam search
patch_sampling()                        # BeamSearchParams
patch_add_cli_args()                    # CLI --catalog-path
patch_OpenAIServingModels_init()        # Catalog
patch_offline_beam_search()             # offline beam search
patch_batch_and_fork()                  # ADD_BATCH/BEAM_FORK
_patch_lmcache_adapter()               # [可选] LMCache 兼容（仅当 lmcache 已安装时）
```
共 7 个必要 patch + 1 个可选 patch，与第三节中 vllm-gr 原始 patch 列表一致。

## 六、文件级别迁移清单

### 6.1 直接复制的文件（不需要修改，vllm-GR 中不存在，无冲突）

| 源文件 (vllm-gr) | 目标文件 (vllm-GR) | 说明 |
|---|---|---|
| `vllm_gr/v1/attention/backends/beam_attn.py` | 同路径 | BeamAttentionBackend, 独立模块 |
| `vllm_gr/v1/attention/backends/__init__.py` | 同路径 | BeamAttentionBackend 导出 |
| `vllm_gr/v1/engine/types.py` | 同路径 | BeamForkRequest |
| `vllm_gr/v1/engine/core.py` | 同路径 | EngineCore ADD_BATCH/BEAM_FORK，`_handle_beam_fork` 中 import Request 会自动得到 GRRequest |
| `vllm_gr/v1/engine/core_client.py` | 同路径 | AsyncMPClient patch，新增方法对 GRAsyncLLM 子类可见 |
| `vllm_gr/v1/engine/core_client_patch.py` | 同路径 | apply_batch_fork_patches |
| `vllm_gr/v1/engine/async_llm.py` | 同路径 | AsyncLLM beam methods，与 GREngineCoreRequest 兼容 |
| `vllm_gr/v1/engine/__init__.py` | 同路径 | 空文件 |
| `vllm_gr/v1/__init__.py` | 同路径 | 空文件 |
| `vllm_gr/v1/metrics/stats.py` | 同路径 | RequestStateStats 扩展，纯新增字段 |
| `vllm_gr/v1/metrics/__init__.py` | 同路径 | 空文件 |
| `vllm_gr/v1/attention/__init__.py` | 同路径 | 空文件 |
| `vllm_gr/logprobs.py` | 同路径 | FlatLogprobs 优化 |
| `vllm_gr/logprobs_patch.py` | 同路径 | patch_flat_logprobs |
| `vllm_gr/sampling_params.py` | 同路径 | BeamSearchParams 扩展 |
| `vllm_gr/entrypoints/llm.py` | 同路径 | 离线 beam search (LLM.beam_search) |
| `vllm_gr/entrypoints/beam_search_patch.py` | 同路径 | 离线 beam search patch |
| `vllm_gr/entrypoints/openai/beam_search_patch.py` | 同路径 | online beam search + BEAM_FORK patch |
| `vllm_gr/entrypoints/openai/protocol.py` | 同路径 | to_beam_search_params |
| `vllm_gr/entrypoints/openai/serving_engine.py` | 同路径 | online beam search 实现 |
| `vllm_gr/entrypoints/openai/serving_models.py` | 同路径 | Catalog + init patch |
| `vllm_gr/entrypoints/openai/__init__.py` | 同路径 | 空文件 |
| `vllm_gr/csrc/minheap.cpp` | 同路径 | C++ MinHeap 扩展 |
| `vllm_gr/3rd_party/lmcache/` (整个目录) | 同路径 | LMCache patch |

### 6.2 需要适配的文件（代码逻辑不改，但需集成到 vllm-GR 的初始化流程）

#### 6.2.1 `vllm_gr/v1/engine/engine_core_patch.py`

**核心逻辑不需要修改**，但需要注意以下集成点：

1. **`apply_engine_core_child_patches()`**: 在子进程中被 `run_engine_core()` 调用，注入 `BeamForkRequest` 和枚举成员。需确保在 vllm-GR 的子进程 patch 流程中被调用。
2. **`run_engine_core()`**: 替换 `EngineCoreProc.run_engine_core`。vllm-GR 当前没有子进程 patch 机制，需要通过 `patch_batch_and_fork()` 在主进程中替换入口函数。
3. **`apply_worker_patches()`**: monkey-patch `GPUModelRunner.execute_model` 和 `_build_attention_metadata`。由于 `GRModelRunner` 没有重写这两个方法，patch 到父类上的版本会通过 MRO 正确传播到子类。
4. **`apply_scheduler_patch()`**: 包装 `Scheduler.schedule`，与 vllm-GR 不冲突（vllm-GR 不 patch Scheduler）。

**集成方式**: 通过 `_apply_gr_beam_patches()` 中的 `patch_batch_and_fork()` 间接调用，在 vllm-GR 的 plugin 初始化流程中执行。

### 6.3 需要修改/合并的文件

#### 6.3.1 `vllm_gr/patch.py` (vllm-GR 现有文件)

**修改内容**: 在现有 `_apply_patches()` 函数末尾追加 beam search patches 调用。

新增 `_apply_gr_beam_patches()` 函数，内部依次调用 7 个必要的 beam search 相关 patch 函数 + 1 个可选的 lmcache patch。

#### 6.3.2 `vllm_gr/__init__.py` (vllm-GR 现有文件)

**修改内容**: 在现有的 `__getattr__` 延迟导入中增加 `BeamAttentionBackend` 的条件导出。

```python
def __getattr__(name):
    if name == "BeamAttentionBackend":
        from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend
        return BeamAttentionBackend
    # ... 现有逻辑保持不变
```

#### 6.3.3 `vllm_gr/engine/arg_utils.py` (vllm-gr 版重命名为 `arg_utils_gr.py`)

**冲突说明**: 两方都有实质内容:
- vllm-GR: `GRCommonEngineArgsMixin`, `GREngineArgs`, `GRAsyncEngineArgs` (dataclass，继承自 vLLM EngineArgs)
- vllm-gr: `ModelConfigGr`, `patch_add_cli_args()` (monkey patch 函数)

**解决**: 将 vllm-gr 的 `arg_utils.py` 重命名为 `arg_utils_gr.py` 放入 vllm-GR，保留 vllm-GR 原有文件不变。`patch_add_cli_args()` 在统一 patch 流程中被调用。

#### 6.3.4 `vllm_gr/entrypoints/cli/main.py` (vllm-GR 现有文件)

**冲突说明**: 两方 CLI 方案不同:
- vllm-GR: 独立 CLI 解析器，使用 FlexibleArgumentParser + 子命令模式
- vllm-gr: 包装原生 vLLM CLI，通过 `--gr` 标志触发 patch

**实际解决**: 保留 vllm-GR 的独立 CLI 方案。`main()` 中新增 `_maybe_launch_beam_search_serve()` 处理 `--gr` 标志（检测 LMCache 配置并设置环境变量，然后移除 `--gr` 避免解析错误）。但实际运行时，**所有 beam search patches 已通过 `vllm.general_plugins` 机制自动加载**，不需要 `--gr` 标志触发。

**在线服务使用方式**:
- HSTU/FuXi 模型: `vllm-gr serve <model> ...`（使用 vllm-GR 自定义 API 服务器，走 GRAsyncLLM + adapter 路径）
- OpenOneRec 模型: `vllm serve <model> --trust-remote-code ...`（使用 vLLM 原生 API 服务器，beam search patches 通过 plugin 自动生效）

#### 6.3.5 `setup.py` (vllm-GR 现有文件)

**修改内容**: 现有内容保持不变，增加 C++ 扩展构建配置 (minheap_cpp)。

新增逻辑:
- import pybind11, numpy
- Extension("minheap_cpp", ["vllm_gr/csrc/minheap.cpp"], ...)
- extra_compile_args=["-std=c++17", "-O3"]

#### 6.3.6 `pyproject.toml` (vllm-GR 现有文件)

修改内容:
- `build-system.requires`: 增加 `"pybind11"`, `"numpy"` (为 C++ 扩展)
- `dependencies`:
  - `pandas` 版本从 `>=2.0.0` 改为 `>=3.0.0` (vllm-gr 要求)
  - 增加 `scikit-learn>=1.8.0`, `pyarrow>=23.0.0` (vllm-gr benchmark 依赖)
- `optional-dependencies`: 增加 `lmcache = ["lmcache==0.4.1"]`

### 6.4 不迁移的文件（vllm-gr 占位符）

| 文件 | 原因 |
|---|---|
| `vllm_gr/core/scheduler.py` | 占位符，实际 scheduler patch 在 engine_core_patch.py |
| `vllm_gr/core/kv_cache_manager.py` | 纯占位符 |
| `vllm_gr/core/__init__.py` | 仅导出占位符类 |
| `vllm_gr/worker/__init__.py` | 纯占位符 |
| `vllm_gr/model_executor/__init__.py` | 纯占位符 |
| `vllm_gr/model_executor/models/__init__.py` | 纯占位符 |
| `vllm_gr/sample/__init__.py` | 纯占位符 |
| `vllm_gr/utils/__init__.py` | 纯占位符 |
| `vllm_gr/engine/__init__.py` (vllm-gr版) | 空文件，vllm-GR 已有自己的 |
| `vllm_gr/attention/__init__.py` (vllm-gr版) | 空文件，vllm-GR 已有自己的 |
| `vllm_gr/init.py` | 自动 patch 机制，改用 plugin 统一管理 |
| `vllm_gr/patch.py` (vllm-gr版) | 功能已合并到 vllm-GR 的 patch.py |
| `vllm_gr/__init__.py` (vllm-gr版) | 功能已合并到 vllm-GR 的 __init__.py |
| `vllm_gr/entrypoints/__init__.py` (vllm-gr版) | 空文件，vllm-GR 已有自己的 |
| `vllm_gr/entrypoints/cli/main.py` (vllm-gr版) | 功能合并到 vllm-GR 的 CLI 中 |
| `collect_env.py` | 环境收集，非核心 |
| `tools/` | pre-commit hooks 等开发工具，暂不迁移 |
| `.github/` | CI 配置，后续单独处理 |
| `.pre-commit-config.yaml` 等配置文件 | 开发工具配置，暂不迁移 |

### 6.5 Benchmark 迁移

整个 `benchmarks/open_one_rec/` 目录复制到 vllm-GR，其中 **2 个文件需要 API 适配**。

包含:
- `one_rec_main.py` — benchmark CLI 入口 **[需适配]**
- `one_rec_patch.py` — vLLM benchmark monkey patch
- `open_one_rec_dataset.py` — OneRecDataset 实现
- `open_one_rec_loader.py` — 数据加载器
- `base_evaluator.py` — 评估器基类
- `tasks/` — 各任务配置和评估器 (video, product, ad, label_cond 等)
- `tools/` — catalog 转换和创建工具
- `one_rec_acc_test.py` — 精度测试 **[需适配]**

Benchmark 中的 import 路径 `benchmarks.open_one_rec.xxx` 保持不变（在项目根目录下，不在 vllm_gr 包内）。

**API 适配**:
- `one_rec_main.py`: `from vllm_gr.patch import run_patch` → `from vllm_gr.patch import re_apply_patches`
- `one_rec_main.py`: `run_patch()` → `re_apply_patches()`
- `one_rec_acc_test.py`: 同上

### 6.6 Examples 迁移

| 源 (vllm-gr) | 目标 (vllm-GR) | 说明 |
|---|---|---|
| `examples/offline_inference/beam_search/offline_beam_search.py` | 同路径 | **[需适配]** `import vllm_gr.init` → `from vllm_gr.patch import re_apply_patches; re_apply_patches()` |
| `examples/online_serving/beam_search/client.sh` | 同路径 | 直接复制，无需修改 |
| `examples/online_serving/catalog/verify_catalog.py` | 同路径 | 直接复制，无需修改 |

### 6.7 测试用例迁移

| 源 (vllm-gr) | 目标 (vllm-GR) | 说明 |
|---|---|---|
| `tests/conftest.py` | `tests/conftest.py` | 新建（vllm-GR 无此文件），定义 `--loops`, `--run-slow` |
| `tests/test_catalog.py` | 同路径 | Catalog 单元测试 |
| `tests/test_serving_catalog.py` | 同路径 | Catalog 集成测试 (需 GPU) |
| `tests/test_batch_and_fork_unit.py` | 同路径 | ADD_BATCH/BEAM_FORK 单元测试 |
| `tests/test_dp_beam_search_unit.py` | 同路径 | DP beam search 单元测试 |
| `tests/test_serving_beam_search.py` | 同路径 | Online beam search 测试 (需 GPU) |
| `tests/test_offline_beam_search.py` | 同路径 | Offline beam search 测试 (需 GPU) |
| `tests/test_task_specific_catalog_program.py` | 同路径 | Catalog 程序测试 |
| `tests/kernels/attention/test_cascade_beam_flash_attn.py` | 同路径 | Beam attention kernel 测试 (需 GPU) |
| `tests/resources/single_one_rec_prompt.txt` | 同路径 | 测试资源文件 |

## 七、迁移后 vllm-GR 新增目录/文件结构

```
vllm-GR/
├── vllm_gr/
│   ├── __init__.py              # 修改: 增加 BeamAttentionBackend 条件导入
│   ├── patch.py                 # 修改: 合并 beam search patches
│   ├── plugin.py                # 不变 (patch.py 内部已合并)
│   ├── logprobs.py              # 新增 (from vllm-gr)
│   ├── logprobs_patch.py        # 新增 (from vllm-gr)
│   ├── sampling_params.py       # 新增 (from vllm-gr)
│   ├── csrc/
│   │   └── minheap.cpp          # 新增 (from vllm-gr)
│   ├── engine/
│   │   ├── arg_utils_gr.py      # 新增 (from vllm-gr engine/arg_utils.py)
│   │   └── ...                  # 现有文件不变
│   ├── entrypoints/
│   │   ├── llm.py               # 新增 (from vllm-gr)
│   │   ├── beam_search_patch.py # 新增 (from vllm-gr)
│   │   ├── cli/
│   │   │   └── main.py          # 修改: 增加 beam search 模式支持
│   │   └── openai/
│   │       ├── __init__.py          # 新增 (from vllm-gr)
│   │       ├── beam_search_patch.py # 新增 (from vllm-gr)
│   │       ├── protocol.py          # 新增 (from vllm-gr)
│   │       ├── serving_engine.py    # 新增 (from vllm-gr)
│   │       └── serving_models.py    # 新增 (from vllm-gr)
│   ├── v1/                          # 整个目录新增 (from vllm-gr)
│   │   ├── __init__.py              # 新增
│   │   ├── attention/
│   │   │   ├── __init__.py          # 新增
│   │   │   └── backends/
│   │   │       ├── __init__.py      # 新增
│   │   │       └── beam_attn.py     # 新增
│   │   ├── engine/
│   │   │   ├── __init__.py              # 新增
│   │   │   ├── core.py                  # 新增 (直接复制)
│   │   │   ├── core_client.py           # 新增 (直接复制)
│   │   │   ├── core_client_patch.py     # 新增 (直接复制)
│   │   │   ├── engine_core_patch.py     # 新增 (需适配集成)
│   │   │   ├── async_llm.py             # 新增 (直接复制)
│   │   │   └── types.py                 # 新增 (直接复制)
│   │   └── metrics/
│   │       ├── __init__.py          # 新增
│   │       └── stats.py             # 新增
│   └── 3rd_party/
│       └── lmcache/...           # 新增 (from vllm-gr)
├── benchmarks/
│   └── open_one_rec/             # 整个目录新增 (from vllm-gr)
├── tests/
│   ├── conftest.py               # 新建
│   ├── test_catalog.py           # 新增
│   ├── test_serving_catalog.py   # 新增
│   ├── test_batch_and_fork_unit.py  # 新增
│   ├── test_dp_beam_search_unit.py  # 新增
│   ├── test_serving_beam_search.py  # 新增
│   ├── test_offline_beam_search.py  # 新增
│   ├── test_task_specific_catalog_program.py  # 新增
│   ├── kernels/attention/        # 新增
│   └── resources/                # 新增 (部分)
├── setup.py                      # 修改: 增加 C++ 扩展
└── pyproject.toml                # 修改: 增加依赖和构建要求
```

## 八、关键冲突点和解决方案

### 冲突 1: 两套 patch.py

| 项目 | 说明 |
|------|------|
| 问题 | 两套 patch 功能完全不同，不能直接覆盖 |
| 方案 | 合并到 vllm-GR 的 patch.py，新增 `_apply_gr_beam_patches()` 函数 |
| 风险 | 低 — 两套 patch 目标完全不重叠 |

### 冲突 2: engine_core_patch.py 子进程 patch 与 vllm-GR 初始化流程

| 项目 | 说明 |
|------|------|
| 问题 | vllm-gr 的 `run_engine_core()` 替换 `EngineCoreProc.run_engine_core`，在子进程中重新应用补丁。vllm-GR 当前没有子进程 patch 机制 |
| 方案 | 通过 `patch_batch_and_fork()` 在 `_apply_gr_beam_patches()` 中调用，在主进程中替换 `EngineCoreProc.run_engine_core`。子进程启动时该包装器自动调用 `apply_engine_core_child_patches()` |
| 风险 | ~~中~~ **低** — 已验证子进程中的 patch 执行正确，所有 UT 和 E2E 测试通过 |

### 冲突 3: GPUModelRunner monkey-patch vs GRModelRunner 继承

| 项目 | 说明 |
|------|------|
| 问题 | vllm-gr 的 `apply_worker_patches()` monkey-patch `GPUModelRunner` 的 `execute_model` 和 `_build_attention_metadata`，而 vllm-GR 有 `GRModelRunner(GPUModelRunner)` 子类 |
| 方案 | **天然兼容** — GRModelRunner 没有重写这两个方法，Python MRO 使 patch 到父类上的版本正确传播到子类。beam_prefix_groups 的 ContextVar 机制在 GRModelRunner 的继承链中正确执行 |
| 风险 | 无 — 已验证兼容性 |

### 冲突 4: 两套 CLI main.py

| 项目 | 说明 |
|------|------|
| 问题 | vllm-GR 有独立 CLI（`vllm-gr` 命令），vllm-gr 包装原生 vLLM CLI |
| 方案 | 保留 vllm-GR 版本，`_maybe_launch_beam_search_serve()` 处理 `--gr` 标志（仅设置 LMCache 环境变量并移除标志）。**OpenOneRec 在线服务应使用 `vllm serve`（vLLM 原生 CLI），beam search patches 通过 `vllm.general_plugins` 自动加载** |
| 风险 | ~~中~~ **低** — 已验证 `vllm serve` 和 `vllm-gr serve` 两条路径均可正常工作 |

### 冲突 5: engine/arg_utils.py

| 项目 | 说明 |
|------|------|
| 问题 | 两方都有实质内容，不能覆盖 |
| 方案 | vllm-gr 版重命名为 `arg_utils_gr.py`，独立存放 |
| 风险 | 低 — 两个文件功能独立 |

### 冲突 6: BeamAttentionBackend vs HSTUAttentionBackend

| 项目 | 说明 |
|------|------|
| 问题 | 两个 attention backend 是否冲突 |
| 方案 | 不冲突 — 在不同路径层级 (v1/attention vs attention)，通过 vLLM 不同注册机制共存 |
| 风险 | 无 — HSTU 使用 HSTU_KV_CACHE，OpenOneRec 使用 CUSTOM |

### 冲突 7: `__init__.py` 自动 patch vs 延迟导入

| 项目 | 说明 |
|------|------|
| 问题 | vllm-gr 导入时自动 patch，vllm-GR 延迟导入 |
| 方案 | 保留 vllm-GR 的延迟导入，增加 BeamAttentionBackend 条件导出 |
| 风险 | 低 — 统一由 plugin 入口管理 |

### 冲突 8: pandas 版本

| 项目 | 说明 |
|------|------|
| 问题 | vllm-GR 要求 `>=2.0.0`，vllm-gr 要求 `>=3.0.0` |
| 方案 | 取 `>=3.0.0` |
| 风险 | 需验证 vllm-GR 代码在 pandas 3.x 下无兼容性问题 |

## 九、建议的迁移顺序

### Phase 1: 基础设施 (1-2 天)
1. 修改 `patch.py` — 统一 monkey-patch 入口
2. 修改 `__init__.py` — 增加 BeamAttentionBackend 导出
3. 不迁移 `init.py` — 统一用 plugin 入口
4. 新建 `conftest.py`
5. 验证 HSTU/FuXi 原有测试仍然通过

### Phase 2: 核心引擎通信 (2-3 天)
1. 复制 `v1/engine/` 下 6 个文件 (types, core, core_client, core_client_patch, async_llm, 以及 `__init__.py`)
2. 复制 `v1/engine/engine_core_patch.py` — 代码逻辑不改，确保通过 `_apply_gr_beam_patches()` 中的 `patch_batch_and_fork()` 在正确时机调用
3. 复制 `v1/metrics/stats.py`
4. 复制 `v1/attention/backends/beam_attn.py`
5. 验证 `engine_core_patch.py` 的子进程 patch 在 vllm-GR plugin 流程中正确集成
6. 运行 `test_batch_and_fork_unit.py` 验证

### Phase 3: Beam Search + Catalog + Logprobs (2-3 天)
1. 复制 `logprobs.py` + `logprobs_patch.py`
2. 复制 `sampling_params.py`
3. 复制 `entrypoints/llm.py` + `beam_search_patch.py`
4. 复制 `entrypoints/openai/` 下 4 个文件
5. 新建 `engine/arg_utils_gr.py` (从 vllm-gr 的 arg_utils.py 重命名)
6. 修改 `cli/main.py` 集成 `--gr` 标志
7. 运行 beam search 相关测试验证

### Phase 4: C++ 扩展 + 配置合并 (1 天)
1. 复制 `csrc/minheap.cpp`
2. 修改 `setup.py` 增加 C++ 扩展
3. 修改 `pyproject.toml` 合并依赖

### Phase 5: Benchmark + Examples + API 适配 (1-2 天)
1. 复制 `benchmarks/open_one_rec/` 整个目录
2. 复制 examples
3. **API 适配**: 将 benchmark 和 example 中的 vllm-gr 旧 API 替换为 vllm-GR 新 API:
   - `from vllm_gr.patch import run_patch; run_patch()` → `from vllm_gr.patch import re_apply_patches; re_apply_patches()`
   - `import vllm_gr.init` → `from vllm_gr.patch import re_apply_patches; re_apply_patches()`
4. 运行 benchmark 冒烟测试

### Phase 6: 测试验证 (1-2 天)
1. 运行所有 HSTU 测试确保无回归
2. 运行所有 vllm-gr 迁移测试
3. 端到端集成测试

**总计: 8-14 天**

## 十、风险和缓解措施

| 风险 | 等级 | 缓解措施 | 状态 |
|------|------|----------|------|
| Patch 执行顺序导致冲突 | 中 | 两套 patch 目标不重叠（数据类型 vs 行为方法），按固定顺序执行 | **已验证通过** |
| pandas 3.x 兼容性 | 低 | 逐模块验证，pandas 3.x 向后兼容性较好 | **已验证通过** |
| vLLM 子进程 patch 集成 (engine_core_patch.py) | ~~中~~ 低 | 通过 `run_engine_core` 包装器在子进程启动时自动应用 | **已验证通过** |
| GRModelRunner 与 beam worker patches 兼容性 | 低 | GRModelRunner 未重写 `execute_model` 和 `_build_attention_metadata`，MRO 正确传播 | **已验证通过** |
| `_handle_beam_fork` 中 Request import | 无 | import 得到的是 GRRequest（已全局替换），beam fork 子请求自动携带 gr_features | **已验证通过** |
| import 循环依赖 | 低 | beam search 模块使用延迟导入，避免循环 | **已验证通过** |
| C++ 扩展编译失败 | 低 | pybind11 成熟稳定，minheap.cpp 代码简单 | **已验证通过** |
| Benchmark/Example 旧 API 引用 | 低 | `run_patch()` → `re_apply_patches()`, `import vllm_gr.init` → `re_apply_patches()` | **已修复并验证** |

## 十一、验证标准

迁移完成的判断标准:
1. vllm-GR 原有的 HSTU/FuXi-alpha 测试全部通过 — **已验证 (358 UT + 1 GPU smoke 通过)**
2. vllm-gr 迁移的 beam search 测试全部通过 — **已验证 (offline 2 + serving 2 + catalog 1 通过)**
3. Catalog 单元测试和集成测试通过 — **已验证**
4. BeamAttention kernel 测试通过 — **已验证 (990 kernel 测试通过)**
5. Benchmark 可以正常运行 — **已验证 (label_pred, product 离线/在线模式均通过)**
6. 不存在 vllm-gr 占位符模块的残留引用 — **已验证**
7. 离线推理示例正常运行 — **已验证 (Vanilla 1.04s, Patched 0.54s)**
8. 在线推理示例正常运行 — **已验证 (128 beam results via API)**

**总计: 1354 测试通过，0 失败。所有验证标准已满足。**

## 十二、实际迁移结果记录

### 12.1 文件迁移状态总结

| 分类 | 文件数 | 说明 |
|------|--------|------|
| 字节级直接复制 | 19 个 | v1/ 目录下 7 个 + logprobs/sampling_params/entrypoints 等 12 个 |
| 需要适配（API 更新） | 3 个 | one_rec_main.py, one_rec_acc_test.py, offline_beam_search.py |
| 需要修改（集成到 vllm-GR） | 5 个 | patch.py, __init__.py, cli/main.py, setup.py, pyproject.toml |
| 需要重命名 | 1 个 | vllm-gr 的 engine/arg_utils.py → vllm-GR 的 engine/arg_utils_gr.py |
| 新建空文件 | 1 个 | entrypoints/openai/__init__.py |
| 不迁移（占位符/合并） | 15 个 | 8 个占位符 + init.py + patch.py(vllm-gr版) + __init__.py(vllm-gr版) 等 |

### 12.2 在线服务使用指南

| 模型类型 | 启动命令 | 说明 |
|----------|----------|------|
| HSTU/FuXi-alpha | `vllm-gr serve <model> ...` | 使用 vllm-GR 自定义 API 服务器，走 GRAsyncLLM + adapter 路径 |
| OpenOneRec | `vllm serve <model> --trust-remote-code --max-logprobs 1024 ...` | 使用 vLLM 原生 API 服务器，beam search patches 通过 `vllm.general_plugins` 自动加载 |

**重要**: OpenOneRec 在线服务不需要 `--gr` 参数。所有 vllm-gr 的 beam search patches（包括 BeamAttentionBackend、Catalog、Logprobs 优化、ADD_BATCH/BEAM_FORK 协议）通过 `vllm_gr.plugin:register` 在 `vllm.general_plugins` 入口点自动加载。

### 12.3 Patch 激活机制

```
vllm serve / vllm-gr serve 启动
    │
    ├─ vLLM load_general_plugins()
    │      └─ vllm_gr.plugin:register()
    │             └─ initialize_runtime()
    │                    └─ re_apply_patches()
    │                           ├─ _apply_gr_core_patches()      [HSTU/FuXi 核心类型替换]
    │                           ├─ _apply_yuanrong_npu_patch()   [NPU Yuanrong 补丁]
    │                           ├─ _apply_kv_merge_patch()       [KV merge 补丁]
    │                           └─ _apply_gr_beam_patches()      [Beam search 相关 patches]
    │                                  ├─ patch_flat_logprobs()
    │                                  ├─ patch_beam_search()
    │                                  ├─ patch_sampling()
    │                                  ├─ patch_add_cli_args()
    │                                  ├─ patch_OpenAIServingModels_init()
    │                                  ├─ patch_offline_beam_search()
    │                                  ├─ patch_batch_and_fork()
    │                                  │      └─ 替换 EngineCoreProc.run_engine_core
    │                                  └─ _patch_lmcache_adapter() [可选]
    │
    └─ [子进程启动时]
           └─ run_engine_core() 包装器
                  └─ apply_engine_core_child_patches()
                         ├─ 注入 BeamForkRequest + ADD_BATCH/BEAM_FORK 枚举
                         ├─ apply_scheduler_patch()    [beam_prefix_groups]
                         └─ apply_worker_patches()     [GPUModelRunner beam 逻辑]
```
