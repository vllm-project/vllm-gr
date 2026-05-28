# vllm-gr → vllm-GR Migration Plan

> Version: v2.0 | Date: 2026-05-28
> v2.0 Update: Retrospectively updated after migration completion to reflect actual code state, supplement missing API adaptation notes and online serving usage.

## 1. Background

- **vllm-GR** (uppercase, `/home/t00464839/merge/vllm-GR`): Generative recommendation inference engine based on vLLM v0.14.1, supporting HSTU/FuXi-alpha models.
- **vllm-gr** (lowercase, `/home/t00464839/merge/vllm-gr`): Recommendation inference framework based on vLLM v0.14.*, supporting OpenOneRec (Qwen architecture) models.

**Goal**: Migrate the core functionality of vllm-gr into the vllm-GR main repository, so the merged codebase supports both HSTU/FuXi-alpha and OpenOneRec model inference.

## 2. Overall Migration Strategy

**Principle**: Use vllm-GR (uppercase) as the primary repository, incrementally adding vllm-gr (lowercase) core functionality into the `vllm_gr` package. The two monkey-patch mechanisms are merged into a unified plugin+patch entry point. Independent features (such as beam search, logprobs optimization) preserve their original code logic and naming as much as possible.

**Core Conflict**: Both repositories use `vllm_gr` as the package name, both depend on `vllm==0.14.*`, but their patch mechanisms are completely different:
- vllm-GR: plugin.py (vLLM general_plugins) + patch.py (core type replacement)
- vllm-gr: init.py auto-patch + patch.py (beam search/catalog/logprobs)

## 3. Architecture Comparison of the Two Codebases

### 3.1 vllm-GR Architecture Characteristics

| Feature | Description |
|---------|-------------|
| Architecture pattern | Adapter pattern + Monkey-Patch + Subclassing |
| Core type replacement | EngineCoreRequest→GREngineCoreRequest, Request→GRRequest, NewRequestData→GRNewRequestData |
| Model support | HSTU (DLRM v3) + FuXi-alpha |
| Inference path | Scoring/Pooling path, no token generation, uses synthetic prompt_token_ids |
| Attention backend | HSTUAttentionBackend (prefix/delta/full three modes) |
| Data flow | GRPrompt → msgpack → gr_features → bypass transmission → feature decoding → model forward |
| Activation mechanism | vLLM general_plugins entry point (`vllm_gr.plugin:register`) |
| CLI entry | Independent CLI parser, registered as `vllm-gr` command |
| Worker | GRWorker (inherits GPUWorker) + GRModelRunner |
| NPU support | Supports GPU + NPU dual platform, Yuanrong KV Pool backend |

### 3.2 vllm-gr Architecture Characteristics

| Feature | Description |
|---------|-------------|
| Architecture pattern | Monkey-Patch plugin |
| Core functionality | BeamAttentionBackend + ADD_BATCH/BEAM_FORK protocol |
| Model support | OpenOneRec/OneRec-1.7B (Qwen architecture), no custom model implementation |
| Inference path | Beam Search (online + offline), token generation path |
| Attention backend | BeamAttentionBackend (shared prefix cascade attention) |
| Data flow | Standard vLLM prompt → tokenizer → beam search → Catalog constraint filtering |
| Activation mechanism | `__init__.py` import-time auto-patch + `VLLM_GR_AUTOPATCH` environment variable |
| CLI entry | Wraps native vLLM CLI, triggers patch via `--gr` flag |
| Online serving method | After migration, OpenOneRec models use `vllm serve` (no `--gr` needed), all patches auto-loaded via `vllm.general_plugins` |
| Catalog | Trie-based valid token constraints, ensuring only valid recommendation results are generated |
| C++ extension | minheap_cpp (pybind11), efficient top-K selection |

### 3.3 Comparison of the Two Patch Mechanisms

#### vllm-GR patch.py

Activation: Through vLLM plugin entry point `vllm_gr.plugin:register` (registered in pyproject.toml)

Core functions: `_apply_patches()` / `re_apply_patches()`

Patch content (3 categories):
1. `_apply_gr_core_patches()` — Replace EngineCoreRequest/Request/NewRequestData with GR variants (via sys.modules global replacement)
2. `_apply_yuanrong_npu_patch()` — NPU platform Yuanrong backend injection
3. `_apply_kv_merge_patch()` — K+V merge KV cache support

Idempotency: Uses `_PATCHES_APPLIED` / `force` parameter to ensure idempotency

Verification: `verify_patches()` checks that key consumer modules have been correctly patched

#### vllm-gr patch.py

Activation: In `__init__.py` via `init.py` auto-calling `run_patch()` (with `VLLM_GR_AUTOPATCH` env var control)

Core function: `run_patch()`

Patch content (7+1 categories):
1. `patch_flat_logprobs()` — FlatLogprobs performance optimization
2. `patch_beam_search()` — OpenAI serving beam search replacement
3. `patch_sampling()` — ChatCompletionRequest.to_beam_search_params replacement
4. `patch_add_cli_args()` — EngineArgs CLI parameter extension (catalog_path)
5. `patch_OpenAIServingModels_init()` — OpenAIServingModels initialization extension (Catalog trie)
6. `patch_offline_beam_search()` — LLM.beam_search replacement
7. `patch_batch_and_fork()` — ADD_BATCH + BEAM_FORK engine pipeline patches
8. Optional: `_patch_lmcache_adapter()` — LMCache adapter patch

**Conclusion: The two patch mechanisms do not overlap at all and can be safely merged.**

**Actual merge result**: vllm-gr's `run_patch()` is not directly migrated; its functionality is decomposed into `_apply_gr_beam_patches()` for unified invocation. vllm-gr's `init.py` auto-patch mechanism is not migrated; instead, it is managed uniformly by vllm-GR's `plugin.py`. All patches are auto-loaded via the `vllm.general_plugins` entry point, requiring no manual invocation or environment variable control.

## 4. v1/ Directory Compatibility Analysis (Critical Issue)

### 4.1 Problem Background

vllm-GR is itself developed based on the vLLM v1 engine; its core EngineCore is based on the vLLM v1 version. vllm-gr's `vllm_gr/v1/` directory also contains monkey-patches for vLLM v1 engine components. The core question is: **Both operate on the vLLM v1 engine; can vllm-gr's v1/ patches be directly copied into vllm-GR?**

### 4.2 Patch Layer Analysis

Although vllm-GR and vllm-gr are both based on the vLLM v1 engine, the layers they patch **do not overlap at all**:

| Dimension | vllm-GR | vllm-gr |
|-----------|---------|---------|
| Patch method | `sys.modules` global replacement of core **data types** | monkey-patch replacement of core **behavior methods** |
| Replacement target | `EngineCoreRequest→GREngineCoreRequest`<br>`Request→GRRequest`<br>`NewRequestData→GRNewRequestData` | `Scheduler.schedule`<br>`GPUModelRunner.execute_model`<br>`EngineCoreProc.run_engine_core`<br>`EngineCoreProc.process_input_sockets` |
| Purpose | Enable GR features (gr_features) to pass through the vLLM pipeline | Enable beam search BEAM_FORK/ADD_BATCH to pass through the engine pipeline |

**Core conclusion: Data type replacement and behavior method replacement do not interfere with each other; the two patch sets can safely coexist.**

### 4.3 File-by-file Compatibility Analysis of v1/

#### 4.3.1 `v1/engine/engine_core_patch.py` — Requires Adaptation (Cannot Copy Directly)

This is the most critical file, containing 6 patch functions, analyzed one by one:

| Patch function | Conflict analysis | Conclusion |
|----------------|-------------------|------------|
| `apply_engine_core_child_patches()` injecting `BeamForkRequest` | Injects a completely new type, does not conflict with vllm-GR's `GREngineCoreRequest`/`GRRequest` | Can use directly |
| `_add_enum_member()` injecting `ADD_BATCH`/`BEAM_FORK` | Extends enum; vllm-GR replaces data classes, not enums | Can use directly |
| `make_patched_engine_core_init()` wrapping `EngineCore.__init__` | vllm-GR does not patch `EngineCore.__init__` | Can use directly |
| `run_engine_core()` replacing `EngineCoreProc.run_engine_core` | **Critical**: vllm-GR currently has no child process patch mechanism, needs integration into plugin initialization flow | Needs adaptation |
| `apply_scheduler_patch()` replacing `Scheduler.schedule` | vllm-GR does not patch Scheduler; its `GRNewRequestData` takes effect via global type replacement | Can use directly |
| `apply_worker_patches()` replacing `GPUModelRunner` methods | **Critical**: vllm-GR has `GRModelRunner(GPUModelRunner)` subclass | See detailed analysis below |

**Compatibility of `apply_worker_patches()` with `GRModelRunner`:**

vllm-GR's `GRModelRunner` inherits from `GPUModelRunner`, overriding `_update_states`, `_prepare_inputs`, `_preprocess`, `_pool`, etc., but **does not override** `execute_model` and `_build_attention_metadata`. vllm-gr's `apply_worker_patches()` monkey-patches beam search logic onto `GPUModelRunner`. Due to Python MRO (Method Resolution Order), `GRModelRunner` instances will use the version patched onto the parent class when calling `execute_model` and `_build_attention_metadata`. **This is actually compatible** — beam_prefix_groups logic executes correctly in the GRModelRunner inheritance chain.

**About `run_engine_core()` child process entry:**

**Actual implementation**: `patch_batch_and_fork()` replaces `EngineCoreProc.run_engine_core` with vllm-gr's wrapper in the main process. This wrapper calls `apply_engine_core_child_patches()` when the child process starts, applying Scheduler/GPUModelRunner and other child-process-level patches. Patches in child processes are split into two layers:
1. **Main process patches** (auto-loaded in all processes via `vllm.general_plugins`): `OpenAIServing.beam_search`, `AsyncMPClient` new methods, `EngineCoreProc.run_engine_core` wrapper, etc.
2. **Child process patches** (applied by the `run_engine_core` wrapper calling `apply_engine_core_child_patches()` at EngineCore child process startup): `ADD_BATCH/BEAM_FORK` enum, `Scheduler.schedule` beam_prefix_groups, `GPUModelRunner._build_attention_metadata`, etc.

**The core logic of the file itself does not need modification**; actual verification has confirmed this mechanism works correctly in vllm-GR.

#### 4.3.2 `v1/engine/core.py` — Can Copy Directly

Defines `_cache_beam_request`, `_handle_beam_fork`, `process_input_sockets` functions.

**Compatibility highlight**: `_handle_beam_fork` has `from vllm.v1.request import Request`. In vllm-GR's child process, `patch.py`'s global replacement has already replaced `Request` with `GRRequest`. Therefore, sub-requests created by beam fork will automatically be `GRRequest` instances with `gr_features` fields. **This does not cause problems but is correct behavior** — beam search sub-requests can also carry GR features.

#### 4.3.3 Other v1/ Files — All Can Copy Directly

| File | Compatibility analysis |
|------|----------------------|
| `v1/engine/types.py` | Pure new type `BeamForkRequest`, does not conflict with any existing type |
| `v1/engine/core_client.py` | Pure new methods `add_requests_async`/`beam_fork_async`, does not modify existing behavior |
| `v1/engine/core_client_patch.py` | Injects into `AsyncMPClient`. vllm-GR's `GRAsyncLLM(AsyncLLM)` inherits from `AsyncLLM`; methods monkey-patched onto `AsyncLLM` are visible to subclasses |
| `v1/engine/async_llm.py` | Defines beam search methods bound to `AsyncLLM`, compatible with `GREngineCoreRequest` (subclass relationship) |
| `v1/attention/backends/beam_attn.py` | Complete independent attention backend, registered as CUSTOM type, does not conflict with HSTUAttentionBackend |
| `v1/metrics/stats.py` | Pure new fields, does not modify existing fields |

### 4.4 v1/ Migration Conclusion (Only for Files in v1/ Directory)

| Conclusion | File count | Description |
|------------|------------|-------------|
| **Can copy directly** | 7 | types.py, core.py, core_client.py, core_client_patch.py, async_llm.py, beam_attn.py, stats.py |
| **Needs adaptation for integration** | 1 | engine_core_patch.py — Core logic unchanged, needs to ensure correct invocation in plugin initialization flow |

## 5. Monkey-Patch Mechanism Merge Plan

### 5.1 Merge Strategy

1. **Keep vllm-GR's plugin.py as the unified entry point** (through vLLM general_plugins), calling both sets of patches in its `initialize_runtime()`
2. **Modify vllm-GR's patch.py**, appending vllm-gr's patch function calls at the end of the existing `_apply_patches()`
3. **Do not migrate vllm-gr's init.py** (auto-import-time patch mechanism); use vllm-GR's plugin for unified management
4. **Do not migrate vllm-gr's patch.py** (functionality already merged into vllm-GR's patch.py)

### 5.2 Modified patch.py Flow

```
_apply_patches(force):
    _apply_gr_core_patches(force)        # vllm-GR: core type replacement
    _apply_yuanrong_npu_patch(force)     # vllm-GR: NPU Yuanrong patch
    _apply_kv_merge_patch(force)         # vllm-GR: KV merge patch
    _apply_gr_beam_patches(force)        # [NEW] vllm-gr: beam search related patches
```

Where `_apply_gr_beam_patches()` internally calls:
```
patch_flat_logprobs()                   # logprobs optimization
patch_beam_search()                     # online beam search
patch_sampling()                        # BeamSearchParams
patch_add_cli_args()                    # CLI --catalog-path
patch_OpenAIServingModels_init()        # Catalog
patch_offline_beam_search()             # offline beam search
patch_batch_and_fork()                  # ADD_BATCH/BEAM_FORK
_patch_lmcache_adapter()               # [Optional] LMCache compatibility (only when lmcache is installed)
```
Total: 7 required patches + 1 optional patch, consistent with the original vllm-gr patch list in Section 3.

## 6. File-Level Migration Checklist

### 6.1 Direct Copy Files (No Modification Needed, Does Not Exist in vllm-GR, No Conflict)

| Source (vllm-gr) | Target (vllm-GR) | Description |
|---|---|---|
| `vllm_gr/v1/attention/backends/beam_attn.py` | Same path | BeamAttentionBackend, independent module |
| `vllm_gr/v1/attention/backends/__init__.py` | Same path | BeamAttentionBackend export |
| `vllm_gr/v1/engine/types.py` | Same path | BeamForkRequest |
| `vllm_gr/v1/engine/core.py` | Same path | EngineCore ADD_BATCH/BEAM_FORK, `import Request` in `_handle_beam_fork` automatically gets GRRequest |
| `vllm_gr/v1/engine/core_client.py` | Same path | AsyncMPClient patch, new methods visible to GRAsyncLLM subclass |
| `vllm_gr/v1/engine/core_client_patch.py` | Same path | apply_batch_fork_patches |
| `vllm_gr/v1/engine/async_llm.py` | Same path | AsyncLLM beam methods, compatible with GREngineCoreRequest |
| `vllm_gr/v1/engine/__init__.py` | Same path | Empty file |
| `vllm_gr/v1/__init__.py` | Same path | Empty file |
| `vllm_gr/v1/metrics/stats.py` | Same path | RequestStateStats extension, pure new fields |
| `vllm_gr/v1/metrics/__init__.py` | Same path | Empty file |
| `vllm_gr/v1/attention/__init__.py` | Same path | Empty file |
| `vllm_gr/logprobs.py` | Same path | FlatLogprobs optimization |
| `vllm_gr/logprobs_patch.py` | Same path | patch_flat_logprobs |
| `vllm_gr/sampling_params.py` | Same path | BeamSearchParams extension |
| `vllm_gr/entrypoints/llm.py` | Same path | Offline beam search (LLM.beam_search) |
| `vllm_gr/entrypoints/beam_search_patch.py` | Same path | Offline beam search patch |
| `vllm_gr/entrypoints/openai/beam_search_patch.py` | Same path | Online beam search + BEAM_FORK patch |
| `vllm_gr/entrypoints/openai/protocol.py` | Same path | to_beam_search_params |
| `vllm_gr/entrypoints/openai/serving_engine.py` | Same path | Online beam search implementation |
| `vllm_gr/entrypoints/openai/serving_models.py` | Same path | Catalog + init patch |
| `vllm_gr/entrypoints/openai/__init__.py` | Same path | Empty file |
| `vllm_gr/csrc/minheap.cpp` | Same path | C++ MinHeap extension |
| `vllm_gr/3rd_party/lmcache/` (entire directory) | Same path | LMCache patch |

### 6.2 Files Requiring Adaptation (Code logic unchanged, but needs integration into vllm-GR initialization flow)

#### 6.2.1 `vllm_gr/v1/engine/engine_core_patch.py`

**Core logic does not need modification**, but note the following integration points:

1. **`apply_engine_core_child_patches()`**: Called by `run_engine_core()` in the child process, injecting `BeamForkRequest` and enum members. Must ensure invocation in vllm-GR's child process patch flow.
2. **`run_engine_core()`**: Replaces `EngineCoreProc.run_engine_core`. vllm-GR currently has no child process patch mechanism; needs to replace the entry function via `patch_batch_and_fork()` in the main process.
3. **`apply_worker_patches()`**: Monkey-patches `GPUModelRunner.execute_model` and `_build_attention_metadata`. Since `GRModelRunner` does not override these two methods, versions patched onto the parent class propagate correctly to the subclass via MRO.
4. **`apply_scheduler_patch()`**: Wraps `Scheduler.schedule`, does not conflict with vllm-GR (vllm-GR does not patch Scheduler).

**Integration method**: Called indirectly through `patch_batch_and_fork()` in `_apply_gr_beam_patches()`, executed during vllm-GR's plugin initialization flow.

### 6.3 Files Requiring Modification/Merging

#### 6.3.1 `vllm_gr/patch.py` (vllm-GR existing file)

**Modification**: Append beam search patches call at the end of the existing `_apply_patches()` function.

Add new `_apply_gr_beam_patches()` function, internally calling 7 required beam search patch functions + 1 optional lmcache patch.

#### 6.3.2 `vllm_gr/__init__.py` (vllm-GR existing file)

**Modification**: Add `BeamAttentionBackend` conditional export in the existing `__getattr__` lazy import.

```python
def __getattr__(name):
    if name == "BeamAttentionBackend":
        from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend
        return BeamAttentionBackend
    # ... existing logic unchanged
```

#### 6.3.3 `vllm_gr/engine/arg_utils.py` (vllm-gr version renamed to `arg_utils_gr.py`)

**Conflict description**: Both sides have substantial content:
- vllm-GR: `GRCommonEngineArgsMixin`, `GREngineArgs`, `GRAsyncEngineArgs` (dataclass, inheriting from vLLM EngineArgs)
- vllm-gr: `ModelConfigGr`, `patch_add_cli_args()` (monkey patch function)

**Resolution**: Rename vllm-gr's `arg_utils.py` to `arg_utils_gr.py` and place in vllm-GR, keeping vllm-GR's original file unchanged. `patch_add_cli_args()` is called in the unified patch flow.

#### 6.3.4 `vllm_gr/entrypoints/cli/main.py` (vllm-GR existing file)

**Conflict description**: Both sides have different CLI approaches:
- vllm-GR: Independent CLI parser, using FlexibleArgumentParser + subcommand mode
- vllm-gr: Wraps native vLLM CLI, triggers patch via `--gr` flag

**Actual resolution**: Keep vllm-GR's independent CLI approach. `_maybe_launch_beam_search_serve()` added to `main()` handles the `--gr` flag (detects LMCache config and sets env var, then removes `--gr` to avoid parse errors). In actual runtime, **all beam search patches are auto-loaded via the `vllm.general_plugins` mechanism**, requiring no `--gr` flag trigger.

**Online serving usage**:
- HSTU/FuXi models: `vllm-gr serve <model> ...` (uses vllm-GR custom API server, via GRAsyncLLM + adapter path)
- OpenOneRec models: `vllm serve <model> --trust-remote-code ...` (uses vLLM native API server, beam search patches auto-activated via plugin)

#### 6.3.5 `setup.py` (vllm-GR existing file)

**Modification**: Keep existing content unchanged, add C++ extension build configuration (minheap_cpp).

New logic:
- import pybind11, numpy
- Extension("minheap_cpp", ["vllm_gr/csrc/minheap.cpp"], ...)
- extra_compile_args=["-std=c++17", "-O3"]

#### 6.3.6 `pyproject.toml` (vllm-GR existing file)

Modifications:
- `build-system.requires`: Add `"pybind11"`, `"numpy"` (for C++ extension)
- `dependencies`:
  - `pandas` version from `>=2.0.0` to `>=3.0.0` (vllm-gr requirement)
  - Add `scikit-learn>=1.8.0`, `pyarrow>=23.0.0` (vllm-gr benchmark dependencies)
- `optional-dependencies`: Add `lmcache = ["lmcache==0.4.1"]`

### 6.4 Files Not Migrated (vllm-gr Placeholders)

| File | Reason |
|---|---|
| `vllm_gr/core/scheduler.py` | Placeholder; actual scheduler patch is in engine_core_patch.py |
| `vllm_gr/core/kv_cache_manager.py` | Pure placeholder |
| `vllm_gr/core/__init__.py` | Only exports placeholder classes |
| `vllm_gr/worker/__init__.py` | Pure placeholder |
| `vllm_gr/model_executor/__init__.py` | Pure placeholder |
| `vllm_gr/model_executor/models/__init__.py` | Pure placeholder |
| `vllm_gr/sample/__init__.py` | Pure placeholder |
| `vllm_gr/utils/__init__.py` | Pure placeholder |
| `vllm_gr/engine/__init__.py` (vllm-gr version) | Empty file; vllm-GR has its own |
| `vllm_gr/attention/__init__.py` (vllm-gr version) | Empty file; vllm-GR has its own |
| `vllm_gr/init.py` | Auto-patch mechanism; replaced by unified plugin management |
| `vllm_gr/patch.py` (vllm-gr version) | Functionality merged into vllm-GR's patch.py |
| `vllm_gr/__init__.py` (vllm-gr version) | Functionality merged into vllm-GR's __init__.py |
| `vllm_gr/entrypoints/__init__.py` (vllm-gr version) | Empty file; vllm-GR has its own |
| `vllm_gr/entrypoints/cli/main.py` (vllm-gr version) | Functionality merged into vllm-GR's CLI |
| `collect_env.py` | Environment collection, non-core |
| `tools/` | Pre-commit hooks and other dev tools, not migrated for now |
| `.github/` | CI configuration, to be handled separately |
| `.pre-commit-config.yaml` and other config files | Dev tool configs, not migrated for now |

### 6.5 Benchmark Migration

The entire `benchmarks/open_one_rec/` directory is copied to vllm-GR, where **2 files require API adaptation**.

Includes:
- `one_rec_main.py` — benchmark CLI entry point **[Needs adaptation]**
- `one_rec_patch.py` — vLLM benchmark monkey patch
- `open_one_rec_dataset.py` — OneRecDataset implementation
- `open_one_rec_loader.py` — Data loader
- `base_evaluator.py` — Evaluator base class
- `tasks/` — Task configs and evaluators (video, product, ad, label_cond, etc.)
- `tools/` — Catalog conversion and creation tools
- `one_rec_acc_test.py` — Accuracy test **[Needs adaptation]**

Benchmark import paths `benchmarks.open_one_rec.xxx` remain unchanged (under project root, not inside the vllm_gr package).

**API adaptation**:
- `one_rec_main.py`: `from vllm_gr.patch import run_patch` → `from vllm_gr.patch import re_apply_patches`
- `one_rec_main.py`: `run_patch()` → `re_apply_patches()`
- `one_rec_acc_test.py`: Same as above

### 6.6 Examples Migration

| Source (vllm-gr) | Target (vllm-GR) | Description |
|---|---|---|
| `examples/offline_inference/beam_search/offline_beam_search.py` | Same path | **[Needs adaptation]** `import vllm_gr.init` → `from vllm_gr.patch import re_apply_patches; re_apply_patches()` |
| `examples/online_serving/beam_search/client.sh` | Same path | Direct copy, no modification needed |
| `examples/online_serving/catalog/verify_catalog.py` | Same path | Direct copy, no modification needed |

### 6.7 Test Case Migration

| Source (vllm-gr) | Target (vllm-GR) | Description |
|---|---|---|
| `tests/conftest.py` | `tests/conftest.py` | New file (vllm-GR has no such file), defines `--loops`, `--run-slow` |
| `tests/test_catalog.py` | Same path | Catalog unit test |
| `tests/test_serving_catalog.py` | Same path | Catalog integration test (requires GPU) |
| `tests/test_batch_and_fork_unit.py` | Same path | ADD_BATCH/BEAM_FORK unit test |
| `tests/test_dp_beam_search_unit.py` | Same path | DP beam search unit test |
| `tests/test_serving_beam_search.py` | Same path | Online beam search test (requires GPU) |
| `tests/test_offline_beam_search.py` | Same path | Offline beam search test (requires GPU) |
| `tests/test_task_specific_catalog_program.py` | Same path | Catalog program test |
| `tests/kernels/attention/test_cascade_beam_flash_attn.py` | Same path | Beam attention kernel test (requires GPU) |
| `tests/resources/single_one_rec_prompt.txt` | Same path | Test resource file |

## 7. New Directory/File Structure After Migration

```
vllm-GR/
├── vllm_gr/
│   ├── __init__.py              # Modified: add BeamAttentionBackend conditional import
│   ├── patch.py                 # Modified: merge beam search patches
│   ├── plugin.py                # Unchanged (patch.py internally merged)
│   ├── logprobs.py              # New (from vllm-gr)
│   ├── logprobs_patch.py        # New (from vllm-gr)
│   ├── sampling_params.py       # New (from vllm-gr)
│   ├── csrc/
│   │   └── minheap.cpp          # New (from vllm-gr)
│   ├── engine/
│   │   ├── arg_utils_gr.py      # New (from vllm-gr engine/arg_utils.py)
│   │   └── ...                  # Existing files unchanged
│   ├── entrypoints/
│   │   ├── llm.py               # New (from vllm-gr)
│   │   ├── beam_search_patch.py # New (from vllm-gr)
│   │   ├── cli/
│   │   │   └── main.py          # Modified: add beam search mode support
│   │   └── openai/
│   │       ├── __init__.py          # New (from vllm-gr)
│   │       ├── beam_search_patch.py # New (from vllm-gr)
│   │       ├── protocol.py          # New (from vllm-gr)
│   │       ├── serving_engine.py    # New (from vllm-gr)
│   │       └── serving_models.py    # New (from vllm-gr)
│   ├── v1/                          # Entire directory new (from vllm-gr)
│   │   ├── __init__.py              # New
│   │   ├── attention/
│   │   │   ├── __init__.py          # New
│   │   │   └── backends/
│   │   │       ├── __init__.py      # New
│   │   │       └── beam_attn.py     # New
│   │   ├── engine/
│   │   │   ├── __init__.py              # New
│   │   │   ├── core.py                  # New (direct copy)
│   │   │   ├── core_client.py           # New (direct copy)
│   │   │   ├── core_client_patch.py     # New (direct copy)
│   │   │   ├── engine_core_patch.py     # New (needs adaptation)
│   │   │   ├── async_llm.py             # New (direct copy)
│   │   │   └── types.py                 # New (direct copy)
│   │   └── metrics/
│   │       ├── __init__.py          # New
│   │       └── stats.py             # New
│   └── 3rd_party/
│       └── lmcache/...           # New (from vllm-gr)
├── benchmarks/
│   └── open_one_rec/             # Entire directory new (from vllm-gr)
├── tests/
│   ├── conftest.py               # New
│   ├── test_catalog.py           # New
│   ├── test_serving_catalog.py   # New
│   ├── test_batch_and_fork_unit.py  # New
│   ├── test_dp_beam_search_unit.py  # New
│   ├── test_serving_beam_search.py  # New
│   ├── test_offline_beam_search.py  # New
│   ├── test_task_specific_catalog_program.py  # New
│   ├── kernels/attention/        # New
│   └── resources/                # New (partial)
├── setup.py                      # Modified: add C++ extension
└── pyproject.toml                # Modified: add dependencies and build requirements
```

## 8. Key Conflict Points and Solutions

### Conflict 1: Two patch.py Files

| Item | Description |
|------|-------------|
| Problem | Two patch sets with completely different functionality, cannot directly overwrite |
| Solution | Merge into vllm-GR's patch.py, add new `_apply_gr_beam_patches()` function |
| Risk | Low — Two patch sets have completely non-overlapping targets |

### Conflict 2: engine_core_patch.py child process patch vs vllm-GR initialization flow

| Item | Description |
|------|-------------|
| Problem | vllm-gr's `run_engine_core()` replaces `EngineCoreProc.run_engine_core`, re-applies patches in child process. vllm-GR currently has no child process patch mechanism |
| Solution | Through `patch_batch_and_fork()` called in `_apply_gr_beam_patches()`, replaces `EngineCoreProc.run_engine_core` in the main process. The wrapper automatically calls `apply_engine_core_child_patches()` at child process startup |
| Risk | ~~Medium~~ **Low** — Verified that child process patches execute correctly; all UT and E2E tests pass |

### Conflict 3: GPUModelRunner monkey-patch vs GRModelRunner inheritance

| Item | Description |
|------|-------------|
| Problem | vllm-gr's `apply_worker_patches()` monkey-patches `GPUModelRunner`'s `execute_model` and `_build_attention_metadata`, while vllm-GR has `GRModelRunner(GPUModelRunner)` subclass |
| Solution | **Naturally compatible** — GRModelRunner does not override these two methods; Python MRO ensures versions patched onto the parent class propagate correctly to the subclass. beam_prefix_groups ContextVar mechanism executes correctly in the GRModelRunner inheritance chain |
| Risk | None — Compatibility verified |

### Conflict 4: Two CLI main.py Files

| Item | Description |
|------|-------------|
| Problem | vllm-GR has independent CLI (`vllm-gr` command); vllm-gr wraps native vLLM CLI |
| Solution | Keep vllm-GR version; `_maybe_launch_beam_search_serve()` handles `--gr` flag (only sets LMCache env var and removes flag). **OpenOneRec online serving should use `vllm serve` (vLLM native CLI), beam search patches auto-loaded via `vllm.general_plugins`** |
| Risk | ~~Medium~~ **Low** — Verified that both `vllm serve` and `vllm-gr serve` paths work correctly |

### Conflict 5: engine/arg_utils.py

| Item | Description |
|------|-------------|
| Problem | Both sides have substantial content, cannot overwrite |
| Solution | vllm-gr version renamed to `arg_utils_gr.py`, stored independently |
| Risk | Low — Two files have independent functionality |

### Conflict 6: BeamAttentionBackend vs HSTUAttentionBackend

| Item | Description |
|------|-------------|
| Problem | Whether the two attention backends conflict |
| Solution | No conflict — At different path levels (v1/attention vs attention), coexist through different vLLM registration mechanisms |
| Risk | None — HSTU uses HSTU_KV_CACHE, OpenOneRec uses CUSTOM |

### Conflict 7: `__init__.py` auto-patch vs lazy import

| Item | Description |
|------|-------------|
| Problem | vllm-gr auto-patches on import, vllm-GR uses lazy import |
| Solution | Keep vllm-GR's lazy import, add BeamAttentionBackend conditional export |
| Risk | Low — Uniformly managed by plugin entry point |

### Conflict 8: pandas version

| Item | Description |
|------|-------------|
| Problem | vllm-GR requires `>=2.0.0`, vllm-gr requires `>=3.0.0` |
| Solution | Use `>=3.0.0` |
| Risk | Need to verify vllm-GR code has no compatibility issues under pandas 3.x |

## 9. Recommended Migration Order

### Phase 1: Infrastructure (1-2 days)
1. Modify `patch.py` — Unified monkey-patch entry point
2. Modify `__init__.py` — Add BeamAttentionBackend export
3. Do not migrate `init.py` — Use plugin entry point uniformly
4. Create new `conftest.py`
5. Verify HSTU/FuXi original tests still pass

### Phase 2: Core Engine Communication (2-3 days)
1. Copy 6 files under `v1/engine/` (types, core, core_client, core_client_patch, async_llm, and `__init__.py`)
2. Copy `v1/engine/engine_core_patch.py` — Code logic unchanged, ensure correct invocation through `patch_batch_and_fork()` in `_apply_gr_beam_patches()`
3. Copy `v1/metrics/stats.py`
4. Copy `v1/attention/backends/beam_attn.py`
5. Verify `engine_core_patch.py` child process patch integrates correctly in vllm-GR plugin flow
6. Run `test_batch_and_fork_unit.py` to verify

### Phase 3: Beam Search + Catalog + Logprobs (2-3 days)
1. Copy `logprobs.py` + `logprobs_patch.py`
2. Copy `sampling_params.py`
3. Copy `entrypoints/llm.py` + `beam_search_patch.py`
4. Copy 4 files under `entrypoints/openai/`
5. Create new `engine/arg_utils_gr.py` (renamed from vllm-gr's arg_utils.py)
6. Modify `cli/main.py` to integrate `--gr` flag
7. Run beam search related tests to verify

### Phase 4: C++ Extension + Config Merge (1 day)
1. Copy `csrc/minheap.cpp`
2. Modify `setup.py` to add C++ extension
3. Modify `pyproject.toml` to merge dependencies

### Phase 5: Benchmark + Examples + API Adaptation (1-2 days)
1. Copy entire `benchmarks/open_one_rec/` directory
2. Copy examples
3. **API adaptation**: Replace old vllm-gr API in benchmarks and examples with vllm-GR new API:
   - `from vllm_gr.patch import run_patch; run_patch()` → `from vllm_gr.patch import re_apply_patches; re_apply_patches()`
   - `import vllm_gr.init` → `from vllm_gr.patch import re_apply_patches; re_apply_patches()`
4. Run benchmark smoke tests

### Phase 6: Test Verification (1-2 days)
1. Run all HSTU tests to ensure no regression
2. Run all vllm-gr migrated tests
3. End-to-end integration testing

**Total: 8-14 days**

## 10. Risks and Mitigation Measures

| Risk | Level | Mitigation | Status |
|------|-------|------------|--------|
| Patch execution order causing conflicts | Medium | Two patch sets have non-overlapping targets (data types vs behavior methods), execute in fixed order | **Verified** |
| pandas 3.x compatibility | Low | Verify module by module; pandas 3.x backward compatibility is good | **Verified** |
| vLLM child process patch integration (engine_core_patch.py) | ~~Medium~~ Low | Through `run_engine_core` wrapper auto-applied at child process startup | **Verified** |
| GRModelRunner vs beam worker patches compatibility | Low | GRModelRunner does not override `execute_model` and `_build_attention_metadata`; MRO propagates correctly | **Verified** |
| `Request` import in `_handle_beam_fork` | None | Import gets GRRequest (globally replaced); beam fork sub-requests automatically carry gr_features | **Verified** |
| Import circular dependency | Low | Beam search modules use lazy imports to avoid circularity | **Verified** |
| C++ extension compilation failure | Low | pybind11 is mature and stable; minheap.cpp code is simple | **Verified** |
| Benchmark/Example old API references | Low | `run_patch()` → `re_apply_patches()`, `import vllm_gr.init` → `re_apply_patches()` | **Fixed and verified** |

## 11. Verification Criteria

Criteria for judging migration completion:
1. vllm-GR's original HSTU/FuXi-alpha tests all pass — **Verified (358 UT + 1 GPU smoke passed)**
2. vllm-gr migrated beam search tests all pass — **Verified (offline 2 + serving 2 + catalog 1 passed)**
3. Catalog unit tests and integration tests pass — **Verified**
4. BeamAttention kernel tests pass — **Verified (990 kernel tests passed)**
5. Benchmark runs correctly — **Verified (label_pred, product offline/online modes both passed)**
6. No residual references to vllm-gr placeholder modules — **Verified**
7. Offline inference example runs correctly — **Verified (Vanilla 1.04s, Patched 0.54s)**
8. Online inference example runs correctly — **Verified (128 beam results via API)**

**Total: 1354 tests passed, 0 failures. All verification criteria met.**

## 12. Actual Migration Results

### 12.1 File Migration Status Summary

| Category | File count | Description |
|----------|------------|-------------|
| Byte-level direct copy | 19 | 7 under v1/ + 12 including logprobs/sampling_params/entrypoints etc. |
| Needs adaptation (API update) | 3 | one_rec_main.py, one_rec_acc_test.py, offline_beam_search.py |
| Needs modification (integration into vllm-GR) | 5 | patch.py, __init__.py, cli/main.py, setup.py, pyproject.toml |
| Needs renaming | 1 | vllm-gr's engine/arg_utils.py → vllm-GR's engine/arg_utils_gr.py |
| New empty file | 1 | entrypoints/openai/__init__.py |
| Not migrated (placeholder/merged) | 15 | 8 placeholders + init.py + patch.py(vllm-gr) + __init__.py(vllm-gr) etc. |

### 12.2 Online Serving Usage Guide

| Model type | Startup command | Description |
|------------|-----------------|-------------|
| HSTU/FuXi-alpha | `vllm-gr serve <model> ...` | Uses vllm-GR custom API server, via GRAsyncLLM + adapter path |
| OpenOneRec | `vllm serve <model> --trust-remote-code --max-logprobs 1024 ...` | Uses vLLM native API server, beam search patches auto-loaded via `vllm.general_plugins` |

**Important**: OpenOneRec online serving does not require the `--gr` flag. All vllm-gr beam search patches (including BeamAttentionBackend, Catalog, Logprobs optimization, ADD_BATCH/BEAM_FORK protocol) are auto-loaded via `vllm_gr.plugin:register` at the `vllm.general_plugins` entry point.

### 12.3 Patch Activation Mechanism

```
vllm serve / vllm-gr serve startup
    │
    ├─ vLLM load_general_plugins()
    │      └─ vllm_gr.plugin:register()
    │             └─ initialize_runtime()
    │                    └─ re_apply_patches()
    │                           ├─ _apply_gr_core_patches()      [HSTU/FuXi core type replacement]
    │                           ├─ _apply_yuanrong_npu_patch()   [NPU Yuanrong patch]
    │                           ├─ _apply_kv_merge_patch()       [KV merge patch]
    │                           └─ _apply_gr_beam_patches()      [Beam search related patches]
    │                                  ├─ patch_flat_logprobs()
    │                                  ├─ patch_beam_search()
    │                                  ├─ patch_sampling()
    │                                  ├─ patch_add_cli_args()
    │                                  ├─ patch_OpenAIServingModels_init()
    │                                  ├─ patch_offline_beam_search()
    │                                  ├─ patch_batch_and_fork()
    │                                  │      └─ Replace EngineCoreProc.run_engine_core
    │                                  └─ _patch_lmcache_adapter() [Optional]
    │
    └─ [At child process startup]
           └─ run_engine_core() wrapper
                  └─ apply_engine_core_child_patches()
                         ├─ Inject BeamForkRequest + ADD_BATCH/BEAM_FORK enum
                         ├─ apply_scheduler_patch()    [beam_prefix_groups]
                         └─ apply_worker_patches()     [GPUModelRunner beam logic]
```
