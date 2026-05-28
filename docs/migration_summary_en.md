# vllm-gr → vllm-GR Migration Completion Summary

> Date: 2026-05-28
> Version: v1.0

## 1. Migration Overview

Core functionality from vllm-gr (lowercase, supporting OpenOneRec models) has been migrated into the vllm-GR (uppercase, supporting HSTU/FuXi-alpha models) main repository. After migration, vllm-GR supports inference for both model types:

| Model type | Startup command | Description |
|------------|-----------------|-------------|
| HSTU/FuXi-alpha | `vllm-gr serve <model> ...` | Custom API server via GRAsyncLLM + adapter path |
| OpenOneRec | `vllm serve <model> --trust-remote-code ...` | vLLM native API server, beam search patches auto-loaded via plugin |

---

## 2. Migrated Functional Modules

| Module | Description | Migration method |
|--------|-------------|-----------------|
| Beam Search Engine | ADD_BATCH/BEAM_FORK protocol, beam search scheduling | Direct copy |
| BeamAttentionBackend | Shared prefix cascade attention backend | Direct copy |
| BeamForkRequest type | Beam fork request type definition | Direct copy |
| EngineCore communication | Core layer communication for beam fork/add_batch | Direct copy |
| Scheduler extension | beam_prefix_groups scheduling support | Direct copy |
| GPUModelRunner extension | Beam search attention metadata construction | Direct copy |
| FlatLogprobs optimization | Logprobs flattening performance optimization | Direct copy |
| BeamSearchParams | Beam search sampling parameter extension | Direct copy |
| Catalog (Trie) | Valid token constraint system | Direct copy |
| Offline Beam Search | LLM.beam_search offline inference | Direct copy |
| Online Beam Search | OpenAI serving beam search online inference | Direct copy |
| LMCache adapter | LMCache connector compatibility patch | Direct copy |
| C++ MinHeap extension | Efficient top-K selection (pybind11) | Direct copy |
| CLI argument extension | --catalog-path and other beam search arguments | Direct copy |
| OpenOneRec Benchmark | Complete benchmark suite (6 task types) | Partial adaptation |
| Examples (offline/online) | Beam search inference examples | Partial adaptation |
| Unit/E2E tests | 9 test files + kernel test | Partial adaptation |
| Documentation | 12 documentation files | Migration + update |

---

## 3. Detailed File Migration List

### 3.1 Direct Copy (Byte-identical, 0 modifications)

The following files were copied from vllm-gr to vllm-GR with identical content, requiring no modifications:

#### Core Engine Modules (v1/)

| # | File path | Description |
|---|-----------|-------------|
| 1 | `vllm_gr/v1/engine/types.py` | BeamForkRequest type definition |
| 2 | `vllm_gr/v1/engine/core.py` | EngineCore ADD_BATCH/BEAM_FORK handling, `_handle_beam_fork` |
| 3 | `vllm_gr/v1/engine/core_client.py` | AsyncMPClient new methods (add_requests_async/beam_fork_async) |
| 4 | `vllm_gr/v1/engine/core_client_patch.py` | apply_batch_fork_patches |
| 5 | `vllm_gr/v1/engine/engine_core_patch.py` | Child process patches: BeamForkRequest injection, enum extension, Scheduler/GPUModelRunner patches |
| 6 | `vllm_gr/v1/engine/async_llm.py` | AsyncLLM beam search methods |
| 7 | `vllm_gr/v1/engine/__init__.py` | Empty file |
| 8 | `vllm_gr/v1/__init__.py` | Empty file |

#### Attention Backend

| # | File path | Description |
|---|-----------|-------------|
| 9 | `vllm_gr/v1/attention/backends/beam_attn.py` | BeamAttentionBackend full implementation |
| 10 | `vllm_gr/v1/attention/backends/__init__.py` | BeamAttentionBackend export |
| 11 | `vllm_gr/v1/attention/__init__.py` | Empty file |

#### Metrics Extension

| # | File path | Description |
|---|-----------|-------------|
| 12 | `vllm_gr/v1/metrics/stats.py` | RequestStateStats extended fields |
| 13 | `vllm_gr/v1/metrics/__init__.py` | Empty file |

#### Independent Modules

| # | File path | Description |
|---|-----------|-------------|
| 14 | `vllm_gr/logprobs.py` | FlatLogprobs optimization implementation |
| 15 | `vllm_gr/logprobs_patch.py` | patch_flat_logprobs |
| 16 | `vllm_gr/sampling_params.py` | BeamSearchParams extension |
| 17 | `vllm_gr/csrc/minheap.cpp` | C++ MinHeap (pybind11) |

#### Entrypoints Module

| # | File path | Description |
|---|-----------|-------------|
| 18 | `vllm_gr/entrypoints/llm.py` | Offline beam search (LLM.beam_search) |
| 19 | `vllm_gr/entrypoints/beam_search_patch.py` | Offline beam search patch |
| 20 | `vllm_gr/entrypoints/openai/beam_search_patch.py` | Online beam search + BEAM_FORK patch |
| 21 | `vllm_gr/entrypoints/openai/protocol.py` | to_beam_search_params |
| 22 | `vllm_gr/entrypoints/openai/serving_engine.py` | Online beam search implementation |
| 23 | `vllm_gr/entrypoints/openai/serving_models.py` | Catalog trie + init patch |

#### 3rd Party

| # | File path | Description |
|---|-----------|-------------|
| 24 | `vllm_gr/3rd_party/lmcache/` (entire directory) | LMCache adapter patch |

#### Benchmark (Direct Copy)

| # | File path | Description |
|---|-----------|-------------|
| 25 | `benchmarks/open_one_rec/__init__.py` | Package init |
| 26 | `benchmarks/open_one_rec/base_evaluator.py` | Evaluator base class |
| 27 | `benchmarks/open_one_rec/one_rec_patch.py` | vLLM benchmark monkey patch |
| 28 | `benchmarks/open_one_rec/open_one_rec_dataset.py` | OneRecDataset implementation |
| 29 | `benchmarks/open_one_rec/open_one_rec_loader.py` | Data loader |
| 30 | `benchmarks/open_one_rec/qwen3_soft_switch.jinja2` | Jinja2 template |
| 31 | `benchmarks/open_one_rec/tools/convert_catalog.py` | Catalog conversion tool |
| 32 | `benchmarks/open_one_rec/tools/create_task_catalog.py` | Catalog creation tool |
| 33 | `benchmarks/open_one_rec/tasks/item_understand/` (4 files) | item_understand task |
| 34 | `benchmarks/open_one_rec/tasks/label_pred/` (4 files) | label_pred task |
| 35 | `benchmarks/open_one_rec/tasks/rec_reason/` (4 files) | rec_reason task |
| 36 | `benchmarks/open_one_rec/tasks/recommendation/` (5 files) | recommendation task |

#### Examples (Direct Copy)

| # | File path | Description |
|---|-----------|-------------|
| 37 | `examples/online_serving/beam_search/client.sh` | Online beam search client script |
| 38 | `examples/online_serving/beam_search/README.md` | Online example docs |
| 39 | `examples/online_serving/catalog/verify_catalog.py` | Catalog verification script |
| 40 | `examples/online_serving/catalog/README.md` | Catalog example docs |
| 41 | `examples/offline_inference/beam_search/README.md` | Offline example docs |

#### Tests (Direct Copy)

| # | File path | Description |
|---|-----------|-------------|
| 42 | `tests/test_catalog.py` | Catalog unit test |
| 43 | `tests/test_dp_beam_search_unit.py` | DP beam search unit test |
| 44 | `tests/test_task_specific_catalog_program.py` | Catalog programmatic construction test |
| 45 | `tests/kernels/attention/test_cascade_beam_flash_attn.py` | Beam attention kernel test |
| 46 | `tests/resources/single_one_rec_prompt.txt` | Test resource file |

**Subtotal: 46+ files, byte-identical.**

---

### 3.2 Adapted Migration (Minor modifications)

The following files were copied from vllm-gr to vllm-GR with API adaptation changes:

| # | File path | Changes | Lines changed |
|---|-----------|---------|---------------|
| 1 | `benchmarks/open_one_rec/one_rec_main.py` | `from vllm_gr.patch import run_patch` → `re_apply_patches` | 2 lines |
| 2 | `benchmarks/open_one_rec/one_rec_acc_test.py` | `from vllm_gr.patch import run_patch` → `re_apply_patches` | 2 lines |
| 3 | `examples/offline_inference/beam_search/offline_beam_search.py` | `import vllm_gr.init` → `from vllm_gr.patch import re_apply_patches; re_apply_patches()` | 3 lines |
| 4 | `tests/test_batch_and_fork_unit.py` | Mock patch path update + `run_patch()` → `_apply_gr_beam_patches(force=True)` | 5 lines |
| 5 | `tests/test_serving_beam_search.py` | Adapted to vllm-GR test framework | Minor |
| 6 | `tests/test_offline_beam_search.py` | Adapted to vllm-GR test framework | Minor |
| 7 | `tests/test_serving_catalog.py` | Adapted to vllm-GR test framework | Minor |

**Unified adaptation pattern**: Replace vllm-gr's `run_patch()` / `import vllm_gr.init` with vllm-GR's `re_apply_patches()`, managed uniformly by the plugin entry point.

---

### 3.3 Renamed Migration

| # | Source (vllm-gr) | Target (vllm-GR) | Description |
|---|---|---|---|
| 1 | `vllm_gr/engine/arg_utils.py` | `vllm_gr/engine/arg_utils_gr.py` | Does not conflict with vllm-GR's arg_utils.py; renamed to avoid overwrite |

---

### 3.4 Newly Created Files

| # | File path | Description |
|---|-----------|-------------|
| 1 | `vllm_gr/entrypoints/openai/__init__.py` | Empty file, package init |
| 2 | `tests/conftest.py` | Test configuration (`--loops`, `--run-slow`) |

---

### 3.5 Modified vllm-GR Existing Files

| # | File path | Changes |
|---|-----------|---------|
| 1 | `vllm_gr/patch.py` | Added `_apply_gr_beam_patches()` function, called at end of `_apply_patches()`, merging 7+1 beam search patches |
| 2 | `vllm_gr/__init__.py` | Added `BeamAttentionBackend` lazy import |
| 3 | `vllm_gr/entrypoints/cli/main.py` | Added `_maybe_launch_beam_search_serve()` for beam search mode |
| 4 | `setup.py` | Added C++ minheap_cpp extension build configuration (pybind11) |
| 5 | `pyproject.toml` | Added pybind11/numpy build deps, pandas>=3.0.0, scikit-learn, pyarrow, lmcache optional dependency |

---

### 3.6 Documentation Migration

Documentation migrated from vllm-gr to vllm-GR:

| # | File path | Migration method |
|---|-----------|-----------------|
| 1 | `docs/DESIGN.md` | Direct copy |
| 2 | `docs/PRD.md` | Direct copy |
| 3 | `docs/PRE_COMMIT.md` | Direct copy |
| 4 | `docs/assets/vLLM-GR.png` | Direct copy |
| 5 | `docs/benchmarks/README.md` | Copy + update (remove `--gr`) |
| 6 | `docs/benchmarks/accuracy.md` | Copy + update (remove `--gr`) |
| 7 | `docs/benchmarks/throughput.md` | Copy + update (remove `--gr`, remove `--gr` argument table row) |
| 8 | `docs/data_parallelism.md` | Copy + update (remove `--gr`) |
| 9 | `docs/lmcache.md` | Copy + update (remove `--gr`) |
| 10 | `docs/quickstart-ascend.md` | Direct copy |
| 11 | `docs/quickstart.md` | Copy + update (remove `--gr`, update references) |
| 12 | `docs/tools/README.md` | Direct copy |

Newly created documentation:

| # | File path | Description |
|---|-----------|-------------|
| 13 | `docs/migration_plan.md` | Migration plan (Chinese) |
| 14 | `docs/migration_plan_en.md` | Migration plan (English) |
| 15 | `docs/migration_summary.md` | This document: migration completion summary (Chinese) |
| 16 | `docs/migration_summary_en.md` | This document: migration completion summary (English) |

---

### 3.7 Files Not Migrated

The following files are placeholders, already merged, or non-core:

| File | Reason |
|------|--------|
| `vllm_gr/core/scheduler.py` | Placeholder; actual patch in engine_core_patch.py |
| `vllm_gr/core/kv_cache_manager.py` | Pure placeholder |
| `vllm_gr/core/__init__.py` | Only exports placeholder classes |
| `vllm_gr/worker/__init__.py` | Pure placeholder |
| `vllm_gr/model_executor/__init__.py` | Pure placeholder |
| `vllm_gr/model_executor/models/__init__.py` | Pure placeholder |
| `vllm_gr/sample/__init__.py` | Pure placeholder |
| `vllm_gr/utils/__init__.py` | Pure placeholder |
| `vllm_gr/init.py` | Auto-patch mechanism; replaced by unified plugin management |
| `vllm_gr/patch.py` (vllm-gr version) | Functionality merged into vllm-GR's patch.py |
| `vllm_gr/__init__.py` (vllm-gr version) | Functionality merged into vllm-GR's __init__.py |
| `vllm_gr/entrypoints/__init__.py` (vllm-gr version) | Empty file; vllm-GR has its own |
| `vllm_gr/entrypoints/cli/main.py` (vllm-gr version) | Functionality merged into vllm-GR's CLI |
| `collect_env.py` | Environment collection, non-core |
| `.github/` | CI configuration, to be handled separately |

---

## 4. Patch Mechanism Merge

### Before Migration (Two Independent Patch Sets)

| | vllm-GR | vllm-gr |
|---|---------|---------|
| Activation | `vllm_gr.plugin:register` (general_plugins) | `__init__.py` import-time auto-patch + `VLLM_GR_AUTOPATCH` env var |
| Entry function | `_apply_patches()` / `re_apply_patches()` | `run_patch()` |
| Patch target | Core data types (EngineCoreRequest, Request, NewRequestData) | Core behavior methods (Scheduler.schedule, GPUModelRunner, EngineCoreProc) |

### After Migration (Unified Entry)

All patches auto-loaded via `vllm_gr.plugin:register`, no `--gr` flag or env var needed:

```
_apply_patches(force):
    _apply_gr_core_patches(force)        # vllm-GR original: core type replacement
    _apply_yuanrong_npu_patch(force)     # vllm-GR original: NPU patch
    _apply_kv_merge_patch(force)         # vllm-GR original: KV merge patch
    _apply_gr_beam_patches(force)        # [NEW] vllm-gr: all beam search patches
        ├─ patch_flat_logprobs()
        ├─ patch_beam_search()
        ├─ patch_sampling()
        ├─ patch_add_cli_args()
        ├─ patch_OpenAIServingModels_init()
        ├─ patch_offline_beam_search()
        ├─ patch_batch_and_fork()
        └─ _patch_lmcache_adapter() [Optional]
```

**Key compatibility**: The two patch sets do not overlap at all (data types vs behavior methods) and can safely coexist. Child-process-level patches are auto-applied via the `run_engine_core()` wrapper at EngineCore child process startup.

---

## 5. Test Verification Results

### 5.1 Unit Tests

| Test file | Content | Result |
|-----------|---------|--------|
| `tests/test_catalog.py` | Catalog Trie construction and lookup | Passed |
| `tests/test_batch_and_fork_unit.py` | ADD_BATCH/BEAM_FORK protocol unit test | Passed |
| `tests/test_dp_beam_search_unit.py` | DP beam search unit test | Passed |
| `tests/test_task_specific_catalog_program.py` | Catalog programmatic construction test | Passed |

### 5.2 Kernel Tests

| Test file | Content | Result |
|-----------|---------|--------|
| `tests/kernels/attention/test_cascade_beam_flash_attn.py` | Beam cascade attention kernel correctness | Passed (990 test cases) |

### 5.3 GPU End-to-End Tests

| Test file | Content | Result |
|-----------|---------|--------|
| `tests/test_serving_beam_search.py` | Online beam search API server test | Passed (2 scenarios) |
| `tests/test_offline_beam_search.py` | Offline beam search inference test | Passed (2 scenarios) |
| `tests/test_serving_catalog.py` | Catalog integration test (requires GPU) | Passed (1 scenario) |

### 5.4 Example Verification

| Example | Content | Result |
|---------|---------|--------|
| `examples/offline_inference/beam_search/` | Offline beam search inference | Passed (Vanilla vLLM: 1.04s, Patched vllm-GR: 0.54s) |
| `examples/online_serving/beam_search/` | Online beam search API | Passed (128 beam results via API) |

### 5.5 Benchmark Verification

| Content | Result |
|---------|--------|
| `label_pred` task offline inference | Passed |
| `product` task offline inference | Passed |
| `label_pred` task online inference | Passed |
| `product` task online inference | Passed |

### 5.6 HSTU/FuXi Regression Tests

| Content | Result |
|---------|--------|
| vllm-GR original 358 unit tests | All passed |
| GPU smoke test | Passed |

### 5.7 Test Summary

```
Total tests: 1354
Passed:      1354
Failed:      0
```

---

## 6. Migration Statistics

| Category | Count |
|----------|-------|
| Byte-identical direct copy files | 46+ |
| Adapted migration files | 7 |
| Renamed files | 1 |
| Newly created files | 2 |
| Modified existing files | 5 |
| Migrated documentation | 12 |
| Newly created documentation | 4 |
| Files not migrated | 15 |
| **Tests passed** | **1354 / 1354** |
