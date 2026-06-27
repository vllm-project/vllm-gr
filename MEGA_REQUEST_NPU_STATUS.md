# Mega Request Beam Search - NPU Support Status

**Document Date:** 2026-06-27
**Base Commit:** `ee771805e91791f3cd2dff77c1e841846fbc8ca1`
**HEAD:** Current HEAD

## Summary of Changes

4 files changed: +1077 insertions, -342 deletions

---

## 1. `vllm_gr/patch.py` (+147 lines)

### New: `_apply_ascend_attention_backend_patch()`

**Purpose:** Preserve CUSTOM attention backend selection on Ascend NPU

The vLLM-ascend platform plugin resets any user-specified attention backend to `None` during `check_and_update_config()`. This patch bypasses the platform's selector by replacing `vllm.v1.attention.selector._cached_get_attn_backend` to directly return `BeamAttentionBackend` when `--attention-backend CUSTOM` is specified.

**Key Mechanism:**
- Uses `VLLM_GR_BEAM_ATTN` environment variable for cross-process signaling
- Main process sets env-var based on CLI args (`--attention-backend CUSTOM`)
- Child processes inherit the env-var, activating the selector patch automatically

```python
def _patched_cached_get_attn_backend(backend, attn_selector_config):
    if os.environ.get(_VLLM_GR_BEAM_ATTN_ENV) == "1":
        from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend
        return BeamAttentionBackend
    return original_cached_fn(backend, attn_selector_config)
```

---

## 2. `vllm_gr/v1/attention/backends/beam_attn.py` (+749/-70 lines)

### 2.1 Device Detection

Added NPU/CUDA device detection at module load:
```python
from vllm_gr.utils.npu_adapter import is_npu_available, is_cuda_available
NPU_DEVICE = is_npu_available()
CUDA_DEVICE = is_cuda_available()
```

### 2.2 Removed Triton Dependency on NPU

**Before:** Used `paged_kv_to_contig_suffix_kernel` (Triton JIT)
**After:** Pure PyTorch `extract_suffix_kv()` using vectorized advanced indexing

### 2.3 New: NPU-Specific Attention Path

The `cascade_attention()` method now has two code paths:

#### CUDA Path (Original)
- Uses `flash_attn_varlen_func` for standard and mega attention
- Uses `merge_attn_states` for combining prefix/suffix outputs

#### NPU Path (New)
- Uses `torch_npu.npu_fused_infer_attention_score_v2` for standard attention
- Uses `extract_suffix_kv_pytorch` (pure PyTorch) for gathering suffix KV
- Uses `merge_attn_states_pytorch` for combining prefix/suffix outputs

### 2.4 New `BeamAttentionMetadata` Fields

```python
block_size: int = 16
num_heads: int = 0
num_kv_heads: int = 0
mega_prefix_cu_seqlens_kv: torch.Tensor | None = None  # NEW: KV seqlens for mega prefix
```

### 2.5 NPU KV Cache Handling

**Key difference:** NPU provides `kv_cache` as a tuple `(key_cache, value_cache)` while GPU provides a stacked tensor `[2, num_blocks, ...]`

```python
if isinstance(kv_cache, tuple):
    key_cache, value_cache = kv_cache[0], kv_cache[1]
else:
    key_cache, value_cache = kv_cache.unbind(0)
```

### 2.6 NPU `_reshape_and_cache` Call

Replaced `reshape_and_cache_flash` with `torch_npu._npu_reshape_and_cache` for NPU.

### 2.7 Unsupported Features on NPU

The following return `False` or raise `NotImplementedError` on NPU:
- FP8 KV cache
- Attention sinks
- CUDA-style compute capability checks (NPU returns `True` unconditionally)

---

## 3. `vllm_gr/v1/attention/backends/merge_tnd_attention_outputs.py` (NEW FILE, +117 lines)

### Pure PyTorch Replacements for NPU

#### `extract_suffix_kv_pytorch()`
Replaces the Triton `paged_kv_to_contig_suffix_kernel` using vectorized advanced indexing:
```python
resolved_slots = slot_mapping[slot_mapping_out.long()]
k_flat = k_cache.view(-1, num_heads, head_dim)
out_k = k_flat[resolved_slots]  # Vectorized gather
```

#### `merge_tnd_attention_outputs()`
Normalizes and merges two TND attention output pairs using weighted average:
```python
max_lse = torch.maximum(lse1, lse2)
w1 = torch.exp(lse1 - max_lse)
w2 = torch.exp(lse2 - max_lse)
normalized_out = (w1 * out1 + w2 * out2) / (w1 + w2)
```

#### `merge_attn_states_pytorch()`
Exact replacement for the C++ `merge_attn_states` kernel:
```python
lse_p = prefix_lse.t().unsqueeze(-1)  # [num_heads, total_q] -> [total_q, num_heads, 1]
weight_p = torch.exp(lse_p - max_lse)
out.copy_((prefix_out * weight_p + suffix_out * weight_s) / weight_sum)
```

---

## 4. `vllm_gr/v1/engine/engine_core_patch.py` (+406/-192 lines)

### 4.1 `apply_worker_patches()` - Dual Runner Support

**Before:** Only patched `GPUModelRunner`
**After:** Patches both `GPUModelRunner` and `NPUModelRunner` based on platform

```python
npu_available = is_npu_available()
if npu_available:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    runners_to_patch.append(("NPUModelRunner", NPUModelRunner))
```

### 4.2 Patched Methods by Runner

| Method | GPUModelRunner | NPUModelRunner |
|--------|---------------|----------------|
| `execute_model` | ✅ Patched | ✅ Patched |
| `_prepare_inputs` | ✅ Patched | ✅ Patched (NPU-specific wrapper, pending validation) |
| `_bookkeeping_sync` | ✅ Patched | ✅ Patched (NPU-specific wrapper, pending validation) |

### 4.3 NPU-specific wrappers (implemented)

The GPU and NPU runner methods have different signatures, so each gets its own
thin wrapper around shared, device-neutral helpers (`_mega_remap_inputs`,
`_mega_collapse_bookkeeping`). The earlier `aclnnInplaceDiv ... 161002` error
was a symptom of applying the GPU wrapper verbatim (its 2-tuple return / wrong
positional binding produced bad `logits_indices`, so the sampler divided
`logits` by a mis-shaped `temperature`) — **not** an inherent incompatibility.

**GPU `_prepare_inputs`:** `(scheduler_output, num_scheduled_tokens) -> (logits_indices, spec_decode_metadata)`

**NPU `_prepare_inputs`:** `(scheduler_output, intermediate_tensors=None) -> (attn_metadata, num_scheduled_tokens, num_input_tokens, num_tokens_across_dp, logits_indices, spec_decode_metadata, max_num_scheduled_tokens)` — 7 values

**GPU `_bookkeeping_sync`:** returns 7 values including `num_nans_in_logits`

**NPU `_bookkeeping_sync`:** returns 6 values (no `num_nans_in_logits`)

Key NPU-specific handling (see `make_patched_prepare_inputs_npu` /
`make_patched_bookkeeping_sync_npu` in `engine_core_patch.py`):

- `req_ids` is published to `MEGA_DATA_VAR` **before** delegating, because the
  NPU attention builder runs *inside* `_prepare_inputs` (the GPU builder runs
  later).
- `_prepare_inputs` (in `execute_model`) and `_bookkeeping_sync` (in the
  separate `sample_tokens` call) straddle two method calls; a defensive
  un-proxy at `_prepare_inputs` entry guards against a stale proxy if an early
  return skips bookkeeping.
- `InputBatchProxy` now also intercepts `num_tokens` (NPU's writeback loop
  updates it; GPU's does not) — inert on GPU.
- Per-beam RoPE positions are written to `self.positions.gpu` in place; no extra
  `copy_to_gpu` is needed and the physical `slot_mapping` (materialized from
  linear positions inside the original) is unaffected.

---

## Current Status

### ✅ Working on NPU
- `execute_model` patch sets up `MEGA_DATA_VAR` context
- `BeamAttentionBackend` selection via custom selector patch
- Standard attention via `npu_fused_infer_attention_score_v2`
- Prefix/suffix KV extraction via pure PyTorch

### 🧪 Implemented, pending NPU validation
- `_prepare_inputs` / `_bookkeeping_sync` NPU wrappers (this change)
- Mega request output processing end-to-end on NPU

### Validation TODO (run on server, async scheduling OFF first)
- `beam_width=2`, single decode step → multi-step accuracy vs GPU / baselines
  (`accuracy_flat_logprobs.txt`, `accuracy_b4_flat_logprobes.txt`).
- Confirm `sampler_output.logprobs_tensors.resize_()` is legal on NPU
  internal-format tensors; if not, switch the in-place remux to allocate+reassign.
- Verify async-scheduling path (`AsyncGPUModelRunnerOutput`) sees the remuxed
  tensor shapes.

See `MEGA_REQUEST_NPU_PREPARE_BOOKKEEPING_PLAN.md` for the full design.

---

## Key Differences Between GPU and NPU Code Paths

| Aspect | GPU | NPU |
|--------|-----|-----|
| Attention | `flash_attn_varlen_func` | `npu_fused_infer_attention_score_v2` |
| KV Gather | Triton kernel | Pure PyTorch |
| Merge | C++ `merge_attn_states` | `merge_attn_states_pytorch` |
| KV Cache Format | `[2, num_blocks, block_size, ...]` | Tuple `(key_cache, value_cache)` |
| Compute Capability Check | Required >= 8.0 | Bypassed |

---

## Files Changed

```
vllm_gr/patch.py                                   |  147 +++-
vllm_gr/v1/attention/backends/beam_attn.py         |  749 ++++++++++++++++-----
vllm_gr/v1/attention/backends/merge_tnd_attention_outputs.py |  117 ++++
vllm_gr/v1/engine/engine_core_patch.py             |  406 ++++++-----
4 files changed, 1077 insertions(+), 342 deletions(-)
```