
# Ascend NPU Migration Status

The current release candidate does not provide an accepted NPU runtime. The
upgrade target is:

| Component | Target | Current status |
| --- | --- | --- |
| vLLM | 0.30.0 | GPU validated; NPU validation pending |
| vLLM-Ascend | 0.30.0rc2 | API adaptation incomplete |
| Triton-Ascend | Undecided | Compatibility version must be established on NPU hardware |
| vllm-gr | 0.30.0+npu | Packaging shape only; runtime release is blocked |

`vllm_gr._env_check` deliberately rejects NPU startup while the supported
Triton-Ascend version is unset. `requirements/npu.txt` also does not install the
vendor runtime: the eventual NPU CI and deployment image must supply the full,
verified vLLM-Ascend stack. These guards prevent the former 0.22.1rc1 recipe
from being mistaken for a supported 0.30 installation.

NPU enablement requires all of the following before an installation procedure
can be published:

1. Select and pin the CANN, PyTorch, torch-npu, Triton-Ascend, and xllm-ops
   versions compatible with vLLM-Ascend 0.30.0rc2.
2. Update the NPU ModelRunner, attention, graph, memory-counter, and process
   lifecycle adapters against the real runtime.
3. Run NPU unit, integration, offline, online, precision, and performance gates
   on supported Ascend hardware.
4. Set the supported Triton-Ascend version only after those results are
   archived, then publish the resulting installation commands.

The rest of this page records the intended decode execution contract for that
work. It is a design reference and has not been accepted on the 0.30 NPU stack.

## Intended NPU decode execution options

Beam Search V1 uses a fixed request width equal to `beam.max_width`
(`vllm_gr_beam_max_width` in the legacy configuration encoding). A smaller
request width is rejected before submission; the NPU batch pool is not a
variable-width view. Native request state advances by one token per request
at Decode, while the NPU model executes `batch_size * beam_width` tokens.

The incremental beam decode branch on Ascend NPU has two execution modes,
selected through the engine's `additional_config`:

- `xllm` (default when the xllm-ops library from step 5 is installed): the
  fused path — a single `x_attention` kernel attends over the paged shared
  prefix and the contiguous unshared beam pool at once, and the fused
  `cache_unshared_kv` / `select_unshared_kv` operators handle pool writes and
  beam-selection reorders.
- `cascade`: the per-operator path — two
  `npu_fused_infer_attention_score_v2` calls plus the pool-to-buffer copies
  and the merge operator chain. This is the default on NPU when xllm-ops is
  missing (a log line explains why) and always the default on CUDA.

These `additional_config` examples describe the intended attention adapter
selection after NPU validation. They are not a complete canonical V1 serving
configuration and are not currently runnable release instructions. Do not
combine them with `--vllm-gr-config`; use the canonical online example for the
V1 service after the NPU gate is enabled.

```bash
# default (xllm on NPU with xllm-ops installed; cascade otherwise)
vllm serve <model> --additional-config '{"vllm_gr_attention_backend": "CUSTOM"}'

# explicit xllm fused decode (fails fast at startup when xllm-ops is missing)
vllm serve <model> --additional-config '{"vllm_gr_attention_backend": "CUSTOM", "vllm_gr_npu_decode_execution": "xllm"}'

# explicit cascade (opt out of the fused path)
vllm serve <model> --additional-config '{"vllm_gr_attention_backend": "CUSTOM", "vllm_gr_npu_decode_execution": "cascade"}'
```

Notes:

- The option is NPU-only; selecting `xllm` on a CUDA host raises at startup.
- Mixed std/beam batches and mixed sliding-window / sink / ALiBI configs are
  not supported by the fused path; the scheduler falls back to the cascade
  operators for those cases, and unsorted or shrunk beam selections always
  reorder through the cascade kernel (the fused `select_unshared_kv` only
  encodes full-width parents grouped by source beam).
