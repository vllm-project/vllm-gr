# Pending upstream follow-ups

Cross-repository issues that are mitigated inside this repository but can
only be fully resolved upstream. Each entry records the symptom, the local
mitigation, and the proposed upstream change so the work can be picked up
later without re-investigation.

## vllm `_maybe_force_spawn()` ignores an initialized NPU runtime

### Symptom

When the current process has already initialized `torch_npu` (for example a
pytest session that ran NPU tests before creating an engine), building a V1
engine fails during EngineCore startup:

```text
RuntimeError: Cannot re-initialize NPU in forked subprocess. To use NPU with
multiprocessing, you must use the 'spawn' start method
```

`get_mp_context()` defaults to the `fork` start method, and the forked
EngineCore child cannot re-initialize a runtime that the parent already
initialized. The same class of problem is handled for CUDA and XPU, which is
why GPU flows never hit this.

### Root cause

`_maybe_force_spawn()` in `vllm/utils/system_utils.py` only inspects
`cuda_is_initialized()` and `xpu_is_initialized()`. NPU support lives in the
out-of-tree vllm-ascend plugin, so vllm core has no branch for it and the
default `fork` method is kept even when it cannot work. The trigger is narrow
(parent-process runtime initialization plus in-process engine creation), which
is why standard serving paths do not surface it but combined test suites do.

### Mitigation in this repository

`beam_search_test()` in `tests/test_serving_beam_search.py` forces
`VLLM_WORKER_MULTIPROC_METHOD=spawn` on NPU before building an engine
(same pattern as `tests/test_offline_beam_search.py` and the spawned-child
variant in the same file). Verified on 910B1: full UT/ST suite green.

### Proposed upstream fix (minimal)

Mirror the CUDA/XPU checks in vllm core:

```diff
--- a/vllm/utils/platform_utils.py
+++ b/vllm/utils/platform_utils.py
@@
 def xpu_is_initialized() -> bool:
     """Check if XPU is initialized."""
     if not torch.xpu._is_compiled():
         return False
     return torch.xpu.is_initialized()
+
+
+def npu_is_initialized() -> bool:
+    """Check if NPU is initialized (e.g. Ascend via torch_npu)."""
+    # torch_npu (unlike torch.cuda/torch.xpu) does not expose _is_compiled(),
+    # and torch.npu only exists once torch_npu is imported.
+    npu = getattr(torch, "npu", None)
+    if npu is None or not hasattr(npu, "is_initialized"):
+        return False
+    return npu.is_initialized()
```

```diff
--- a/vllm/utils/system_utils.py
+++ b/vllm/utils/system_utils.py
@@
     if cuda_is_initialized():
         reasons.append("CUDA is initialized")
     elif xpu_is_initialized():
         reasons.append("XPU is initialized")
+    elif npu_is_initialized():
+        reasons.append("NPU is initialized")
```

The `getattr` probe keeps the check inert on builds without `torch_npu`, so
GPU behavior is unchanged.

### Proposed upstream fix (preferred: platform hook)

Hardcoding device knowledge in core contradicts the direction that moved NPU
support out of tree. The cleaner shape is a `Platform` hook so every plugin
platform declares its own fork restriction:

```diff
--- a/vllm/platforms/interface.py
+++ b/vllm/platforms/interface.py
@@
 class Platform:
+    def is_device_runtime_initialized(self) -> bool:
+        """Whether this process already initialized the device runtime.
+
+        Runtimes that cannot be re-initialized in forked children report
+        True so multiprocessing switches to the spawn start method.
+        """
+        return False
```

```diff
--- a/vllm/utils/system_utils.py
+++ b/vllm/utils/system_utils.py
@@
     if cuda_is_initialized():
         reasons.append("CUDA is initialized")
     elif xpu_is_initialized():
         reasons.append("XPU is initialized")
+    elif current_platform.is_device_runtime_initialized():
+        reasons.append("Device runtime is initialized")
```

with the NPU implementation in vllm-ascend:

```diff
--- a/vllm_ascend/platform.py
+++ b/vllm_ascend/platform.py
@@ class NPUPlatform(Platform):
+    def is_device_runtime_initialized(self) -> bool:
+        npu = getattr(torch, "npu", None)
+        if npu is None or not hasattr(npu, "is_initialized"):
+            return False
+        return bool(npu.is_initialized())
```

Monkeypatching `_maybe_force_spawn` from the plugin is not a good alternative:
`vllm/v1/executor/multiproc_executor.py` imports it by name, so a module
attribute patch does not propagate to existing bindings.

### Action items

- [ ] Submit the platform-hook variant to vllm core and the matching NPU
      implementation to vllm-ascend (fallback: the minimal core-only diff).
- [ ] Once released, drop the NPU-only `spawn` workaround from
      `tests/test_serving_beam_search.py`.
