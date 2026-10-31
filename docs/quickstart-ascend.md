# Installation for Ascend NPU

## Prerequisites

- An Ascend NPU host (verified on 910B) with the matching driver
- A container runtime (Docker or Podman)
- The vendor image `quay.io/ascend/vllm-ascend:v0.30.0rc2`
- 32GB+ RAM (recommended)

The vendor image already contains the complete validated runtime: Python
3.12, CANN 9.2.0, PyTorch / torch-npu 2.10.0, Triton-Ascend 3.6.0, vLLM
0.30.0, vLLM-Ascend 0.30.0rc2, and the matching numpy / cv2 pair. No part
of that stack is installed by hand.

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/JiusiServe/vllm-gr.git
cd vllm-gr
```

2. **Enter the validated base environment**

Run all remaining steps inside a container based on the vendor image
`quay.io/ascend/vllm-ascend:v0.30.0rc2`. That image is the validated base
environment of this vllm-gr release: it supplies the complete vendor stack
(vLLM 0.30.0, vLLM-Ascend 0.30.0rc2, Triton-Ascend 3.6.0, torch /
torch-npu 2.10.0, CANN 9.2.0) together with the matching numpy and cv2
pair, and the NPU runtime validation of this release was performed inside
it.

This vllm-gr release supports the vLLM 0.30.0 API only.

`vllm_gr._env_check` pins the validated vendor pair (vLLM-Ascend 0.30.0rc2
with Triton-Ascend 3.6.0) and rejects a mismatched runtime at startup, so a
different vllm-ascend release fails fast instead of serving.

Installing the vendor stack with plain pip is not a supported alternative:

- Triton-Ascend is not published on PyPI.
- The torch build used by the image has no public PyPI distribution.
- Every opencv wheel for Python 3.9+ declares `numpy>=2`, while
  triton-ascend declares `numpy==1.26.4`; no single pip resolution can
  satisfy both.

See the [vllm-ascend v0.30.0rc2 installation reference](https://github.com/vllm-project/vllm-ascend/tree/v0.30.0rc2).

3. **Install xllm-ops**

```bash
# Clone the xllm-ops repository
git clone https://github.com/xLLM-AI/xllm-ops.git

# Prepare the xllm-ops build
cd xllm-ops
git submodule init
git submodule update

# Load the CANN environment supplied by the base image
source "${ASCEND_TOOLKIT_HOME}/set_env.sh"

# Build the operator subset vllm-gr calls. CANN 9.2.0's op store ships a
# built-in CausalConv1d that collides with xllm-ops' custom operator of the
# same type name, so build only the required operators instead of the full
# set; the --no-build-* flags skip the unrelated third-party operator
# libraries.
bash build.sh -c ascend910b \
  -n "beam_search_group;cache_unshared_kv;select_unshared_kv;x_attention;onerec_final_beam_select;rec_constrained_top_k" \
  --no-build-transformer --no-build-flash

# Load the freshly installed custom operator library
source "${ASCEND_HOME_PATH}/opp/vendors/custom_xllm_math/bin/set_env.bash"

# Install the xllm-ops library
cd test/python_test
bash build_and_run.sh
```

Return to the vllm-gr directory before continuing.

4. **Install vllm-gr**

```bash
VLLM_GR_TARGET_DEVICE=npu pip install -e . --no-build-isolation
```

For developer mode:

```bash
VLLM_GR_TARGET_DEVICE=npu pip install -e ".[dev]" --no-build-isolation
```

`requirements/npu.txt` installs the shared dependencies only; the vendor
runtime comes from the base image. Confirm that the complete Ascend runtime
pair was resolved correctly:

```bash
python -c 'import importlib.metadata as m; print("vllm", m.version("vllm")); print("vllm-ascend", m.version("vllm-ascend")); print("vllm-gr", m.version("vllm-gr"))'
```

5. **Verify installation**

```bash
python -c "import vllm_gr; print(vllm_gr.__version__)"
python -c "from vllm_gr._env_check import check_environment; print(check_environment())"
```

## Dependency checking on Ascend

`python -m pip check` is not a validity criterion on this platform. The
validated image is assembled stepwise, so pip never re-validates the final
combination, and two classes of pre-existing reports are expected:

- Declaration conflict: the image's opencv-python-headless 5.x wheel declares
  `numpy>=2` while triton-ascend 3.6.0 declares `numpy==1.26.4`. The
  conflict is metadata-only at runtime: the image's cv2 binary is built
  against numpy 2 headers and is ABI-compatible with the installed numpy
  1.26.4, and vllm loads cv2 lazily with a placeholder fallback.
- Metadata drift: the image's vllm-ascend metadata declares slightly older
  companions (triton-ascend 3.2.2, torch-npu 2.10.0.post4) than the image
  actually ships (3.6.0, post7).

On Ascend the acceptance criterion for a vllm-gr environment is therefore
that it runs correctly inside the validated vendor image, not pip-level
dependency consistency; the latter is unreachable while Triton-Ascend is
not published on PyPI.

## Quick Start

Refer to [quickstart](quickstart.md#quick-start) for deployment.

After installation, use the [online V1 serving example](../examples/online_serving/beam_search/README.md)
with `constraint_table.backend: npu` and the NPU execution settings below.

## NPU decode execution options

Beam Search V1 uses a fixed request width equal to `beam.max_width`
(`vllm_gr_beam_max_width` in the legacy configuration encoding). A smaller
request width is rejected before submission; the NPU batch pool is not a
variable-width view. Native request state advances by one token per request
at Decode, while the NPU model executes `batch_size * beam_width` tokens.

The incremental beam decode branch on Ascend NPU has two execution modes,
selected through the engine's `additional_config`:

- `xllm` (default when the xllm-ops library is installed): the
  fused path — a single `x_attention` kernel attends over the paged shared
  prefix and the contiguous unshared beam pool at once, and the fused
  `cache_unshared_kv` / `select_unshared_kv` operators handle pool writes and
  beam-selection reorders.
- `cascade`: the per-operator path — two
  `npu_fused_infer_attention_score_v2` calls plus the pool-to-buffer copies
  and the merge operator chain. This is the default on NPU when xllm-ops is
  missing (a log line explains why) and always the default on CUDA.

These `additional_config` examples select the attention adapter and the NPU
decode execution mode only. They are not a complete canonical V1 serving
configuration. Do not combine them with `--vllm-gr-config`; use the canonical
online example for the V1 service.

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

## Optional BSND Prefill graphs

For full prompts without cached history, the opt-in [BSND Prefill graph path](npu_prefill_bsnd.md) avoids per-layer FIA parameter updates. It is disabled by default and requires no import overlay.
