# Installation for Ascend NPU

## Prerequisites

- An Ascend NPU host (verified on 910B/950) with the matching driver
- A container runtime (Docker or Podman)
- 32GB+ RAM (recommended)
- A vendor image matching the host hardware and operating system; see the
  image variants below

The vendor publishes hardware- and OS-specific image variants:

| Image tag | Hardware | Operating system |
| --- | --- | --- |
| `quay.io/ascend/vllm-ascend:v0.30.0rc2` | A2 | Ubuntu |
| `quay.io/ascend/vllm-ascend:v0.30.0rc2-openeuler` | A2 | openEuler |
| `quay.io/ascend/vllm-ascend:v0.30.0rc2-a5` | A5 | Ubuntu |
| `quay.io/ascend/vllm-ascend:v0.30.0rc2-a5-openeuler` | A5 | openEuler |

Use the unsuffixed tag for Ubuntu on A2, the `-openeuler` tag for openEuler
on A2, the `-a5` tag for Ubuntu on A5, and the `-a5-openeuler` tag for
openEuler on A5. All four variants provide vLLM-Ascend 0.30.0rc2, which is
the runtime pinned by `vllm_gr._env_check`.

The selected vendor image already contains the complete validated runtime: Python
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

Run all remaining steps inside a container based on the vendor image selected
from the [Prerequisites](#prerequisites). That image is the validated base
environment of this vllm-gr release: it supplies the complete vendor stack
(vLLM 0.30.0, vLLM-Ascend 0.30.0rc2, Triton-Ascend 3.6.0, torch /
torch-npu 2.10.0, CANN 9.2.0) together with the matching numpy and cv2 pair.

This vllm-gr release supports the vLLM 0.30.0 API only.

`vllm_gr._env_check` pins the validated vendor pair (vLLM-Ascend 0.30.0rc2
with Triton-Ascend 3.6.0) and rejects a mismatched runtime at startup, so a
different vllm-ascend release fails fast instead of serving.

Installing the vendor stack with plain pip is not a supported alternative:

- Triton-Ascend is not published on PyPI.
- The torch build used by the image has no public PyPI distribution.
- The image's opencv distribution declares `numpy>=2`, while its
  triton-ascend distribution declares `numpy==1.26.4`; those particular
  metadata requirements cannot both be satisfied by a pip resolver.

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
bash build.sh \
  -n "beam_search_group;cache_unshared_kv;select_unshared_kv;x_attention;onerec_final_beam_select;rec_constrained_top_k" \
  --no-build-transformer --no-build-flash

# Load the freshly installed custom operator library
source "${ASCEND_HOME_PATH}/opp/vendors/custom_xllm_math/bin/set_env.bash"

# Install the xllm-ops library
cd test/python_test
bash build_and_run.sh
```

Return to the vllm-gr directory before continuing.

For the optional caller-owned XAttention output binding, see
[NPU XAttention output buffers](../../features/attention.md#npu-xattention-output-binding). Its separate build
reuses the installed kernel and checks runtime versions and recorded binary
identities. Validate the rebuilt binding as described in that guide; normal
installation retains the allocating path.

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
python -c "import numpy, cv2; print('numpy', numpy.__version__, 'cv2', cv2.__version__)"
```

## Dependency checking on Ascend

### Jenkins test runtime

`tools/run_test.sh` checks the active NPU Python against the versions declared
by the checkout before running resource tests, selecting a device or loading
custom operators. The check uses `tools/ci/resolve_runtime.py`, as the GitHub
Actions runtime setup does. It reports the required and installed versions
and fails when they disagree.

An older Jenkins job or Conda environment named `vllm-gr-0.22` or
`xGr-install-22` does not become a 0.30 runtime after installing this
repository. The NPU extra intentionally leaves the vendor stack to the base
environment. Run the job inside the validated vendor image described above,
and ensure `CONDA_PREFIX`, `PATH` and the CANN environment all select that
same runtime. Do not copy an old CANN custom-operator build into the new
image; rebuild it for the image's CANN version.

Before installing GR or launching the gate, the selected Python can check
the base environment without importing vLLM or allocating an NPU:

```bash
python3 tools/ci/resolve_runtime.py resolve --device npu \
  --output /tmp/vllm-gr-npu-runtime.json
VLLM_GR_TARGET_DEVICE=npu python3 -m pip install -e ".[dev]" --no-build-isolation
bash tools/run_test.sh
```

GPU-only attention tests skip on the NPU target before importing CUDA-only
vLLM APIs. That device boundary does not make an older vendor runtime
compatible; the runtime check remains required.

### Vendor package metadata

Keep the output of `python -m pip check` as an environment diagnostic. The
validated image has known metadata conflicts, so that command alone cannot
establish whether this vendor runtime works. Recorded reports include:

- Declaration conflict: the image's opencv-python-headless 5.x wheel declares
  `numpy>=2` while triton-ascend 3.6.0 declares `numpy==1.26.4`. The
  successful runtime validation used the vendor-supplied cv2 build. Confirm
  its actual import with NumPy 1.26.4; a version declaration or vLLM's lazy
  import fallback is not evidence of binary compatibility.
- Metadata drift: the image's vllm-ascend metadata declares slightly older
  companions (triton-ascend 3.2.2, torch-npu 2.10.0.post4) than the image
  actually ships (3.6.0, post7).

Treat these recorded vendor-image discrepancies separately from new missing
or incompatible dependencies. Validate runtime versions, direct numpy/cv2
imports, custom operator loading, and inference inside the selected image.
Do not ignore unrelated `pip check` errors or replace vendor packages merely
to silence the known reports. Package availability on PyPI and consistency
of installed package metadata are separate questions.

## Quick Start

- To [run OneRec offline on Ascend](https://github.com/JiusiServe/vllm-gr/blob/main/examples/README.md#run-onerec-offline),
  prepare the model first, then run the Ascend command on the examples page.
- Follow the [online V1 serving example](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/beam_search/README.md)
  with `gr.constraint_table.backend: npu` and the NPU execution settings below.

## NPU decode execution options

Beam Search V1 uses a fixed request width equal to the first entry of
`gr.beam_width`. A smaller
request width is rejected before submission; the NPU batch pool is not a
variable-width view. Native request state advances by one token per request
at Decode, while the NPU model executes `batch_size * beam_width` tokens.

The incremental beam decode branch on Ascend NPU has two execution modes,
selected through `gr.npu_decode_execution`:

Both modes require an active Beam session and persistent suffix KV pool.
The old paged-suffix recompute path has been removed: missing session state
or invalid Decode metadata raises an error. The `xllm` fallback to `cascade`
still uses incremental KV; ordinary requests, Prefill and standard prefix
segments retain their paged-cache execution.

- `xllm` (default when the xllm-ops library is installed): the
  fused path — a single `x_attention` kernel attends over the paged shared
  prefix and the contiguous unshared beam pool at once, and the fused
  `cache_unshared_kv` / `select_unshared_kv` operators handle pool writes and
  beam-selection reorders.
- `cascade`: the per-operator path — two
  `npu_fused_infer_attention_score_v2` calls plus the pool-to-buffer copies
  and the merge operator chain. This is the default on NPU when xllm-ops is
  missing (a log line explains why) and always the default on CUDA.

Use the [OneRec serving configuration](https://github.com/JiusiServe/vllm-gr/blob/main/examples/online_serving/catalog/README.md#write-the-server-configuration)
as the base YAML. Omit `gr.npu_decode_execution` to use automatic selection.

```bash
# explicit xllm fused decode (fails fast at startup when xllm-ops is missing)
vllm serve /path/to/model --gr --config /path/to/config.yaml --gr.npu_decode_execution xllm

# explicit cascade (opt out of the fused path)
vllm serve /path/to/model --gr --config /path/to/config.yaml --gr.npu_decode_execution cascade
```

Notes:

- The option is NPU-only; selecting `xllm` on a CUDA host raises at startup.
- NPU `CUSTOM` Beam Attention rejects sliding windows, ALiBi slopes, and
  nonzero Attention logits softcap at initialization. Neither `xllm` nor
  `cascade` implements these features. Sink attention and FP8 KV caches are
  also unsupported by this backend. These restrictions do not describe GPU
  or native vLLM-Ascend attention backends.
- Mixed std/beam batches use the cascade implementation. Unsorted or shrunk
  beam selections reorder through the cascade kernel because the fused
  `select_unshared_kv` only encodes full-width parents grouped by source beam.

## Optional BSND Prefill graphs

For full prompts without cached history, the opt-in [BSND Prefill graph path](../../features/attention.md#npu-bsnd-prefill) avoids per-layer FIA parameter updates. It is disabled by default and requires no import overlay.
