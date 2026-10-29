
# Installation for Ascend NPU

## Prerequisites

- Python 3.11 or 3.12
- Pip 23+
- CANN 9.2.0
- PyTorch / torch-npu 2.10.0
- Triton-Ascend 3.6.0
- Conda 24.11+
- 32GB+ RAM (recommended)

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/JiusiServe/vllm-gr.git
cd vllm-gr
```

2. **Create environment and install CANN**

- Follow miniconda installation procedure on [conda installation](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install) if not yet installed.
- For details about CANN installation on various platforms, please refer to [CANN installation](https://ascend.github.io/docs/sources/ascend/quick_install.html)
- This example was verified for 910B architecture

```bash
export MY_ENV=xGr-install # any valid env name will do
conda create -n ${MY_ENV}  -c conda-forge python=3.11 numactl gcc=10.3.0  gxx=10.3.0
conda activate ${MY_ENV}
```

If device is 910b install 910b ops

```bash
conda install -c https://repo.huaweicloud.com/ascend/repos/conda/ ascend-cann-toolkit==9.2.0 ascend-cann-910b-ops==9.2.0
```

If device is 910c install a3 ops

```bash
conda install -c https://repo.huaweicloud.com/ascend/repos/conda/ ascend-cann-toolkit==9.2.0 ascend-cann-a3-ops==9.2.0
```

For any other devices please follow the official [CANN installation documentation](https://ascend.github.io/docs/sources/ascend/quick_install.html).

```bash
source ${CONDA_PREFIX}/Ascend/ascend-toolkit/set_env.sh
conda install -c https://repo.huaweicloud.com/ascend/repos/conda/ ascend-cann-nnal==9.2.0
source ${CONDA_PREFIX}/Ascend/nnal/atb/set_env.sh
```

3. **Install vllm**

follow [vllm installation](https://docs.vllm.ai/en/latest/getting_started/installation) for more details.

```bash
python -m pip install vllm==0.30.0
```

This vllm-gr release supports the vLLM 0.30.0 API only.

4. **Install vllm-ascend (base environment)**

vllm-gr does **not** install vllm-ascend. Install the Ascend platform stack first:

```bash
python -m pip install vllm-ascend==0.30.0rc2
```

To install the matching release from source instead:

```bash
cd <working-dir>
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout v0.30.0rc2
pip install .
```

Return to vllm-gr folder for next steps.

`vllm_gr._env_check` pins the validated vendor pair (vLLM-Ascend 0.30.0rc2
with Triton-Ascend 3.6.0) and rejects a mismatched runtime at startup, so a
different vllm-ascend release fails fast instead of serving.

See the [vllm-ascend v0.30.0rc2 installation reference](https://github.com/vllm-project/vllm-ascend/tree/v0.30.0rc2).

5. **Install xllm-ops**

```bash
# Clone the xllm-ops repository
git clone https://github.com/xLLM-AI/xllm-ops.git

# Prepare the xllm-ops build
cd xllm-ops
git submodule init
git submodule update

# Load the CANN environment
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

6. **Install vllm-gr**

```bash
VLLM_GR_TARGET_DEVICE=npu pip install -e . --no-build-isolation
```

For developer mode:

```bash
VLLM_GR_TARGET_DEVICE=npu pip install -e ".[dev]" --no-build-isolation
```

`requirements/npu.txt` installs the shared dependencies only; the vendor
runtime comes from steps 3 and 4. Confirm that the complete Ascend runtime
pair was resolved correctly:

```bash
python -c 'import importlib.metadata as m; print("vllm", m.version("vllm")); print("vllm-ascend", m.version("vllm-ascend")); print("vllm-gr", m.version("vllm-gr"))'
python -m pip check
```

7. **Verify installation**

```bash
python -c "import vllm_gr; print(vllm_gr.__version__)"
```

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
