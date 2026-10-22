#!/usr/bin/env bash
# Build an isolated vendor package and the matching Python extension.
set -euo pipefail
GR_FFN_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${ASCEND_HOME_PATH:?Source the installed CANN environment first}"
GR_FFN_PYTHON=${GR_FFN_PYTHON:-python3}
GR_FFN_PYTHON=$(command -v "${GR_FFN_PYTHON}")
export PATH="$(dirname -- "${GR_FFN_PYTHON}"):${PATH}"
GR_FFN_BUILD_DIR=${GR_FFN_BUILD_DIR:-"${GR_FFN_ROOT}/build"}
mkdir -p -- "${GR_FFN_BUILD_DIR}"
GR_FFN_BUILD_DIR=$(cd -- "${GR_FFN_BUILD_DIR}" && pwd)
export CUSTOM_OPS_BUILD_JOBS=${GR_FFN_BUILD_JOBS:-4}
export CMAKE_BUILD_PARALLEL_LEVEL=${CUSTOM_OPS_BUILD_JOBS}
cmake -G "Unix Makefiles" -S "${GR_FFN_ROOT}" -B "${GR_FFN_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release -DCHECK_COMPATIBLE=OFF \
  -DPython3_EXECUTABLE="${GR_FFN_PYTHON}" \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
  -DASCEND_OP_NAME=gr_gate_up_swi_glu -DVENDOR_NAME=gr_ffn \
  -DASCEND_COMPUTE_UNIT=ascend910b
cmake --build "${GR_FFN_BUILD_DIR}" --target package -j "${CUSTOM_OPS_BUILD_JOBS}"
cd "${GR_FFN_ROOT}"
"${GR_FFN_PYTHON}" setup.py build_ext --inplace
# Runtime uses the local CPack staging tree, not a global OPP installation.
printf 'Inspect staged package, then source %s/set_env.sh\n' "${GR_FFN_ROOT}"
