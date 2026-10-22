#!/usr/bin/env bash
GR_FFN_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GR_FFN_BUILD_DIR=${GR_FFN_BUILD_DIR:-"${GR_FFN_ROOT}/build"}
GR_FFN_BUILD_DIR=$(cd -- "${GR_FFN_BUILD_DIR}" && pwd) || return 1
GR_FFN_VENDOR_DIR=${GR_FFN_VENDOR_DIR:-"${GR_FFN_BUILD_DIR}/_CPack_Packages/Linux/External/gr_ffn.run/packages/vendors/gr_ffn"}
GR_FFN_VENDOR_DIR=$(cd -- "${GR_FFN_VENDOR_DIR}" && pwd) || return 1
if [[ ! -f "${GR_FFN_VENDOR_DIR}/op_api/lib/libgr_ffn_opapi.so" ]]; then
  printf 'Missing local Gr FFN OPAPI library in %s\n' "${GR_FFN_VENDOR_DIR}" >&2
  return 1
fi
export ASCEND_CUSTOM_OPP_PATH="${GR_FFN_VENDOR_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export GR_FFN_OPAPI_LIBRARY="${GR_FFN_VENDOR_DIR}/op_api/lib/libgr_ffn_opapi.so"
export PYTHONPATH="${GR_FFN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
