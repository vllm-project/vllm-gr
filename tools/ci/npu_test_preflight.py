# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Inspect the prepared Ascend runtime; never install or replace it."""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path


def inspect() -> dict[str, object]:
    import torch
    import torch_npu
    from vllm.platforms import current_platform

    import vllm_gr
    from vllm_gr._env_check import check_ascend_environment

    # Torch loads libc10/libtorch before the native extension is imported.
    custom_ops_lib = importlib.import_module("custom_ops_lib")
    root = Path(__file__).resolve().parents[2]
    if Path(vllm_gr.__file__).resolve().parent != root / "vllm_gr":
        raise RuntimeError("vllm_gr is imported from a different checkout")
    check_ascend_environment(require_npu_platform=True)
    if current_platform.device_type != "npu" or not torch.npu.is_available():
        raise RuntimeError("The active platform must be an available NPU")
    if torch.npu.device_count() != 1:
        raise RuntimeError("This entry requires exactly one visible NPU")
    required = (
        "rec_constrained_topk",
        "cache_unshared_kv",
        "select_unshared_kv",
        "beam_search_group",
        "beam_search_rec_final_select",
        "x_attention",
    )
    for name in required:
        if not callable(getattr(custom_ops_lib, name, None)):
            raise RuntimeError(f"Missing required custom_ops_lib export: {name}")
    if not callable(getattr(torch_npu, "npu_fused_infer_attention_score_v2", None)):
        raise RuntimeError("Missing FIA v2 export")
    torch.npu.set_device(0)
    torch.npu.synchronize()
    packages = ("torch", "torch-npu", "vllm", "vllm-ascend", "triton-ascend", "pytest")
    origins = {}
    for name in ("acl", "torch", "torch_npu", "vllm", "vllm_ascend", "vllm_gr", "custom_ops_lib"):
        origins[name] = str(importlib.import_module(name).__file__)
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: metadata.version(name) for name in packages},
        "origins": origins,
        "ascend_visible_device": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "logical_device": "npu:0",
        "device_name": torch.npu.get_device_name(0),
        "physical_mapping_verified": False,
        "mapping_note": "The caller owns allocation and container-to-host mapping verification.",
        "required_ops": list(required),
        "configuration": {
            name: os.environ.get(name)
            for name in (
                "ASCEND_TOOLKIT_HOME",
                "VLLM_USE_V2_MODEL_RUNNER",
                "VLLM_ASCEND_ENABLE_NZ",
                "VLLM_GR_NPU_FUSED_FFN",
                "VLLM_ENABLE_PREFILL_CUDAGRAPH",
                "VLLM_GR_NPU_STAGE_LM_HEAD",
                "VLLM_GR_TEST_MODEL",
            )
        },
    }


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(inspect(), indent=2) + "\n", encoding="utf-8")
