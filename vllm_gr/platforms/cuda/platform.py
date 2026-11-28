# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA platform extensions for GR."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

from vllm.platforms.cuda import CudaPlatform, CudaPlatformBase

from .. import cuda_platform_plugin
from ..interface import BeamAttentionExecution, GRPlatform

if TYPE_CHECKING:
    from vllm_gr.v1.worker.prefill_graph_common import BasePrefillGraphRunner


class CudaGRPlatform(GRPlatform, CudaPlatform):
    native_platform_type = CudaPlatformBase
    backend_id = "cuda"
    beam_attention_layout = "cuda"
    canonical_constraint_backend = "cuda"
    default_use_v2_model_runner = False
    default_worker_decision = True
    supports_unconstrained = True
    supports_dynamic_width = True

    def get_gr_worker_cls(self) -> str:
        return "vllm_gr.v1.worker.cuda.worker.CudaGRWorker"

    def check_and_update_gr_config(self, config) -> None:
        from vllm_gr.config import GRConfig

        gr = GRConfig.from_vllm_config(config)
        if gr.npu_decode_execution == "xllm":
            raise ValueError("CUDA does not support npu_decode_execution='xllm'")
        if gr.beam_sid_stage_vocab:
            raise ValueError("beam.sid_stage_vocab requires NPU ModelRunner V1")
        compilation = getattr(config, "compilation_config", None)
        if (
            gr.beam_max_width is not None
            and not getattr(config, "use_v2_model_runner", False)
            and compilation is not None
            and compilation.backend == "inductor"
        ):
            compilation.inductor_compile_config.setdefault("dynamic_scale_rblock", False)

    def check_environment(self) -> dict[str, str]:
        from ..compat import check_vllm_runtime

        if cuda_platform_plugin() is None:
            raise RuntimeError("CUDA integration requires the native CUDA platform")

        report = check_vllm_runtime()
        report["platform"] = "cuda"
        return report

    def get_attention_execution(self) -> BeamAttentionExecution:
        return cast(
            BeamAttentionExecution,
            import_module("vllm_gr.v1.attention.backends.beam_attn_gpu"),
        )

    def get_prefill_graph_runner_class(self) -> type[BasePrefillGraphRunner]:
        from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import (
            GPUPrefillGraphRunner,
        )

        return GPUPrefillGraphRunner
