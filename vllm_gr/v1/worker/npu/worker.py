# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Ascend GR Worker; device initialization remains with vllm-ascend."""

from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.worker.worker import NPUWorker

from vllm_gr.v1.worker.gr_context import get_beam_graph_runtime_stats
from vllm_gr.v1.worker.gr_gc import freeze_gc_after_warmup
from vllm_gr.v1.worker.hot_update import reload_trie


class AscendGRWorker(NPUWorker):
    get_gr_beam_graph_runtime_stats = get_beam_graph_runtime_stats
    gr_freeze_gc_after_warmup = freeze_gc_after_warmup
    reload_trie = reload_trie

    def __init__(self, *args, **kwargs):
        from vllm_gr.platforms.ascend.config_patch import install_ascend_config_filter

        install_ascend_config_filter()
        super().__init__(*args, **kwargs)

    def init_device(self):
        from .kv_layout_patch import apply_npu_contiguous_kv_cache_patch
        from .stage_projection import install_npu_stage_weight_update_hooks
        from .startup_patch import apply_penalties_warmup_oom_guard

        if self.use_v2_model_runner:
            raise ValueError("Ascend GR requires ModelRunner V1")
        apply_npu_contiguous_kv_cache_patch()
        apply_penalties_warmup_oom_guard()
        install_npu_stage_weight_update_hooks(type(self))
        self.device = self._init_device()
        init_workspace_manager(self.device, 1)
        from .model_runner import AscendGRModelRunner

        self.model_runner = AscendGRModelRunner(self.vllm_config, self.device)
        if self.rank == 0:
            report_usage_stats(self.vllm_config)
