# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""GR ModelRunner V1; the native class remains unchanged."""

from functools import partialmethod

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import (
    _stage_graph_status,
    cancel_finished_stage,
    execute_cuda_stage,
    execute_gr_retirement,
)
from vllm_gr.v1.worker.cuda.gpu_postprocess_workspace import (
    make_load_model_with_beam_postprocess_workspace,
)
from vllm_gr.v1.worker.cuda.stage_capture import capture_cuda_v1_forward
from vllm_gr.v1.worker.cuda.v1 import gpu_model_runner_patch as gr
from vllm_gr.v1.worker.cuda.v1.gpu_beam_stage_runner_v1 import CudaStageHooksV1
from vllm_gr.v1.worker.gr_gc import freeze_gc_after_graph_capture
from vllm_gr.v1.worker.model_runner_common import make_patched_execute_model
from vllm_gr.v1.worker.model_runner_patch import (
    make_load_model_with_constraint_backend,
    make_shutdown_with_constraint_backend,
)
from vllm_gr.v1.worker.runner import load_gr_model, sample_gr_tokens


class CudaGRModelRunnerV1(GPUModelRunner):
    execute_model = make_patched_execute_model(GPUModelRunner.execute_model)
    _prepare_inputs = gr.make_patched_prepare_inputs(GPUModelRunner._prepare_inputs)
    _model_forward = gr.make_patched_model_forward(GPUModelRunner._model_forward)
    _determine_batch_execution_and_padding = gr.make_patched_determine_batch_execution_and_padding(
        GPUModelRunner._determine_batch_execution_and_padding
    )
    _vllm_gr_native_bookkeeping = GPUModelRunner._bookkeeping_sync
    sample_tokens = sample_gr_tokens(gr.make_patched_sample_tokens(GPUModelRunner.sample_tokens))
    load_model = load_gr_model(
        make_load_model_with_beam_postprocess_workspace(
            make_load_model_with_constraint_backend(GPUModelRunner.load_model)
        )
    )
    shutdown = make_shutdown_with_constraint_backend(GPUModelRunner.shutdown)
    execute_gr_stage = partialmethod(execute_cuda_stage, hooks_type=CudaStageHooksV1)
    retire_gr_stage = execute_gr_retirement
    cancel_finished_gr = cancel_finished_stage
    gr_graph_status = _stage_graph_status

    def sample_gr_stage(self, grammar_output=None):
        return self._gr_gpu_stage_runner.sample(grammar_output)

    def capture_gr_startup(self, allow_eager_fallback):
        return capture_cuda_v1_forward(
            self, hooks_type=CudaStageHooksV1, allow_eager_fallback=allow_eager_fallback
        )

    def capture_model(self, *args, **kwargs):
        result = super().capture_model(*args, **kwargs)
        from vllm_gr.v1.worker.gr_context import _get_gr_config

        if _get_gr_config(self).beam_max_width is not None:
            self.capture_gr_startup(False)
        else:
            gr.capture_beam_decode_graph(self)
        gr.capture_prefill_graphs(self)
        freeze_gc_after_graph_capture(self)
        return result
