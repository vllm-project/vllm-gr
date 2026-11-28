# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""GR ModelRunner V2, reusing vLLM request state and native graph execution."""

from functools import partialmethod

from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_gr.v1.beam.batch_contracts import SUPPORTED_BEAM_BATCH_SIZES
from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import (
    _stage_graph_status,
    cancel_finished_stage,
    execute_cuda_stage,
    execute_gr_retirement,
)
from vllm_gr.v1.worker.cuda.gpu_postprocess_workspace import (
    make_load_model_with_beam_postprocess_workspace,
)
from vllm_gr.v1.worker.cuda.v2 import gpu_model_runner_v2_patch as gr
from vllm_gr.v1.worker.cuda.v2 import gpu_model_runner_v2_resources as resources
from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import CudaStageHooksV2
from vllm_gr.v1.worker.model_runner_patch import (
    make_load_model_with_constraint_backend,
    make_shutdown_with_constraint_backend,
)
from vllm_gr.v1.worker.runner import load_gr_model, sample_gr_tokens


class CudaGRModelRunnerV2(GPUModelRunner):
    execute_model = gr.make_patched_execute_model_v2(GPUModelRunner.execute_model)
    prepare_inputs = gr.make_patched_prepare_inputs_v2(GPUModelRunner.prepare_inputs)
    add_requests = gr.make_patched_add_requests_v2(GPUModelRunner.add_requests)
    sample_tokens = sample_gr_tokens(gr.make_patched_sample_tokens_v2(GPUModelRunner.sample_tokens))
    capture_model = gr.make_patched_capture_model_v2(GPUModelRunner.capture_model)
    initialize_kv_cache = gr.make_patched_initialize_kv_cache_v2(GPUModelRunner.initialize_kv_cache)
    profile_cudagraph_memory = gr.make_patched_profile_cudagraph_memory_v2(
        GPUModelRunner.profile_cudagraph_memory
    )
    load_model = load_gr_model(
        make_load_model_with_constraint_backend(
            make_load_model_with_beam_postprocess_workspace(GPUModelRunner.load_model),
            max_batch=max(SUPPORTED_BEAM_BATCH_SIZES),
            prepare_backend=gr._warmup_constraint_backend,
        ),
        close_runtime=resources.close_v2_runtime,
    )
    shutdown = make_shutdown_with_constraint_backend(
        GPUModelRunner.shutdown, close_runtime=resources.close_v2_runtime
    )
    execute_gr_stage = partialmethod(execute_cuda_stage, hooks_type=CudaStageHooksV2)
    retire_gr_stage = execute_gr_retirement
    cancel_finished_gr = cancel_finished_stage
    gr_graph_status = _stage_graph_status

    def sample_gr_stage(self, grammar_output=None):
        return self._gr_gpu_stage_runner.sample(grammar_output)

    def capture_gr_startup(self, allow_eager_fallback):
        return resources.capture_multibatch_forward(self, allow_eager_fallback=allow_eager_fallback)
