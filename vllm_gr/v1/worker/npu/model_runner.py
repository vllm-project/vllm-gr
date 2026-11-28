# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Ascend GR Runner with native allocation and Beam-specific execution."""

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

from vllm_gr.v1.beam.batch_contracts import SUPPORTED_BEAM_BATCH_SIZES
from vllm_gr.v1.worker.gr_gc import freeze_gc_after_graph_capture
from vllm_gr.v1.worker.model_runner_common import make_patched_execute_model
from vllm_gr.v1.worker.model_runner_patch import (
    make_load_model_with_constraint_backend,
    make_shutdown_with_constraint_backend,
)
from vllm_gr.v1.worker.runner import load_gr_model, sample_gr_tokens

from . import npu_model_runner_patch as gr
from .npu_beam_stage_runner import (
    _stage_graph_status,
    capture_ascend_forward,
    execute_npu_gr_retirement,
    execute_npu_stage,
)


class AscendGRModelRunner(NPUModelRunner):
    execute_model = make_patched_execute_model(NPUModelRunner.execute_model)
    _update_states = gr.make_patched_update_states(NPUModelRunner._update_states)
    _prepare_inputs = gr.make_patched_prepare_inputs(NPUModelRunner._prepare_inputs)
    _model_forward = gr.make_patched_model_forward(NPUModelRunner._model_forward)
    _determine_batch_execution_and_padding = gr.make_patched_determine_batch_execution_and_padding(
        NPUModelRunner._determine_batch_execution_and_padding
    )
    _vllm_gr_native_bookkeeping = NPUModelRunner._bookkeeping_sync
    sample_tokens = sample_gr_tokens(gr.make_patched_sample_tokens(NPUModelRunner.sample_tokens))
    load_model = load_gr_model(
        gr.make_patched_load_model(
            make_load_model_with_constraint_backend(
                NPUModelRunner.load_model, max_batch=max(SUPPORTED_BEAM_BATCH_SIZES)
            )
        )
    )
    shutdown = gr.make_patched_shutdown(
        make_shutdown_with_constraint_backend(getattr(NPUModelRunner, "shutdown", None))
    )
    reload_weights = gr.make_patched_reload_weights(NPUModelRunner.reload_weights)
    execute_gr_stage = execute_npu_stage
    retire_gr_stage = execute_npu_gr_retirement
    capture_gr_startup = capture_ascend_forward
    gr_graph_status = _stage_graph_status

    def sample_gr_stage(self, grammar_output=None):
        return self._gr_npu_stage_runner.sample(grammar_output)

    def cancel_finished_gr(self, finished):
        # Ascend releases the owner through the explicit retirement frame.
        pass

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
