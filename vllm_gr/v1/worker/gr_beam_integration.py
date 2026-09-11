# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-session GPU ModelRunner V1 adapter for A3 stages and A4/A5 state."""

from copy import copy
from functools import wraps

import torch

from vllm_gr.constraints.backend import ConstraintMetadata
from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.beam.persistent_contracts import GRBeamSessionSpec, GRWorkerStageSpec
from vllm_gr.v1.engine.beam_search_v1 import GRWorkerResult
from vllm_gr.v1.worker.gr_beam_output import GRBeamFinalOutput
from vllm_gr.v1.worker.gr_persistent_beam import (
    GPUModelRunnerPersistentBeamBackend,
    GRDecodeDeviceInputs,
    GRPrefillDeviceInputs,
)


class _ConstraintAdapter:
    def __init__(self, execution, width, steps):
        self.execution, self.width, self.steps = execution, width, steps

    def select_out(self, logits, metadata, **outputs):
        prefix = metadata.prefix_token_ids
        sequence = (
            prefix.reshape(-1)[:1].view(1, 1, 1)
            if metadata.current_step == 0
            else prefix.view(1, self.width, self.steps)
        )
        return self.execution.run_out(
            logits,
            sequence,
            metadata.temperatures,
            metadata.current_step,
            metadata.top_k,
            **outputs,
        )


class GRBeamWorkerIntegration:
    def __init__(self, runner):
        from vllm.platforms import current_platform
        from vllm_gr.v1.worker.constraint_hook import get_or_create_execution_adapter
        from vllm_gr.v1.worker.gpu_model_runner_patch import initialize_beam_postprocess_workspaces
        from vllm_gr.v1.worker.model_runner_common import _create_beam_context, _get_gr_config

        if not current_platform.is_cuda():
            raise NotImplementedError("persistent beam_search_v1 requires CUDA")
        if (
            runner.parallel_config.tensor_parallel_size != 1
            or runner.parallel_config.pipeline_parallel_size != 1
        ):
            raise NotImplementedError("beam_search_v1 integration currently requires TP=PP=1")
        if runner.speculative_config is not None or runner.model_config.is_hybrid:
            raise NotImplementedError("beam_search_v1 does not support speculative/hybrid models")
        config = _get_gr_config(runner)
        context = getattr(runner, "_beam_context", None)
        if context is None:
            context = _create_beam_context(runner, config)
            runner._beam_context = context
        pool = initialize_beam_postprocess_workspaces(runner)
        if pool is None:
            raise RuntimeError("beam_search_v1 requires prewarmed Beam workspaces")
        self.runner = runner
        self.constraint = _ConstraintAdapter(
            get_or_create_execution_adapter(runner), context.max_beams, 3
        )
        self.backend = GPUModelRunnerPersistentBeamBackend(
            context=context, workspace_pool=pool, constraint_backends={"default": self.constraint}
        )
        self.output = GRBeamFinalOutput(self.backend)
        self.logits_indices = torch.arange(context.max_beams, device=runner.device)
        self.active = None
        self.pending = None

    def cleanup_finished(self, scheduler_output):
        for sid in scheduler_output.finished_req_ids:
            if sid in self.backend.registry:
                self.output.cleanup_session(sid, abort=True)
        self.output.reap_completed()

    def prepare_inputs(self):
        metadata = self.active
        if metadata is None or metadata.stage_index == 0:
            return None
        state = self.backend.registry.get(metadata.session_id)
        width = state.spec.beam_width
        self.backend.prepare_decode_inputs(
            self._stage(metadata), GRDecodeDeviceInputs(self.runner.input_ids.gpu[:width])
        )
        self.runner.positions[:width].fill_(metadata.num_prompt_tokens + metadata.stage_index - 1)
        return self.logits_indices[:width]

    @staticmethod
    def _stage(metadata):
        return GRWorkerStageSpec(
            metadata.session_id, metadata.stage_index, metadata.kind, metadata.is_last_stage
        )

    def execute(self, original, scheduler_output, *args, **kwargs):
        from vllm.v1.outputs import ModelRunnerOutput

        self.cleanup_finished(scheduler_output)
        stages = scheduler_output.gr_stage_metadata
        if len(stages) != 1 or set(stages) != set(scheduler_output.num_scheduled_tokens):
            raise NotImplementedError("beam_search_v1 requires one unmixed session")
        if self.pending is not None:
            raise RuntimeError("previous GR stage output has not been consumed")
        metadata = next(iter(stages.values()))
        sid = metadata.session_id
        if metadata.stage_index == 0:
            # Native preparation gathers BW placeholder rows before the GPU
            # adapter replaces their IDs. Reserve room for that bounded gather.
            capacity = self.runner.input_batch.token_ids_cpu.shape[1]
            required = (
                metadata.num_prompt_tokens
                + metadata.beam_params.max_tokens
                + metadata.beam_params.beam_width
                - 2
            )
            if required > capacity:
                raise ValueError("beam_search_v1 prompt exceeds native Decode buffer capacity")
        if metadata.stage_index == 0 and sid not in self.backend.registry:
            params = metadata.beam_params
            self.output.create_session(
                GRBeamSessionSpec(
                    sid,
                    sid,
                    params.beam_width,
                    params.max_tokens,
                    params.constraint_resource_id,
                    temperature=params.temperature,
                    ignore_stop_tokens=params.ignore_stop_tokens,
                ),
                metadata.output_options,
            )
        state = self.backend.registry.get(sid)
        state.validate_stage(self._stage(metadata))
        worker_output = copy(scheduler_output)
        # Match legacy constraint-prefill classification. In particular, do
        # not route cache-hit Prefill through the ordinary padded Prefill graph.
        worker_output.beam_data = {sid: {"is_constraint_prefill": True}}
        if metadata.stage_index:
            width = state.spec.beam_width
            worker_output.num_scheduled_tokens = {sid: width}
            worker_output.total_num_scheduled_tokens = width
            worker_output.beam_data = {
                sid: {
                    "is_beam_decode": True,
                    "beam_width": width,
                    "prefix_len": metadata.num_prompt_tokens,
                    "cache_len": metadata.num_prompt_tokens,
                    "beam_decode_steps": metadata.stage_index,
                    "decode_step": metadata.stage_index - 1,
                    "max_decode_steps": state.spec.max_tokens,
                }
            }
        beam_info = {
            "beam_data": worker_output.beam_data,
            "req_ids": [sid],
            "beam_session": state.beam_session,
        }
        self.active = metadata
        self.runner._current_beam_info = beam_info
        self.runner._current_scheduler_output = worker_output
        token = BEAM_DATA_VAR.set(beam_info)
        try:
            result = original(self.runner, worker_output, *args, **kwargs)
            if result is not None or self.runner.execute_model_state is None:
                raise RuntimeError("GR requires native GPU ExecuteModelState logits")
            logits = self.runner.execute_model_state.logits
            self.runner.execute_model_state = None
            if metadata.produces_output:
                stage = self._stage(metadata)
                if metadata.stage_index == 0:
                    ws = state.workspace
                    width = state.spec.beam_width
                    tokens = ws.topk_token_ids.view(-1, width)[:1]
                    scores = ws.topk_scores.view(-1, width)[:1]
                    valid = ws.topk_valid_mask.view(-1, width)[:1]
                    ws.prefix_token_ids.zero_()
                    self.constraint.select_out(
                        logits,
                        ConstraintMetadata(
                            ws.prefix_token_ids[0, :1], ws.temperatures[0, :1], 0, width
                        ),
                        out_token_ids=tokens,
                        out_logprobs=scores,
                        out_valid_mask=valid,
                        out_valid_counts=ws.topk_valid_counts.view(-1)[:1],
                        out_wire_logprobs=None,
                    )
                    self.backend.bootstrap_from_prefill(
                        stage,
                        GRPrefillDeviceInputs(tokens, scores, valid, metadata.num_prompt_tokens),
                    )
                else:
                    self.backend.advance_beam(stage, logits)
                # Native request bookkeeping carries a control placeholder only.
                batch = self.runner.input_batch
                index = batch.req_id_to_index[sid]
                position = batch.num_tokens_no_spec[index]
                batch.token_ids_cpu[index, position] = 0
                batch.is_token_ids[index, position] = True
                batch.num_tokens_no_spec[index] = position + 1
                self.runner.requests[sid].output_token_ids.append(0)
                if metadata.is_last_stage:
                    self.output.enqueue_final(sid)
            output = ModelRunnerOutput([sid], {sid: 0}, [[0]] if metadata.produces_output else [[]])
            output.gr_worker_results = {
                sid: GRWorkerResult(
                    sid,
                    metadata.stage_index,
                    int(metadata.produces_output),
                    metadata.output_device_state,
                    metadata.is_last_stage and metadata.produces_output,
                )
            }
            self.pending = (metadata, output)
            return None  # native BatchQueue consumes sample_tokens(non_block=True)
        except Exception:
            self.runner.execute_model_state = None
            self.output.cleanup_session(sid, abort=True)
            raise
        finally:
            self.active = None
            self.runner._current_beam_info = None
            BEAM_DATA_VAR.reset(token)

    def sample(self):
        metadata, output = self.pending
        self.pending = None
        if metadata.is_last_stage and metadata.produces_output:
            sid = metadata.session_id
            try:
                output.gr_batch_results = {sid: self.output.get_result(sid, wait=True)}
            finally:
                self.output.cleanup_session(sid)
        return output


def make_patched_execute_model_gpu(original):
    from vllm_gr.v1.worker.model_runner_common import make_patched_execute_model

    legacy = make_patched_execute_model(original)

    @wraps(legacy)
    def execute(self, scheduler_output, *args, **kwargs):
        integration = getattr(self, "_gr_beam_integration", None)
        if getattr(scheduler_output, "gr_stage_metadata", None):
            if integration is None:
                integration = GRBeamWorkerIntegration(self)
                self._gr_beam_integration = integration
            return integration.execute(original, scheduler_output, *args, **kwargs)
        if integration is not None:
            integration.cleanup_finished(scheduler_output)
        return legacy(self, scheduler_output, *args, **kwargs)

    return execute
