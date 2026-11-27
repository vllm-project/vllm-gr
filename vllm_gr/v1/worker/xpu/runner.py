# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Independent Intel XPU V2 adapter for the persistent GR session protocol."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np
import torch
from vllm.v1.outputs import ModelRunnerOutput

from vllm_gr.platforms.xpu.provider import RUNNER_ID
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchOutput,
    BeamBatchResult,
    BeamOutputSequence,
    BeamRequestOutput,
)
from vllm_gr.v1.beam.dispatch import (
    GR_RETIRE_OUTPUT_ATTR,
    GR_RETIRE_RESULT_ATTR,
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
    GRRetireResult,
    GRStageKind,
    GRStageMetadata,
    GRWorkerResult,
    validate_dispatch,
    validate_dispatch_session,
)


def prepare_native_prefill(scheduled: Any) -> Any:
    """Adapt GR arrays only at the XPU 0.30 worker's list-based staging boundary."""
    requests = [
        replace(request, prefill_token_ids=request.prefill_token_ids.tolist())
        if isinstance(request.prefill_token_ids, np.ndarray)
        else request
        for request in scheduled.scheduled_new_reqs
    ]
    return replace(scheduled, scheduled_new_reqs=requests)


def select_candidates(
    logits: torch.Tensor,
    scores: torch.Tensor,
    width: int,
    candidate_width: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize over the full vocabulary, then retain global Beam candidates."""
    scaled = logits.float()
    if temperature > 0:
        scaled = scaled / temperature
    logprobs = scaled.log_softmax(dim=-1)
    count = min(candidate_width, logits.shape[-1])
    # Stable token ordering makes tied scores reproducible across row layouts.
    tokens = torch.argsort(logprobs, dim=-1, descending=True, stable=True)[:, :count]
    values = logprobs.gather(1, tokens) + scores[:, None]
    order = torch.argsort(values.flatten(), descending=True, stable=True)[:width]
    parents = order // count
    return tokens.flatten()[order], parents, values.flatten()[order]


class XPUBeamStageRunner:
    def __init__(self, worker: Any, scheduled: Any) -> None:
        self.worker = worker
        self.session = validate_dispatch_session(scheduled)
        self.previous: GRStageMetadata | None = None
        self.pending: tuple[GRStageMetadata, torch.Tensor] | None = None
        self.execution_error: Exception | None = None
        ((_, metadata),) = getattr(scheduled, GR_STAGE_OUTPUT_ATTR).items()
        request = metadata.session_init
        if request is None or len(request.items) != 1:
            raise ValueError("XPU Beam requires one prompt per session")
        params = request.beam_params
        if params.dynamic_beam_widths is not None or params.per_parent_beam_widths is not None:
            raise ValueError("XPU Beam requires fixed width")
        self.width = params.beam_width
        self.max_tokens = params.max_tokens
        self.request = request
        self.owner_generation = metadata.owner_generation
        self.prompt_length = metadata.num_prompt_tokens
        self.prefix_blocks: tuple[int, ...] = ()
        self.layer_kv: dict[str, tuple[torch.Tensor, ...]] = {}
        self.sequence = torch.empty((1, 0), dtype=torch.long, device=worker.device)
        self.parents = torch.empty_like(self.sequence)
        self.scores = torch.zeros(1, dtype=torch.float32, device=worker.device)
        self.closed = False

    def execute(self, scheduled: Any, original: Any, *args: Any, **kwargs: Any) -> None:
        if self.execution_error is not None:
            raise RuntimeError("Previous XPU Beam dispatch failed") from self.execution_error
        if self.closed or self.pending is not None:
            raise RuntimeError("XPU Beam dispatch requires an open owner without pending sampling")
        (metadata,) = validate_dispatch(scheduled, session=self.session, previous=self.previous)
        if metadata.beam_params.beam_width != self.width:
            raise ValueError("XPU Beam width changed")
        if metadata.kind is GRStageKind.PREFILL:
            if not metadata.kv_blocks or len(metadata.kv_blocks) != 1:
                raise ValueError("XPU Beam requires published prompt KV blocks")
            self.prefix_blocks = metadata.kv_blocks[0]
            result = original(self.worker, prepare_native_prefill(scheduled), *args, **kwargs)
            state = self.worker.execute_model_state
            if result is not None or state is None or state.hidden_states is None:
                raise RuntimeError("Native XPU Prefill did not publish hidden states")
            self.worker.execute_model_state = None
            batch = state.input_batch
            logits = self.worker.model.compute_logits(state.hidden_states[batch.logits_indices])
            if logits.shape[0] != 1:
                raise RuntimeError("XPU Beam Prefill requires exactly one logit row")
            self.worker.postprocess_num_computed_tokens(batch)
        else:
            self.consume_native_state(scheduled)
            logits = self.decode(metadata)
        self.pending = (metadata, logits)

    def consume_native_state(self, scheduled: Any) -> None:
        worker = self.worker
        worker.finish_requests(scheduled)
        worker.free_states(scheduled)
        if scheduled.scheduled_new_reqs:
            raise ValueError("XPU Beam Decode cannot add native requests")
        worker.update_requests(scheduled)
        worker.block_tables.apply_staged_writes()

    def decode(self, metadata: Any) -> torch.Tensor:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        from vllm_gr.v1.attention.backends.beam_attn_xpu import XPUDecodeMetadata

        worker = self.worker
        layer_names = [
            name for groups in worker.attn_groups for group in groups for name in group.layer_names
        ]
        attention = XPUDecodeMetadata(self, metadata.stage_index - 1, self.width)
        slots = torch.full((self.width,), -1, dtype=torch.long, device=worker.device)
        positions = torch.full(
            (self.width,),
            self.prompt_length + metadata.stage_index - 1,
            dtype=torch.long,
            device=worker.device,
        )
        with set_forward_context(
            {name: attention for name in layer_names},
            worker.vllm_config,
            num_tokens=self.width,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping={name: slots for name in layer_names},
        ):
            hidden = worker.model(input_ids=self.sequence[:, -1].contiguous(), positions=positions)
        return worker.model.compute_logits(hidden)

    def sample(self, scheduled: Any) -> ModelRunnerOutput:
        if self.execution_error is not None:
            raise RuntimeError("XPU Beam execute failed before sampling") from self.execution_error
        if self.pending is None:
            raise RuntimeError("XPU Beam sampling has no matching dispatch")
        metadata, logits = self.pending
        params = metadata.beam_params
        tokens, parents, scores = select_candidates(
            logits,
            self.scores,
            self.width,
            params.candidate_width or self.width,
            params.temperature,
        )
        if tokens.numel() != self.width:
            raise ValueError("XPU Beam candidate capacity cannot supply the configured width")
        self.sequence = torch.cat((self.sequence[parents], tokens[:, None]), dim=1)
        self.parents = torch.cat((self.parents[parents], parents[:, None]), dim=1)
        self.scores = scores
        # Prefix KV is shared and immutable. Only suffix history follows parents.
        for _, _, suffix_k, suffix_v in self.layer_kv.values():
            suffix_k.copy_(suffix_k[parents])
            suffix_v.copy_(suffix_v[parents])
        terminal = self.final_output() if metadata.is_last_stage else None
        if metadata.is_last_stage:
            self.close(scheduled)
        output = ModelRunnerOutput(
            req_ids=list(metadata.request_ids),
            req_id_to_index={request_id: i for i, request_id in enumerate(metadata.request_ids)},
            sampled_token_ids=[[0]],
        )
        setattr(
            output,
            GR_WORKER_RESULT_ATTR,
            {
                metadata.session_id: GRWorkerResult(
                    metadata.session_id,
                    metadata.dispatch_id,
                    metadata.stage_index,
                    1,
                    metadata.output_device_state,
                    metadata.is_last_stage,
                    terminal,
                    resources_released=metadata.is_last_stage,
                    owner_generation=metadata.owner_generation,
                )
            },
        )
        self.previous = metadata
        self.pending = None
        return output

    def final_output(self) -> BeamBatchResult:
        options = self.request.output_options
        count = min(options.num_return_sequences, self.width)
        tokens = self.sequence[:count].cpu().tolist()
        scores = self.scores[:count].cpu().tolist()
        parents = self.parents[:count].cpu().tolist()
        sequences = tuple(
            BeamOutputSequence(
                tuple(tokens[i]),
                scores[i] if options.include_cumulative_logprob else None,
                tuple(parents[i]) if options.include_parent_beam_indices else (),
                "length",
                None,
            )
            for i in range(count)
        )
        item = self.request.items[0]
        return BeamBatchResult(
            BEAM_BATCH_PROTOCOL_VERSION,
            self.request.batch_id,
            BeamBatchOutput(
                (BeamRequestOutput(item.item_index, item.request.request_id, sequences),)
            ),
            None,
        )

    def close(self, scheduled: Any) -> None:
        if self.closed:
            return
        self.worker.finish_requests(
            replace(
                scheduled, finished_req_ids=scheduled.finished_req_ids | {self.request.batch_id}
            )
        )
        self.worker.execute_model_state = None
        self.release()

    def release(self) -> None:
        torch.xpu.synchronize(self.worker.device)
        self.layer_kv.clear()
        self.pending = None
        self.sequence = self.sequence.new_empty((0, 0))
        self.parents = self.parents.new_empty((0, 0))
        self.scores = self.scores.new_empty((0,))
        self.closed = True


def discover_worker_runners() -> tuple[tuple[str, type[Any]], ...]:
    from vllm.v1.worker.xpu_model_runner import XPUModelRunnerV2

    return (("XPUModelRunnerV2", XPUModelRunnerV2),)


def is_patched(runner_class: type[Any]) -> bool:
    from vllm_gr.v1.worker.runner_lifecycle import has_runner_lifecycle

    return has_runner_lifecycle(runner_class, RUNNER_ID)


def install() -> None:
    from vllm.v1.worker.xpu_worker import XPUWorker

    from vllm_gr.v1.worker.model_runner_patch import install_worker_utility_hooks
    from vllm_gr.v1.worker.runner_binding import (
        bind_after_model_load,
        close_backend_binding,
    )
    from vllm_gr.v1.worker.runner_lifecycle import install_runner_lifecycle

    install_worker_utility_hooks(XPUWorker)
    ((_, runner_class),) = discover_worker_runners()
    if is_patched(runner_class):
        return
    original_execute = runner_class.execute_model
    original_sample = runner_class.sample_tokens
    original_shutdown = runner_class.shutdown

    @wraps(original_execute)
    def execute(self: Any, scheduled: Any, *args: Any, **kwargs: Any) -> Any:
        retirement = getattr(scheduled, GR_RETIRE_OUTPUT_ATTR, None)
        if retirement is not None:
            owner = getattr(self, "_gr_xpu_stage_runner", None)
            if owner is not None:
                if (owner.session.session_id, owner.owner_generation) != (
                    retirement.session_id,
                    retirement.owner_generation,
                ):
                    raise RuntimeError("XPU retirement does not match the execution owner")
                owner.close(scheduled)
            self._gr_xpu_scheduled = None
            output = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
            setattr(
                output,
                GR_RETIRE_RESULT_ATTR,
                GRRetireResult(
                    retirement.session_id, retirement.owner_generation, retirement.retire_id, True
                ),
            )
            return output
        stages = getattr(scheduled, GR_STAGE_OUTPUT_ATTR, None)
        if not stages:
            return original_execute(self, scheduled, *args, **kwargs)
        ((_, metadata),) = stages.items()
        owner = getattr(self, "_gr_xpu_stage_runner", None)
        if metadata.kind is GRStageKind.PREFILL:
            if owner is not None and not owner.closed:
                raise RuntimeError("XPU Beam already has an active owner")
            owner = XPUBeamStageRunner(self, scheduled)
            self._gr_xpu_stage_runner = owner
        if owner is None:
            raise RuntimeError("XPU Beam Decode has no Prefill owner")
        self._gr_xpu_scheduled = scheduled
        try:
            return owner.execute(scheduled, original_execute, *args, **kwargs)
        except Exception as error:
            owner.execution_error = error
            raise

    @wraps(original_sample)
    def sample(self: Any, grammar_output: Any, *args: Any, **kwargs: Any) -> Any:
        scheduled = getattr(self, "_gr_xpu_scheduled", None)
        if scheduled is None:
            return original_sample(self, grammar_output, *args, **kwargs)
        try:
            output = self._gr_xpu_stage_runner.sample(scheduled)
        except Exception as error:
            self._gr_xpu_stage_runner.execution_error = error
            raise
        self._gr_xpu_scheduled = None
        return output

    @wraps(original_shutdown)
    def shutdown(self: Any, *args: Any, **kwargs: Any) -> Any:
        close_backend_binding(self)
        owner = getattr(self, "_gr_xpu_stage_runner", None)
        if owner is not None and not owner.closed:
            owner.release()
        self._gr_xpu_scheduled = None
        return original_shutdown(self, *args, **kwargs)

    bound_load = bind_after_model_load(runner_class.load_model)

    @wraps(bound_load)
    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        owner = getattr(self, "_gr_xpu_stage_runner", None)
        if owner is not None and not owner.closed:
            raise RuntimeError("XPU model reload requires retirement of the active Beam owner")
        return bound_load(self, *args, **kwargs)

    runner_class.execute_model = execute
    runner_class.sample_tokens = sample
    runner_class.shutdown = shutdown
    runner_class.load_model = load_model
    install_runner_lifecycle(runner_class, RUNNER_ID, prepare="prepare_inputs")
