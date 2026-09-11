# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import deque

import torch

from vllm_gr.constraints.backend import ConstraintOutput
from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
from vllm_gr.v1.beam.beam_decision import (
    BeamPostprocessWorkspace,
    BeamPostprocessWorkspacePool,
)
from vllm_gr.v1.beam.persistent_contracts import (
    GRBeamSessionSpec,
    GRStageKind,
    GRWorkerStageSpec,
)
from vllm_gr.v1.worker.gr_persistent_beam import (
    GRPrefillDeviceInputs,
    GPUModelRunnerPersistentBeamBackend,
)


class _QueuedConstraintBackend:
    def __init__(self, outputs: list[ConstraintOutput]) -> None:
        self.outputs = deque(outputs)
        self.metadata = []

    @property
    def info(self):
        return None

    def select(self, logits, metadata) -> ConstraintOutput:
        del logits
        self.metadata.append(metadata)
        return self.outputs.popleft()

    def close(self) -> None:
        pass


def _context(
    *,
    device: torch.device = torch.device("cpu"),
    max_batch: int = 1,
    width: int = 2,
    max_tokens: int = 3,
) -> BeamAttentionContext:
    return BeamAttentionContext(
        max_batch=max_batch,
        max_beams=width,
        num_kv_heads=1,
        max_decode_steps=max_tokens,
        head_dim=8,
        num_layers=1,
        device=device,
        dtype=torch.float16,
    )


def _workspace_pool(
    context: BeamAttentionContext,
    *,
    width: int,
    max_tokens: int,
) -> BeamPostprocessWorkspacePool:
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=context.max_batch,
        beam_width=width,
        top_k=width,
        max_decode_steps=max_tokens,
        device=context.device,
        logits_dtype=context.dtype,
        pin_host=False,
    )
    pool = BeamPostprocessWorkspacePool()
    pool.add(workspace)
    return pool


def _spec(
    *,
    session_id: str = "session-0",
    width: int = 2,
    max_tokens: int = 3,
) -> GRBeamSessionSpec:
    return GRBeamSessionSpec(
        session_id=session_id,
        request_id=f"request-{session_id}",
        beam_width=width,
        max_tokens=max_tokens,
        constraint_handle="default",
        temperature=1.0,
        ignore_stop_tokens=True,
    )


def _stage(session_id: str, index: int, max_tokens: int) -> GRWorkerStageSpec:
    return GRWorkerStageSpec(
        session_id=session_id,
        stage_index=index,
        stage_kind=GRStageKind.PREFILL if index == 0 else GRStageKind.DECODE,
        is_last_stage=index == max_tokens - 1,
    )


def _output(tokens: list[list[int]], scores: list[list[float]]) -> ConstraintOutput:
    token_ids = torch.tensor(tokens, dtype=torch.int32)
    logprobs = torch.tensor(scores, dtype=torch.float32)
    return ConstraintOutput(
        token_ids,
        logprobs,
        torch.ones_like(token_ids, dtype=torch.bool),
    )


def _backend(
    outputs: list[ConstraintOutput],
    *,
    context: BeamAttentionContext | None = None,
) -> tuple[GPUModelRunnerPersistentBeamBackend, _QueuedConstraintBackend]:
    context = context or _context()
    constraints = _QueuedConstraintBackend(outputs)
    backend = GPUModelRunnerPersistentBeamBackend(
        context=context,
        constraint_backends={"default": constraints},  # type: ignore[dict-item]
        workspace_pool=_workspace_pool(
            context,
            width=context.max_beams,
            max_tokens=context.max_decode_steps,
        ),
    )
    return backend, constraints


def _prefill_inputs(device: torch.device = torch.device("cpu")) -> GRPrefillDeviceInputs:
    return GRPrefillDeviceInputs(
        candidate_token_ids=torch.tensor(
            [[10, 11]], dtype=torch.int32, device=device
        ),
        candidate_logprobs=torch.tensor(
            [[-0.1, -0.2]], dtype=torch.float32, device=device
        ),
        valid_mask=torch.ones((1, 2), dtype=torch.bool, device=device),
        prefix_len=7,
    )
