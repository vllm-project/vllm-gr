# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Reusable CUDA request runner builder with independent state and workspace."""

from __future__ import annotations

import torch

from vllm_gr.v1.beam.attention_context import BeamAttentionContext
from vllm_gr.v1.beam.session import BeamSearchSession
from vllm_gr.v1.beam.state import BeamSearchState
from vllm_gr.v1.worker.cuda.beam_search_runner import BeamSearchRunner
from vllm_gr.v1.worker.cuda.gpu_postprocess_workspace import BeamPostprocessWorkspace

# mypy: disable-error-code="no-untyped-def,no-untyped-call"


def make_request_runner(
    width=4,
    steps=7,
    *,
    dtype=torch.float16,
    stops=(),
    ignore=False,
    batch=1,
    state_index=0,
    physical_steps=None,
    candidate_width=None,
):
    context = BeamAttentionContext(
        batch, width, 1, physical_steps or steps, 8, 2, torch.device("cuda"), dtype
    )
    state = BeamSearchState(
        context.decision,
        beam_width=width,
        max_new_tokens=steps,
        stop_token_ids=stops,
        ignore_stop_tokens=ignore,
        candidate_width=candidate_width,
    )
    slot = context.registry.allocate("s", width, 0)
    session = BeamSearchSession("s", slot, context)
    session.begin_session(width, 0, decision_index=state_index)
    state.reset(state_index)
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=width,
        top_k=width if candidate_width is None else candidate_width,
        max_decode_steps=physical_steps or steps,
        device="cuda",
        logits_dtype=dtype,
    )
    return BeamSearchRunner(session, state, workspace, state_index=state_index)
