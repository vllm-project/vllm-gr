# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Reusable CUDA stage builders; allocations and ownership stay local to each call."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import torch

from tests.constraints.test_table import make_table
from tests.helpers.beam_requests import make_batch_request
from tests.helpers.gr_config import engine_config
from vllm_gr.config import SUPPORTED_BEAM_BATCH_SIZES
from vllm_gr.config.models import DEFAULT_DEPLOYMENT
from vllm_gr.constraints.loader import _upload_cuda_backend
from vllm_gr.constraints.versioned_table import VersionedConstraintTable
from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionImpl
from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.beam.beam_state import BeamSearchState
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
from vllm_gr.v1.beam.gpu_beam_postprocess_workspace import BeamPostprocessWorkspace
from vllm_gr.v1.engine.beam_search_v1 import (
    GRDeviceStateRef,
    GRStageKind,
    GRStageMetadata,
)
from vllm_gr.v1.engine.gr_async_scheduler import GR_STAGE_OUTPUT_ATTR
from vllm_gr.v1.worker.cuda.beam_search_runner import (
    BeamBatchSearchRunner,
    BeamSearchRunner,
)
from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import (
    BeamRequestBinding,
    GPUBeamStageRunner,
)
from vllm_gr.v1.worker.cuda.gr_inputs import GRInputBuffers
from vllm_gr.v1.worker.cuda.v1.gpu_beam_stage_runner_v1 import CudaStageHooksV1

# mypy: disable-error-code="no-untyped-def,no-untyped-call"


def metadata(
    stage: int = 0,
    dispatch: int | None = None,
    output: bool = True,
    n: int = 3,
    candidate_width: int | None = None,
) -> GRStageMetadata:
    request = make_batch_request(n)
    return GRStageMetadata(
        session_id=request.batch_id,
        dispatch_id=stage if dispatch is None else dispatch,
        stage_index=stage,
        kind=GRStageKind.PREFILL if stage == 0 else GRStageKind.DECODE,
        decode_step=stage - 1 if stage else None,
        is_last_stage=output and stage == n - 1,
        input_device_state=GRDeviceStateRef(request.batch_id, stage - 1) if stage else None,
        output_device_state=GRDeviceStateRef(request.batch_id, stage),
        beam_params=msgspec.structs.replace(
            request.beam_params, stop_token_ids=(), candidate_width=candidate_width
        ),
        output_options=request.output_options,
        num_prompt_tokens=7,
        produces_output=output,
        request_ids=(request.batch_id,),
        num_prompt_tokens_by_request=(7,),
    )


def dispatch(m: GRStageMetadata) -> SimpleNamespace:
    result = SimpleNamespace(num_scheduled_tokens={m.session_id: 7 if m.stage_index == 0 else 1})
    setattr(result, GR_STAGE_OUTPUT_ATTR, {m.session_id: m})
    return result


def runtime(
    dtype=torch.float16,
    n=3,
    stop_token_ids=(),
    candidate_width=None,
    *,
    unconstrained=False,
    gr_options=None,
    hooks_cls=CudaStageHooksV1,
):
    context = BeamAttentionContext(3, 4, 1, 5, 8, 2, torch.device("cuda"), dtype)
    top_k = 4 if candidate_width is None else candidate_width
    state = BeamSearchState(
        context.decision,
        beam_width=4,
        max_new_tokens=n,
        stop_token_ids=stop_token_ids,
        ignore_stop_tokens=True,
        candidate_width=top_k,
    )
    # A one-request batch owns request-major state, workspace, and KV row 0.
    slot = context.registry.allocate("session-0", 4, 7)
    session = BeamSearchSession("session-0", slot, context)
    session.begin_session(4, 7, decision_index=0)
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=3,
        beam_width=4,
        top_k=top_k,
        max_decode_steps=5,
        device="cuda",
        logits_dtype=dtype,
    )
    runner = BeamSearchRunner(session, state, workspace)
    beam = BeamBatchSearchRunner(
        (runner,),
        BeamBatchExecutionView.from_stage_metadata(metadata(n, candidate_width=candidate_width)),
    )
    backend, _ = _upload_cuda_backend(
        make_table(), device=torch.device("cuda"), max_top_k=max(4, top_k), max_rows=4
    )
    backend._mark_ready()
    worker = SimpleNamespace(
        vllm_config=engine_config(**(gr_options or {})),
        _beam_sessions={"session-0": session},
        model_config=SimpleNamespace(max_model_len=64),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))]
        ),
        execute_model_state=None,
        _update_states=Mock(),
    )
    owner = GPUBeamStageRunner(
        worker,
        beam,
        (BeamRequestBinding("session-0", 0, "session-0", 0, 0, 0),),
        None if unconstrained else backend,
        hooks=hooks_cls(worker),
    )
    owner.metadata = SimpleNamespace(
        suffix_cu_seqlens_q_device=torch.arange(5, device="cuda", dtype=torch.int32),
        suffix_seqlens_kv_device=torch.empty(4, device="cuda", dtype=torch.int32),
        suffix_cu_seqlens_k_device=torch.empty(5, device="cuda", dtype=torch.int32),
    )
    owner.temperatures.fill_(1)
    owner.prefixes.fill_(-1)
    return owner


def prepare_binding(owner, entry):
    """Prime the CPU/upload contract for tests that isolate device binding."""
    if owner.inputs is None:
        owner.inputs = GRInputBuffers(owner.beam.state.beam_width, 64, 16, owner.beam.state.device)
    slot = owner.pending[1] if owner.pending is not None else 0
    owner.prepared = owner.inputs.prepare_cpu(
        entry.metadata, entry.binding, block_ids=[[1]], slot=slot
    )
    owner.device_positions = owner.inputs.upload(owner.prepared)["positions"]
    owner.prepare_decode(entry)


def admission_worker(
    sliding_windows: tuple[int | None, ...] = (None, None),
    dtype: torch.dtype = torch.float16,
    *,
    max_rows: int = 4,
) -> SimpleNamespace:
    device = torch.device("cuda", torch.accelerator.current_device_index())
    backend, _ = _upload_cuda_backend(make_table(), device=device, max_top_k=4, max_rows=max_rows)
    backend._mark_ready()
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=max(SUPPORTED_BEAM_BATCH_SIZES),
        beam_width=4,
        top_k=4,
        max_decode_steps=3,
        device=device,
        logits_dtype=dtype,
    )
    layers = {
        f"layer.{i}": SimpleNamespace(
            impl=BeamAttentionImpl(
                num_heads=1,
                head_size=32,
                scale=32**-0.5,
                num_kv_heads=1,
                alibi_slopes=None,
                sliding_window=window,
                kv_cache_dtype="auto",
            )
        )
        for i, window in enumerate(sliding_windows)
    }
    return SimpleNamespace(
        device=device,
        use_async_scheduling=True,
        speculative_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
        ),
        model_config=SimpleNamespace(
            dtype=dtype,
            max_model_len=64,
            is_encoder_decoder=False,
            is_hybrid=False,
            hf_config=SimpleNamespace(architectures=[]),
            get_vocab_size=lambda: backend.info.table_metadata.vocab_size,
            get_head_size=lambda: 32,
            get_num_kv_heads=lambda _: 1,
            get_num_attention_heads=lambda _: 1,
            get_num_layers=lambda _: len(layers),
        ),
        supports_mm_inputs=False,
        vllm_config=SimpleNamespace(
            additional_config={
                "vllm_gr": {"gr": {"beam_width": [128] * 3, "candidate_width": [128] * 3}}
            },
            scheduler_config=SimpleNamespace(async_scheduling=True),
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", sliding_window=None),
        _vllm_gr_config=SimpleNamespace(
            deployment=DEFAULT_DEPLOYMENT,
            beam_max_width=4,
            beam_max_decode_steps=3,
            enable_short_suffix_attention=False,
            # resolve_candidate_width reads this; None keeps K == W for the
            # fixture's W=4 workspace.
            constraint_table=None,
        ),
        constraint_backend=VersionedConstraintTable(backend, loader=lambda _: backend),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=list(layers), kv_cache_spec=SimpleNamespace(block_size=16)
                )
            ]
        ),
        compilation_config=SimpleNamespace(static_forward_context=layers),
        _beam_postprocess_workspaces=SimpleNamespace(
            keys=(workspace.key,), require=Mock(return_value=workspace)
        ),
        _update_states=Mock(),
    )


def install_decode_model(
    worker: SimpleNamespace, forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> None:
    """Use a tiny real model with the production graph wrapper and context."""
    from vllm.config import CompilationConfig, ParallelConfig

    class Model(torch.nn.Module):  # type: ignore[misc]
        def forward(self, input_ids, positions):
            return forward(input_ids, positions)

        def compute_logits(self, hidden):
            return hidden

    worker.model = Model()
    worker.vllm_config.compilation_config = CompilationConfig(mode=0)
    worker.vllm_config.parallel_config = ParallelConfig()
    worker._model_forward = lambda **kw: worker.model(
        input_ids=kw["input_ids"], positions=kw["positions"]
    )
