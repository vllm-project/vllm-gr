# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest
import torch

from vllm_gr.constraints.backend import ConstraintOutput
from tests.constraints.test_table import make_table
from vllm_gr.config import ConstraintTableBackendEnum, ConstraintTableConfig
from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.backend import ConstraintMetadata
from vllm_gr.constraints.loader import load_constraint_backend
from vllm_gr.ops.onerec_decode import (
    _beam_search_group_torch,
    _beam_search_rec_final_select_torch,
)
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
    GRDecodeDeviceInputs,
    GRPrefillDeviceInputs,
    GPUModelRunnerPersistentBeamBackend,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _QueuedConstraintBackend:
    def __init__(self, outputs: list[ConstraintOutput]) -> None:
        self.outputs = deque(outputs)

    @property
    def info(self):  # pragma: no cover - direct harness only
        return None

    def select(self, logits, metadata) -> ConstraintOutput:
        del logits, metadata
        return self.outputs.popleft()

    def close(self) -> None:
        pass


def _stage(index: int, max_tokens: int = 3) -> GRWorkerStageSpec:
    return GRWorkerStageSpec(
        session_id="cuda-session",
        stage_index=index,
        stage_kind=GRStageKind.PREFILL if index == 0 else GRStageKind.DECODE,
        is_last_stage=index == max_tokens - 1,
    )


@pytest.mark.parametrize("width", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_direct_call_matches_frozen_reference_and_reorders_kv(
    dtype: torch.dtype, width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = torch.device("cuda:0")
    max_tokens = 3
    context = BeamAttentionContext(
        max_batch=1,
        max_beams=width,
        num_kv_heads=1,
        max_decode_steps=max_tokens,
        head_dim=8,
        num_layers=1,
        device=device,
        dtype=dtype,
    )
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=width,
        top_k=width,
        max_decode_steps=max_tokens,
        device=device,
        logits_dtype=dtype,
        pin_host=False,
    )
    workspace_pool = BeamPostprocessWorkspacePool()
    workspace_pool.add(workspace)

    prefill_tokens = torch.arange(1000, 1000 + width, dtype=torch.int32, device=device).view(
        1, width
    )
    prefill_scores = (
        -torch.arange(1, width + 1, dtype=torch.float32, device=device).view(1, width) / 100.0
    )
    flat = torch.arange(width * width, dtype=torch.int32, device=device).view(width, width)
    decode1_tokens = flat + 2000
    decode2_tokens = flat + 10000
    decode1_logprobs = -(flat.to(torch.float32) + 1.0) / 10000.0
    decode2_logprobs = -(torch.flip(flat, dims=(1,)).to(torch.float32) + 1.0) / 20000.0
    valid = torch.ones((width, width), dtype=torch.bool, device=device)
    constraints = _QueuedConstraintBackend(
        [
            ConstraintOutput(decode1_tokens, decode1_logprobs, valid),
            ConstraintOutput(decode2_tokens, decode2_logprobs, valid),
        ]
    )
    backend = GPUModelRunnerPersistentBeamBackend(
        context=context,
        constraint_backends={"default": constraints},  # type: ignore[dict-item]
        workspace_pool=workspace_pool,
    )
    backend.create_session(
        GRBeamSessionSpec(
            session_id="cuda-session",
            request_id="request-0",
            beam_width=width,
            max_tokens=max_tokens,
            constraint_handle="default",
            temperature=1.0,
            ignore_stop_tokens=True,
        )
    )
    state = backend.registry.get("cuda-session")
    state_ptrs = state.buffers.data_ptrs()
    workspace_ptrs = tuple(tensor.data_ptr() for tensor in workspace.device_tensors())

    def forbidden_host_operation(*args, **kwargs):
        del args, kwargs
        raise AssertionError("persistent CUDA path attempted a Host materialization or sync")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_host_operation)
    monkeypatch.setattr(torch.Tensor, "item", forbidden_host_operation)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden_host_operation)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden_host_operation)

    backend.bootstrap_from_prefill(
        _stage(0),
        GRPrefillDeviceInputs(
            prefill_tokens,
            prefill_scores,
            torch.ones_like(prefill_tokens, dtype=torch.bool),
            prefix_len=32,
        ),
    )

    # Model forward has written the first decode suffix KV before selection.
    kv = context.pools[0].unshared_key[0]
    vv = context.pools[0].unshared_value[0]
    kv.copy_(torch.arange(width, dtype=dtype, device=device).view(width, 1, 1, 1).expand_as(kv))
    vv.copy_(kv + 100)
    kv_before = kv.clone()
    vv_before = vv.clone()

    input_ids = torch.empty(width, dtype=torch.int32, device=device)
    backend.prepare_decode_inputs(_stage(1), GRDecodeDeviceInputs(input_ids))
    backend.advance_beam(_stage(1), torch.zeros((width, 16), dtype=dtype, device=device))
    parent_after_decode1 = state.buffers.parent_beam_indices[0, :, 1].clone()

    backend.prepare_decode_inputs(_stage(2), GRDecodeDeviceInputs(input_ids))
    terminal_ack = backend.advance_beam(
        _stage(2), torch.zeros((width, 16), dtype=dtype, device=device)
    )
    assert terminal_ack.terminal
    monkeypatch.undo()
    torch.cuda.synchronize(device)

    # The score accumulator is seeded with the Prefill scores (matching the
    # legacy frontend), so the reference group must start from prefill_scores,
    # not from the first Decode column.
    sequence0 = torch.zeros((1, width, max_tokens), dtype=torch.int32)
    sequence0[:, :, 0].copy_(prefill_tokens.cpu())
    ref1 = _beam_search_group_torch(
        prefill_scores.cpu().reshape(width, 1),
        decode1_tokens.cpu(),
        decode1_logprobs.cpu(),
        sequence0,
        1,
        width,
    )
    ref2 = _beam_search_rec_final_select_torch(
        ref1[2].reshape(width, 1),
        decode2_tokens.cpu(),
        decode2_logprobs.cpu(),
        ref1[4],
        2,
        width,
    )
    final = backend.get_final_device_view("cuda-session")
    torch.testing.assert_close(final.sampled_token_ids.cpu(), ref2[0].reshape(1, width))
    torch.testing.assert_close(
        final.cumulative_logprobs.cpu(), ref2[2].reshape(1, width), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(final.generated_token_ids.cpu(), ref2[3])
    assert final.valid_mask.all()

    ref_parent1 = ref1[1].reshape(width).to(torch.int64)
    ref_parent2 = ref2[1].reshape(width).to(torch.int64)
    expected_lineage = torch.zeros((1, width, max_tokens), dtype=torch.int32)
    expected_lineage[0, :, 1] = ref_parent1[ref_parent2].to(torch.int32)
    expected_lineage[0, :, 2] = ref_parent2.to(torch.int32)
    torch.testing.assert_close(final.parent_beam_indices.cpu(), expected_lineage)

    # Non-terminal Decode reordered KV in the same stream; terminal Decode did not.
    parent_index = parent_after_decode1.to(torch.int64)
    torch.testing.assert_close(kv[:, :, :1].cpu(), kv_before[parent_index, :, :1].cpu())
    torch.testing.assert_close(vv[:, :, :1].cpu(), vv_before[parent_index, :, :1].cpu())
    torch.testing.assert_close(kv[:, :, 1:].cpu(), kv_before[:, :, 1:].cpu())
    torch.testing.assert_close(vv[:, :, 1:].cpu(), vv_before[:, :, 1:].cpu())
    assert state.buffers.data_ptrs() == state_ptrs
    assert tuple(tensor.data_ptr() for tensor in workspace.device_tensors()) == workspace_ptrs

    backend.release_session("cuda-session")
    backend.create_session(
        GRBeamSessionSpec(
            session_id="cuda-session-reused",
            request_id="request-reused",
            beam_width=width,
            max_tokens=max_tokens,
            constraint_handle="default",
            temperature=1.0,
            ignore_stop_tokens=True,
        )
    )
    reused = backend.registry.get("cuda-session-reused")
    assert reused.slot_id == 0
    assert reused.buffers.data_ptrs() == state_ptrs
    assert tuple(tensor.data_ptr() for tensor in workspace.device_tensors()) == workspace_ptrs


@pytest.mark.parametrize("layout", ["contiguous", "row_padded", "vocab_strided"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_raw_logits_run_real_constrained_topk_through_final(
    tmp_path: Path,
    layout: str,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    width = 4
    max_tokens = 3
    table_path = tmp_path / "persistent-table.safetensors"
    digest = save_constraint_table_artifact(make_table(), table_path)
    constraint_backend = load_constraint_backend(
        ConstraintTableConfig(
            enabled=True,
            backend=ConstraintTableBackendEnum.CUDA,
            path=str(table_path),
            artifact_digest=digest,
            tokenizer_digest="sha256:" + "b" * 64,
            max_top_k=width,
        ),
        device=device,
        vocab_size=6,
        dtype=dtype,
        max_rows=width,
    )
    assert constraint_backend is not None
    context = BeamAttentionContext(
        max_batch=1,
        max_beams=width,
        num_kv_heads=1,
        max_decode_steps=max_tokens,
        head_dim=8,
        num_layers=1,
        device=device,
        dtype=dtype,
    )
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=width,
        top_k=width,
        max_decode_steps=max_tokens,
        device=device,
        logits_dtype=dtype,
        pin_host=False,
    )
    workspace_pool = BeamPostprocessWorkspacePool()
    workspace_pool.add(workspace)
    backend = GPUModelRunnerPersistentBeamBackend(
        context=context,
        constraint_backends={"default": constraint_backend},
        workspace_pool=workspace_pool,
    )
    backend.create_session(
        GRBeamSessionSpec(
            "cuda-session",
            "request-0",
            width,
            max_tokens,
            "default",
            temperature=1.0,
            ignore_stop_tokens=True,
        )
    )

    prefill_logits = torch.tensor([[-8, -5, 1, 7, 3, 9]], dtype=dtype, device=device)
    prefill = constraint_backend.select(
        prefill_logits,
        ConstraintMetadata(
            prefix_token_ids=torch.zeros((1, 3), dtype=torch.int32, device=device),
            temperatures=torch.ones(1, dtype=torch.float32, device=device),
            current_step=0,
            top_k=width,
        ),
    )
    backend.bootstrap_from_prefill(
        _stage(0),
        GRPrefillDeviceInputs(
            prefill.token_ids,
            prefill.logprobs,
            prefill.valid_mask,
            prefix_len=16,
        ),
    )
    input_ids = torch.empty(width, dtype=torch.int32, device=device)
    reference_sequence = torch.zeros((1, width, max_tokens), dtype=torch.int32)
    reference_sequence[:, :, 0].copy_(prefill.token_ids.cpu())
    reference_scores = prefill.logprobs.cpu().masked_fill(
        ~prefill.valid_mask.cpu(), float("-inf")
    )
    seen_layouts = []
    original_select_out = constraint_backend.select_out

    def select_out_spy(self, logits, *args, **kwargs):
        assert self is constraint_backend
        seen_layouts.append((logits.data_ptr(), logits.stride()))
        return original_select_out(logits, *args, **kwargs)

    monkeypatch.setattr(type(constraint_backend), "select_out", select_out_spy)
    for stage_index in (1, 2):
        backend.prepare_decode_inputs(_stage(stage_index), GRDecodeDeviceInputs(input_ids))
        row_bias = torch.arange(width, dtype=dtype, device=device)[:, None]
        logits = torch.tensor([[-8, -5, 1, 7, 3, 9]], dtype=dtype, device=device).expand(width, -1)
        logits = (logits + row_bias).contiguous()
        if layout != "contiguous":
            storage = torch.empty((width, 12), dtype=dtype, device=device)
            view = storage[:, :6] if layout == "row_padded" else storage[:, ::2]
            view.copy_(logits)
            logits = view
        candidates = constraint_backend.select(
            logits.contiguous(),
            ConstraintMetadata(
                prefix_token_ids=backend.registry.get("cuda-session").buffers.generated_token_ids[
                    0
                ],
                temperatures=torch.ones(width, dtype=torch.float32, device=device),
                current_step=stage_index,
                top_k=width,
            ),
        )
        scores = candidates.logprobs.cpu()
        scores = scores.masked_fill(~candidates.valid_mask.cpu(), float("-inf"))
        select = (
            _beam_search_group_torch if stage_index == 1 else _beam_search_rec_final_select_torch
        )
        expected = select(
            reference_scores.reshape(width, 1),
            candidates.token_ids.cpu(),
            scores,
            reference_sequence,
            stage_index,
            width,
        )
        reference_scores = expected[2].reshape(1, width)
        reference_sequence = expected[4] if stage_index == 1 else expected[3]
        backend.advance_beam(_stage(stage_index), logits)
        assert seen_layouts[-1] == (logits.data_ptr(), logits.stride())
        buffers = backend.registry.get("cuda-session").buffers
        torch.testing.assert_close(buffers.generated_token_ids.cpu(), reference_sequence)
        torch.testing.assert_close(
            buffers.cumulative_logprobs.cpu(), reference_scores, rtol=0, atol=1e-6
        )

    torch.cuda.synchronize(device)
    final = backend.get_final_device_view("cuda-session")
    final_sequences = {tuple(row) for row in final.generated_token_ids[0].cpu().tolist()}
    legal_sequences = {
        (1, 2, 0),
        (1, 2, 1),
        (1, 4, 3),
        (3, 5, 1),
        (3, 5, 4),
    }
    assert final_sequences <= legal_sequences
    # With fewer than W valid Prefill candidates the beam may retain duplicate
    # paths. Freeze the stage results above, not a uniqueness assumption.
    assert final.generated_token_ids.shape == (1, width, max_tokens)
    assert final.valid_mask.all()
    assert backend.metrics.beam_decisions == 2
    assert backend.metrics.kv_reorders == 1
    constraint_backend.close()
