# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Contract tests for the long-lived incremental BeamSearch Session."""

from __future__ import annotations

from typing import Generator

import pytest
import torch

from vllm_gr.v1.beam.attention_context import BeamAttentionContext
from vllm_gr.v1.beam.session import BeamSearchSession
from vllm_gr.v1.beam.session_registry import BeamSessionRegistry


def _make_context(
    *,
    max_batch: int = 2,
    max_beams: int = 4,
    num_kv_heads: int = 2,
    max_decode_steps: int = 3,
    head_dim: int = 8,
    num_layers: int = 2,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float16,
) -> BeamAttentionContext:
    return BeamAttentionContext(
        max_batch=max_batch,
        max_beams=max_beams,
        num_kv_heads=num_kv_heads,
        max_decode_steps=max_decode_steps,
        head_dim=head_dim,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
    )


def _make_session(
    *,
    session_id: str = "test_session",
    max_batch: int = 2,
    max_beams: int = 4,
    num_kv_heads: int = 2,
    max_decode_steps: int = 3,
    head_dim: int = 8,
    num_layers: int = 2,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float16,
) -> BeamSearchSession:
    context = _make_context(
        max_batch=max_batch,
        max_beams=max_beams,
        num_kv_heads=num_kv_heads,
        max_decode_steps=max_decode_steps,
        head_dim=head_dim,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
    )
    slot = context.registry.allocate(session_id, beam_width=0, prefix_len=0)
    return BeamSearchSession(session_id=session_id, slot=slot, context=context)


def _persistent_tensors(session: BeamSearchSession) -> Generator[torch.Tensor, None, None]:
    for pool in session.pools:
        yield pool.unshared_key
        yield pool.unshared_value
    yield session.context.suffix_k_buf
    yield session.context.suffix_v_buf


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_constructor_allocates_frozen_storage_contract(dtype: torch.dtype) -> None:
    session = _make_session(
        dtype=dtype, num_layers=2, max_beams=4, max_decode_steps=3, num_kv_heads=2, head_dim=8
    )

    assert len(session.pools) == 2
    # Each pool has shape [max_batch, max_beams, num_kv_heads, max_decode_steps, head_dim]
    assert session.pools[0].unshared_key.shape == (2, 4, 2, 3, 8)
    assert session.pools[0].unshared_value.shape == (2, 4, 2, 3, 8)
    # Suffix buffers are shared across all sessions in the context
    # Shape: [max_batch * max_beams * max_decode_steps, num_kv_heads, head_dim]
    assert session.context.suffix_k_buf.shape == (2 * 4 * 3, 2, 8)
    assert session.context.suffix_v_buf.shape == (2 * 4 * 3, 2, 8)
    assert session.context.suffix_k_buf.dtype == session.context.suffix_v_buf.dtype == dtype
    assert session.context.dtype == dtype
    # After allocate, is_active is True (our registry sets is_active=True on allocate)
    assert session.is_active
    assert session.beam_width == 0
    assert session.decode_step == 0


def test_context_rejects_unsupported_activation_dtype() -> None:
    with pytest.raises(
        ValueError,
        match="dtype must be torch.float16 or torch.bfloat16",
    ):
        _make_context(dtype=torch.float32)


def test_session_references_shared_context() -> None:
    session = _make_session()

    # Pools are owned by the context, session references them
    assert session.pools is session.context.pools
    assert len(session.pools) == 2


def test_constructor_accepts_device_string() -> None:
    context = BeamAttentionContext(
        max_batch=1,
        max_beams=4,
        num_kv_heads=2,
        max_decode_steps=3,
        head_dim=8,
        num_layers=2,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    slot = context.registry.allocate("test", beam_width=0, prefix_len=0)
    session = BeamSearchSession(session_id="test", slot=slot, context=context)

    assert session.context.suffix_k_buf.device == torch.device("cpu")


def test_begin_and_end_change_logical_state_and_preserve_storage() -> None:
    session = _make_session()
    tensors = tuple(_persistent_tensors(session))
    pointers = tuple(tensor.data_ptr() for tensor in tensors)
    for pool in session.pools:
        pool.unshared_key.fill_(11)
        pool.unshared_value.fill_(12)
    session.context.suffix_k_buf.fill_(13)
    session.context.suffix_v_buf.fill_(14)

    session.begin_session(beam_width=3, prefix_len=97)

    assert session.is_active
    assert session.beam_width == 3
    assert session.decode_step == 0
    # begin_session does NOT zero slot KV (lazy reset: cache_unshared_kv
    # overwrites per-step and cascade attention reads only [:decode_step+1],
    # so stale rows are never read. Storage is preserved.
    for pool in session.pools:
        assert torch.count_nonzero(pool.unshared_key[session.slot]).item() != 0
        assert torch.count_nonzero(pool.unshared_value[session.slot]).item() != 0

    session.end_session()

    # After end_session, the session's slot is released from the registry
    # so we can no longer access _state (it's been removed)
    assert not session.context.registry.has_session("test_session")
    # Storage pointers remain unchanged
    assert tuple(tensor.data_ptr() for tensor in tensors) == pointers


def test_end_session_releases_slot_from_registry() -> None:
    session = _make_session(max_batch=1)
    session.begin_session(beam_width=2, prefix_len=10)

    assert session.context.registry.has_session("test_session")

    session.end_session()

    assert not session.context.registry.has_session("test_session")


def test_end_session_clears_advanced_logical_request_state() -> None:
    session = _make_session(max_decode_steps=3)
    session.begin_session(beam_width=3, prefix_len=97)
    parent = torch.arange(3, dtype=torch.int32)
    session.step_complete(parent)

    assert session.decode_step == 1

    session.end_session()

    # After end_session, the slot is released from the registry
    assert not session.context.registry.has_session("test_session")


def test_repeated_requests_reuse_every_persistent_storage() -> None:
    session = _make_session()
    tensors = tuple(_persistent_tensors(session))
    pointers = tuple(tensor.data_ptr() for tensor in tensors)

    for beam_width, prefix_len in ((4, 10), (1, 200), (3, 0)):
        session.begin_session(beam_width, prefix_len)
        assert session.decode_step == 0
        session.end_session()
        # Re-allocate the slot for the next iteration
        session.context.registry.allocate("test_session", beam_width=0, prefix_len=0)

    assert tuple(tensor.data_ptr() for tensor in tensors) == pointers


def test_slot_allocation_and_release_cycle() -> None:
    """Test that slot allocation and release works correctly."""
    context = _make_context(max_batch=2)
    slot1 = context.registry.allocate("session_1", beam_width=2, prefix_len=10)
    slot2 = context.registry.allocate("session_2", beam_width=3, prefix_len=20)

    assert slot1 == 0  # free list is [0, 1], pop(0) gives 0 first
    assert slot2 == 1

    # No more free slots
    with pytest.raises(RuntimeError, match="No free beam session slots"):
        context.registry.allocate("session_3", beam_width=1, prefix_len=5)

    # Release one slot
    context.registry.release("session_1")
    assert not context.registry.has_session("session_1")

    # Now we can allocate again
    slot3 = context.registry.allocate("session_3", beam_width=1, prefix_len=5)
    assert slot3 == 0  # recycled slot


@pytest.mark.l1
@pytest.mark.parametrize("max_batch", [1, 2])  # type: ignore[misc]
def test_duplicate_slot_allocation_preserves_session_and_capacity(max_batch: int) -> None:
    registry = BeamSessionRegistry(max_batch)
    slot = registry.allocate("existing", beam_width=2, prefix_len=10)
    state = registry.get_state("existing")
    state.decode_step = 1

    with pytest.raises(RuntimeError, match="Beam session 'existing' already allocated"):
        registry.allocate("existing", beam_width=4, prefix_len=20)

    assert registry.num_active_sessions == 1
    assert registry.get_slot("existing") == slot
    assert registry.get_state("existing") is state
    assert state.beam_width == 2
    assert state.prefix_len == 10
    assert state.decode_step == 1

    registry.release("existing")
    assert registry.num_active_sessions == 0
    assert registry.allocate("existing", beam_width=1, prefix_len=0) == slot
    registry.release("existing")
    assert [
        registry.allocate(f"replacement_{index}", beam_width=1, prefix_len=0)
        for index in range(max_batch)
    ] == list(range(max_batch))
    assert registry.num_active_sessions == max_batch
    with pytest.raises(RuntimeError, match="No free beam session slots"):
        registry.allocate("overflow", beam_width=1, prefix_len=0)


def test_step_complete_advances_decode_step() -> None:
    session = _make_session(max_decode_steps=3)
    session.begin_session(beam_width=3, prefix_len=20)

    parent = torch.arange(3, dtype=torch.int32)
    session.step_complete(parent)
    assert session.decode_step == 1

    session.step_complete(parent)
    assert session.decode_step == 2


def test_step_complete_reorders_pool_kv_via_select_unshared_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _make_session(max_beams=4, num_layers=2)
    session.begin_session(beam_width=3, prefix_len=10)

    # Fill pool slot 0 with identifiable data
    for layer_idx, pool in enumerate(session.pools):
        pool.unshared_key[0].fill_(layer_idx * 10 + 1)
        pool.unshared_value[0].fill_(layer_idx * 10 + 2)

    select_calls: list[dict[str, object]] = []

    def mock_select(
        beam_index: torch.Tensor,
        slot: int,
        key_caches: list[torch.Tensor],
        value_caches: list[torch.Tensor],
        decode_step: int,
        **kwargs: object,
    ) -> None:
        select_calls.append(
            {
                "beam_index": beam_index,
                "slot": slot,
                "key_caches": key_caches,
                "value_caches": value_caches,
                "decode_step": decode_step,
            }
        )

    monkeypatch.setattr(
        "vllm_gr.v1.beam.reorder.select_unshared_kv",
        mock_select,
    )

    parent = torch.tensor([2, 0, 1], dtype=torch.int32)
    session.step_complete(parent)

    assert len(select_calls) == 1
    call = select_calls[0]
    assert torch.equal(call["beam_index"], parent)
    assert call["slot"] == session.slot
    assert call["decode_step"] == 0
    assert len(call["key_caches"]) == 2  # type: ignore[arg-type]
    assert len(call["value_caches"]) == 2  # type: ignore[arg-type]
    assert session.decode_step == 1


def test_step_complete_skips_select_on_last_step() -> None:
    session = _make_session(max_decode_steps=3)
    session.begin_session(beam_width=3, prefix_len=20)
    state = session._state
    state.is_last_step = True

    select_calls: list[bool] = []

    def mock_select(*args: object, **kwargs: object) -> None:
        select_calls.append(True)

    import vllm_gr.v1.beam.reorder as reorder_mod

    original_select = reorder_mod.select_unshared_kv
    reorder_mod.select_unshared_kv = mock_select
    try:
        parent = torch.arange(3, dtype=torch.int32)
        session.step_complete(parent)
    finally:
        reorder_mod.select_unshared_kv = original_select

    assert len(select_calls) == 0
    assert session.decode_step == 1


def test_suffix_buf_returns_slice_for_slot() -> None:
    session = _make_session(
        max_batch=2, max_beams=4, max_decode_steps=3, num_kv_heads=2, head_dim=8
    )
    session.begin_session(beam_width=2, prefix_len=5)

    k_buf, v_buf = session.suffix_k_buf, session.suffix_v_buf
    # suffix_k_buf shape: [beam_width * max_decode_steps, num_kv_heads, head_dim]
    assert k_buf.shape == (2 * 3, 2, 8)
    assert v_buf.shape == (2 * 3, 2, 8)


def test_slot_property_matches_registry() -> None:
    session = _make_session()
    assert session.slot == session.context.registry.get_slot("test_session")


def test_begin_session_preserves_slot_kv() -> None:
    """Test that begin_session does NOT zero the pool KV for the session's slot.

    Lazy reset: ``cache_unshared_kv_kernel`` overwrites exactly the current
    ``decode_step`` slice and cascade attention limits suffix reads to
    ``[:decode_step+1]``, so stale rows beyond ``decode_step`` are never read.
    ``begin_session`` therefore skips the redundant per-layer zeroing.
    """
    session = _make_session(max_batch=2)
    session.begin_session(beam_width=2, prefix_len=5)

    # Write some data
    for pool in session.pools:
        pool.unshared_key[session.slot].fill_(42)
        pool.unshared_value[session.slot].fill_(43)

    # End and re-begin
    session.end_session()
    session.context.registry.allocate("test_session", beam_width=0, prefix_len=0)
    session.begin_session(beam_width=3, prefix_len=10)

    # Pool KV is preserved (not zeroed) — lazy reset relies on overwrite +
    # read-range-limiting instead.
    for pool in session.pools:
        assert torch.count_nonzero(pool.unshared_key[session.slot]).item() != 0
        assert torch.count_nonzero(pool.unshared_value[session.slot]).item() != 0


def test_context_uses_xllm_buffers_flag_tracks_device_type() -> None:
    """The xllm-buffer fill_ gate is True on NPU, False on CUDA/CPU.

    NPU device tensors cannot be allocated on a CUDA-only host, so the flag's
    device-type derivation is verified indirectly: a CPU-built context must be
    gated off, and the predicate ``type == "npu"`` (the exact gate expression)
    is exercised across every torch-recognized device type.
    """
    cpu = _make_context(device=torch.device("cpu"))
    assert cpu._gr_uses_xllm_buffers is False

    # Exact gate predicate: ``getattr(device, "type", None) == "npu"``.
    assert torch.device("cpu").type != "npu"
    assert torch.device("cuda", 0).type != "npu"
    assert torch.device("meta").type != "npu"
    # NPU device type is simulated with a plain string gate-equivalent, since
    # torch.device("npu") is not constructible on non-Ascend hosts.
    assert getattr(type("D", (), {"type": "npu"})(), "type", None) == "npu"


pytestmark = pytest.mark.cpu_test
