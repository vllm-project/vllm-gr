# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration tests for Session active views and OneRec decode operators."""

from __future__ import annotations

import pytest
import torch

from vllm_gr.ops import beam_search_group, cache_unshared_kv, select_unshared_kv
from vllm_gr.v1.beam import BeamSearchSession


def _devices() -> list[pytest.ParameterSet]:
    return [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda"),
            id="cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
        ),
    ]


def _make_session(
    device: torch.device,
    dtype: torch.dtype,
    *,
    layers: int = 2,
    max_beams: int = 6,
    max_decode_steps: int = 3,
    heads: int = 2,
    dim: int = 4,
) -> BeamSearchSession:
    return BeamSearchSession(
        layer_names=tuple(f"model.layers.{index}.self_attn" for index in range(layers)),
        max_beams=max_beams,
        local_num_kv_heads=heads,
        max_decode_steps=max_decode_steps,
        head_dim=dim,
        device=device,
        dtype=dtype,
    )


@pytest.mark.parametrize("device", _devices())  # type: ignore[misc]
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_public_cache_uses_session_active_view_and_fixed_abi_buffers(
    device: torch.device, dtype: torch.dtype
) -> None:
    session = _make_session(device, dtype, layers=1, max_beams=4)
    session.begin_session(beam_width=2, prefix_len=51)
    pool = session.pools[0]
    pool.unshared_key.fill_(-1)
    pool.unshared_value.fill_(-2)
    key_ptr = pool.unshared_key.data_ptr()
    value_ptr = pool.unshared_value.data_ptr()
    curr_shape = (2, 2, 4)
    curr_key = torch.arange(16, dtype=dtype, device=device).reshape(curr_shape)
    curr_value = curr_key + 100

    returned_key, returned_value = cache_unshared_kv(
        pool.active_key(session.beam_width),
        pool.active_value(session.beam_width),
        curr_key,
        curr_value,
        session.block_table_buf,
        session.decode_step_buf,
    )

    assert returned_key.data_ptr() == key_ptr
    assert returned_value.data_ptr() == value_ptr
    torch.testing.assert_close(pool.unshared_key[0, :2, :, 0, :], curr_key, rtol=0, atol=0)
    torch.testing.assert_close(pool.unshared_value[0, :2, :, 0, :], curr_value, rtol=0, atol=0)
    assert torch.all(pool.unshared_key[0, :2, :, 1:, :] == -1)
    assert torch.all(pool.unshared_value[0, :2, :, 1:, :] == -2)
    assert torch.all(pool.unshared_key[0, 2:] == -1)
    assert torch.all(pool.unshared_value[0, 2:] == -2)


def _prefix_for_expanded_parents(parents: list[int], beam_width: int) -> torch.Tensor:
    counts = [parents.count(parent) for parent in range(beam_width)]
    prefix: list[int] = []
    total = 0
    for count in counts:
        total += count
        prefix.append(total)
    return torch.tensor(prefix, dtype=torch.int32)


@pytest.mark.parametrize("device", _devices())  # type: ignore[misc]
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
@pytest.mark.parametrize(
    "expanded_parents",
    [[0, 1, 2, 3], [0, 0, 2, 2], [1, 1, 1, 3]],
    ids=["identity", "duplicate-groups", "three-from-one"],
)  # type: ignore[misc]
def test_prepare_selects_complete_time_axis_for_every_layer(
    device: torch.device,
    dtype: torch.dtype,
    expanded_parents: list[int],
) -> None:
    session = _make_session(device, dtype, layers=3, max_beams=6)
    session.begin_session(beam_width=4, prefix_len=80)
    originals: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer, pool in enumerate(session.pools):
        values = torch.arange(pool.unshared_key.numel(), device=device).reshape(
            pool.unshared_key.shape
        )
        pool.unshared_key.copy_(values.to(dtype) + layer * 1000)
        pool.unshared_value.copy_(values.to(dtype) + layer * 2000 + 17)
        originals.append((pool.unshared_key.clone(), pool.unshared_value.clone()))

    beam_index = torch.tensor(expanded_parents, dtype=torch.int32, device=device)
    group_token_num = _prefix_for_expanded_parents(expanded_parents, 4).to(device)
    session.prepare_next_step(beam_index, group_token_num)
    parent_index = torch.tensor(expanded_parents, dtype=torch.int64, device=device)

    for pool, (original_key, original_value) in zip(session.pools, originals):
        torch.testing.assert_close(
            pool.unshared_key[0, :4], original_key[0, parent_index], rtol=0, atol=0
        )
        torch.testing.assert_close(
            pool.unshared_value[0, :4], original_value[0, parent_index], rtol=0, atol=0
        )
        torch.testing.assert_close(pool.unshared_key[0, 4:], original_key[0, 4:], rtol=0, atol=0)
        torch.testing.assert_close(
            pool.unshared_value[0, 4:], original_value[0, 4:], rtol=0, atol=0
        )
    assert session.current_step == 1
    assert torch.equal(session.decode_step_buf.cpu(), torch.tensor([2], dtype=torch.int32))


@pytest.mark.parametrize("device", _devices())  # type: ignore[misc]
def test_beam_search_group_outputs_feed_session_without_parent_conversion(
    device: torch.device,
) -> None:
    beam_width, step = 4, 1
    session = _make_session(device, torch.float16, layers=2, max_beams=5)
    session.begin_session(beam_width, prefix_len=12)
    originals: list[torch.Tensor] = []
    for layer, pool in enumerate(session.pools):
        beam_values = torch.arange(beam_width, dtype=torch.float16, device=device).reshape(
            1, beam_width, 1, 1, 1
        )
        pool.unshared_key.fill_(-9)
        pool.unshared_value.fill_(-8)
        pool.unshared_key[:, :beam_width].copy_(beam_values.expand(1, -1, 2, 3, 4) + layer * 10)
        pool.unshared_value[:, :beam_width].copy_(
            beam_values.expand(1, -1, 2, 3, 4) + layer * 10 + 100
        )
        originals.append(pool.unshared_key.clone())

    log_probs = torch.zeros((beam_width, 1), dtype=torch.float32, device=device)
    top_probs = torch.full((beam_width, beam_width), -10.0, dtype=torch.float32, device=device)
    top_probs[3, 0] = 4.0
    top_probs[1, :3] = torch.tensor([3.0, 2.0, 1.0], device=device)
    top_tokens = torch.arange(beam_width * beam_width, dtype=torch.int32, device=device).reshape(
        beam_width, beam_width
    )
    sequence = torch.arange(beam_width * (step + 1), dtype=torch.int32, device=device).reshape(
        1, beam_width, step + 1
    )

    _, beam_index, _, group_token_num, _ = beam_search_group(
        log_probs, top_tokens, top_probs, sequence, step, beam_width
    )
    assert torch.equal(beam_index.cpu(), torch.tensor([[1, 1, 1, 3]], dtype=torch.int32))
    assert torch.equal(group_token_num.cpu(), torch.tensor([[0, 3, 3, 4]], dtype=torch.int32))

    session.prepare_next_step(beam_index.flatten(), group_token_num.flatten())

    expected_parent = torch.tensor([1, 1, 1, 3], dtype=torch.int64, device=device)
    for pool, original in zip(session.pools, originals):
        torch.testing.assert_close(
            pool.unshared_key[0, :beam_width], original[0, expected_parent], rtol=0, atol=0
        )
        assert torch.all(pool.unshared_key[0, beam_width:] == -9)


def test_select_validates_every_layer_before_first_write() -> None:
    session = _make_session(torch.device("cpu"), torch.float16, layers=2, max_beams=4)
    session.begin_session(beam_width=3, prefix_len=10)
    first_key = session.pools[0].active_key(3)
    first_value = session.pools[0].active_value(3)
    first_key.copy_(torch.arange(first_key.numel()).reshape(first_key.shape))
    first_value.copy_(torch.arange(first_value.numel()).reshape(first_value.shape) + 100)
    original_key = first_key.clone()
    original_value = first_value.clone()
    bad_key = session.pools[1].active_key(3).to(torch.float32)
    bad_value = session.pools[1].active_value(3).to(torch.float32)
    identity = torch.arange(3, dtype=torch.int32)
    prefix = torch.arange(1, 4, dtype=torch.int32)

    with pytest.raises(TypeError, match="layer 1 K/V cache dtype mismatch"):
        select_unshared_kv(
            identity,
            [first_key, bad_key],
            [first_value, bad_value],
            session.block_table_buf,
            prefix,
            1,
            3,
            2,
        )

    torch.testing.assert_close(first_key, original_key, rtol=0, atol=0)
    torch.testing.assert_close(first_value, original_value, rtol=0, atol=0)


def test_select_rejects_layer_list_length_before_first_write() -> None:
    session = _make_session(torch.device("cpu"), torch.float16, layers=2, max_beams=4)
    session.begin_session(beam_width=3, prefix_len=10)
    first_key = session.pools[0].active_key(3)
    first_value = session.pools[0].active_value(3)
    first_key.fill_(7)
    first_value.fill_(8)
    identity = torch.arange(3, dtype=torch.int32)
    prefix = torch.arange(1, 4, dtype=torch.int32)

    with pytest.raises(ValueError, match="tensor-list lengths must equal layer_num"):
        select_unshared_kv(
            identity,
            [first_key],
            [first_value],
            session.block_table_buf,
            prefix,
            1,
            3,
            2,
        )

    assert torch.all(first_key == 7)
    assert torch.all(first_value == 8)


def test_public_cache_entry_returns_original_tensor_objects() -> None:
    session = _make_session(torch.device("cpu"), torch.bfloat16, layers=1, max_beams=2)
    session.begin_session(2, 5)
    pool = session.pools[0]
    active_key = pool.active_key(2)
    active_value = pool.active_value(2)
    current = torch.zeros((2, 2, 4), dtype=torch.bfloat16)

    returned_key, returned_value = cache_unshared_kv(
        active_key,
        active_value,
        current,
        current,
        session.block_table_buf,
        session.decode_step_buf,
    )

    assert returned_key is active_key
    assert returned_value is active_value
