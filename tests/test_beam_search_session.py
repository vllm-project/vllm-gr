# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contract tests for the long-lived incremental BeamSearch Session."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch

from vllm_gr.v1.beam import BeamSearchSession
from vllm_gr.v1.beam import beam_search_session as session_module
from vllm_gr.v1.beam.beam_search_session import (
    BeamSearchSession as SessionImplementation,
)


def _make_session(
    *,
    layer_names: tuple[str, ...] = ("model.layers.0.self_attn", "model.layers.2.self_attn"),
    max_beams: int = 4,
    local_num_kv_heads: int = 2,
    max_decode_steps: int = 3,
    head_dim: int = 8,
    device: torch.device | str = torch.device("cpu"),
    dtype: torch.dtype = torch.float16,
) -> BeamSearchSession:
    return BeamSearchSession(
        layer_names=layer_names,
        max_beams=max_beams,
        local_num_kv_heads=local_num_kv_heads,
        max_decode_steps=max_decode_steps,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )


def _persistent_tensors(session: BeamSearchSession) -> Iterator[torch.Tensor]:
    for pool in session.pools:
        yield pool.unshared_key
        yield pool.unshared_value
    yield session.suffix_k_buf
    yield session.suffix_v_buf
    yield session.block_table_buf
    yield session.decode_step_buf
    yield session.actual_shared_kvlen_buf


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # type: ignore[misc]
def test_constructor_allocates_frozen_storage_contract(dtype: torch.dtype) -> None:
    session = _make_session(dtype=dtype)

    assert session.layer_names == ("model.layers.0.self_attn", "model.layers.2.self_attn")
    assert len(session.pools) == 2
    assert session.suffix_k_buf.shape == (4 * 3, 2, 8)
    assert session.suffix_v_buf.shape == (4 * 3, 2, 8)
    assert session.suffix_k_buf.dtype == session.suffix_v_buf.dtype == dtype
    assert session.block_table_buf.dtype == torch.int32
    assert session.decode_step_buf.dtype == torch.int32
    assert session.actual_shared_kvlen_buf.dtype == torch.int32
    assert torch.equal(session.block_table_buf, torch.tensor([0], dtype=torch.int32))
    assert not session.is_active
    assert session.beam_width == session.current_step == session.prefix_len == 0


def test_layer_mapping_is_immutable_and_uses_canonical_names() -> None:
    session = _make_session()

    assert session.pool_index("model.layers.0.self_attn") == 0
    assert session.pool_index("model.layers.2.self_attn") == 1
    assert session.pool_for_layer("model.layers.2.self_attn") is session.pools[1]
    with pytest.raises(TypeError):
        session.layer_name_to_index["new"] = 2
    with pytest.raises(KeyError, match="unknown layer name"):
        session.pool_for_layer("model.layers.1.self_attn")
    with pytest.raises(TypeError, match="must be a string"):
        session.pool_index(0)


@pytest.mark.parametrize(
    ("layer_names", "exception"),
    [
        ((), ValueError),
        (("",), ValueError),
        (("layer", "layer"), ValueError),
        (("layer", 1), TypeError),
        ("layer", TypeError),
    ],
)  # type: ignore[misc]
def test_constructor_rejects_invalid_layer_names(
    layer_names: Any, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        _make_session(layer_names=layer_names)


def test_constructor_accepts_device_string() -> None:
    session = _make_session(device="cpu")

    assert session.decode_step_buf.device == torch.device("cpu")


def test_constructor_rejects_non_device_like_value() -> None:
    with pytest.raises(TypeError, match="torch.device or string"):
        _make_session(device=None)


def test_constructor_rejects_meta_device_string() -> None:
    with pytest.raises(ValueError, match="meta device is not supported"):
        _make_session(device="meta")


def test_begin_and_end_only_change_logical_state_and_scalar_contents() -> None:
    session = _make_session()
    tensors = tuple(_persistent_tensors(session))
    identities = tuple(id(tensor) for tensor in tensors)
    pointers = tuple(tensor.data_ptr() for tensor in tensors)
    for pool in session.pools:
        pool.unshared_key.fill_(11)
        pool.unshared_value.fill_(12)
    session.suffix_k_buf.fill_(13)
    session.suffix_v_buf.fill_(14)

    session.begin_session(beam_width=3, prefix_len=97)

    assert session.is_active
    assert session.beam_width == 3
    assert session.current_step == 0
    assert session.prefix_len == 97
    assert torch.equal(session.decode_step_buf, torch.tensor([1], dtype=torch.int32))
    assert torch.equal(session.actual_shared_kvlen_buf, torch.tensor([97], dtype=torch.int32))
    assert all(torch.all(pool.unshared_key == 11) for pool in session.pools)
    assert all(torch.all(pool.unshared_value == 12) for pool in session.pools)
    assert torch.all(session.suffix_k_buf == 13)
    assert torch.all(session.suffix_v_buf == 14)

    session.end_session()

    assert not session.is_active
    assert session.beam_width == session.current_step == session.prefix_len == 0
    assert torch.equal(session.decode_step_buf, torch.tensor([1], dtype=torch.int32))
    assert tuple(id(tensor) for tensor in tensors) == identities
    assert tuple(tensor.data_ptr() for tensor in tensors) == pointers


def test_end_session_clears_advanced_logical_request_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _make_session()
    monkeypatch.setattr(session_module, "select_unshared_kv", lambda *args: None)
    group = torch.arange(3, dtype=torch.int32)
    session.begin_session(beam_width=3, prefix_len=97)
    session.prepare_next_step(group, group)

    assert session.current_step == 1

    session.end_session()

    assert not session.is_active
    assert session.beam_width == session.current_step == session.prefix_len == 0
    assert torch.equal(session.decode_step_buf, torch.tensor([2], dtype=torch.int32))


def test_repeated_requests_reuse_every_persistent_storage() -> None:
    session = _make_session()
    tensors = tuple(_persistent_tensors(session))
    pointers = tuple(tensor.data_ptr() for tensor in tensors)

    for beam_width, prefix_len in ((4, 10), (1, 200), (3, 0)):
        session.begin_session(beam_width, prefix_len)
        assert session.current_step == 0
        assert torch.equal(session.decode_step_buf, torch.tensor([1], dtype=torch.int32))
        session.end_session()

    assert tuple(tensor.data_ptr() for tensor in tensors) == pointers


@pytest.mark.parametrize(
    ("beam_width", "prefix_len", "exception"),
    [
        (0, 0, ValueError),
        (5, 0, ValueError),
        (True, 0, TypeError),
        (1.5, 0, TypeError),
        (1, -1, ValueError),
        (1, True, TypeError),
        (1, torch.iinfo(torch.int32).max + 1, ValueError),
    ],
)  # type: ignore[misc]
def test_begin_rejects_invalid_capacity_or_prefix(
    beam_width: Any, prefix_len: Any, exception: type[Exception]
) -> None:
    session = _make_session()
    with pytest.raises(exception):
        session.begin_session(beam_width, prefix_len)
    assert not session.is_active
    assert session.current_step == 0
    assert torch.count_nonzero(session.decode_step_buf) == 0


def test_nested_begin_and_inactive_transitions_are_rejected() -> None:
    session = _make_session()
    identity = torch.arange(2, dtype=torch.int32)
    prefix = torch.arange(1, 3, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="no BeamSearch session"):
        session.prepare_next_step(identity, prefix)
    with pytest.raises(RuntimeError, match="no BeamSearch session"):
        session.end_session()

    session.begin_session(2, 8)
    with pytest.raises(RuntimeError, match="already active"):
        session.begin_session(2, 8)


def test_prepare_next_step_uses_zero_based_host_and_one_based_device_steps() -> None:
    session = _make_session(max_decode_steps=3)
    identity = torch.arange(3, dtype=torch.int32)
    prefix = torch.arange(1, 4, dtype=torch.int32)
    session.begin_session(3, 20)

    session.prepare_next_step(identity, prefix)
    assert session.current_step == 1
    assert torch.equal(session.decode_step_buf, torch.tensor([2], dtype=torch.int32))

    session.prepare_next_step(identity, prefix)
    assert session.current_step == 2
    assert torch.equal(session.decode_step_buf, torch.tensor([3], dtype=torch.int32))

    with pytest.raises(ValueError, match="beyond max_decode_steps"):
        session.prepare_next_step(identity, prefix)
    assert session.current_step == 2
    assert torch.equal(session.decode_step_buf, torch.tensor([3], dtype=torch.int32))
    assert session.is_active


@pytest.mark.parametrize(
    ("bad_name", "bad_input"),
    [
        ("beam_index", torch.arange(3, dtype=torch.int64)),
        ("beam_index", torch.arange(2, dtype=torch.int32)),
        ("group_token_num", torch.arange(3, dtype=torch.int64)),
        ("group_token_num", torch.arange(2, dtype=torch.int32)),
        ("group_token_num", torch.arange(3, dtype=torch.int32).reshape(1, 3)),
    ],
)  # type: ignore[misc]
def test_prepare_validation_failure_does_not_advance(
    bad_name: str, bad_input: torch.Tensor
) -> None:
    session = _make_session()
    valid = torch.arange(3, dtype=torch.int32)
    session.begin_session(3, 20)

    with pytest.raises((TypeError, ValueError)):
        if bad_name == "beam_index":
            session.prepare_next_step(bad_input, valid)
        else:
            session.prepare_next_step(valid, bad_input)

    assert session.current_step == 0
    assert torch.equal(session.decode_step_buf, torch.tensor([1], dtype=torch.int32))
    assert session.is_active


def test_prepare_rejects_tensors_on_a_different_device() -> None:
    session = _make_session()
    session.begin_session(3, 20)
    valid = torch.arange(3, dtype=torch.int32)
    other_device = torch.empty(3, dtype=torch.int32, device="meta")

    with pytest.raises(ValueError, match="must be on"):
        session.prepare_next_step(valid, other_device)

    assert session.current_step == 0
    assert torch.equal(session.decode_step_buf, torch.tensor([1], dtype=torch.int32))


def test_select_failure_does_not_advance_host_or_device_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _make_session()
    session.begin_session(3, 20)
    valid = torch.arange(3, dtype=torch.int32)

    def failing_select(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected select failure")

    monkeypatch.setattr(session_module, "select_unshared_kv", failing_select)
    with pytest.raises(RuntimeError, match="injected select failure"):
        session.prepare_next_step(valid, valid + 1)

    assert session.current_step == 0
    assert torch.equal(session.decode_step_buf, torch.tensor([1], dtype=torch.int32))
    assert session.is_active


def test_reset_buffers_zeros_in_place_without_changing_state_or_addresses() -> None:
    session = _make_session(dtype=torch.bfloat16)
    session.begin_session(2, 33)
    tensors = tuple(_persistent_tensors(session))
    pointers = tuple(tensor.data_ptr() for tensor in tensors)
    for pool in session.pools:
        pool.unshared_key.fill_(1)
        pool.unshared_value.fill_(2)
    session.suffix_k_buf.fill_(3)
    session.suffix_v_buf.fill_(4)

    session.reset_buffers()

    for pool in session.pools:
        assert torch.count_nonzero(pool.unshared_key) == 0
        assert torch.count_nonzero(pool.unshared_value) == 0
    assert torch.count_nonzero(session.suffix_k_buf) == 0
    assert torch.count_nonzero(session.suffix_v_buf) == 0
    assert session.is_active
    assert session.prefix_len == 33
    assert torch.equal(session.decode_step_buf, torch.tensor([1], dtype=torch.int32))
    assert tuple(tensor.data_ptr() for tensor in tensors) == pointers


def test_public_export_is_stable() -> None:
    assert BeamSearchSession is SessionImplementation
