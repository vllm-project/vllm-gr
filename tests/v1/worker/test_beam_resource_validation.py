# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm_gr.v1.worker.cuda import beam_search_runner as module

Runner = module.BeamSearchRunner


def _resources() -> Any:
    def fp(*shape: int) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.float16)

    context = SimpleNamespace(
        pools=[
            SimpleNamespace(unshared_key=fp(2, 4, 1, 3, 8), unshared_value=fp(2, 4, 1, 3, 8))
            for _ in range(2)
        ],
        decision=SimpleNamespace(
            sequence=torch.zeros(1, 4, 4, dtype=torch.int32),
            beam_scores=fp(1, 4),
            group_token_num=torch.zeros(1),
        ),
        aux_key_buf=fp(2, 4, 1, 3, 8),
        aux_value_buf=fp(2, 4, 1, 3, 8),
        pool_key_ptrs=torch.zeros(2, dtype=torch.int64),
        pool_value_ptrs=torch.zeros(2, dtype=torch.int64),
    )
    workspace = SimpleNamespace(
        key=SimpleNamespace(beam_width=4, top_k=4),
        max_batch=1,
        max_decode_steps=4,
        topk_token_ids=torch.zeros(1, 4, 4, dtype=torch.int32),
        selected_token_ids=torch.zeros(1, 4, dtype=torch.int32),
        selected_parent_ids=torch.zeros(1, 4, dtype=torch.int32),
        selected_scores=fp(1, 4),
        selected_flat_indices=torch.zeros(1, 4, dtype=torch.int32),
        next_sequence=torch.zeros(1, 4, 4, dtype=torch.int32),
    )
    return context, workspace


def _validate(context: Any, workspace: Any) -> Any:
    return Runner.validate_resources(context, workspace, beam_width=4, max_new_tokens=3, kv_slot=0)


def _full(context: Any, workspace: Any) -> Any:
    return Runner._validate_resources_full(
        context, workspace, beam_width=4, max_new_tokens=3, kv_slot=0
    )


PATHS = [
    "context.pools.0.unshared_key",
    "context.pools.0.unshared_value",
    "context.pools.1.unshared_key",
    "context.pools.1.unshared_value",
    "context.decision.sequence",
    "context.decision.beam_scores",
    "context.decision.group_token_num",
    "context.aux_key_buf",
    "context.aux_value_buf",
    "context.pool_key_ptrs",
    "context.pool_value_ptrs",
    "workspace.selected_token_ids",
    "workspace.selected_parent_ids",
    "workspace.selected_scores",
    "workspace.selected_flat_indices",
    "workspace.next_sequence",
]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("mutation", ["replace", "data", "set"])
def test_every_bound_address_is_live(path: str, mutation: str) -> None:
    with torch.inference_mode():
        context, workspace = _resources()
        key = _validate(context, workspace)
        parts = path.split(".")
        parent = context if parts[0] == "context" else workspace
        for part in parts[1:-1]:
            parent = parent[int(part)] if part.isdecimal() else getattr(parent, part)
        tensor = getattr(parent, parts[-1])
        address, identity = tensor.data_ptr(), id(tensor)
        if mutation == "replace":
            setattr(parent, parts[-1], tensor.clone())
        elif mutation == "data":
            tensor.data = tensor.clone()
        else:
            tensor.set_(tensor.clone())
        current = getattr(parent, parts[-1])
        assert current.data_ptr() != address
        if mutation != "replace":
            assert id(current) == identity
        actual = _validate(context, workspace)
        assert actual != key
        assert actual == _full(context, workspace)


@pytest.mark.parametrize(
    "case",
    [
        "pool_shape",
        "pool_stride",
        "pool_dtype",
        "aux_dtype",
        "aux_shape",
        "ptr_dtype",
        "ptr_none",
        "topk_device",
        "empty_pools",
        "fewer_pools",
        "width",
        "top_k",
        "batch",
        "steps",
    ],
)
def test_invalid_live_changes_match_full_validator(case: str) -> None:
    context, workspace = _resources()
    _validate(context, workspace)
    if case == "pool_shape":
        context.pools[1].unshared_value.resize_(2, 4, 1, 2, 8)
    elif case == "pool_stride":
        t = context.pools[1].unshared_value
        context.pools[1].unshared_value = torch.empty_strided(
            t.shape, (96, 24, 24, 1, 3), dtype=t.dtype
        )
    elif case == "pool_dtype":
        context.pools[1].unshared_value = context.pools[1].unshared_value.float()
    elif case == "aux_dtype":
        context.aux_value_buf = context.aux_value_buf.float()
    elif case == "aux_shape":
        context.aux_value_buf.resize_(1)
    elif case == "ptr_dtype":
        context.pool_key_ptrs = context.pool_key_ptrs.int()
    elif case == "ptr_none":
        context.pool_key_ptrs = None
    elif case == "topk_device":
        workspace.topk_token_ids = workspace.topk_token_ids.to("meta")
    elif case == "empty_pools":
        context.pools.clear()
    elif case == "fewer_pools":
        context.pools.pop()
    elif case == "width":
        workspace.key.beam_width = 8
    elif case == "top_k":
        workspace.key.top_k = 8
    elif case == "batch":
        workspace.max_batch = 0
    else:
        workspace.max_decode_steps = 1
    with pytest.raises(ValueError) as expected:
        _full(context, workspace)
    with pytest.raises(ValueError) as actual:
        _validate(context, workspace)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("kwargs", [{"beam_width": 5}, {"kv_slot": 2}, {"max_new_tokens": 8}])
def test_changed_call_parameters_cannot_reuse_validation(kwargs: Any) -> None:
    context, workspace = _resources()
    _validate(context, workspace)
    parameters = {"beam_width": 4, "max_new_tokens": 3, "kv_slot": 0}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        Runner.validate_resources(context, workspace, **parameters)


@pytest.mark.parametrize("factory", [None, Mock(side_effect=TypeError("unsupported API"))])
def test_unavailable_guard_api_uses_full_validation(monkeypatch: Any, factory: Any) -> None:
    context, workspace = _resources()
    monkeypatch.setattr(module, "_TensorGuards", factory)
    assert _validate(context, workspace) == _full(context, workspace)
    context.pool_value_ptrs = None
    with pytest.raises(ValueError, match="pointer tables"):
        _validate(context, workspace)


def test_storage_offset_is_included_by_live_address() -> None:
    context, workspace = _resources()
    key = _validate(context, workspace)
    workspace.selected_token_ids = torch.empty(5, dtype=torch.int32)[1:].view(1, 4)
    assert _validate(context, workspace) != key
    assert _validate(context, workspace) == _full(context, workspace)


def test_dispatch_mode_change_revalidates_safely() -> None:
    with torch.inference_mode():
        context, workspace = _resources()
        key = _validate(context, workspace)
    assert _validate(context, workspace) == key


def _ping_pong_resources() -> Any:
    context, workspace = _resources()
    context.enable_kv_pool_ping_pong = True
    shape = (len(context.pools), *context.pools[0].unshared_key.shape)
    context.standby_key_buf = torch.empty(shape, dtype=torch.float16)
    context.standby_value_buf = torch.empty_like(context.standby_key_buf)
    context.standby_key_ptrs = torch.zeros(len(context.pools), dtype=torch.int64)
    context.standby_value_ptrs = torch.zeros_like(context.standby_key_ptrs)
    for layer, pool in enumerate(context.pools):
        pool.standby_key = context.standby_key_buf[layer]
        pool.standby_value = context.standby_value_buf[layer]
    return context, workspace


@pytest.mark.l1
def test_ping_pong_key_tracks_standby_view_replacement() -> None:
    context, workspace = _ping_pong_resources()
    key = _validate(context, workspace)
    context.pools[0].standby_key = context.pools[0].standby_key.clone()
    assert _validate(context, workspace) != key


@pytest.mark.l1
@pytest.mark.parametrize("malformed", ["missing", "overlap"])
def test_ping_pong_rejects_invalid_standby_storage(malformed: str) -> None:
    context, workspace = _ping_pong_resources()
    _validate(context, workspace)
    if malformed == "missing":
        context.pools[0].standby_key = None
    else:
        # Cross K/V aliasing is unsafe even when the key bank does not alias
        # its corresponding source key. All source/destination banks must be
        # disjoint before a duplicate-parent gather can run safely.
        context.pools[0].standby_key = context.pools[0].unshared_value
    with pytest.raises(ValueError, match="ping-pong"):
        _validate(context, workspace)


pytestmark = pytest.mark.cpu_test
