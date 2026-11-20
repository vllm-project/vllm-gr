# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Reorder dispatcher tests.

Covers ``vllm_gr.v1.beam.reorder``:

- ``group_token_num`` round-trip consistency: the prefix sums built by
  ``_group_token_num_from_parents`` must parse back to the same parents via
  ``_parent_beams_from_prefix`` (the inverse used by the torch reference of
  the xllm kernel) for every *sorted* parent — the ordering the fused kernel
  requires.  Unsorted parents round-trip to their sorted grouping, which is
  exactly why the dispatcher falls back for them.
- cascade-mode parity with the Torch reference for
  arbitrary parent orderings (permutation, duplicated source, shrink).
- xllm-mode parity on NPU: the fused ``select_unshared_kv`` vs the
  cascade reference vs the package's ``_select_unshared_kv_torch``
  golden for full-width sorted parents, and fallback correctness for
  unsorted / shrunk parents.
"""

from __future__ import annotations

import pytest
import torch

from vllm_gr.ops.onerec_decode import _parent_beams_from_prefix
from vllm_gr.v1.beam.reorder import (
    _group_token_num_from_parents,
    reorder_unshared_kv,
)

# ---------------------------------------------------------------------------
# group_token_num round trip (device-agnostic)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parent",
    [
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),  # identity, full width
        torch.tensor([0, 0, 1, 2], dtype=torch.int32),  # duplicated source
        torch.tensor([1, 3, 3, 3], dtype=torch.int32),  # duplicated source
    ],
)  # type: ignore[misc]
def test_group_token_num_round_trips_sorted_parents(parent: torch.Tensor) -> None:
    group_token_num = _group_token_num_from_parents(parent)
    restored = _parent_beams_from_prefix(group_token_num, 1, parent.numel())
    torch.testing.assert_close(restored.reshape(-1).to(torch.int32), parent, rtol=0, atol=0)


def test_group_token_num_unsorted_parent_yields_sorted_grouping() -> None:
    """The prefix sums can only encode the sorted grouping of the parents.

    This is the kernel contract: unsorted callers must not use the fused op
    (the dispatcher detects this and falls back).
    """
    parent = torch.tensor([2, 0, 1, 0], dtype=torch.int32)
    group_token_num = _group_token_num_from_parents(parent)
    assert group_token_num.tolist() == [2, 3, 4, 4]
    restored = _parent_beams_from_prefix(group_token_num, 1, 4).reshape(-1)
    torch.testing.assert_close(restored.to(torch.int32), parent.sort().values)


def test_group_token_num_matches_bincount_cumsum() -> None:
    parent = torch.tensor([0, 0, 1, 2], dtype=torch.int32)
    expected = torch.bincount(parent.to(torch.int64), minlength=4).cumsum(0)
    torch.testing.assert_close(
        _group_token_num_from_parents(parent), expected.to(torch.int32), rtol=0, atol=0
    )


# ---------------------------------------------------------------------------
# Parity harness
# ---------------------------------------------------------------------------


class _Pool:
    def __init__(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.unshared_key = key
        self.unshared_value = value


def _make_pools(
    shape: tuple[int, ...],
    layers: int,
    device: torch.device,
    seed: int,
) -> list[_Pool]:
    torch.manual_seed(seed)
    return [
        _Pool(
            torch.randn(shape, dtype=torch.float16).to(device),
            torch.randn(shape, dtype=torch.float16).to(device),
        )
        for _ in range(layers)
    ]
    # note: randn on CPU then moved, so all paths share identical inputs


def _reference_reorder(
    parent: torch.Tensor, pools: list[_Pool], slot: int, decode_step: int
) -> None:
    """Replay the cascade reference on CPU."""
    from vllm_gr.v1.beam.reorder import select_unshared_kv

    select_unshared_kv(
        parent.cpu(),
        slot,
        [p.unshared_key.cpu() for p in pools],
        [p.unshared_value.cpu() for p in pools],
        decode_step,
    )


_MAX_BEAMS, _KV_HEADS, _STEPS, _HEAD_DIM, _LAYERS = 4, 2, 3, 8, 2
_POOL_SHAPE = (1, _MAX_BEAMS, _KV_HEADS, _STEPS, _HEAD_DIM)


@pytest.mark.parametrize(
    "parent",
    [
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),  # identity
        torch.tensor([2, 0, 1, 3], dtype=torch.int32),  # permutation (unsorted)
        torch.tensor([0, 0, 1, 2], dtype=torch.int32),  # duplicated source
        torch.tensor([3, 1], dtype=torch.int32),  # shrink 4 -> 2
    ],
)  # type: ignore[misc]
def test_reorder_cascade_mode_matches_beam_kernels(
    monkeypatch: pytest.MonkeyPatch,
    parent: torch.Tensor,
) -> None:
    """The dispatcher in cascade mode must equal the original kernel."""
    from vllm_gr.v1.beam import reorder as reorder_mod

    monkeypatch.setattr(reorder_mod, "_resolve_mode", lambda: "cascade")

    seed = 10 + int(parent.sum())
    pools = _make_pools(_POOL_SHAPE, _LAYERS, torch.device("cpu"), seed)
    expected = _make_pools(_POOL_SHAPE, _LAYERS, torch.device("cpu"), seed)

    decode_step = 2
    _reference_reorder(parent, expected, slot=0, decode_step=decode_step)
    reorder_unshared_kv(parent, 0, pools, decode_step)

    for pool, ref in zip(pools, expected):
        torch.testing.assert_close(pool.unshared_key, ref.unshared_key, rtol=0, atol=0)
        torch.testing.assert_close(pool.unshared_value, ref.unshared_value, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# NPU parity: fused kernel + fallback
# ---------------------------------------------------------------------------

try:
    import torch_npu  # noqa: F401

    _NPU_IMPORTABLE = True
except (ImportError, RuntimeError):
    _NPU_IMPORTABLE = False

_NPU_AVAILABLE = _NPU_IMPORTABLE and hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.npu_test  # type: ignore[misc]
@pytest.mark.skipif(not _NPU_AVAILABLE, reason="Ascend NPU is not available")  # type: ignore[misc]
@pytest.mark.parametrize(
    "parent",
    [
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),  # identity, sorted
        torch.tensor([0, 0, 1, 2], dtype=torch.int32),  # duplicated source
        torch.tensor([1, 3, 3, 3], dtype=torch.int32),  # duplicated source
    ],
)  # type: ignore[misc]
def test_reorder_xllm_mode_fused_matches_references_on_npu(
    monkeypatch: pytest.MonkeyPatch,
    parent: torch.Tensor,
) -> None:
    """Full-width sorted parents: fused op == cascade reference == torch golden."""
    from vllm_gr.v1.beam import reorder as reorder_mod

    monkeypatch.setattr(reorder_mod, "_resolve_mode", lambda: "xllm")

    device = torch.device("npu")
    seed = 100 + int(parent.sum())
    pools = _make_pools(_POOL_SHAPE, _LAYERS, device, seed)
    expected = _make_pools(_POOL_SHAPE, _LAYERS, torch.device("cpu"), seed)

    decode_step = 2
    _reference_reorder(parent, expected, slot=0, decode_step=decode_step)
    reorder_unshared_kv(parent.to(device), 0, pools, decode_step)
    torch.npu.synchronize()

    for pool, ref in zip(pools, expected):
        torch.testing.assert_close(pool.unshared_key.cpu(), ref.unshared_key, rtol=0, atol=0)
        torch.testing.assert_close(pool.unshared_value.cpu(), ref.unshared_value, rtol=0, atol=0)


@pytest.mark.npu_test  # type: ignore[misc]
@pytest.mark.skipif(not _NPU_AVAILABLE, reason="Ascend NPU is not available")  # type: ignore[misc]
@pytest.mark.parametrize(
    "parent",
    [
        torch.tensor([2, 0, 1, 3], dtype=torch.int32),  # unsorted permutation
        torch.tensor([3, 1], dtype=torch.int32),  # shrink 4 -> 2
    ],
)  # type: ignore[misc]
def test_reorder_xllm_mode_falls_back_for_unsorted_or_shrunk(
    monkeypatch: pytest.MonkeyPatch,
    parent: torch.Tensor,
) -> None:
    """Unsorted / shrunk parents must stay correct via the cascade kernel."""
    from vllm_gr.v1.beam import reorder as reorder_mod

    monkeypatch.setattr(reorder_mod, "_resolve_mode", lambda: "xllm")
    monkeypatch.setattr(reorder_mod, "_FALLBACK_LOGGED", False)

    device = torch.device("npu")
    seed = 200 + int(parent.sum())
    pools = _make_pools(_POOL_SHAPE, _LAYERS, device, seed)
    expected = _make_pools(_POOL_SHAPE, _LAYERS, torch.device("cpu"), seed)

    decode_step = 2
    _reference_reorder(parent, expected, slot=0, decode_step=decode_step)
    reorder_unshared_kv(parent.to(device), 0, pools, decode_step)
    torch.npu.synchronize()

    for pool, ref in zip(pools, expected):
        torch.testing.assert_close(pool.unshared_key.cpu(), ref.unshared_key, rtol=0, atol=0)
        torch.testing.assert_close(pool.unshared_value.cpu(), ref.unshared_value, rtol=0, atol=0)


@pytest.mark.npu_test  # type: ignore[misc]
@pytest.mark.skipif(not _NPU_AVAILABLE, reason="Ascend NPU is not available")  # type: ignore[misc]
def test_reorder_xllm_fused_agrees_with_onerec_golden_on_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused-op call built by the dispatcher matches _select_unshared_kv_torch."""
    from vllm_gr.ops.onerec_decode import _select_unshared_kv_torch
    from vllm_gr.v1.beam import reorder as reorder_mod

    monkeypatch.setattr(reorder_mod, "_resolve_mode", lambda: "xllm")

    device = torch.device("npu")
    parent = torch.tensor([0, 0, 1, 2], dtype=torch.int32)
    seed = 4242
    pools = _make_pools(_POOL_SHAPE, _LAYERS, device, seed)
    golden = _make_pools(_POOL_SHAPE, _LAYERS, torch.device("cpu"), seed)

    decode_step = 2
    reorder_unshared_kv(parent.to(device), 0, pools, decode_step)
    torch.npu.synchronize()

    group_token_num = _group_token_num_from_parents(parent)
    _select_unshared_kv_torch(
        parent,
        [p.unshared_key for p in golden],
        [p.unshared_value for p in golden],
        torch.tensor([0], dtype=torch.int32),
        group_token_num,
        decode_step + 1,
        _MAX_BEAMS,
        _LAYERS,
    )

    for pool, ref in zip(pools, golden):
        torch.testing.assert_close(pool.unshared_key.cpu(), ref.unshared_key, rtol=0, atol=0)
        torch.testing.assert_close(pool.unshared_value.cpu(), ref.unshared_value, rtol=0, atol=0)


pytestmark = pytest.mark.cpu_test
