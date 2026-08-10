# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference tensor operators for incremental BeamSearch KV storage."""

from __future__ import annotations

from collections.abc import Sequence

import torch

_SUPPORTED_KV_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)


def _validate_step(step: int, max_decode_steps: int) -> None:
    if type(step) is not int:
        raise TypeError("step must be an int")
    if step < 0 or step >= max_decode_steps:
        raise ValueError(f"step must be in [0, {max_decode_steps}), got {step}")


def _validate_pool(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 5 or tensor.shape[0] != 1:
        raise ValueError(
            f"{name} must have shape [1, max_beams, num_kv_heads, max_decode_steps, head_dim]"
        )
    if any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name} dimensions must be positive")
    if tensor.dtype not in _SUPPORTED_KV_DTYPES:
        raise TypeError(f"{name} must use torch.float16 or torch.bfloat16")


def _validate_pool_pair(
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    *,
    key_name: str = "unshared_key",
    value_name: str = "unshared_value",
) -> None:
    _validate_pool(key_name, unshared_key)
    _validate_pool(value_name, unshared_value)
    if unshared_value.shape != unshared_key.shape:
        raise ValueError(f"{value_name} must have the same shape as {key_name}")
    if unshared_value.dtype != unshared_key.dtype:
        raise TypeError(f"{value_name} must have the same dtype as {key_name}")
    if unshared_value.device != unshared_key.device:
        raise ValueError(f"{value_name} must be on {unshared_key.device}")


def _validate_cache_inputs(
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    step: int,
) -> int:
    _validate_pool_pair(unshared_key, unshared_value)
    _, max_beams, num_kv_heads, max_decode_steps, head_dim = unshared_key.shape
    _validate_step(step, max_decode_steps)

    if curr_key.ndim != 3:
        raise ValueError("curr_key must have shape [beam_width, num_kv_heads, head_dim]")
    if curr_value.shape != curr_key.shape:
        raise ValueError("curr_value must have the same shape as curr_key")

    beam_width = curr_key.shape[0]
    expected_shape = (beam_width, num_kv_heads, head_dim)
    if beam_width <= 0 or beam_width > max_beams:
        raise ValueError(f"active beam width must be in [1, {max_beams}], got {beam_width}")
    if curr_key.shape != expected_shape:
        raise ValueError(f"curr_key must have shape {expected_shape}, got {tuple(curr_key.shape)}")

    for name, tensor in (("curr_key", curr_key), ("curr_value", curr_value)):
        if tensor.dtype != unshared_key.dtype:
            raise TypeError(f"{name} must have the same dtype as unshared_key")
        if tensor.device != unshared_key.device:
            raise ValueError(f"{name} must be on {unshared_key.device}")
    return beam_width


def cache_unshared_kv(
    unshared_key: torch.Tensor,
    unshared_value: torch.Tensor,
    curr_key: torch.Tensor,
    curr_value: torch.Tensor,
    step: int,
) -> None:
    """Write one incremental decode step into fixed-address KV pools.

    ``curr_key`` and ``curr_value`` contain one token for each active beam and
    have shape ``[beam_width, num_kv_heads, head_dim]``. The function updates
    only active beams at ``step`` and never replaces either pool tensor.
    """

    beam_width = _validate_cache_inputs(unshared_key, unshared_value, curr_key, curr_value, step)
    unshared_key[0, :beam_width, :, step, :].copy_(curr_key)
    unshared_value[0, :beam_width, :, step, :].copy_(curr_value)


def _validate_select_inputs(
    parent: torch.Tensor,
    unshared_key_caches: Sequence[torch.Tensor],
    unshared_value_caches: Sequence[torch.Tensor],
    step: int,
) -> int:
    if len(unshared_key_caches) == 0:
        raise ValueError("at least one KV cache layer is required")
    if len(unshared_value_caches) != len(unshared_key_caches):
        raise ValueError("key and value cache lists must have the same length")
    if parent.ndim != 1:
        raise ValueError("parent must be one-dimensional")
    if parent.dtype not in _SUPPORTED_INDEX_DTYPES:
        raise TypeError("parent must use torch.int32 or torch.int64")

    first_key = unshared_key_caches[0]
    first_value = unshared_value_caches[0]
    _validate_pool_pair(first_key, first_value, key_name="key cache 0", value_name="value cache 0")
    _, max_beams, _, max_decode_steps, _ = first_key.shape
    _validate_step(step, max_decode_steps)

    beam_width = parent.numel()
    if beam_width <= 0 or beam_width > max_beams:
        raise ValueError(f"active beam width must be in [1, {max_beams}], got {beam_width}")
    if parent.device != first_key.device:
        raise ValueError(f"parent must be on {first_key.device}")

    expected_shape = first_key.shape
    expected_dtype = first_key.dtype
    expected_device = first_key.device
    for layer, (key, value) in enumerate(zip(unshared_key_caches, unshared_value_caches)):
        _validate_pool_pair(
            key,
            value,
            key_name=f"key cache {layer}",
            value_name=f"value cache {layer}",
        )
        if key.shape != expected_shape:
            raise ValueError(f"key cache {layer} must have shape {tuple(expected_shape)}")
        if key.dtype != expected_dtype:
            raise TypeError(f"key cache {layer} must use {expected_dtype}")
        if key.device != expected_device:
            raise ValueError(f"key cache {layer} must be on {expected_device}")

    if torch.any(parent < 0) or torch.any(parent >= beam_width):
        raise ValueError(f"parent values must be in [0, {beam_width})")
    return beam_width


def select_unshared_kv(
    parent: torch.Tensor,
    unshared_key_caches: Sequence[torch.Tensor],
    unshared_value_caches: Sequence[torch.Tensor],
    step: int,
) -> None:
    """Rearrange executed KV history using a new-beam-to-old-parent map.

    ``parent[i] = j`` means new beam ``i`` inherits history from old beam
    ``j``. A cloned source snapshot makes permutations and duplicate parents
    safe while the destination pools are updated in place.
    """

    beam_width = _validate_select_inputs(parent, unshared_key_caches, unshared_value_caches, step)
    parent_index = parent.to(dtype=torch.long)
    history = slice(0, step + 1)
    for key, value in zip(unshared_key_caches, unshared_value_caches):
        active_key = key[0, :beam_width, :, history, :]
        active_value = value[0, :beam_width, :, history, :]
        source_key = active_key.clone()
        source_value = active_value.clone()
        active_key.copy_(source_key.index_select(0, parent_index))
        active_value.copy_(source_value.index_select(0, parent_index))


__all__ = ["cache_unshared_kv", "select_unshared_kv"]
