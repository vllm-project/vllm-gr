# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Long-lived storage and logical lifecycle for incremental BeamSearch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import torch

from vllm_gr.ops import select_unshared_kv
from vllm_gr.v1.beam.beam_attention_pool import BeamAttentionPool

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_INT32_MAX = torch.iinfo(torch.int32).max


def _require_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")


def _require_positive_int(name: str, value: int) -> None:
    _require_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")


class BeamSearchSession:
    """Own persistent OneRec-compatible buffers for one logical BeamSearch."""

    layer_names: tuple[str, ...]
    layer_name_to_index: Mapping[str, int]
    pools: tuple[BeamAttentionPool, ...]
    block_table_buf: torch.Tensor
    decode_step_buf: torch.Tensor
    actual_shared_kvlen_buf: torch.Tensor
    suffix_k_buf: torch.Tensor
    suffix_v_buf: torch.Tensor

    def __init__(
        self,
        layer_names: Sequence[str],
        max_beams: int,
        local_num_kv_heads: int,
        max_decode_steps: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if isinstance(layer_names, (str, bytes)) or not isinstance(layer_names, Sequence):
            raise TypeError("layer_names must be a sequence of strings")
        frozen_layer_names = tuple(layer_names)
        if not frozen_layer_names:
            raise ValueError("layer_names must not be empty")
        for layer_name in frozen_layer_names:
            if not isinstance(layer_name, str):
                raise TypeError("every layer name must be a string")
            if not layer_name:
                raise ValueError("layer names must not be empty")
        if len(set(frozen_layer_names)) != len(frozen_layer_names):
            raise ValueError("layer names must be unique")

        _require_positive_int("max_beams", max_beams)
        _require_positive_int("local_num_kv_heads", local_num_kv_heads)
        _require_positive_int("max_decode_steps", max_decode_steps)
        _require_positive_int("head_dim", head_dim)
        if isinstance(device, str):
            device = torch.device(device)
        elif not isinstance(device, torch.device):
            raise TypeError(f"device must be a torch.device or string, got {type(device).__name__}")
        if device.type == "meta":
            raise ValueError("meta device is not supported")
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}")
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be torch.float16 or torch.bfloat16, got {dtype}")

        self.layer_names = frozen_layer_names
        self.layer_name_to_index = MappingProxyType(
            {layer_name: index for index, layer_name in enumerate(self.layer_names)}
        )
        self.pools = tuple(
            BeamAttentionPool(
                max_beams=max_beams,
                local_num_kv_heads=local_num_kv_heads,
                max_decode_steps=max_decode_steps,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in self.layer_names
        )

        suffix_shape = (max_beams * max_decode_steps, local_num_kv_heads, head_dim)
        self.suffix_k_buf = torch.empty(suffix_shape, dtype=dtype, device=device)
        self.suffix_v_buf = torch.empty(suffix_shape, dtype=dtype, device=device)
        self.block_table_buf = torch.tensor([0], dtype=torch.int32, device=device)
        self.decode_step_buf = torch.zeros(1, dtype=torch.int32, device=device)
        self.actual_shared_kvlen_buf = torch.zeros(1, dtype=torch.int32, device=device)

        self._max_beams = max_beams
        self._max_decode_steps = max_decode_steps
        # Bare accelerator devices resolve to a concrete index during tensor
        # allocation (for example, cuda becomes cuda:0). Keep the resolved
        # device so validation agrees with tensors created from the same input.
        self._device = self.decode_step_buf.device
        self._is_active = False
        self._beam_width = 0
        self._current_step = 0
        self._prefix_len = 0

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def beam_width(self) -> int:
        return self._beam_width

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def prefix_len(self) -> int:
        return self._prefix_len

    def pool_index(self, layer_name: str) -> int:
        """Return the local Pool index for a canonical attention layer name."""

        if not isinstance(layer_name, str):
            raise TypeError("layer_name must be a string")
        try:
            return self.layer_name_to_index[layer_name]
        except KeyError as exc:
            raise KeyError(f"unknown layer name: {layer_name!r}") from exc

    def pool_for_layer(self, layer_name: str) -> BeamAttentionPool:
        """Return the Pool owned by a canonical attention layer name."""

        return self.pools[self.pool_index(layer_name)]

    def begin_session(self, beam_width: int, prefix_len: int) -> None:
        """Begin a logical request without replacing persistent storage."""

        if self._is_active:
            raise RuntimeError("a BeamSearch session is already active")
        _require_positive_int("beam_width", beam_width)
        if beam_width > self._max_beams:
            raise ValueError(
                f"beam_width must not exceed max_beams={self._max_beams}, got {beam_width}"
            )
        _require_int("prefix_len", prefix_len)
        if prefix_len < 0:
            raise ValueError(f"prefix_len must be non-negative, got {prefix_len}")
        if prefix_len > _INT32_MAX:
            raise ValueError(f"prefix_len must fit in int32, got {prefix_len}")

        self.decode_step_buf.fill_(1)
        self.actual_shared_kvlen_buf.fill_(prefix_len)
        self._beam_width = beam_width
        self._current_step = 0
        self._prefix_len = prefix_len
        self._is_active = True

    def prepare_next_step(
        self,
        beam_index: torch.Tensor,
        group_token_num: torch.Tensor,
    ) -> None:
        """Select active-beam history and advance only after success."""

        if not self._is_active:
            raise RuntimeError("no BeamSearch session is active")
        if self._current_step + 1 >= self._max_decode_steps:
            raise ValueError("cannot prepare another step beyond max_decode_steps")
        self._validate_group_tensor("beam_index", beam_index)
        self._validate_group_tensor("group_token_num", group_token_num)

        active_keys = [pool.active_key(self._beam_width) for pool in self.pools]
        active_values = [pool.active_value(self._beam_width) for pool in self.pools]
        select_unshared_kv(
            beam_index,
            active_keys,
            active_values,
            self.block_table_buf,
            group_token_num,
            self._current_step + 1,
            self._beam_width,
            len(self.pools),
        )

        next_step = self._current_step + 1
        self.decode_step_buf.fill_(next_step + 1)
        self._current_step = next_step

    def end_session(self) -> None:
        """End the logical request without touching persistent storage."""

        if not self._is_active:
            raise RuntimeError("no BeamSearch session is active")
        self._is_active = False
        self._beam_width = 0
        self._current_step = 0
        self._prefix_len = 0

    def reset_buffers(self) -> None:
        """Zero K/V and suffix buffers in place for tests and debugging."""

        for pool in self.pools:
            pool.reset()
        self.suffix_k_buf.zero_()
        self.suffix_v_buf.zero_()

    def _validate_group_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must use torch.int32")
        if tensor.device != self._device:
            raise ValueError(f"{name} must be on {self._device}, got {tensor.device}")
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if tensor.numel() != self._beam_width:
            raise ValueError(f"{name} must contain beam_width={self._beam_width} values")


__all__ = ["BeamSearchSession"]
