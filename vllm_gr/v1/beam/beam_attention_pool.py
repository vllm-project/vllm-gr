# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixed-address storage for one local Beam Attention layer."""

from __future__ import annotations

import torch

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")


class BeamAttentionPool:
    """Own fixed-address incremental decode K/V storage for one local layer."""

    unshared_key: torch.Tensor
    unshared_value: torch.Tensor

    def __init__(
        self,
        max_beams: int,
        local_num_kv_heads: int,
        max_decode_steps: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        _validate_positive_int("max_beams", max_beams)
        _validate_positive_int("local_num_kv_heads", local_num_kv_heads)
        _validate_positive_int("max_decode_steps", max_decode_steps)
        _validate_positive_int("head_dim", head_dim)

        if not isinstance(device, torch.device):
            raise TypeError(f"device must be a torch.device, got {type(device).__name__}")
        if device.type == "meta":
            raise ValueError("meta device is not supported")

        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}")
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be torch.float16 or torch.bfloat16, got {dtype}")

        shape = (1, max_beams, local_num_kv_heads, max_decode_steps, head_dim)
        self.unshared_key = torch.empty(shape, dtype=dtype, device=device)
        self.unshared_value = torch.empty(shape, dtype=dtype, device=device)

    def active_key(self, beam_width: int) -> torch.Tensor:
        """Return the active-beam view of the key storage."""
        self._validate_beam_width(beam_width)
        return self.unshared_key[:, :beam_width]

    def active_value(self, beam_width: int) -> torch.Tensor:
        """Return the active-beam view of the value storage."""
        self._validate_beam_width(beam_width)
        return self.unshared_value[:, :beam_width]

    def reset(self) -> None:
        """Clear both storages in place without changing their addresses."""
        self.unshared_key.zero_()
        self.unshared_value.zero_()

    def nbytes(self) -> int:
        """Return the bytes owned by the key and value storages."""
        return (
            self.unshared_key.numel() * self.unshared_key.element_size()
            + self.unshared_value.numel() * self.unshared_value.element_size()
        )

    def _validate_beam_width(self, beam_width: int) -> None:
        _validate_positive_int("beam_width", beam_width)
        max_beams = self.unshared_key.shape[1]
        if beam_width > max_beams:
            raise ValueError(f"beam_width must not exceed max_beams={max_beams}, got {beam_width}")
