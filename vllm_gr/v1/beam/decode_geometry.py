# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static packed-Decode layout, independent of session admission capabilities."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import torch


@dataclass(frozen=True)
class BeamDecodeGeometry:
    """Pure GR requests in native request order, one query per physical Beam.

    This is request geometry, not Beam state. It owns no tensors or workspace.
    Writers run during input preparation, outside forward Graph capture. Their
    outputs are caller-owned, address-stable buffers. Supporting several groups
    here does not enable B>1 admission or dynamic widths in a persistent backend.
    Mixed/noncontiguous native request slots need a separate explicit mapping.
    """

    request_widths: tuple[int, ...]
    prefix_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        widths = tuple(self.request_widths)
        prefixes = tuple(self.prefix_lengths)
        if not widths or len(widths) != len(prefixes):
            raise ValueError("Decode geometry needs matching nonempty request groups")
        if any(isinstance(w, bool) or not isinstance(w, Integral) or w <= 0 for w in widths):
            raise ValueError("Decode geometry requires positive integer widths")
        if any(isinstance(p, bool) or not isinstance(p, Integral) or p < 0 for p in prefixes):
            raise ValueError("Decode geometry requires nonnegative integer prefix lengths")
        object.__setattr__(self, "request_widths", tuple(int(w) for w in widths))
        object.__setattr__(self, "prefix_lengths", tuple(int(p) for p in prefixes))

    @property
    def num_requests(self) -> int:
        return len(self.request_widths)

    @property
    def num_rows(self) -> int:
        return sum(self.request_widths)

    @property
    def query_offsets(self) -> tuple[int, ...]:
        offsets = [0]
        for width in self.request_widths:
            offsets.append(offsets[-1] + width)
        return tuple(offsets)

    def matches_request_indices(self, indices: Sequence[int]) -> bool:
        if torch.is_tensor(indices):
            raise TypeError("request indices validation requires a Host integer sequence")
        if len(indices) != self.num_rows:
            return False
        offset = 0
        for request_index, width in enumerate(self.request_widths):
            if any(index != request_index for index in indices[offset : offset + width]):
                return False
            offset += width
        return True

    @staticmethod
    def _check_output(output: torch.Tensor, size: int, name: str, max_value: int) -> None:
        if (
            output.ndim != 1
            or output.numel() != size
            or output.dtype not in (torch.int32, torch.int64)
            or not output.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous integer vector of length {size}")
        if max_value > torch.iinfo(output.dtype).max:
            raise ValueError(f"{name} exceeds output integer dtype capacity")

    def write_request_indices(self, output: torch.Tensor) -> None:
        self._check_output(output, self.num_rows, "request indices", self.num_requests - 1)
        offset = 0
        for request_index, width in enumerate(self.request_widths):
            output[offset : offset + width].fill_(request_index)
            offset += width

    def write_static_seqlens(
        self,
        prefix_kv: torch.Tensor,
        prefix_cu_q: torch.Tensor,
        suffix_cu_q: torch.Tensor,
    ) -> None:
        # Validate every destination before writing any of them. No Host
        # tensor creation/copy is needed: all values are static layout data.
        self._check_output(prefix_kv, self.num_requests, "prefix KV lengths", max(self.prefix_lengths))
        self._check_output(prefix_cu_q, self.num_requests + 1, "prefix query offsets", self.num_rows)
        self._check_output(suffix_cu_q, self.num_rows + 1, "suffix query offsets", self.num_rows)
        if not (prefix_kv.device == prefix_cu_q.device == suffix_cu_q.device):
            raise ValueError("Decode geometry outputs must share a device")
        if len(set(self.prefix_lengths)) == 1:
            prefix_kv.fill_(self.prefix_lengths[0])
        else:
            for index, length in enumerate(self.prefix_lengths):
                prefix_kv[index : index + 1].fill_(length)
        if len(set(self.request_widths)) == 1:
            width = self.request_widths[0]
            torch.arange(
                0,
                (self.num_requests + 1) * width,
                step=width,
                dtype=prefix_cu_q.dtype,
                device=prefix_cu_q.device,
                out=prefix_cu_q,
            )
        else:
            for index, offset in enumerate(self.query_offsets):
                prefix_cu_q[index : index + 1].fill_(offset)
        torch.arange(
            self.num_rows + 1,
            dtype=suffix_cu_q.dtype,
            device=suffix_cu_q.device,
            out=suffix_cu_q,
        )
