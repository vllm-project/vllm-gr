# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Bulk CPU-array staging for GR Prefill, retaining native device writes."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch


class NumpyStagedContents:
    """Concatenate array chunks in bulk rather than list.extend per token."""

    def __init__(self, dtype: Any) -> None:
        self.dtype = np.dtype(dtype)
        self.chunks: list[npt.NDArray[Any]] = []
        self.length = 0

    def __len__(self) -> int:
        return self.length

    def extend(self, values: Any) -> None:
        array = np.asarray(values, dtype=self.dtype)
        if array.ndim != 1:
            raise ValueError("staged GR tokens must be one-dimensional")
        self.chunks.append(array)
        self.length += array.size

    def append(self, value: int) -> None:
        self.extend(np.asarray([value], dtype=self.dtype))

    def array(self) -> npt.NDArray[Any]:
        if not self.chunks:
            return np.empty(0, dtype=self.dtype)
        if len(self.chunks) == 1:
            return self.chunks[0]
        return np.concatenate(self.chunks)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> npt.NDArray[Any]:
        """Materialize once for native UVA NumPy assignment.

        Without this protocol, assignment visits __getitem__ for every token
        and concatenates all chunks on every visit. Keep copy optional for
        NumPy 1.x callers while honoring the NumPy 2.x no-copy contract.
        """
        if copy is False and len(self.chunks) != 1:
            raise ValueError("Cannot avoid a copy when combining staged token chunks")
        values = self.array()
        if dtype is not None and np.dtype(dtype) != values.dtype:
            if copy is False:
                raise ValueError("Cannot avoid a copy when converting staged token dtype")
            return values.astype(dtype)
        return values.copy() if copy else values

    def __getitem__(self, index: Any) -> Any:
        return self.array()[index]

    def clear(self) -> None:
        self.chunks.clear()
        self.length = 0


def array_h2d(values: npt.NDArray[Any], dtype: torch.dtype, device: Any) -> torch.Tensor:
    """Copy into PyTorch-owned storage before asynchronous device transfer.

    np.copyto accepts read-only received buffers; torch.from_numpy would warn
    and expose writable Tensor access to those buffers. The pinned allocator
    and .to() retain transfer lifetime through the device copy.
    """
    from vllm.utils.platform_utils import is_pin_memory_available

    pin = torch.device(device).type != "cpu" and is_pin_memory_available()
    cpu = torch.empty(values.size, dtype=dtype, device="cpu", pin_memory=pin)
    np.copyto(cpu.numpy(), values, casting="same_kind")
    return cpu.to(device=device, non_blocking=True)


def install_numpy_staging() -> None:
    from vllm.v1.worker.gpu import buffer_utils

    cls = buffer_utils.StagedWriteTensor
    if getattr(cls.stage_write, "_gr_numpy_tokens", False):
        return
    original_stage = cls.stage_write
    original_h2d = buffer_utils.async_tensor_h2d

    def stage_write(self: Any, index: int, start: int, values: Any) -> None:
        if not isinstance(values, np.ndarray):
            return original_stage(self, index, start, values)
        if index < 0 or start < 0 or values.ndim != 1:
            raise ValueError("invalid array staged write")
        if values.size == 0:
            return
        if not isinstance(self._staged_write_contents, NumpyStagedContents):
            contents = NumpyStagedContents(str(self.dtype).removeprefix("torch."))
            if self._staged_write_contents:
                contents.extend(self._staged_write_contents)
            self._staged_write_contents = contents
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(start)
        self._staged_write_contents.extend(values)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def async_tensor_h2d(data: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(data, NumpyStagedContents):
            data = data.array()
            # vLLM 0.30 accepts arrays directly. Keep received read-only storage
            # out of torch.from_numpy, which exposes writable Tensor access.
            if not data.flags.writeable:
                data = data.copy()
        return original_h2d(data, *args, **kwargs)

    setattr(stage_write, "_gr_numpy_tokens", True)
    cls.stage_write = stage_write  # type: ignore[method-assign]
    buffer_utils.async_tensor_h2d = async_tensor_h2d
