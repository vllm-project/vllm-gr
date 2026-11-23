# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Canonical GR CPU token storage. No token-sized Python containers."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
import numpy.typing as npt


class NumpyTokenBuffer:
    """Native Request's mutable token interface backed by a bounded array.

    Prompt contents are copied in bulk once by native Request construction;
    only the short generated tail is appended. Slices/copies return arrays.

    ``_items`` exports the storage as a buffer so full iteration produces
    Python ints without boxing a NumPy scalar per token (``map(int, ...)``
    measured ~5x slower). Exporting a buffer forbids ``resize``; the array is
    sized exactly once in ``__init__`` and only item-assigned afterwards.
    """

    __slots__ = ("_items", "length", "storage")

    def __init__(self, owned_prompt: npt.NDArray[np.int64], max_new_tokens: int) -> None:
        self.length = len(owned_prompt)
        owned_prompt.resize((self.length + max_new_tokens,), refcheck=False)
        self.storage = owned_prompt
        self._items = memoryview(owned_prompt)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: Any) -> Any:
        """Read one token or a token view.

        Only integer and slice indices are supported. The list-backed request
        this buffer replaces also accepted NumPy fancy indices, which the engine
        never uses; those now raise instead of returning an array by accident.
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(self.length)
            if step < 0:
                # Normalized -1 stops are sentinels, not backing-array indices.
                return self.storage[: self.length][index]
            # Forward slices can address the backing array directly.
            return self.storage[start:stop:step]
        try:
            position = operator.index(index)
        except TypeError as error:
            raise TypeError("GR token index must be an int or a slice") from error
        if position < 0:
            position += self.length
        if position < 0 or position >= self.length:
            raise IndexError("GR token index out of range")
        return int(self.storage[position])

    def __iter__(self) -> Any:
        return iter(self._items[: self.length])

    def copy(self) -> npt.NDArray[np.int64]:
        return self.storage[: self.length].copy()

    def append(self, value: int) -> None:
        if self.length >= self.storage.size:
            raise ValueError("GR generated token capacity exceeded")
        self.storage[self.length] = value
        self.length += 1

    def extend(self, values: Any) -> None:
        count = len(values)
        if self.length + count > self.storage.size:
            raise ValueError("GR generated token capacity exceeded")
        self.storage[self.length : self.length + count] = values
        self.length += count


def initialize_numpy_request(request: Any, block_hasher: Any) -> Any:
    """Use native initialization, then adapt its mutable token state.

    Hashing is delayed until array state is installed. Native initialization
    already creates the owned prompt copy; resize it before exposing views.
    """
    from vllm.v1.request import Request
    from vllm.v1.utils import ConstantList

    original = getattr(Request, "_gr_numpy_original", Request.from_engine_core_request.__func__)
    native = original(Request, request, None)
    native._all_token_ids = NumpyTokenBuffer(native._all_token_ids, native.max_tokens)
    native.all_token_ids = ConstantList(native._all_token_ids)
    native._block_hasher = block_hasher
    native.update_block_hashes()
    return native


def install_numpy_request_factory() -> None:
    """Adapt only canonical GR requests; leave native ADD construction alone."""
    from vllm.v1.request import Request

    from vllm_gr.v1.beam.batch_contracts import BeamEngineCoreRequest

    if hasattr(Request, "_gr_numpy_original"):
        return
    original = Request.from_engine_core_request.__func__
    setattr(Request, "_gr_numpy_original", original)

    def from_engine_core_request(cls: Any, request: Any, block_hasher: Any) -> Any:
        if isinstance(request, BeamEngineCoreRequest):
            return initialize_numpy_request(request, block_hasher)
        return original(cls, request, block_hasher)

    Request.from_engine_core_request = classmethod(  # type: ignore[method-assign,assignment]
        from_engine_core_request
    )


def install_numpy_scheduler_data() -> None:
    """Snapshot native V2's prefill token container into a contiguous array."""
    from vllm.v1.core.sched.output import NewRequestData

    if getattr(NewRequestData.from_request, "_gr_numpy_tokens", False):
        return
    original = NewRequestData.from_request.__func__

    def from_request(
        cls: Any,
        request: Any,
        block_ids: Any,
        prefill_token_ids: Any = None,
        uses_mrope: bool = False,
    ) -> Any:
        if isinstance(prefill_token_ids, NumpyTokenBuffer):
            prefill_token_ids = prefill_token_ids.copy()
        return original(cls, request, block_ids, prefill_token_ids, uses_mrope=uses_mrope)

    setattr(from_request, "_gr_numpy_tokens", True)
    NewRequestData.from_request = classmethod(  # type: ignore[method-assign,assignment]
        from_request
    )
