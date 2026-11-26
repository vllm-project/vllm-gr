# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Lazy public Beam output fields shared by the V1 frontend."""

from dataclasses import dataclass
from typing import Any

from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence


class _LazyBeamField:
    """Data descriptor deferring ``tokens``/``logprobs``/``text`` materialization.

    Assignments land directly in the instance ``__dict__`` (the same place a
    plain dataclass field lives), so construction and in-place updates are
    unchanged.  A read first hands the beam to the per-request materializer
    exactly once; afterwards the field behaves like an ordinary attribute.
    """

    __slots__ = ("name",)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        materializer = obj.__dict__.get("_lazy_materializer")
        if materializer is not None:
            if obj.__dict__.get("_lazy_materializing", False):
                raise RuntimeError("recursive Beam output materialization")
            obj.__dict__["_lazy_materializing"] = True
            try:
                materializer(obj)
            finally:
                obj.__dict__.pop("_lazy_materializing", None)
            # A failed attempt must retain the callback for subsequent access.
            obj.__dict__["_lazy_materializer"] = None
        return obj.__dict__.get(self.name)

    def __set__(self, obj: Any, value: Any) -> None:
        obj.__dict__[self.name] = value


@dataclass
class _LazyMaterializedBeamSequence(BeamSearchSequence):  # type: ignore[misc]
    """Beam whose final fields materialize on first access.

    The class body rebinds the three expensive fields to descriptors without
    redeclaring them as dataclass fields (no annotations), so ``fields()``,
    ``replace()`` and the generated ``__init__`` keep working unchanged.
    """

    tokens = _LazyBeamField()
    logprobs = _LazyBeamField()
    text = _LazyBeamField()
