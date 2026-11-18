# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Capture/replay contract for the opt-in dense CUDA Prefill.

Dense Prefill attends over a layer's freshly projected K/V instead of the paged
cache, which is only correct when a request has no earlier context. A CPU check
at call time is enough for eager execution, but a CUDA graph freezes the choice
made at capture: a decode graph captured from ``seq_len == query_len == 1``
dummies would replay over cached context and silently drop it.

The contract therefore has two halves:

* Capture: while a stream is capturing, dense attention may only be selected
  inside :func:`fresh_prefill_capture`, which only the GR Prefill graph runners
  open. Native decode, mixed and other captures keep the paged path.
* Replay: when dense Prefill is requested, those runners replay a Prefill graph
  only if every scheduled request has ``num_computed_tokens == 0``
  (:func:`requires_fresh_prefill_replay`).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

ENV_DENSE_PREFILL = "VLLM_GR_DENSE_PREFILL"
DENSE_PREFILL_MODES = ("fa2", "cudnn")

_fresh_prefill_capture: ContextVar[bool] = ContextVar(
    "vllm_gr_fresh_prefill_capture", default=False
)


def requested_dense_prefill_mode() -> str | None:
    """The env-selected mode before config gating, or None when off or invalid."""
    raw = os.environ.get(ENV_DENSE_PREFILL, "off").strip().lower()
    return raw if raw in DENSE_PREFILL_MODES else None


def requires_fresh_prefill_replay() -> bool:
    """Whether Prefill graph replay must be limited to requests without context."""
    return requested_dense_prefill_mode() is not None


@contextmanager
def fresh_prefill_capture() -> Iterator[None]:
    """Mark a capture whose graphs only replay for fresh prompts."""
    token = _fresh_prefill_capture.set(True)
    try:
        yield
    finally:
        _fresh_prefill_capture.reset(token)


def in_fresh_prefill_capture() -> bool:
    return _fresh_prefill_capture.get()
