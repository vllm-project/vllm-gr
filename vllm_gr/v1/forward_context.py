# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Forward state shared by GR workers and attention backends."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_fresh_prefill_capture: ContextVar[bool] = ContextVar(
    "vllm_gr_fresh_prefill_capture", default=False
)


@contextmanager
def fresh_prefill_capture() -> Iterator[None]:
    """Mark graphs whose runners guarantee fresh prompts on every replay.

    A decode capture can look like a fresh prompt but replay with cached KV.
    Only GR Prefill runners establish this scope for dense attention.
    """
    token = _fresh_prefill_capture.set(True)
    try:
        yield
    finally:
        _fresh_prefill_capture.reset(token)


def in_fresh_prefill_capture() -> bool:
    return _fresh_prefill_capture.get()
