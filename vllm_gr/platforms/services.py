# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Backend-owned execution services, so shared code never selects by device name.

The provider layer already answers "which backend am I". This module answers the
next question: "which implementation performs this piece of execution". Every
answer is returned by the selected provider, so a call site asks for a service
instead of comparing a device string.

Each service is intentionally narrow, and its hardware module imports live inside
method bodies. Shared code therefore never imports a CUDA or Ascend module merely
to decide what to do, and ``vllm_gr.platforms`` keeps its import isolation.

The service set mirrors the RFC #706 execution table. Stage execution, native
retirement and startup graph capture are the responsibilities that still carried
device-name branches in shared code at provider API version 2. Attention,
constraint and ModelRunner patch selection already flow through provider
factories on their production paths.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StageExecutionService(Protocol):
    """Execute, retire and pre-capture one GR Beam stage for a backend.

    ``capture_startup_graph`` carries the per-backend startup capture policy. A
    backend returns ``True`` when the capture it owns succeeded, ``False`` when
    capture does not apply to this runner, and raises when the capture was
    attempted and failed. Shared startup code therefore never inspects a device
    type to decide which of those three outcomes is expected.

    ``admit_unconstrained_beam`` carries the per-backend admission rules for an
    unconstrained session. A backend that cannot serve that configuration raises;
    a backend that can returns normally.
    """

    def execute(
        self,
        worker: Any,
        scheduled: Any,
        original: Callable[..., Any],
        *args: Any,
        hooks_type: Any = None,
        **kwargs: Any,
    ) -> Any: ...

    def retire(self, worker: Any, scheduled: Any) -> Any: ...

    def capture_startup_graph(
        self,
        worker: Any,
        *,
        hooks_type: Any = None,
        allow_eager_fallback: bool = False,
    ) -> bool: ...

    def admit_unconstrained_beam(self, worker: Any) -> None: ...


__all__ = ["StageExecutionService"]
