# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Platform adapter factory for canonical constraint tables."""

from __future__ import annotations

from typing import Any


def create_constraint_execution(
    *,
    device: str | Any | None = None,
) -> Any:
    """Create the platform constrained-top-k adapter, failing explicitly."""
    from vllm_gr.platforms.loader import activate_current_backend

    provider = activate_current_backend()
    adapter = provider.create_constrained_topk_execution(device=device)
    if adapter is None:
        platform = provider.device_type
        raise RuntimeError(
            "constraint.mode='table' cannot start: the active platform "
            f"{platform!r} does not implement Constraint Table execution."
        )
    if not adapter.is_available():
        platform = provider.device_type
        raise RuntimeError(
            "constraint.mode='table' cannot start: the active platform "
            f"{platform!r} provides a Constraint Table adapter but its custom "
            "operator is unavailable (is_available()=False)."
        )
    return adapter
