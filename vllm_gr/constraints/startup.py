# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Platform adapter factory for canonical constraint tables."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Test-only override that forces the Torch oracle adapter regardless of the
# active platform. Used by CI / development only.
_CONSTRAINT_EXECUTION_ENV = "VLLM_GR_CONSTRAINT_EXECUTION"
_TORCH_OVERRIDE = "torch"


def create_constraint_execution(
    *,
    device: str | Any | None = None,
) -> Any:
    """Create the platform constrained-top-k adapter, failing explicitly."""
    override = os.environ.get(_CONSTRAINT_EXECUTION_ENV)
    if override is not None:
        if override != _TORCH_OVERRIDE:
            raise ValueError(
                f"unsupported {_CONSTRAINT_EXECUTION_ENV}={override!r}; "
                f"expected {_TORCH_OVERRIDE!r} or unset"
            )
        from vllm_gr.constraints.execution import TorchConstrainedTopKExecution

        logger.info(
            "vllm-gr: constraint execution overridden to Torch oracle (%s=%s); CI/dev only",
            _CONSTRAINT_EXECUTION_ENV,
            override,
        )
        return TorchConstrainedTopKExecution(device=device)

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
