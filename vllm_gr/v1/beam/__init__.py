# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Context shared by Beam workers and attention backends."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

BEAM_DATA_VAR: ContextVar[dict[str, Any] | None] = ContextVar("beam_data", default=None)

__all__ = ["BEAM_DATA_VAR"]
