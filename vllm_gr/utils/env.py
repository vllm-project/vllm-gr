# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared environment-value parsing helpers."""

import os

_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "y", "on"})


def is_truthy_env(name: str) -> bool:
    """Return whether an environment variable contains a supported true value."""
    return os.getenv(name, "").lower() in _TRUTHY_ENV_VALUES
