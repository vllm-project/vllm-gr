# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Format platform class names for diagnostics and wire compatibility."""

from __future__ import annotations


def qualified_class_name(cls: type) -> str:
    """Return ``module.qualname`` without importing or instantiating the class."""
    module = getattr(cls, "__module__", "") or ""
    qualname = getattr(cls, "__qualname__", "") or ""
    return f"{module}.{qualname}".lstrip(".")
