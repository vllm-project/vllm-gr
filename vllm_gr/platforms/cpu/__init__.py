# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Experimental eager CPU backend (Apple Silicon macOS, Linux CPU)."""

from .provider import CPU_BACKEND, CPUBackendProvider

__all__ = ["CPU_BACKEND", "CPUBackendProvider"]
