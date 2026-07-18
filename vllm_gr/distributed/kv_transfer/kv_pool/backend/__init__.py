# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility namespace for KV pool backends.

vLLM-Ascend 0.22 owns the YuanRong backend and KV-pool implementation.  No
backend is re-exported here so importing vllm-gr does not eagerly require the
optional YuanRong runtime.
"""

__all__: list[str] = []
