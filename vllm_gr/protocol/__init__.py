# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned messages shared by vLLM-GR engine transports.

Keep this package initializer lightweight. In particular, importing session
errors must not import vLLM's ``SamplingParams`` and its runtime dependencies.
Consumers should import concrete types from ``messages``, ``codec``,
``version``, or ``errors``.
"""
