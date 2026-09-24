# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Beam runtime primitives."""

from __future__ import annotations

from vllm_gr.v1.beam.context import BEAM_DATA_VAR
from vllm_gr.v1.beam.geometry import _compute_beam_bounds

__all__ = ["BEAM_DATA_VAR", "_compute_beam_bounds"]
