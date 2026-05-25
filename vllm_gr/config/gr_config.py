# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GRConfig — typed runtime configuration for GR inference.

Separates GR-specific *runtime / serving* parameters from the model
architecture parameters that live in HSTUConfig (config.json).

Feature names (uih, candidate, primary_id, task_names, etc.) are NOT
stored here — they are read directly from ``HSTUConfig`` (hf_config),
which is the single source of truth.

* ``max_seq_len``  — runtime truncation length for user-history sequences.
  Must be <= the model's ``HSTUConfig.max_seq_len`` (architecture limit).
* ``max_candidates`` — runtime per-request candidate limit.
  Must be <= the model's ``HSTUConfig.max_num_candidates`` (architecture limit).
"""

from dataclasses import dataclass


@dataclass
class GRConfig:
    """Typed GR runtime config.

    Separates GR-specific runtime/serving parameters from model architecture
    parameters that live in HSTUConfig (config.json).
    """

    max_seq_len: int = 512
    """Runtime truncation length for user-history (can be <= model max_seq_len)."""

    max_candidates: int = 1000
    """Runtime per-request candidate limit (can be <= model max_num_candidates)."""
