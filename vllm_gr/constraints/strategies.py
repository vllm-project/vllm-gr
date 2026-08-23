# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Frontend request marker and fixed constraint protocol limits."""

from __future__ import annotations

MAX_CONSTRAINT_GENERATION_STEPS = 3
MAX_CONSTRAINT_BEAM_WIDTH = 256


class ConstraintTableStrategy:
    """Stateless compatibility marker for transitional Beam metadata.

    It deliberately owns no path, digest, tokenizer, catalog, table, or
    Backend. Runtime resources live only on ``ModelRunner.constraint_backend``.
    """

    __slots__ = ()


def validate_constraint_generation_limits(*, beam_width: int, max_generate_steps: int) -> None:
    if beam_width > MAX_CONSTRAINT_BEAM_WIDTH:
        raise ValueError(
            f"constraint_table beam_width must be <= {MAX_CONSTRAINT_BEAM_WIDTH}, got {beam_width}"
        )
    if max_generate_steps > MAX_CONSTRAINT_GENERATION_STEPS:
        raise ValueError(
            "constraint_table supports at most "
            f"{MAX_CONSTRAINT_GENERATION_STEPS} generated steps, got {max_generate_steps}"
        )


def validate_constraint_table_protocol(
    *,
    is_constraint_table: bool,
    beam_protocol_available: bool,
    required: str,
    hint: str = "",
) -> None:
    if is_constraint_table and not beam_protocol_available:
        suffix = f" {hint}" if hint else ""
        raise RuntimeError(
            "constraint_table requires the beam-request protocol path "
            f"({required}); the fallback cannot carry step/prefix metadata.{suffix}"
        )
