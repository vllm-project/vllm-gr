# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from collections.abc import Sequence

SUPPORTED_BEAM_WIDTH_SCHEDULE_STEPS = (1, 2, 3)


def per_parent_width_error(
    beam_width: int,
    dynamic_beam_widths: Sequence[int] | None,
    per_parent_beam_widths: Sequence[int],
) -> str | None:
    """Check candidate capacity after the caller validates both schedules."""
    retained_widths = dynamic_beam_widths or (beam_width,) * len(per_parent_beam_widths)
    if per_parent_beam_widths[0] != retained_widths[0]:
        return "the first per-parent beam width must equal the first retained width"

    input_width = retained_widths[0]
    for candidate_width, output_width in zip(per_parent_beam_widths[1:], retained_widths[1:]):
        if output_width > input_width * candidate_width:
            return "each retained width must not exceed input_width * per_parent_beam_width"
        input_width = output_width
    return None


def final_retained_width(beam_width: int, dynamic_beam_widths: Sequence[int] | None) -> int:
    """Return the final retained width of a validated request."""
    return beam_width if dynamic_beam_widths is None else dynamic_beam_widths[-1]
