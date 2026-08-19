# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import msgspec

BeamLogprobsMode = Literal["none", "selected", "top_k"]


class BeamOutputSpec(msgspec.Struct, frozen=True):
    """Caller-visible Beam output contract."""

    result_width: int
    logprobs_mode: BeamLogprobsMode = "none"
    top_logprobs: int = 0

    def validate(self, beam_width: int) -> None:
        if not 1 <= self.result_width <= beam_width:
            raise ValueError(
                f"result_width must be between 1 and beam_width ({beam_width}), "
                f"got {self.result_width}"
            )
        if self.logprobs_mode not in ("none", "selected", "top_k"):
            raise ValueError(f"unsupported logprobs_mode {self.logprobs_mode!r}")
        if self.logprobs_mode == "top_k":
            if not 1 <= self.top_logprobs <= beam_width:
                raise ValueError(
                    "top_logprobs must be between 1 and beam_width when logprobs_mode='top_k'"
                )
        elif self.top_logprobs != 0:
            raise ValueError("top_logprobs must be 0 unless logprobs_mode='top_k'")
