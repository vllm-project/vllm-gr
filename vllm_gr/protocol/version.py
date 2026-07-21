# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-GR EngineCore protocol version contract."""

from __future__ import annotations

import msgspec

from vllm_gr.protocol.errors import GRIncompatibleVersionError


class GRProtocolVersion(msgspec.Struct, frozen=True, kw_only=True):
    """Semantic version for the GR EngineCore protocol.

    A different major version is always incompatible. A newer minor version
    may contain fields or commands this implementation cannot understand, so
    it is rejected until explicit capability negotiation is introduced.
    """

    major: int
    minor: int

    def ensure_supported_by(self, supported: GRProtocolVersion) -> None:
        """Raise when this peer version cannot be consumed by ``supported``."""
        if self.major != supported.major or self.minor > supported.minor:
            raise GRIncompatibleVersionError(received=self, supported=supported)


CURRENT_GR_PROTOCOL_VERSION = GRProtocolVersion(major=1, minor=0)
