# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Static runtime descriptors for the closed-set Phase 1 registry.

Pure data only — must not import torch, vllm, or other heavy dependencies.
"""

from __future__ import annotations

from typing import Final, FrozenSet

FAMILY_NATIVE: Final = "native"
FAMILY_LLM: Final = "llm"
FAMILY_GR_RECOMMEND: Final = "gr_recommend"

FAMILIES: Final[FrozenSet[str]] = frozenset(
    {FAMILY_NATIVE, FAMILY_LLM, FAMILY_GR_RECOMMEND}
)

MODEL_ARCH_HSTU: Final = "hstu"

MODEL_ARCHS_BY_FAMILY: Final[dict[str, FrozenSet[str]]] = {
    FAMILY_GR_RECOMMEND: frozenset({MODEL_ARCH_HSTU}),
}

CAPABILITY_GR_BEAM: Final = "gr_beam"

CAPABILITIES: Final[FrozenSet[str]] = frozenset({CAPABILITY_GR_BEAM})

CAPABILITY_HOST_FAMILY: Final[dict[str, str]] = {
    CAPABILITY_GR_BEAM: FAMILY_LLM,
}

# Families that must not carry any capability in Phase 1.
CAPABILITY_FORBIDDEN_FAMILIES: Final[FrozenSet[str]] = frozenset(
    {FAMILY_NATIVE, FAMILY_GR_RECOMMEND}
)

# User-facing VLLM_GR_MODE values (internal ``llm`` is not exposed).
USER_MODE_NATIVE: Final = "native"
USER_MODE_GR_RECOMMEND: Final = "gr_recommend"
USER_MODE_GR_BEAM: Final = "gr_beam"

USER_MODES: Final[FrozenSet[str]] = frozenset(
    {USER_MODE_NATIVE, USER_MODE_GR_RECOMMEND, USER_MODE_GR_BEAM}
)

DIAG_SOURCE_CLI: Final = "cli"
DIAG_SOURCE_API: Final = "api"
DIAG_SOURCE_ENV: Final = "env"
DIAG_SOURCE_AUTO: Final = "auto"
DIAG_SOURCE_DEFAULT: Final = "default"

DIAG_SOURCES: Final[FrozenSet[str]] = frozenset(
    {
        DIAG_SOURCE_CLI,
        DIAG_SOURCE_API,
        DIAG_SOURCE_ENV,
        DIAG_SOURCE_AUTO,
        DIAG_SOURCE_DEFAULT,
    }
)
