# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Constraint strategy abstractions shared by serving and offline beam search."""

from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintBackend,
    ConstraintBackendInfo,
    ConstraintMetadata,
    ConstraintOutput,
)
from vllm_gr.constraints.coordinator import BeamConstraintCoordinator
from vllm_gr.constraints.loader import load_constraint_backend
from vllm_gr.constraints.models import (
    CandidateLayout,
    ParentCandidateBatch,
    ParentCandidateBuilder,
)
from vllm_gr.constraints.strategies import (
    ConstraintStrategy,
    ConstraintTableStrategy,
    TrieConstraintStrategy,
    create_constraint_strategy,
)

__all__ = [
    "BeamConstraintCoordinator",
    "CandidateLayout",
    "ConstraintBackend",
    "ConstraintBackendInfo",
    "ConstraintMetadata",
    "ConstraintOutput",
    "ConstraintStrategy",
    "ConstraintTableStrategy",
    "ParentCandidateBatch",
    "ParentCandidateBuilder",
    "TrieConstraintStrategy",
    "create_constraint_strategy",
    "INVALID_LOGIT",
    "load_constraint_backend",
]
