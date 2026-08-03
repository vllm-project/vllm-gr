# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Thin RECIF model integration built on vLLM's Qwen3-MoE model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch.nn as nn

from vllm_gr.entrypoints.beam_search_config import (
    BeamSearchStep,
    register_beam_search_step_builder,
)

VOCAB = 8192
NUM_SID_LEVELS = 3
FULL_VOCAB = NUM_SID_LEVELS * VOCAB
RECIF_ARCHITECTURE = "RecifForCausalLM"

# RECIF's hierarchy is model semantics, not a per-request sampling option.
_RECIF_BRANCH_FACTORS = (8, 8, 8)
_RECIF_ALLOWED_TOKEN_RANGES = tuple(
    range(level * VOCAB, (level + 1) * VOCAB) for level in range(NUM_SID_LEVELS)
)
RECIF_MAX_LOGPROBS = max(_RECIF_BRANCH_FACTORS)
RECIF_MAX_BEAM_WIDTH = 8 * 8 * 8

try:
    from vllm.model_executor.models.qwen3_moe import (
        Qwen3MoeForCausalLM as _VllmQwen3MoeForCausalLM,
    )
except ImportError:  # pragma: no cover - helper-only test environment

    class _VllmQwen3MoeForCausalLM(nn.Module):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__()
            raise RuntimeError("vLLM is required to instantiate RecifForCausalLM")


class RecifForCausalLM(_VllmQwen3MoeForCausalLM):
    """RECIF execution model using an already converted HF checkpoint."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        logprobs_mode = vllm_config.model_config.logprobs_mode
        if logprobs_mode != "processed_logprobs":
            raise ValueError(
                f"RECIF requires logprobs_mode='processed_logprobs', got {logprobs_mode!r}."
            )
        config = vllm_config.model_config.hf_text_config
        if int(config.vocab_size) != FULL_VOCAB:
            raise ValueError(f"RECIF requires vocab_size={FULL_VOCAB}, got {config.vocab_size}.")
        if bool(getattr(config, "tie_word_embeddings", False)):
            raise ValueError(
                "RECIF requires tie_word_embeddings=false because the SID "
                "heads are independent from the input embedding table."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)


def build_beam_search_steps(
    max_tokens: int,
    beam_width: int,
    *,
    branch_factors: Sequence[int] | None = None,
) -> list[BeamSearchStep]:
    """Build RECIF's intrinsic hierarchical candidate-selection policy.

    ``branch_factors`` is intentionally not exposed as a request parameter.
    The keyword-only hook is reserved for a future startup-level model option.
    """

    factors = tuple(branch_factors or _RECIF_BRANCH_FACTORS)
    if not 1 <= beam_width <= RECIF_MAX_BEAM_WIDTH:
        raise ValueError(
            f"RECIF beam_width must be between 1 and {RECIF_MAX_BEAM_WIDTH}, got {beam_width}"
        )
    if max_tokens != NUM_SID_LEVELS:
        raise ValueError(
            f"RECIF beam search requires max_tokens={NUM_SID_LEVELS}, got {max_tokens}"
        )
    if len(factors) != NUM_SID_LEVELS or any(width <= 0 for width in factors):
        raise ValueError(f"RECIF branch factors must contain {NUM_SID_LEVELS} positive values")

    steps: list[BeamSearchStep] = []
    candidate_count = 1
    for expand_width, allowed_token_ids in zip(
        factors,
        _RECIF_ALLOWED_TOKEN_RANGES,
        strict=True,
    ):
        candidate_count *= expand_width
        steps.append(
            BeamSearchStep(
                expand_width=expand_width,
                keep_width=min(candidate_count, beam_width),
                allowed_token_ids=list(allowed_token_ids),
            )
        )
    return steps


def register_recif_model() -> None:
    """Register RECIF's execution class and model-owned search policy."""

    register_beam_search_step_builder(RECIF_ARCHITECTURE, build_beam_search_steps)
    try:
        from vllm import ModelRegistry
    except ImportError:  # pragma: no cover - helper-only test environment
        return
    ModelRegistry.register_model(
        RECIF_ARCHITECTURE,
        "vllm_gr.models.recif:RecifForCausalLM",
    )
