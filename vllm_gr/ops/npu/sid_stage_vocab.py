# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Tokenizer-derived SID stage vocabularies, independent of catalog tables.

Each stage permits every tokenizer SID token of its corresponding family.
This is a stage-format constraint; it does not check whether a generated
SA/SB/SC triple exists in a recommendation catalog.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import regex as re
import torch

from vllm_gr.constraints.backend import ConstraintOutput
from vllm_gr.ops.npu.unconstrained_topk import select_unconstrained_candidates

_SID_TOKEN = re.compile(r"<s_([abc])_([0-9]+)>")


@dataclass(frozen=True, slots=True)
class NPUSIDStageVocab:
    """Three ascending global-ID maps and their common projection capacity."""

    full_vocab_size: int
    vocab_size: int
    stage_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    @property
    def device_memory_bytes(self) -> int:
        return sum(ids.numel() * ids.element_size() for ids in self.stage_token_ids)

    @property
    def stage_token_counts(self) -> tuple[int, int, int]:
        a, b, c = self.stage_token_ids
        return int(a.numel()), int(b.numel()), int(c.numel())


def build_sid_stage_vocab(
    tokenizer: Any, vocab_size: int, device: torch.device | str
) -> NPUSIDStageVocab:
    """Validate tokenizer IDs on the host once, then publish device maps.

    IDs need not be contiguous and families need not have equal sizes. The
    exact spelling is ``<s_a_N>``, ``<s_b_N>``, or ``<s_c_N>`` with an ASCII
    decimal index. Duplicate IDs or duplicate numeric names are rejected,
    rather than silently changing normalization or tie ordering.
    """
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError("SID stage vocabulary requires a positive model vocab_size")
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise TypeError("SID stage vocabulary requires tokenizer.get_vocab()")
    vocabulary = get_vocab()
    if not isinstance(vocabulary, Mapping):
        raise TypeError("tokenizer.get_vocab() must return a token-to-ID mapping")
    stages: dict[str, list[int]] = {family: [] for family in "abc"}
    seen_ids: dict[int, str] = {}
    seen_names: set[tuple[str, int]] = set()
    for token, token_id in vocabulary.items():
        if not isinstance(token, str):
            raise TypeError("tokenizer vocabulary keys must be strings")
        match = _SID_TOKEN.fullmatch(token)
        if match is None:
            continue
        if type(token_id) is not int or not 0 <= token_id < vocab_size:
            raise ValueError(
                f"SID token {token!r} has an invalid model vocabulary ID: {token_id!r}"
            )
        if token_id > torch.iinfo(torch.int32).max:
            raise ValueError(f"SID token {token!r} does not fit the int32 candidate ID contract")
        family, number = match.group(1), int(match.group(2))
        if token_id in seen_ids:
            raise ValueError(f"SID tokens {seen_ids[token_id]!r} and {token!r} share ID {token_id}")
        if (family, number) in seen_names:
            raise ValueError(f"SID stage {family!r} contains duplicate numeric index {number}")
        seen_ids[token_id] = token
        seen_names.add((family, number))
        stages[family].append(token_id)
    missing = [family for family, ids in stages.items() if not ids]
    if missing:
        raise ValueError(f"SID stage vocabulary has no tokens for families: {', '.join(missing)}")

    # Ascending global IDs make stable local TopK ties identical to the
    # full-head stage-mask baseline, including the cutoff boundary.
    def ids(family: str) -> torch.Tensor:
        return torch.tensor(sorted(stages[family]), dtype=torch.int64, device=device)

    return NPUSIDStageVocab(
        vocab_size, max(map(len, stages.values())), (ids("a"), ids("b"), ids("c"))
    )


def select_sid_stage_candidates(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: int,
    stage_index: int,
    stage_vocab: NPUSIDStageVocab,
    *,
    compact_logits: bool = False,
) -> ConstraintOutput:
    """Normalize across the complete current SID family, then select TopK.

    Full logits index the tokenizer's global rows. Compact logits contain
    those rows in map order, followed by projection padding. Padding is
    removed before normalization, so it can never acquire probability mass.
    """
    if type(stage_index) is not int or stage_index not in (0, 1, 2):
        raise ValueError("SID stage_index must be 0, 1, or 2")
    if logits.ndim != 2:
        raise ValueError("SID stage logits must be two-dimensional")
    ids = stage_vocab.stage_token_ids[stage_index]
    if ids.device != logits.device:
        raise ValueError("SID stage token IDs must share logits.device")
    expected_width = stage_vocab.vocab_size if compact_logits else stage_vocab.full_vocab_size
    if logits.shape[1] != expected_width:
        raise ValueError(f"SID stage logits width must be {expected_width}, got {logits.shape[1]}")
    selected_logits = logits[:, : ids.numel()] if compact_logits else logits.index_select(1, ids)
    local = select_unconstrained_candidates(selected_logits, temperatures, top_k)
    mapped = ids.index_select(0, local.token_ids.reshape(-1).to(torch.int64))
    mapped = mapped.reshape_as(local.token_ids).to(torch.int32)
    global_ids = torch.where(local.valid_mask, mapped, torch.zeros_like(mapped))
    return ConstraintOutput(global_ids, local.logprobs, local.valid_mask)
