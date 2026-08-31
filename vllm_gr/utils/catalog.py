# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Catalog-constraint utilities for beam search."""

import json
import random
from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input
from vllm.logprobs import FlatLogprobs
from vllm.lora.request import LoRARequest
from vllm.tokenizers import TokenizerLike

from vllm_gr.logger import init_logger

logger = init_logger(__name__)


def compute_catalog_slice_config(
    catalog_path: str,
    tokenizer_or_model: Any,
    trust_remote_code: bool = False,
):
    """Calculate (vocab_offset, vocab_slice_size) from a JSON catalog file."""
    try:
        if isinstance(tokenizer_or_model, str):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_or_model, trust_remote_code=trust_remote_code
            )
        else:
            tokenizer = tokenizer_or_model

        with open(catalog_path) as f:
            catalog_data = json.load(f)

        min_tid = float("inf")
        max_tid = -float("inf")

        for seq in catalog_data:
            token_ids = tokenizer.convert_tokens_to_ids(seq)
            for token_str, token_id in zip(seq, token_ids):
                if token_id is not None and token_id != -1:
                    if not (isinstance(token_str, str) and token_str.startswith("<|sid_")):
                        if token_id < min_tid:
                            min_tid = token_id
                        if token_id > max_tid:
                            max_tid = token_id

        if min_tid != float("inf") and max_tid != -float("inf"):
            return int(min_tid), int(max_tid - min_tid + 1)
    except Exception as e:
        logger.warning("vllm_gr: Failed to compute catalog slice config: %s", e)
    return None, None


class Catalog:
    # Sentinel key used to mark nodes that are valid terminal completions
    # (end of a catalog sequence). Token IDs are integers, so a string key
    # cannot collide with actual trie entries.
    _TERMINAL_MARKER = "__catalog_terminal__"

    def __init__(self, tokenizer: TokenizerLike):
        self.tokenizer = tokenizer
        self.trie: dict[int, Any] = {}
        # Compact sequence storage: one flat uint32 array plus cumulative
        # offsets.  Avoids keeping a second full Python list-of-lists token
        # representation (O(entries x sequence length) object overhead) next
        # to the trie, which can OOM for catalogs with millions of entries.
        self._seq_data: np.ndarray[Any, Any] = np.empty(0, dtype=np.uint32)
        self._seq_offsets: np.ndarray[Any, Any] = np.array([0], dtype=np.int64)
        self.vocab_offset: int | None = None
        self.vocab_slice_size: int | None = None

    def load(self, path: str):
        try:
            with open(path) as f:
                catalog_data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Catalog file not found: {path}")
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from catalog file: {path}")

        flat_tokens: list[int] = []
        offsets: list[int] = [0]
        for seq_idx, seq in enumerate(catalog_data):
            token_ids = self.tokenizer.convert_tokens_to_ids(seq)
            node = self.trie
            for i, (token_str, token_id) in enumerate(zip(seq, token_ids)):
                if token_id is None or token_id == -1:
                    raise ValueError(
                        f"Failed to map token {token_str!r} to a valid token ID using the tokenizer "
                        f"(got {token_id!r}). Token occurs at index {i} of sequence #{seq_idx} in catalog {path!r}. "
                        "Please verify that special/semantic tokens are registered in the model's tokenizer."
                    )
                if token_id not in node:
                    node[token_id] = {}
                node = node[token_id]
            node[self._TERMINAL_MARKER] = True
            flat_tokens.extend(token_ids)
            offsets.append(len(flat_tokens))

        self._seq_data = np.asarray(flat_tokens, dtype=np.uint32)
        self._seq_offsets = np.asarray(offsets, dtype=np.int64)

        offset, slice_size = compute_catalog_slice_config(path, self.tokenizer)
        if offset is not None and slice_size is not None:
            self.vocab_offset = offset
            self.vocab_slice_size = slice_size
            from vllm_gr.ops.sliced_logits import set_vocab_slice_config

            set_vocab_slice_config(self.vocab_offset, self.vocab_slice_size)
            logger.info(
                "vllm_gr: Catalog vocabulary slice configured: offset=%d, slice_size=%d (min_token_id=%d, max_token_id=%d)",
                self.vocab_offset,
                self.vocab_slice_size,
                self.vocab_offset,
                self.vocab_offset + self.vocab_slice_size - 1,
            )

    def __len__(self) -> int:
        """Return the number of stored catalog sequences."""
        return len(self._seq_offsets) - 1

    def sequence_at(self, i: int) -> list[int]:
        """Return the token IDs of the ``i``-th stored catalog sequence."""
        start = int(self._seq_offsets[i])
        end = int(self._seq_offsets[i + 1])
        return list(self._seq_data[start:end].tolist())

    def is_terminal(self, token_ids: list[int]) -> bool:
        """Returns True if token_ids exactly matches a complete catalog item."""
        node = self.trie
        for tid in token_ids:
            if tid in node:
                node = node[tid]
            else:
                return False
        return self._TERMINAL_MARKER in node

    def valid(self, token_ids: list[int]) -> set[int]:
        node = self.trie
        for tid in token_ids:
            if tid in node:
                node = node[tid]
            else:
                return set()
        return {k for k in node if k != self._TERMINAL_MARKER}

    def sample_sequences(self, n: int, rng: random.Random | None = None) -> list[list[int]]:
        """Return ``n`` random full catalog sequences.

        Prefers unique picks (sampling without replacement); falls back to
        duplicates with replacement when the catalog has fewer entries than
        requested.
        """
        if n <= 0 or len(self) == 0:
            return []
        rand = rng if rng is not None else random
        picked = rand.sample(range(len(self)), min(n, len(self)))
        sequences = [self.sequence_at(i) for i in picked]
        while len(sequences) < n:
            sequences.append(self.sequence_at(rand.randrange(len(self))))
        return sequences

    def apply_catalog_filter(
        self,
        token_ids_pos: list[int],
        logprobs_pos: list[float],
        valid_tokens_set: set[int],
    ) -> list[float]:
        """Mask logprobs of candidates outside the catalog-valid set to -inf.

        Token IDs are preserved unchanged so candidate geometry stays aligned
        with the caller's ranks/decoded metadata.
        """
        return mask_logprobs_by_token_set(token_ids_pos, logprobs_pos, valid_tokens_set)


def mask_logprobs_by_token_set(
    token_ids: Sequence[int],
    logprobs: Sequence[float],
    valid_tokens_set: Collection[int],
) -> list[float]:
    """Mask logprobs of token IDs outside ``valid_tokens_set`` to ``-inf``.

    Single shared implementation behind both the Offline
    ``mask_candidate_logprobs`` helper and the OpenAI
    ``Catalog.apply_catalog_filter`` method so the two entry-point paths
    cannot silently diverge.
    """
    return [
        logprob if token_id in valid_tokens_set else -float("inf")
        for token_id, logprob in zip(token_ids, logprobs)
    ]


def complete_catalog_beam(beam: BeamSearchSequence) -> BeamSearchSequence:
    """Copy ``beam`` as a completed catalog-terminal sequence.

    Used when a beam's generated tokens match a complete catalog item, so no
    valid continuation exists.  The copy keeps the beam's accumulated score
    and parent-pointer chain but finishes immediately (``stop``) instead of
    being kept alive with masked candidates and later dropped by the
    finite-score filter.
    """
    completed = BeamSearchSequence(
        orig_prompt=beam.orig_prompt,
        tokens=list(beam.tokens),
        logprobs=beam.logprobs,
        lora_request=beam.lora_request,
        cum_logprob=beam.cum_logprob,
        finish_reason="stop",
        stop_reason=None,
    )
    completed._lp_parent = beam._lp_parent
    completed._lp_step_data = beam._lp_step_data
    return completed


def fill_catalog_beams(
    best_beams: list[BeamSearchSequence],
    beam_width: int,
    catalog: Catalog,
    *,
    prompt_token_ids: Sequence[int],
    prompt_text: str | None,
    sid_begin_token_id: int | None,
    sid_end_token_id: int | None,
    initial_logprobs: FlatLogprobs,
    max_tokens: int,
    lora_request: LoRARequest | None = None,
    rng: random.Random | None = None,
) -> list[BeamSearchSequence]:
    """Top up ``best_beams`` to ``beam_width`` with random catalog sequences.

    Fill beams carry ``cum_logprob=-10000.0`` so they never outrank real
    beams, and a ``stop`` finish reason (they are complete catalog items).
    Begin/end tokens already handled by the prompt/session are stripped from
    the sampled sequence.

    Catalog entries longer than the remaining generation budget
    (``max_tokens`` minus the reserved begin/end tokens) are skipped, and
    entries already present in ``best_beams`` are not duplicated unless the
    catalog runs out of distinct sequences that fit.
    """
    missing = beam_width - len(best_beams)
    if missing <= 0 or len(catalog) == 0:
        return best_beams
    if rng is None:
        rng = random.Random()

    pre_calc = int(sid_begin_token_id is not None) + int(sid_end_token_id is not None)
    remaining_budget = max_tokens - pre_calc
    if remaining_budget <= 0:
        logger.warning(
            "vllm_gr: cannot fill catalog beams: max_tokens=%d leaves no budget "
            "after reserving %d begin/end tokens",
            max_tokens,
            pre_calc,
        )
        return best_beams

    prompt_len = len(prompt_token_ids)
    used: set[tuple[int, ...]] = set()
    for beam in best_beams:
        generated = tuple(beam.tokens[prompt_len:])
        if sid_begin_token_id is not None and generated and generated[0] == sid_begin_token_id:
            generated = generated[1:]
        if sid_end_token_id is not None and generated and generated[-1] == sid_end_token_id:
            generated = generated[:-1]
        used.add(generated)

    def _prepare(tokens: list[int]) -> tuple[list[int], tuple[int, ...]] | None:
        tokens = list(tokens)
        if sid_begin_token_id is not None and tokens and tokens[0] == sid_begin_token_id:
            tokens.pop(0)
        if sid_end_token_id is not None and tokens and tokens[-1] == sid_end_token_id:
            tokens.pop()
        if len(tokens) > remaining_budget:
            return None
        key = tuple(tokens)
        if sid_end_token_id is not None:
            tokens.append(sid_end_token_id)
        return tokens, key

    def _build(tokens: list[int]) -> BeamSearchSequence:
        full_tokens = list(prompt_token_ids) + tokens
        beam = BeamSearchSequence(
            orig_prompt=tokens_input(full_tokens, prompt=prompt_text),
            tokens=full_tokens,
            logprobs=initial_logprobs,
            lora_request=lora_request,
            cum_logprob=-10000.0,
            finish_reason="stop",
            stop_reason=sid_end_token_id,
        )
        beam._lp_parent = None
        beam._lp_step_data = None
        return beam

    logger.info("vllm_gr: filling for %d beams", missing)
    skipped_oversized = 0
    sample_size = min(max(missing * 4, 32), len(catalog))
    order = rng.sample(range(len(catalog)), sample_size)
    for idx in order:
        if len(best_beams) >= beam_width:
            break
        prepared = _prepare(catalog.sequence_at(idx))
        if prepared is None:
            skipped_oversized += 1
            continue
        tokens, key = prepared
        if key in used:
            continue
        used.add(key)
        best_beams.append(_build(tokens))

    if skipped_oversized:
        logger.warning(
            "vllm_gr: skipped %d catalog entries longer than the remaining "
            "budget of %d generated tokens",
            skipped_oversized,
            remaining_budget,
        )
    if len(best_beams) < beam_width and len(catalog) > 0:
        # The catalog ran out of distinct sequences that fit the budget;
        # fall back to duplicates so the requested beam width is preserved.
        attempts = 0
        max_attempts = min(max(beam_width * 10, 100), max(2 * len(catalog), beam_width))
        while len(best_beams) < beam_width and attempts < max_attempts:
            attempts += 1
            prepared = _prepare(catalog.sequence_at(rng.randrange(len(catalog))))
            if prepared is None:
                continue
            best_beams.append(_build(prepared[0]))
        if len(best_beams) < beam_width:
            logger.warning(
                "vllm_gr: catalog fill produced only %d/%d beams",
                len(best_beams),
                beam_width,
            )

    return best_beams
