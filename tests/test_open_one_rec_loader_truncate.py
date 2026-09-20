# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Truncation-side behavior of the OpenOneRec prompt length normalizer."""

from types import SimpleNamespace

import pytest

from benchmarks.open_one_rec.open_one_rec_loader import BaseLoader, DataLoaderWrapper


class WordTokenizer:
    """Round-trip fake tokenizer over whitespace-separated words."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._inv: dict[int, str] = {}

    def _encode_word(self, word: str) -> int:
        if word not in self._vocab:
            token_id = len(self._vocab) + 1
            self._vocab[word] = token_id
            self._inv[token_id] = word
        return self._vocab[word]

    def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[self._encode_word(w) for w in text.split()])

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return " ".join(self._inv[i] for i in token_ids)


def _loader(custom_input_len: int, truncate_side: str = "head") -> BaseLoader:
    return BaseLoader(
        task_config={"name": "video"},
        data_dir=None,
        tokenizer=WordTokenizer(),
        custom_input_len=custom_input_len,
        truncate_side=truncate_side,
    )


def test_head_truncation_keeps_first_tokens() -> None:
    loader = _loader(4, truncate_side="head")
    prompt = "a b c d e f"
    assert loader._enforce_prompt_token_length(prompt, "0") == "a b c d"


def test_tail_truncation_keeps_last_tokens() -> None:
    loader = _loader(4, truncate_side="tail")
    prompt = "a b c d e f"
    assert loader._enforce_prompt_token_length(prompt, "0") == "c d e f"


def test_tail_truncation_preserves_trailing_question() -> None:
    loader = _loader(4, truncate_side="tail")
    prompt = "history1 history2 history3 history4 history5 recommend_the_next_video"
    normalized = loader._enforce_prompt_token_length(prompt, "0")
    assert normalized.endswith("recommend_the_next_video")


def test_short_prompt_is_extended_regardless_of_side() -> None:
    for side in ("head", "tail"):
        loader = _loader(6, truncate_side=side)
        normalized = loader._enforce_prompt_token_length("a b", "0")
        assert normalized == "a b a b a b"


def test_default_side_is_head() -> None:
    loader = _loader(4)
    assert loader.truncate_side == "head"
    assert loader._enforce_prompt_token_length("a b c d e f", "0") == "a b c d"


def test_invalid_truncate_side_rejected() -> None:
    with pytest.raises(ValueError, match="truncate_side"):
        DataLoaderWrapper(tokenizer=WordTokenizer(), data_dir=".", truncate_side="middle")
    with pytest.raises(ValueError, match="truncate_side"):
        _loader(4, truncate_side="middle")
