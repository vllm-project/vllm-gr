# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_gr.entrypoints.beam_search.boundary_tokens import boundary_token_id

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def _tokenizer(**fields: object) -> Any:
    return SimpleNamespace(**fields)


def _unreachable_vocab() -> dict[str, int]:
    raise AssertionError("vocabulary must not be consulted for a non-UNK token")


def test_none_token_returns_none() -> None:
    assert boundary_token_id(object(), None, "begin_token") is None


@pytest.mark.parametrize("error", [KeyError, ValueError])  # type: ignore[misc]
def test_converter_exceptions_are_wrapped(error: type[Exception]) -> None:
    def converter(token: str) -> int:
        raise error(token)

    tokenizer = _tokenizer(convert_tokens_to_ids=converter)

    with pytest.raises(ValueError, match="begin_token must be a valid tokenizer token") as caught:
        boundary_token_id(tokenizer, "<bad>", "begin_token")

    assert isinstance(caught.value.__cause__, error)


@pytest.mark.parametrize("token_id", [None, True, "3", 1.5, -1])  # type: ignore[misc]
def test_rejects_missing_non_integer_or_negative_ids(token_id: object) -> None:
    tokenizer = _tokenizer(convert_tokens_to_ids=lambda _token: token_id)

    with pytest.raises(ValueError, match="end_token must be a valid tokenizer token"):
        boundary_token_id(tokenizer, "<end>", "end_token")


def test_accepts_unk_when_vocabulary_maps_token() -> None:
    tokenizer = _tokenizer(
        convert_tokens_to_ids=lambda _token: 3,
        unk_token_id=3,
        get_vocab=lambda: {"<unk>": 3},
    )

    assert boundary_token_id(tokenizer, "<unk>", "begin_token") == 3


def test_rejects_unk_when_vocabulary_maps_a_different_token() -> None:
    tokenizer = _tokenizer(
        convert_tokens_to_ids=lambda _token: 3,
        unk_token_id=3,
        get_vocab=lambda: {"<other>": 4},
    )

    with pytest.raises(ValueError, match="begin_token must be a valid tokenizer token"):
        boundary_token_id(tokenizer, "<unk>", "begin_token")


def test_rejects_unk_after_vocabulary_miss_without_reverse_fallback() -> None:
    tokenizer = _tokenizer(
        convert_tokens_to_ids=lambda _token: 3,
        unk_token_id=3,
        get_vocab=lambda: {"<other>": 4},
        convert_ids_to_tokens=lambda token_id: "<unk>" if token_id == 3 else "<other>",
    )

    with pytest.raises(ValueError, match="begin_token must be a valid tokenizer token"):
        boundary_token_id(tokenizer, "<unk>", "begin_token")


def test_accepts_unk_through_reverse_mapping() -> None:
    tokenizer = _tokenizer(
        convert_tokens_to_ids=lambda _token: 3,
        unk_token_id=3,
        convert_ids_to_tokens=lambda token_id: "<unk>" if token_id == 3 else "<other>",
    )

    assert boundary_token_id(tokenizer, "<unk>", "begin_token") == 3


def test_rejects_unk_when_reverse_mapping_differs() -> None:
    tokenizer = _tokenizer(
        convert_tokens_to_ids=lambda _token: 3,
        unk_token_id=3,
        convert_ids_to_tokens=lambda _token_id: "<other>",
    )

    with pytest.raises(ValueError, match="begin_token must be a valid tokenizer token"):
        boundary_token_id(tokenizer, "<unk>", "begin_token")


def test_accepts_unk_from_tokenizer_marker() -> None:
    tokenizer = _tokenizer(
        convert_tokens_to_ids=lambda _token: 3,
        unk_token_id=3,
        unk_token="<unk>",
    )

    assert boundary_token_id(tokenizer, "<unk>", "begin_token") == 3


def test_rejects_unk_when_tokenizer_marker_differs() -> None:
    tokenizer = _tokenizer(
        convert_tokens_to_ids=lambda _token: 3,
        unk_token_id=3,
        unk_token="<other>",
    )

    with pytest.raises(ValueError, match="begin_token must be a valid tokenizer token"):
        boundary_token_id(tokenizer, "<unk>", "begin_token")


def test_non_unk_id_skips_vocabulary_validation() -> None:
    tokenizer = _tokenizer(
        convert_tokens_to_ids=lambda _token: 7,
        unk_token_id=3,
        get_vocab=_unreachable_vocab,
    )

    assert boundary_token_id(tokenizer, "<good>", "begin_token") == 7
