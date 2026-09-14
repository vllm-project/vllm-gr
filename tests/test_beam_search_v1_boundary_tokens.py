# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from vllm_gr.entrypoints.beam_search_v1.exceptions import BeamSearchRequestError
from vllm_gr.entrypoints.beam_search_v1.input_processor import _get_token_id
from vllm_gr.entrypoints.beam_search_v1.output_processor import _token_id


@pytest.mark.parametrize("resolve", [_get_token_id, _token_id])  # type: ignore[misc]
def test_wordlevel_boundary_resolution_preserves_explicit_unk(
    resolve: Callable[[Any, str | None, str], int | None],
) -> None:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "<start>": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    assert tokenizer.convert_tokens_to_ids("<missing>") == 0
    assert resolve(tokenizer, "<start>", "begin_token") == 1
    assert resolve(tokenizer, "[UNK]", "begin_token") == 0
    assert resolve(tokenizer, None, "begin_token") is None
    error = BeamSearchRequestError if resolve is _get_token_id else ValueError
    with pytest.raises(error, match="begin_token must be a valid tokenizer token"):
        resolve(tokenizer, "<missing>", "begin_token")


@pytest.mark.parametrize("resolve", [_get_token_id, _token_id])  # type: ignore[misc]
def test_boundary_resolution_with_reverse_lookup_only(
    resolve: Callable[[Any, str | None, str], int | None],
) -> None:
    tokenizer = SimpleNamespace(
        unk_token_id=17,
        convert_tokens_to_ids=lambda token: 17,
        convert_ids_to_tokens=lambda token_id: "<unk>",
    )
    assert resolve(tokenizer, "<unk>", "end_token") == 17
    with pytest.raises(ValueError, match="end_token must be a valid tokenizer token"):
        resolve(tokenizer, "<unknown-end>", "end_token")
