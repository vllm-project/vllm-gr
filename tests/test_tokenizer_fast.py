# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Parity tests for ``vllm_gr.tokenizer_fast``.

The raw Rust backend bypass must stay byte-identical to the HuggingFace
wrapper.  Added-token handling is covered by the singleton probe in
``_rust_decode_verified``; the tests here pin the *sequence-level* axis:
``clean_up_tokenization_spaces`` post-processing (``"hello ,"`` ->
``"hello,"``) only manifests on multi-token sequences, so it must be
covered explicitly for both helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vllm_gr.tokenizer_fast import (
    _rust_backend,
    _rust_decode_verified,
    fast_batch_decode,
    fast_decode,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

# WordLevel vocab whose multi-token decode exposes wrapper cleanup:
# without cleanup "hello ,", with cleanup "hello,".
VOCAB = {"hello": 0, ",": 1, "extra": 2, "world": 3}


def _make_tokenizer(clean_up_tokenization_spaces: bool) -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(VOCAB)),
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )
    tokenizer.add_tokens(["extra"])
    return tokenizer


SEQUENCES = [[0, 1], [0, 1, 2], [0, 0], [3, 1, 0, 2], [0]]


def test_raw_bypass_disabled_while_cleanup_enabled() -> None:
    """The singleton added-token probe passes either way; only the cleanup
    flag keeps the raw bypass off (multi-token cleanup divergence)."""
    with_cleanup = _make_tokenizer(True)
    assert getattr(with_cleanup, "clean_up_tokenization_spaces", False)
    assert _rust_decode_verified(with_cleanup, _rust_backend(with_cleanup)) is False

    without_cleanup = _make_tokenizer(False)
    assert _rust_decode_verified(without_cleanup, _rust_backend(without_cleanup)) is True


def test_fast_decode_multi_token_cleanup_parity() -> None:
    tokenizer = _make_tokenizer(True)
    for sequence in SEQUENCES:
        for skip_special_tokens in (False, True):
            assert fast_decode(tokenizer, sequence, skip_special_tokens) == (
                tokenizer.decode(sequence, skip_special_tokens=skip_special_tokens)
            ), (sequence, skip_special_tokens)


def test_fast_batch_decode_multi_token_cleanup_parity() -> None:
    tokenizer = _make_tokenizer(True)
    assert fast_batch_decode(tokenizer, SEQUENCES) == tokenizer.batch_decode(SEQUENCES)
    assert fast_batch_decode(tokenizer, SEQUENCES, skip_special_tokens=True) == (
        tokenizer.batch_decode(SEQUENCES, skip_special_tokens=True)
    )


def test_fast_paths_match_wrapper_without_cleanup() -> None:
    """With cleanup disabled the raw bypass stays on and must still agree."""
    tokenizer = _make_tokenizer(False)
    for sequence in SEQUENCES:
        assert fast_decode(tokenizer, sequence) == tokenizer.decode(sequence)
    assert fast_batch_decode(tokenizer, SEQUENCES) == tokenizer.batch_decode(SEQUENCES)


def test_cleanup_change_after_cached_probe_preserves_decode_parity() -> None:
    tokenizer = _make_tokenizer(False)
    backend = _rust_backend(tokenizer)
    assert _rust_decode_verified(tokenizer, backend) is True

    for cleanup in (True, False, True):
        tokenizer.clean_up_tokenization_spaces = cleanup
        assert _rust_decode_verified(tokenizer, backend) is not cleanup
        for skip_special_tokens in (False, True):
            for sequence in SEQUENCES:
                assert fast_decode(tokenizer, sequence, skip_special_tokens) == tokenizer.decode(
                    sequence, skip_special_tokens=skip_special_tokens
                )
            assert fast_batch_decode(tokenizer, SEQUENCES, skip_special_tokens) == (
                tokenizer.batch_decode(SEQUENCES, skip_special_tokens=skip_special_tokens)
            )


@pytest.mark.l1
@pytest.mark.parametrize("skip_special_tokens", [False, True])
def test_special_tokens_use_verified_backend_with_explicit_flags(
    monkeypatch, skip_special_tokens
) -> None:
    tokenizer = _make_tokenizer(False)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<sid>"]})
    special = tokenizer.convert_tokens_to_ids("<sid>")
    backend = _rust_backend(tokenizer)
    assert backend.decode([special]) == ""
    assert tokenizer.decode([special], skip_special_tokens=False) == "<sid>"
    sequences = [[special], [special, 0, 1, special], [0, special, 3]]
    expected = tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)
    assert _rust_decode_verified(tokenizer, backend) is True

    def unexpected_wrapper(*args, **kwargs):
        raise AssertionError("Compatible special-token tokenizer should use the raw backend")

    monkeypatch.setattr(tokenizer, "_decode", unexpected_wrapper)
    assert fast_batch_decode(tokenizer, sequences, skip_special_tokens) == expected
    assert [fast_decode(tokenizer, seq, skip_special_tokens) for seq in sequences] == expected


@pytest.mark.l1
def test_matching_flags_do_not_bypass_wrapper_specific_behavior(monkeypatch) -> None:
    tokenizer = _make_tokenizer(False)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<sid>"]})
    original = tokenizer.decode

    def custom_decode(ids, **kwargs):
        return original(ids, **kwargs) + " wrapper-specific"

    monkeypatch.setattr(tokenizer, "decode", custom_decode)
    assert _rust_decode_verified(tokenizer, _rust_backend(tokenizer)) is False


pytestmark = pytest.mark.cpu_test
