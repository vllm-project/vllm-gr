# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest


class _Tokenizer(SimpleNamespace):
    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self.added_vocab)


def _tokenizer(**overrides: object) -> _Tokenizer:
    values: dict[str, object] = {
        "vocab": {"a": 0, "b": 1, "<extra>": 4},
        "added_vocab": {"<extra>": 4},
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "unk_token_id": 3,
        "sep_token_id": None,
        "cls_token_id": None,
        "mask_token_id": None,
        "additional_special_tokens_ids": [4],
        "all_special_ids": [0, 1, 2, 3, 4],
    }
    values.update(overrides)
    return _Tokenizer(**values)


def test_digest_is_stable_across_mapping_order() -> None:
    first = _tokenizer()
    second = _tokenizer(
        vocab={"<extra>": 4, "b": 1, "a": 0},
        added_vocab={"<extra>": 4},
    )
    assert compute_tokenizer_digest(first) == compute_tokenizer_digest(second)


@pytest.mark.parametrize(  # type: ignore[misc]
    "changed",
    [
        {"vocab": {"a": 0, "b": 2, "<extra>": 4}},
        {"added_vocab": {"<extra>": 5}},
        {"bos_token_id": 1, "eos_token_id": 0},
        {"additional_special_tokens_ids": [5]},
        {"all_special_ids": [0, 1, 2, 3]},
    ],
)
def test_digest_covers_full_tokenizer_state(changed: dict[str, object]) -> None:
    assert compute_tokenizer_digest(_tokenizer()) != compute_tokenizer_digest(_tokenizer(**changed))


def test_digest_has_sha256_prefix() -> None:
    digest = compute_tokenizer_digest(_tokenizer())
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


def test_rejects_invalid_vocab_and_special_ids() -> None:
    with pytest.raises(ValueError, match="vocab"):
        compute_tokenizer_digest(_tokenizer(vocab={"a": True}))
    with pytest.raises(ValueError, match="bos_token_id"):
        compute_tokenizer_digest(_tokenizer(bos_token_id=True))
