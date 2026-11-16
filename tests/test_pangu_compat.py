# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_gr.pangu_compat import (
    _PATCH_MARKER,
    _make_patched_maybe_untie,
    apply_pangu_tied_embeddings_patch,
)


def _model_config(*, architecture: str, tied: bool) -> SimpleNamespace:
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[architecture],
            tie_word_embeddings=tied,
        )
    )


def test_official_pangu_placeholder_does_not_untie_embeddings() -> None:
    calls: list[Any] = []
    patched = _make_patched_maybe_untie(calls.append)
    config = _model_config(architecture="PanguEmbeddedForCausalLM", tied=True)

    patched(config)

    assert calls == []


@pytest.mark.parametrize(
    ("architecture", "tied"),
    [("OtherForCausalLM", True), ("PanguEmbeddedForCausalLM", False)],
)  # type: ignore[misc]
def test_other_weight_layouts_keep_vllm_loader_decision(architecture: str, tied: bool) -> None:
    calls: list[Any] = []
    patched = _make_patched_maybe_untie(calls.append)
    config = _model_config(architecture=architecture, tied=tied)

    patched(config)

    assert calls == [config]


def test_installer_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.config import ModelConfig

    calls: list[Any] = []
    monkeypatch.setattr(ModelConfig, "maybe_untie_word_embeddings", calls.append)

    assert apply_pangu_tied_embeddings_patch() is True
    assert apply_pangu_tied_embeddings_patch() is False
    assert getattr(ModelConfig.maybe_untie_word_embeddings, _PATCH_MARKER, False)

    config = _model_config(architecture="OtherForCausalLM", tied=True)
    ModelConfig.maybe_untie_word_embeddings(config)
    assert calls == [config]


pytestmark = pytest.mark.cpu_test
