# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest import mock

from vllm_gr.platform.npu import capabilities


def test_non_npu_requires_no_bridge() -> None:
    with mock.patch.object(capabilities, "is_npu_platform", return_value=False):
        result = capabilities.detect_npu_attention_capabilities()

    assert not result.is_npu
    assert not result.requires_custom_backend_bridge


def test_selector_contract_accepts_vllm_0221_signature() -> None:
    def selector(
        backend: Any, attn_selector_config: Any, num_heads: int | None = None
    ) -> tuple[Any, Any, int | None]:
        return backend, attn_selector_config, num_heads

    assert capabilities._selector_api_is_compatible(selector)


def test_selector_contract_rejects_unknown_signature() -> None:
    def selector(
        backend: Any, config: Any, extra: int | None = None
    ) -> tuple[Any, Any, int | None]:
        return backend, config, extra

    assert not capabilities._selector_api_is_compatible(selector)


def test_native_custom_backend_is_detected() -> None:
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

    expected = f"{BeamAttentionBackend.__module__}.{BeamAttentionBackend.__qualname__}"
    platform = SimpleNamespace(get_attn_backend_cls=mock.Mock(return_value=expected))

    assert capabilities._platform_selects_registered_custom_backend(platform)


def test_ascend_0221_behavior_requires_bridge() -> None:
    platform = SimpleNamespace(
        get_attn_backend_cls=mock.Mock(
            return_value="vllm_ascend.attention.attention_v1.AscendAttentionBackend"
        )
    )

    assert not capabilities._platform_selects_registered_custom_backend(platform)
