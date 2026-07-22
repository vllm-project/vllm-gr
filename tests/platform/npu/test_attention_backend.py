# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest import mock

import pytest
import torch

import vllm_gr.platform.npu.attention_backend as attention_backend
from vllm_gr.platform.npu.capabilities import NPUAttentionCapabilities


def _capabilities(
    *, is_npu: bool = True, native: bool = False, compatible: bool = True
) -> NPUAttentionCapabilities:
    return NPUAttentionCapabilities(
        is_npu=is_npu,
        native_custom_backend=native,
        selector_api_compatible=compatible,
    )


@pytest.fixture(autouse=True)  # type: ignore[misc]
def restore_selector() -> Iterator[None]:
    yield
    if attention_backend._INSTALLED_SELECTOR is not None:
        attention_backend.uninstall_npu_attention_backend_compat()


def _selector_config(*, dtype: torch.dtype = torch.float16) -> Any:
    from vllm.v1.attention.selector import AttentionSelectorConfig

    return AttentionSelectorConfig(
        head_size=128,
        dtype=dtype,
        kv_cache_dtype=None,
        block_size=128,
    )


def test_gpu_does_not_modify_vllm_selector() -> None:
    import vllm.v1.attention.selector as selector_mod

    original = selector_mod._cached_get_attn_backend
    with mock.patch.object(
        attention_backend,
        "detect_npu_attention_capabilities",
        return_value=_capabilities(is_npu=False),
    ):
        assert attention_backend.install_npu_attention_backend_compat() == 0

    assert selector_mod._cached_get_attn_backend is original


def test_native_npu_support_does_not_modify_selector() -> None:
    import vllm.v1.attention.selector as selector_mod

    original = selector_mod._cached_get_attn_backend
    with mock.patch.object(
        attention_backend,
        "detect_npu_attention_capabilities",
        return_value=_capabilities(native=True),
    ):
        assert attention_backend.install_npu_attention_backend_compat() == 0

    assert selector_mod._cached_get_attn_backend is original


def test_affected_npu_installs_bridge_idempotently() -> None:
    import vllm.v1.attention.selector as selector_mod

    original = selector_mod._cached_get_attn_backend
    with mock.patch.object(
        attention_backend,
        "detect_npu_attention_capabilities",
        return_value=_capabilities(),
    ):
        assert attention_backend.install_npu_attention_backend_compat() == 1
        installed = selector_mod._cached_get_attn_backend
        assert attention_backend.install_npu_attention_backend_compat(force=True) == 0

    assert installed is not original
    assert selector_mod._cached_get_attn_backend is installed


def test_explicit_custom_returns_beam_backend_and_is_cached() -> None:
    import vllm.v1.attention.selector as selector_mod
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

    with (
        mock.patch.object(
            attention_backend,
            "detect_npu_attention_capabilities",
            return_value=_capabilities(),
        ),
        mock.patch.object(attention_backend, "_validate_custom_backend") as validate,
    ):
        attention_backend.install_npu_attention_backend_compat()
        first = selector_mod._cached_get_attn_backend(
            AttentionBackendEnum.CUSTOM, _selector_config()
        )
        second = selector_mod._cached_get_attn_backend(
            AttentionBackendEnum.CUSTOM, _selector_config()
        )

    assert first is BeamAttentionBackend
    assert second is BeamAttentionBackend
    validate.assert_called_once()


def test_marker_restores_custom_backend_when_ascend_erased_it() -> None:
    import vllm.v1.attention.selector as selector_mod

    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

    with (
        mock.patch.object(
            attention_backend,
            "detect_npu_attention_capabilities",
            return_value=_capabilities(),
        ),
        mock.patch.object(attention_backend, "_marker_requests_custom_backend", return_value=True),
        mock.patch.object(attention_backend, "_validate_custom_backend"),
    ):
        attention_backend.install_npu_attention_backend_compat()
        selected = selector_mod._cached_get_attn_backend(None, _selector_config())

    assert selected is BeamAttentionBackend


def test_non_custom_request_delegates_to_original_selector() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    original_call = mock.Mock(return_value="native-backend")

    def original(backend: Any, attn_selector_config: Any, num_heads: int | None = None) -> str:
        return str(original_call(backend, attn_selector_config, num_heads=num_heads))

    selector = attention_backend._build_selector_bridge(original)
    selected = selector(AttentionBackendEnum.FLASH_ATTN, _selector_config())

    assert selected == "native-backend"
    original_call.assert_called_once()


def test_marker_state_is_part_of_the_cache_key() -> None:
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

    original_call = mock.Mock(return_value="native-backend")

    def original(backend: Any, attn_selector_config: Any, num_heads: int | None = None) -> str:
        return str(original_call(backend, attn_selector_config, num_heads=num_heads))

    selector = attention_backend._build_selector_bridge(original)
    selector_config = _selector_config()
    with (
        mock.patch.object(
            attention_backend,
            "_marker_requests_custom_backend",
            side_effect=[True, False],
        ),
        mock.patch.object(attention_backend, "_validate_custom_backend"),
    ):
        custom = selector(None, selector_config)
        native = selector(None, selector_config)

    assert custom is BeamAttentionBackend
    assert native == "native-backend"
    original_call.assert_called_once()


def test_invalid_custom_configuration_fails_closed() -> None:
    import vllm.v1.attention.selector as selector_mod
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    with (
        mock.patch.object(
            attention_backend,
            "detect_npu_attention_capabilities",
            return_value=_capabilities(),
        ),
        mock.patch.object(
            attention_backend,
            "_validate_custom_backend",
            side_effect=ValueError("dtype not supported"),
        ),
    ):
        attention_backend.install_npu_attention_backend_compat()
        with pytest.raises(ValueError, match="dtype not supported"):
            selector_mod._cached_get_attn_backend(AttentionBackendEnum.CUSTOM, _selector_config())


def test_unknown_selector_api_is_not_patched() -> None:
    import vllm.v1.attention.selector as selector_mod

    original = selector_mod._cached_get_attn_backend
    with (
        mock.patch.object(
            attention_backend,
            "detect_npu_attention_capabilities",
            return_value=_capabilities(compatible=False),
        ),
        mock.patch.object(attention_backend.logger, "warning") as warning,
    ):
        assert attention_backend.install_npu_attention_backend_compat() == 0

    assert selector_mod._cached_get_attn_backend is original
    warning.assert_called_once()
    assert "cannot be installed safely" in warning.call_args.args[0]


def test_explicit_custom_request_rejects_unknown_selector_api() -> None:
    config = mock.Mock(additional_config={}, attention_config=mock.Mock(backend=None))
    unsupported = NPUAttentionCapabilities(
        is_npu=True,
        native_custom_backend=False,
        selector_api_compatible=False,
        reason="unsupported test selector",
    )
    with (
        mock.patch.object(attention_backend, "is_npu_platform", return_value=True),
        mock.patch.object(
            attention_backend,
            "detect_npu_attention_capabilities",
            return_value=unsupported,
        ),
        pytest.raises(
            attention_backend.NPUAttentionCompatibilityError,
            match="unsupported test selector",
        ),
    ):
        attention_backend.preserve_custom_backend_intent(config, requested=True)


def test_required_kv_cache_layout_is_preserved() -> None:
    class BackendWithLayout:
        @staticmethod
        def get_required_kv_cache_layout() -> str:
            return "HND"

    with mock.patch("vllm.v1.attention.backends.utils.set_kv_cache_layout") as set_layout:
        attention_backend._apply_required_kv_cache_layout(BackendWithLayout)

    set_layout.assert_called_once_with("HND")
