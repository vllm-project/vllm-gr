# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Independent GR selection does not replace or contaminate native selection."""

from functools import cache
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import vllm.platforms
from vllm.v1.attention import selector
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_gr import platforms
from vllm_gr.v1.attention import selector_patch

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize("backend", [None, AttentionBackendEnum.CUSTOM])  # type: ignore[misc]
def test_gr_selection_uses_selected_platform_and_isolates_native_cache(
    monkeypatch: pytest.MonkeyPatch, backend: Any
) -> None:
    calls = Mock(return_value=dict)
    native = cache(calls)
    monkeypatch.setattr(selector, "_cached_get_attn_backend", native)
    choose = Mock(return_value="builtins.list")
    platform = SimpleNamespace(get_gr_attention_backend_cls=choose)
    monkeypatch.setattr(platforms, "get_current_gr_platform", lambda **kw: platform)
    original_platform = vllm.platforms.current_platform
    selector_patch.install_gr_attention_selector()
    installed = selector._cached_get_attn_backend
    selector_patch.install_gr_attention_selector()
    assert installed is selector._cached_get_attn_backend
    geometry = object()
    for enabled in (False, True, False):
        monkeypatch.setattr(selector_patch, "_gr_attention_is_custom", lambda: enabled)
        assert installed(backend, geometry) is (list if enabled else dict)
    assert calls.call_count == 1
    choose.assert_called_once_with(backend, geometry, num_heads=None)
    assert vllm.platforms.current_platform is original_platform


def test_gr_keeps_explicit_native_backend_and_custom_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = Mock(return_value=dict)
    monkeypatch.setattr(selector, "_cached_get_attn_backend", cache(calls))
    monkeypatch.setattr(selector_patch, "_gr_attention_is_custom", lambda: True)
    choose = Mock(side_effect=AssertionError("explicit backend reached GR selection"))
    monkeypatch.setattr(
        platforms,
        "get_current_gr_platform",
        lambda **kw: SimpleNamespace(get_gr_attention_backend_cls=choose),
    )
    selector_patch.install_gr_attention_selector()
    assert selector._cached_get_attn_backend(AttentionBackendEnum.FLASH_ATTN, 32) is dict
    choose.assert_not_called()


def test_external_platforms_on_same_hardware_keep_their_own_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(selector, "_cached_get_attn_backend", cache(lambda *a, **k: dict))
    monkeypatch.setattr(selector_patch, "_gr_attention_is_custom", lambda: True)
    selector_patch.install_gr_attention_selector()
    for name, expected in [("builtins.list", list), ("builtins.tuple", tuple)]:
        platform = SimpleNamespace(get_gr_attention_backend_cls=lambda *a, **k: name)
        monkeypatch.setattr(platforms, "get_current_gr_platform", lambda **kw: platform)
        assert selector._cached_get_attn_backend(None, 64) is expected


def test_real_engine_context_controls_the_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends import registry

    from vllm_gr.config import GRAttentionBackend, GRConfig

    plain = VllmConfig()
    beam = VllmConfig()
    GRConfig(attention_backend=GRAttentionBackend.CUSTOM).write_to_vllm_config(beam)
    native = Mock(return_value=dict)
    monkeypatch.setattr(selector, "_cached_get_attn_backend", cache(native))
    monkeypatch.setattr(
        platforms,
        "get_current_gr_platform",
        lambda **kw: SimpleNamespace(get_gr_attention_backend_cls=lambda *a, **k: "builtins.list"),
    )
    before = dict(registry._ATTN_OVERRIDES)
    selector_patch.install_gr_attention_selector()
    for config, expected in [(plain, dict), (beam, list), (plain, dict)]:
        with set_current_vllm_config(config):
            assert selector._cached_get_attn_backend(None, 64) is expected
    assert registry._ATTN_OVERRIDES == before
    native.assert_called_once_with(None, 64, num_heads=None)
