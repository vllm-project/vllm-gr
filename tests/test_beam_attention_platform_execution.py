# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform isolation and delegation contracts for Beam Attention."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest
import torch
from vllm.platforms.interface import DeviceCapability

from vllm_gr.v1.attention.backends import beam_attn


def test_facade_import_does_not_load_platform_execution_modules() -> None:
    project_root = Path(__file__).resolve().parents[1]
    code = """
import sys

from vllm.v1.attention.backends.registry import AttentionBackendEnum

assert not AttentionBackendEnum.CUSTOM.is_overridden()

import vllm_gr.v1.attention.backends.beam_attn

assert not AttentionBackendEnum.CUSTOM.is_overridden()
assert "vllm_gr.v1.attention.backends.beam_attn_gpu" not in sys.modules
assert "vllm_gr.v1.attention.backends.beam_attn_npu" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_facade_resolves_attention_hooks_from_active_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr import platforms

    hooks = object()
    provider = SimpleNamespace(get_attention_hooks=lambda: hooks)
    get_backend = mock.Mock(return_value=provider)
    monkeypatch.setattr(platforms, "get_active_backend", get_backend)

    assert beam_attn._get_attention_hooks() is hooks
    get_backend.assert_called_once_with()


def test_impl_resolves_and_stores_attention_hooks_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hooks = SimpleNamespace(
        get_flash_attn_version=lambda **kwargs: 3,
        supports_batch_invariance=lambda: True,
        supports_kv_cache_dtype=lambda dtype: True,
    )
    resolve = mock.Mock(return_value=hooks)
    monkeypatch.setattr(beam_attn, "_get_attention_hooks", resolve)

    impl = beam_attn.BeamAttentionImpl(
        num_heads=2,
        head_size=16,
        scale=0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    assert impl._attention_hooks is hooks
    resolve.assert_called_once_with()


def test_backend_capabilities_delegate_to_active_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder_cls = type("Builder", (), {})
    capability = DeviceCapability(8, 0)
    hooks = SimpleNamespace(
        supports_batch_invariance=lambda: True,
        supports_kv_cache_dtype=lambda dtype: dtype == "auto",
        supports_sinks=lambda: False,
        supports_compute_capability=lambda value: value == capability,
        get_metadata_builder_cls=lambda: builder_cls,
        get_kv_cache_shape=lambda *args: (2, *args[:4]),
        get_kv_cache_stride_order=lambda include_layers=False: (
            (2, 0, 1) if include_layers else (0, 1)
        ),
        validate_combination=lambda *args: "platform decision",
    )
    monkeypatch.setattr(beam_attn, "_get_attention_hooks", lambda: hooks)

    backend = beam_attn.BeamAttentionBackend
    assert backend.supports_batch_invariance() is True
    assert backend.supports_kv_cache_dtype("auto") is True
    assert backend.supports_kv_cache_dtype("fp8") is False
    assert backend.supports_sink() is False
    assert backend.supports_compute_capability(capability) is True
    assert backend.get_builder_cls() is builder_cls
    assert backend.get_kv_cache_shape(4, 16, 2, 128) == (2, 4, 16, 2, 128)
    assert backend.get_kv_cache_stride_order(True) == (2, 0, 1)
    assert (
        backend.supports_combination(
            128,
            torch.float16,
            "auto",
            16,
            False,
            False,
            False,
            capability,
        )
        == "platform decision"
    )


@pytest.mark.parametrize("tuple_cache", [False, True])  # type: ignore[misc]
def test_forward_preserves_cache_and_attention_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tuple_cache: bool,
) -> None:
    calls: dict[str, Any] = {}

    def reshape_and_cache(*args: Any) -> None:
        calls["reshape"] = args

    def cascade_attention(**kwargs: Any) -> None:
        calls["cascade"] = kwargs

    def split_kv_cache(kv_cache: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple_cache:
            return cast(tuple[torch.Tensor, torch.Tensor], kv_cache)
        key_cache, value_cache = kv_cache.unbind(0)
        return key_cache, value_cache

    def get_kv_scales(layer: Any) -> tuple[Any, Any]:
        if tuple_cache:
            return None, None
        return layer._k_scale, layer._v_scale

    hooks = SimpleNamespace(
        split_kv_cache=split_kv_cache,
        get_kv_scales=get_kv_scales,
        reshape_and_cache=reshape_and_cache,
        cascade_attention=cascade_attention,
    )
    monkeypatch.setattr(
        beam_attn,
        "_get_attention_hooks",
        mock.Mock(side_effect=AssertionError("forward resolved the registry")),
    )

    impl = object.__new__(beam_attn.BeamAttentionImpl)
    impl._attention_hooks = hooks
    impl.kv_sharing_target_layer_name = None
    impl.kv_cache_dtype = "auto"
    impl.sliding_window = (6, 0)
    impl.scale = 0.5
    impl.logits_soft_cap = 9.0
    impl.vllm_flash_attn_version = 3
    impl.sinks = torch.empty(1)
    impl.alibi_slopes = torch.empty(2)

    key_cache = torch.zeros(2, 2)
    value_cache = torch.ones(2, 2)
    kv_cache: Any
    if tuple_cache:
        kv_cache = (key_cache, value_cache)
    else:
        kv_cache = torch.stack((key_cache, value_cache))

    key = torch.empty(1)
    value = torch.empty(1)
    query = torch.empty(1)
    output = torch.empty(1)
    slot_mapping = torch.tensor([0])
    metadata = SimpleNamespace(slot_mapping=slot_mapping)
    k_scale = torch.tensor(1.0)
    v_scale = torch.tensor(2.0)
    layer = (
        SimpleNamespace() if tuple_cache else SimpleNamespace(_k_scale=k_scale, _v_scale=v_scale)
    )

    returned = impl.forward(
        layer=layer,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=metadata,
        output=output,
    )

    assert returned is output
    reshape_args = calls["reshape"]
    assert reshape_args[0] is key
    assert reshape_args[1] is value
    torch.testing.assert_close(reshape_args[2], key_cache)
    torch.testing.assert_close(reshape_args[3], value_cache)
    assert reshape_args[4] is slot_mapping
    assert reshape_args[5] == "auto"
    assert reshape_args[6] is (None if tuple_cache else k_scale)
    assert reshape_args[7] is (None if tuple_cache else v_scale)

    cascade_args = calls["cascade"]
    assert cascade_args["output"] is output
    assert cascade_args["query"] is query
    torch.testing.assert_close(cascade_args["key_cache"], key_cache)
    torch.testing.assert_close(cascade_args["value_cache"], value_cache)
    assert cascade_args["attn_metadata"] is metadata
    assert cascade_args["sliding_window_size"] == [6, 0]
    assert cascade_args["scale"] == 0.5
    assert cascade_args["logits_soft_cap"] == 9.0
    assert cascade_args["vllm_flash_attn_version"] == 3
    assert cascade_args["s_aux"] is impl.sinks
    assert cascade_args["alibi_slopes"] is impl.alibi_slopes
