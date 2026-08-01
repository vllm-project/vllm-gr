# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform isolation and delegation contracts for Beam Attention."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.attention.backends import beam_attn


def test_facade_import_does_not_load_platform_execution_modules() -> None:
    project_root = Path(__file__).resolve().parents[1]
    code = """
import sys

from vllm_gr.v1.attention.backends import beam_attn_registration

beam_attn_registration.register_beam_attention_backend = lambda: (_ for _ in ()).throw(
    AssertionError("facade import registered Attention backend")
)

import vllm_gr.v1.attention.backends.beam_attn

assert "vllm_gr.v1.attention.backends.beam_attn_gpu" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_facade_resolves_hooks_from_active_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hooks = object()
    provider = SimpleNamespace(get_attention_hooks=lambda: hooks)
    monkeypatch.setattr("vllm_gr.platforms.get_active_backend", lambda: provider)

    assert beam_attn._get_active_attention_hooks() is hooks


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

    execution = SimpleNamespace(
        split_kv_cache=(
            (lambda kv_cache: (kv_cache[0], kv_cache[1]))
            if tuple_cache
            else (lambda kv_cache: tuple(kv_cache.unbind(0)))
        ),
        get_kv_scales=(
            (lambda layer: (None, None))
            if tuple_cache
            else (lambda layer: (layer._k_scale, layer._v_scale))
        ),
        reshape_and_cache=reshape_and_cache,
        cascade_attention=cascade_attention,
    )
    monkeypatch.setattr(
        beam_attn,
        "_get_active_attention_hooks",
        lambda: (_ for _ in ()).throw(AssertionError("hot path queried Registry")),
    )

    impl = object.__new__(beam_attn.BeamAttentionImpl)
    impl._hooks = execution
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
