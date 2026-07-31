# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform isolation and Provider delegation contracts for Beam Attention."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.platforms.errors import BackendNotActiveError
from vllm_gr.platforms.registry import _reset_backend_registry_for_tests
from vllm_gr.v1.attention.backends import beam_attn


def test_facade_import_does_not_load_or_resolve_platform_hooks() -> None:
    project_root = Path(__file__).resolve().parents[1]
    code = """
import sys

import vllm_gr.v1.attention.backends.beam_attn

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


def test_attention_hooks_require_an_active_provider() -> None:
    _reset_backend_registry_for_tests()
    with pytest.raises(BackendNotActiveError):
        beam_attn.BeamAttentionBackend.supports_batch_invariance()


def test_platform_modules_implement_attention_hook_surface() -> None:
    project_root = Path(__file__).resolve().parents[1]
    required = {
        "is_available",
        "get_flash_attn_version",
        "supports_batch_invariance",
        "supports_kv_cache_dtype",
        "supports_sinks",
        "supports_compute_capability",
        "validate_combination",
        "get_kv_cache_shape",
        "get_kv_cache_stride_order",
        "split_kv_cache",
        "get_kv_scales",
        "reshape_and_cache",
        "cascade_attention",
        "get_metadata_builder_cls",
    }

    for filename in ("beam_attn_gpu.py", "beam_attn_npu.py"):
        path = project_root / "vllm_gr" / "v1" / "attention" / "backends" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert required <= functions

    gpu_source = (project_root / "vllm_gr/v1/attention/backends/beam_attn_gpu.py").read_text(
        encoding="utf-8"
    )
    assert "torch_npu" not in gpu_source
    assert "vllm_ascend" not in gpu_source


def test_cascade_attention_resolves_provider_hooks_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    hooks = SimpleNamespace(cascade_attention=lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(beam_attn, "_get_active_attention_hooks", lambda: hooks)

    output = torch.empty(1)
    metadata = SimpleNamespace()
    beam_attn.BeamAttentionImpl.cascade_attention(
        output=output,
        query=torch.empty(2),
        key_cache=torch.empty(3),
        value_cache=torch.empty(4),
        attn_metadata=metadata,
        sliding_window_size=[7, 0],
        scale=0.25,
        logits_soft_cap=12.0,
        vllm_flash_attn_version=3,
    )

    assert captured["output"] is output
    assert captured["attn_metadata"] is metadata
    assert captured["sliding_window_size"] == [7, 0]
    assert captured["vllm_flash_attn_version"] == 3


def test_forward_uses_hooks_cached_by_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    key_cache = torch.zeros(2, 2)
    value_cache = torch.ones(2, 2)
    k_scale = torch.tensor(1.0)
    v_scale = torch.tensor(2.0)

    hooks = SimpleNamespace(
        split_kv_cache=lambda cache: (cache[0], cache[1]),
        get_kv_scales=lambda layer: (layer._k_scale, layer._v_scale),
        reshape_and_cache=lambda *args: calls.setdefault("reshape", args),
        cascade_attention=lambda **kwargs: calls.setdefault("cascade", kwargs),
    )
    monkeypatch.setattr(
        beam_attn,
        "_get_active_attention_hooks",
        lambda: pytest.fail("forward must not re-read the Provider registry"),
    )

    impl = object.__new__(beam_attn.BeamAttentionImpl)
    impl._hooks = hooks
    impl.kv_sharing_target_layer_name = None
    impl.kv_cache_dtype = "auto"
    impl.sliding_window = (6, 0)
    impl.scale = 0.5
    impl.logits_soft_cap = 9.0
    impl.vllm_flash_attn_version = 3
    impl.sinks = None
    impl.alibi_slopes = None

    output = torch.empty(1)
    slot_mapping = torch.tensor([0])
    metadata = SimpleNamespace(slot_mapping=slot_mapping)
    returned = impl.forward(
        layer=SimpleNamespace(_k_scale=k_scale, _v_scale=v_scale),
        query=torch.empty(1),
        key=torch.empty(1),
        value=torch.empty(1),
        kv_cache=(key_cache, value_cache),
        attn_metadata=metadata,
        output=output,
    )

    assert returned is output
    assert calls["reshape"][2] is key_cache
    assert calls["reshape"][3] is value_cache
    assert calls["reshape"][6] is k_scale
    assert calls["reshape"][7] is v_scale
    assert calls["cascade"]["attn_metadata"] is metadata
