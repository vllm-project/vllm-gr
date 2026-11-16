# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disallow-untyped-decorators=False

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm_gr.config import (
    ConstraintTableConfig,
    GRConfig,
    gr_config_from_mapping,
    gr_config_to_mapping,
)
from vllm_gr.v1.attention.backends import beam_attn, beam_attn_gpu


@pytest.fixture
def flash_backend(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # These tests exercise policy selection without loading CUDA extensions on NPU.
    flash = ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
    monkeypatch.setattr(flash, "is_fa_version_supported", lambda version: True, raising=False)
    monkeypatch.setattr(
        flash, "fa_version_unsupported_reason", lambda version: "test device", raising=False
    )
    monkeypatch.setitem(sys.modules, flash.__name__, flash)
    return flash


@pytest.mark.parametrize("version", [None, "auto", 2, 3])
@pytest.mark.parametrize("short_suffix", [False, True])
def test_prefill_config_roundtrip(version: Any, short_suffix: bool) -> None:
    config = GRConfig(
        prefill_flash_attn_version=version, enable_short_suffix_attention=short_suffix
    )
    worker = SimpleNamespace(additional_config={})
    config.write_to_vllm_config(worker)
    assert GRConfig.from_vllm_config(worker) == config
    canonical = GRConfig(
        prefill_flash_attn_version=version,
        enable_short_suffix_attention=short_suffix,
        constraint_table=ConstraintTableConfig(max_top_k=256),
    )
    mapping = gr_config_to_mapping(canonical)
    assert gr_config_from_mapping(mapping) == canonical
    if short_suffix:
        assert "enable_short_suffix_attention" not in mapping["beam"]
    else:
        assert mapping["beam"]["enable_short_suffix_attention"] is False
    if version == "auto":
        assert "prefill_flash_attn_version" not in mapping["beam"]
    elif version is None:
        assert mapping["beam"]["prefill_flash_attn_version"] is None


@pytest.mark.parametrize("version", [True, False, 0, 4, 3.0, "3", "fast", [], {}])
def test_prefill_config_rejects_invalid_version(version: Any) -> None:
    with pytest.raises(ValueError, match="prefill_flash_attn_version"):
        GRConfig(prefill_flash_attn_version=version)


def _resolve(requested: Any, decode: int = 2, **changes: Any) -> int:
    options = dict(
        head_size=128,
        kv_cache_dtype="auto",
        requires_alibi=False,
        has_sinks=False,
        batch_invariant=False,
    )
    options.update(changes)
    return cast(int, beam_attn_gpu.resolve_prefill_flash_attn_version(requested, decode, **options))


@pytest.mark.parametrize("available", [True, False])
def test_auto_prefers_fa3_but_explicit_request_must_be_supported(
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
    flash_backend: ModuleType,
) -> None:
    monkeypatch.setattr(flash_backend, "is_fa_version_supported", lambda version: available)
    assert _resolve(None) == 2
    assert _resolve("auto") == (3 if available else 2)
    assert _resolve("auto", decode=4) == 4  # Do not downgrade an existing backend.
    if available:
        assert _resolve(3) == 3
    else:
        with pytest.raises(ValueError, match="test device"):
            _resolve(3)


@pytest.mark.parametrize(
    "options",
    [
        {"kv_cache_dtype": "fp8"},
        {"requires_alibi": True},
        {"has_sinks": True},
        {"batch_invariant": True},
        {"head_size": 7},
    ],
)
def test_special_features_keep_existing_version_or_fail_explicitly(options: dict[str, Any]) -> None:
    assert _resolve("auto", **options) == 2
    with pytest.raises(ValueError, match="Cannot select prefill"):
        _resolve(3, **options)


def _impl() -> Any:
    impl = object.__new__(beam_attn.BeamAttentionImpl)
    impl.kv_sharing_target_layer_name = None
    impl.kv_cache_dtype = "auto"
    impl.sliding_window = (-1, -1)
    impl.scale = 0.125
    impl.logits_soft_cap = 0.0
    impl.vllm_flash_attn_version = 2
    impl.prefill_flash_attn_version = 3
    impl.head_size = 8
    impl.sinks = None
    impl.alibi_slopes = None
    return impl


@pytest.mark.parametrize("graph", [False, True])
def test_forward_routes_both_prefill_entries_without_changing_decode(
    monkeypatch: pytest.MonkeyPatch,
    graph: bool,
) -> None:
    calls: list[dict[str, Any]] = []
    execution = SimpleNamespace(
        reshape_and_cache=lambda *args: None,
        cascade_attention=lambda **kwargs: calls.append(kwargs),
        prefill_attention=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", True)
    monkeypatch.setattr(beam_attn, "NPU_DEVICE", False)
    monkeypatch.setattr(beam_attn, "_get_platform_execution_module", lambda: execution)
    impl = _impl()
    q = torch.zeros(4, 2, 8)
    key_cache = torch.zeros(2, 16, 2, 8)
    cache = torch.cat((key_cache, torch.zeros_like(key_cache)), dim=-1).transpose(1, 2)
    output = torch.empty_like(q)
    metadata = SimpleNamespace(is_prefill_graph_exec=graph, slot_mapping=torch.arange(4))
    impl.forward(SimpleNamespace(_k_scale=None, _v_scale=None), q, q, q, cache, metadata, output)
    assert impl.vllm_flash_attn_version == 2
    assert calls[0]["vllm_flash_attn_version"] == (3 if graph else 2)
    if not graph:
        assert calls[0]["prefill_flash_attn_version"] == 3


@pytest.mark.parametrize("q_len", [1, 8])
def test_mixed_cascade_keeps_prefix_suffix_on_fa2(
    monkeypatch: pytest.MonkeyPatch,
    q_len: int,
) -> None:
    versions: list[int] = []

    def flash(**kwargs: Any) -> Any:
        versions.append(kwargs["fa_version"])
        out = torch.zeros_like(kwargs["q"])
        return (out, torch.zeros(2, out.shape[0])) if kwargs.get("return_softmax_lse") else out

    monkeypatch.setattr(beam_attn_gpu, "flash_attn_varlen_func", flash)
    monkeypatch.setattr(beam_attn_gpu, "merge_attn_states", lambda *args: None)

    # This CPU routing test uses nonempty prefixes; device-state repair is
    # covered by the attention kernel tests.
    def clear_empty(state: Any, cu_q: torch.Tensor, lengths: torch.Tensor) -> Any:
        assert torch.all(lengths > 0)
        return state

    monkeypatch.setattr(beam_attn_gpu, "clear_empty_prefix_states", clear_empty)
    monkeypatch.setattr(
        beam_attn_gpu, "extract_suffix_kv", lambda *args: (torch.zeros(1), torch.zeros(1))
    )

    md = SimpleNamespace(
        std_indices=torch.tensor([0]),
        num_std_indices=1,
        prefix_indices=torch.tensor([1]),
        std_cu_seqlens_q_device=torch.tensor([0, 1]),
        std_max_q_len=q_len,
        std_seqlens_kv_device=torch.tensor([8]),
        std_max_seq_len=8,
        causal=True,
        std_block_table=torch.zeros(1, 1, dtype=torch.int32),
        prefix_cu_seqlens_q_device=torch.tensor([0, 1]),
        prefix_max_q_len=1,
        prefix_seqlens_kv_device=torch.tensor([8]),
        prefix_max_seq_len=8,
        prefix_scheduler_metadata=None,
        prefix_block_table=torch.zeros(1, 1, dtype=torch.int32),
        max_num_splits=0,
        beam_session=None,
        suffix_slot_mapping_out=torch.zeros(1, dtype=torch.int64),
        suffix_total_tokens=1,
        block_size=16,
        suffix_cu_seqlens_q_device=torch.tensor([0, 1]),
        suffix_cu_seqlens_k_device=torch.tensor([0, 1]),
        suffix_max_q_len=1,
        suffix_max_seq_len=1,
        num_heads=2,
        num_kv_heads=2,
    )
    q = torch.zeros(2, 2, 8)
    beam_attn_gpu.cascade_attention(
        torch.empty_like(q), q, q, q, md, (-1, -1), 0.125, 0, 2, prefill_flash_attn_version=3
    )
    assert versions == ([2, 2, 2] if q_len == 1 else [3, 2, 2])


@pytest.mark.parametrize("policy", [{}, {"vllm_gr_prefill_flash_attn_version": 3}])
@pytest.mark.parametrize("decode_version", [2, 3, 4])
@pytest.mark.parametrize("short_suffix", [False, True])
def test_initialization_preserves_global_decode_config(
    monkeypatch: pytest.MonkeyPatch,
    policy: dict[str, Any],
    flash_backend: ModuleType,
    decode_version: int,
    short_suffix: bool,
) -> None:
    import vllm.config

    config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=decode_version),
        additional_config={
            "vllm_gr_beam_max_width": 128,
            "vllm_gr_enable_short_suffix_attention": short_suffix,
            **policy,
        },
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", True)
    monkeypatch.setattr(beam_attn, "NPU_DEVICE", False)
    monkeypatch.setattr(beam_attn.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(beam_attn, "_get_platform_execution_module", lambda: beam_attn_gpu)
    monkeypatch.setattr(beam_attn_gpu, "get_flash_attn_version", lambda **kwargs: decode_version)
    impl = beam_attn.BeamAttentionImpl(16, 128, 128**-0.5, 8, None, None, "auto")
    assert impl.prefill_flash_attn_version == (
        3 if policy or decode_version == 2 else decode_version
    )
    assert impl.vllm_flash_attn_version == decode_version
    assert config.attention_config.flash_attn_version == decode_version
    assert impl.supports_quant_query_input is (not short_suffix)


def test_omitted_prefill_policy_defaults_to_auto() -> None:
    assert GRConfig().prefill_flash_attn_version == "auto"
    assert (
        GRConfig.from_vllm_config(SimpleNamespace(additional_config={})).prefill_flash_attn_version
        == "auto"
    )
    canonical = GRConfig(constraint_table=ConstraintTableConfig(max_top_k=256))
    mapping = gr_config_to_mapping(canonical)
    assert "prefill_flash_attn_version" not in mapping["beam"]
    assert gr_config_from_mapping(mapping).prefill_flash_attn_version == "auto"


@pytest.mark.parametrize("policy", [None, "auto"])
def test_npu_prefill_default_and_inherit_are_noops(
    monkeypatch: pytest.MonkeyPatch, policy: Any
) -> None:
    import vllm.config

    config = SimpleNamespace(additional_config={"vllm_gr_prefill_flash_attn_version": policy})
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", False)
    monkeypatch.setattr(beam_attn, "NPU_DEVICE", True)
    execution = SimpleNamespace(get_flash_attn_version=lambda **kw: None)
    monkeypatch.setattr(beam_attn, "_get_platform_execution_module", lambda: execution)
    monkeypatch.setattr(
        beam_attn_gpu,
        "resolve_prefill_flash_attn_version",
        lambda *args, **kw: pytest.fail("NPU must not resolve CUDA FA"),
    )
    impl = beam_attn.BeamAttentionImpl(16, 128, 128**-0.5, 8, None, None, "auto")
    assert impl.prefill_flash_attn_version is None


pytestmark = pytest.mark.cpu_test
