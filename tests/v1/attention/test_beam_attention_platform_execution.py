# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Platform isolation and delegation contracts for Beam Attention."""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_gr.platforms as table
from tests.helpers.native_platform import native_platform
from tests.helpers.paths import REPO_ROOT
from vllm_gr.platforms.errors import UnsupportedBackendError
from vllm_gr.v1.attention.backends import beam_attn

try:
    import torch_npu  # noqa: F401

    _TORCH_NPU_IMPORTABLE = True
except (ImportError, RuntimeError):
    _TORCH_NPU_IMPORTABLE = False


def test_facade_import_does_not_load_platform_execution_modules() -> None:
    project_root = REPO_ROOT
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


def test_xllm_module_import_fails_fast_without_operator_library() -> None:
    """Importing beam_attn_npu_xllm without xllm-ops must raise immediately."""
    project_root = REPO_ROOT
    code = """
import importlib.util
import os
import sys
import types
from unittest import mock

# Keep torch from auto-loading the (stubbed) torch_npu backend extension.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

# Simulate a machine without the xllm-ops library: every probe reports
# unavailable and names the missing binding.  torch_npu is stubbed with a
# real module carrying a spec (a bare stub breaks find_spec in transformers).
stub = types.ModuleType("torch_npu")
stub._C = mock.MagicMock()
stub.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
stub.__file__ = "/fake/torch_npu/__init__.py"
sys.modules["torch_npu"] = stub
import vllm_gr.ops.npu.beam_ops as beam_ops

mock_probe = mock.patch.object(beam_ops, "is_available", return_value=False)
mock_reason = mock.patch.object(
    beam_ops, "unavailable_reason", return_value="custom_ops_lib cannot be imported"
)
with mock_probe, mock_reason:
    try:
        import vllm_gr.v1.attention.backends.beam_attn_npu_xllm
    except RuntimeError as exc:
        assert "requires the xllm-ops library" in str(exc), exc
        assert "custom_ops_lib cannot be imported" in str(exc), exc
    else:
        raise AssertionError("expected RuntimeError without xllm-ops")

# The failed import must not linger in sys.modules.
assert "vllm_gr.v1.attention.backends.beam_attn_npu_xllm" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    ("platform_enum", "placement", "lineage", "expected"),
    [
        ("CUDA", "cuda", (), "gpu"),
        ("OOT", "npu", ("vllm_ascend.platform.NPUPlatform",), "npu"),
        ("ROCM", "cuda", (), None),
        ("OOT", "npu", ("other.platform.Accelerator",), None),
    ],
)
def test_platform_dispatch_selects_exactly_one_execution_module(
    monkeypatch: pytest.MonkeyPatch,
    platform_enum: str,
    placement: str,
    lineage: tuple[str, ...],
    expected: str | None,
) -> None:
    import vllm.platforms

    native = native_platform(platform_enum, placement)
    if lineage == ("other.platform.Accelerator",):
        from tests.helpers.native_platform import VendorNative

        native = VendorNative()
        native.device_type = "npu"
    monkeypatch.setattr(vllm.platforms, "current_platform", native)
    modules = {
        name: ModuleType(f"vllm_gr.v1.attention.backends.beam_attn_{name}")
        for name in ("gpu", "npu")
    }
    for module in modules.values():
        monkeypatch.setitem(sys.modules, module.__name__, module)

    if expected is None:
        with pytest.raises(UnsupportedBackendError, match="No GR integration"):
            beam_attn._get_platform_execution_module()
    else:
        assert beam_attn._get_platform_execution_module() is modules[expected]


@pytest.mark.npu_test  # type: ignore[misc]
@pytest.mark.skipif(  # type: ignore[misc]
    not _TORCH_NPU_IMPORTABLE, reason="torch_npu is not installed"
)
@pytest.mark.parametrize(  # type: ignore[misc]
    ("mode", "expected_module_name"),
    [
        ("cascade", "vllm_gr.v1.attention.backends.beam_attn_npu"),
        ("xllm", "vllm_gr.v1.attention.backends.beam_attn_npu_xllm"),
    ],
)
def test_npu_dispatch_selects_execution_mode_module(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_module_name: str,
) -> None:
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", False)
    monkeypatch.setattr(beam_attn, "NPU_DEVICE", True)
    from tests.helpers.ascend_platform import AscendGRPlatform
    from vllm_gr.platforms.ascend import decode

    monkeypatch.setattr(table, "get_current_gr_platform", AscendGRPlatform)
    monkeypatch.setattr(decode, "engine_mode", lambda config: mode)
    assert beam_attn._get_platform_execution_module().__name__ == expected_module_name


def test_ascend_decode_choices_are_engine_local(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.ops.npu import beam_ops
    from vllm_gr.platforms.ascend.decode import engine_mode

    auto = SimpleNamespace(additional_config={})
    explicit = SimpleNamespace(
        additional_config={"vllm_gr": {"gr": {"npu_decode_execution": "cascade"}}}
    )
    monkeypatch.setattr(beam_ops, "is_available", lambda: True)
    assert engine_mode(auto) == "xllm"
    assert engine_mode(explicit) == "cascade"
    assert engine_mode(None) == "cascade"
    assert engine_mode(auto) == "xllm"


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
        reshape_and_cache=reshape_and_cache,
        cascade_attention=cascade_attention,
    )
    monkeypatch.setattr(
        beam_attn,
        "_get_platform_execution_module",
        lambda: execution,
    )
    monkeypatch.setattr(beam_attn, "NPU_DEVICE", tuple_cache)
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", not tuple_cache)

    impl = object.__new__(beam_attn.BeamAttentionImpl)
    impl.kv_sharing_target_layer_name = None
    impl.kv_cache_dtype = "auto"
    impl.sliding_window = (6, 0)
    impl.scale = 0.5
    impl.logits_soft_cap = 9.0
    impl.vllm_flash_attn_version = 3
    impl.prefill_flash_attn_version = impl.vllm_flash_attn_version
    impl.sinks = torch.empty(1)
    impl.alibi_slopes = torch.empty(2)

    impl.head_size = 8
    key_cache = torch.zeros(2, 16, 2, 8)
    value_cache = torch.ones_like(key_cache)
    kv_cache: Any
    if tuple_cache:
        kv_cache = (key_cache, value_cache)
    else:
        kv_cache = torch.cat((key_cache, value_cache), dim=-1).transpose(1, 2)

    key = torch.empty(1)
    value = torch.empty(1)
    query = torch.empty(1)
    output = torch.empty(1)
    slot_mapping = torch.tensor([0])
    metadata = SimpleNamespace(slot_mapping=slot_mapping, is_beam_decode=False, beam_session=None)
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


def test_forward_incremental_uses_beam_pool_and_skips_paged_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the incremental beam decode path uses beam_pool and
    skips reshape_and_cache for the paged KV cache."""
    from vllm_gr.v1.beam.attention_context import BeamAttentionContext
    from vllm_gr.v1.beam.session import BeamSearchSession

    calls: dict[str, Any] = {}

    def unexpected_reshape(*args: Any) -> None:
        raise AssertionError("incremental forward must not write the Paged KV Cache")

    def cascade_attention(**kwargs: Any) -> None:
        calls.update(kwargs)

    from vllm_gr.v1.beam.attention_pool import write_eager_step

    execution = SimpleNamespace(
        reshape_and_cache=unexpected_reshape,
        cascade_attention=cascade_attention,
        cache_unshared_kv=write_eager_step,
    )
    monkeypatch.setattr(beam_attn, "_get_platform_execution_module", lambda: execution)
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", True)
    monkeypatch.setattr(beam_attn, "NPU_DEVICE", False)

    # Create a session using our 3-layer architecture
    context = BeamAttentionContext(
        max_batch=1,
        max_beams=4,
        num_kv_heads=2,
        max_decode_steps=3,
        head_dim=8,
        num_layers=1,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    slot = context.registry.allocate("test", beam_width=0, prefix_len=0)
    session = BeamSearchSession(session_id="test", slot=slot, context=context)
    session.begin_session(beam_width=2, prefix_len=5)

    metadata = SimpleNamespace(
        is_beam_decode=True,
        beam_session=session,
        incremental_suffix_kv_len=1,
        suffix_max_q_len=1,
        prefix_max_seq_len=5,
        prefix_indices=torch.arange(2),
    )

    impl = object.__new__(beam_attn.BeamAttentionImpl)
    impl.kv_sharing_target_layer_name = None
    impl.kv_cache_dtype = "auto"
    impl.sliding_window = (-1, -1)
    impl.scale = 0.5
    impl.logits_soft_cap = 0.0
    impl.vllm_flash_attn_version = 3
    impl.prefill_flash_attn_version = impl.vllm_flash_attn_version
    impl.sinks = None
    impl.alibi_slopes = None

    query = torch.zeros(2, 4, 8, dtype=torch.float16)
    impl.head_size = 8
    key = torch.arange(2 * 2 * 8, dtype=torch.float16).reshape(2, 2, 8)
    value = key + 100
    output = torch.empty_like(query)
    key_cache = torch.empty(1, 1, 2, 8, dtype=torch.float16)
    value_cache = torch.empty_like(key_cache)

    pool = session.pools[0]
    stale_context = BeamAttentionContext(
        max_batch=1,
        max_beams=4,
        num_kv_heads=2,
        max_decode_steps=3,
        head_dim=8,
        num_layers=1,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    stale_pool = stale_context.pools[0]
    layer = SimpleNamespace(
        layer_name="model.layers.0.self_attn",
        _k_scale=None,
        _v_scale=None,
        _beam_pool=stale_pool,
    )

    returned = impl.forward(
        layer=layer,
        query=query,
        key=key,
        value=value,
        kv_cache=torch.cat((key_cache, value_cache), dim=-1).transpose(1, 2),
        attn_metadata=metadata,
        output=output,
    )

    assert returned is output
    # Verify that cascade_attention was called with beam_pool
    assert not stale_pool.unshared_key.count_nonzero()
    assert not stale_pool.unshared_value.count_nonzero()
    assert "beam_pool" in calls
    assert calls["beam_pool"] is pool
    # Verify pool KV was written by cache_unshared_kv
    torch.testing.assert_close(pool.unshared_key[0, :2, :, 0, :], key)
    torch.testing.assert_close(pool.unshared_value[0, :2, :, 0, :], value)


def test_npu_prefill_attention_consumes_common_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fia_kwargs: dict[str, Any] = {}

    def fake_fia_v2(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        fia_kwargs.update(kwargs)
        return torch.zeros(7, 8, 64), torch.empty(1)

    fake_torch_npu = SimpleNamespace(npu_fused_infer_attention_score_v2=fake_fia_v2)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    npu_name = "vllm_gr.v1.attention.backends.beam_attn_npu"
    # ``from pkg import beam_attn_npu`` would return a stale module on machines
    # where the module was already imported (the package attribute short-circuits
    # _handle_fromlist), so resolve it explicitly and patch its torch_npu global.
    try:
        beam_attn_npu = sys.modules[npu_name]
    except KeyError:
        import importlib

        beam_attn_npu = importlib.import_module(npu_name)
    monkeypatch.setattr(beam_attn_npu, "torch_npu", fake_torch_npu)

    from vllm_gr.v1.attention.backends.beam_attn_metadata import BeamAttentionMetadata

    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    metadata = BeamAttentionMetadata(
        num_actual_tokens=7,
        max_query_len=4,
        query_start_loc=torch.tensor([0, 3, 7, 0, 0], dtype=torch.int32),
        max_seq_len=7,
        seq_lens=torch.tensor([7, 4, 0], dtype=torch.int32),
        block_table=block_table,
        slot_mapping=torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64),
        causal=True,
        is_prefill_graph_exec=True,
        num_reqs=2,
        num_heads=8,
        num_kv_heads=8,
        block_size=16,
        query_start_loc_cpu=[0, 3, 7, 0, 0],
        seq_lens_cpu=[7, 4, 0],
    )
    output = torch.zeros(7, 8, 64)
    # Padded batch: 8 query rows vs 7 actual tokens (vllm-ascend pads to
    # num_tokens_padded) — the fast path must slice to the real rows.
    query = torch.zeros(8, 8, 64)
    key_cache = torch.zeros(8, 16, 8, 64)
    value_cache = torch.zeros(8, 16, 8, 64)

    beam_attn_npu.prefill_attention(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        attn_metadata=metadata,
        sliding_window_size=None,
        scale=0.125,
        logits_soft_cap=0.0,
        vllm_flash_attn_version=0,
    )

    assert fia_kwargs["query"].shape == (7, 8, 64)
    assert fia_kwargs["actual_seq_qlen"] == [3, 7]
    assert fia_kwargs["actual_seq_kvlen"] == [7, 4]
    assert fia_kwargs["block_table"] is block_table
    assert fia_kwargs["num_query_heads"] == 8
    assert fia_kwargs["num_key_value_heads"] == 8
    assert fia_kwargs["block_size"] == 16
    assert fia_kwargs["input_layout"] == "TND"
    assert torch.equal(output[:4], torch.zeros(4, 8, 64))


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "invalid",
    ["missing-session", "inactive-session", "old-metadata", "multi-query", "missing-pool"],
)
def test_incremental_forward_rejects_invalid_state_before_cache_write(
    monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    from vllm_gr.v1.attention.backends import beam_attn

    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid beam state reached a device operation")

    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", True)
    monkeypatch.setattr(beam_attn, "NPU_DEVICE", False)
    monkeypatch.setattr(
        beam_attn,
        "_get_platform_execution_module",
        lambda: SimpleNamespace(
            reshape_and_cache=unexpected, cache_unshared_kv=unexpected, cascade_attention=unexpected
        ),
    )
    session = SimpleNamespace(is_active=invalid != "inactive-session", pools=[])
    metadata = SimpleNamespace(
        is_beam_decode=invalid != "old-metadata",
        prefix_indices=torch.arange(2),
        suffix_max_q_len=2 if invalid == "multi-query" else 1,
        beam_session=None if invalid == "missing-session" else session,
    )
    impl = object.__new__(beam_attn.BeamAttentionImpl)
    impl.vllm_flash_attn_version = 2
    impl.head_size = 8
    tensor = torch.empty(2, 1, 8)
    with pytest.raises((RuntimeError, ValueError), match="Beam decode requires"):
        impl.forward(
            SimpleNamespace(layer_name="model.layers.0.self_attn"),
            tensor,
            tensor,
            tensor,
            torch.empty(1, 1, 16, 16),
            metadata,
            output=tensor,
        )
