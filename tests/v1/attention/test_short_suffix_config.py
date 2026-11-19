# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disallow-untyped-decorators=False

import inspect
import subprocess
import sys
from textwrap import dedent
from types import ModuleType, SimpleNamespace
from typing import Any, Literal

import pytest
import torch

from tests.helpers.attention import bind_attention
from vllm_gr.config import GRConfig, gr_config_from_mapping, gr_config_to_mapping
from vllm_gr.platforms.legacy import legacy_config_service
from vllm_gr.v1.attention.backends.beam_attn_gpu import (
    GPUBeamAttentionBatch,
    validate_short_suffix_attention,
)
from vllm_gr.v1.attention.backends.beam_attn_metadata import BeamAttentionMetadata
from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext


@pytest.fixture
def flash_backend(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # Policy tests must not import CUDA extensions on NPU or depend on local GPUs.
    flash = ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
    monkeypatch.setattr(flash, "is_fa_version_supported", lambda version: True, raising=False)
    monkeypatch.setattr(
        flash, "fa_version_unsupported_reason", lambda version: "test device", raising=False
    )
    monkeypatch.setitem(sys.modules, flash.__name__, flash)
    return flash


def test_short_suffix_import_does_not_require_gluon() -> None:
    # Use a fresh interpreter so a Gluon import from another test cannot hide
    # the regression. NPU Triton builds may ship an incompatible Gluon module.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""
                import importlib.abc
                import sys

                class BlockGluon(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "triton.experimental.gluon" or fullname.startswith(
                            "triton.experimental.gluon."
                        ):
                            raise ImportError("Gluon deliberately unavailable")

                sys.meta_path.insert(0, BlockGluon())
                from vllm_gr.v1.attention.backends import beam_short_suffix

                assert callable(beam_short_suffix.merge_short_suffix_attention)
                assert "vllm_gr.v1.attention.backends.beam_short_suffix_fa2" not in sys.modules
            """),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("capacity", [1, 3, 4, 8, 16, 32, 128])
def test_config_roundtrip(enabled: bool, capacity: int) -> None:
    raw = {
        "schema_version": 1,
        "beam": {
            "max_width": 256,
            "max_decode_steps": capacity,
            "enable_short_suffix_attention": enabled,
        },
        "constraint_table": {"max_top_k": 256},
    }
    config = gr_config_from_mapping(raw)

    assert config.enable_short_suffix_attention is enabled
    assert gr_config_from_mapping(gr_config_to_mapping(config)) == config

    worker = SimpleNamespace(additional_config={"vllm_gr": gr_config_to_mapping(config)})
    assert GRConfig.from_vllm_config(worker).enable_short_suffix_attention is enabled

    legacy = GRConfig(
        beam_max_width=256,
        beam_max_decode_steps=capacity,
        enable_short_suffix_attention=enabled,
    )
    worker = SimpleNamespace(additional_config={})
    legacy.write_to_vllm_config(worker)

    assert GRConfig.from_vllm_config(worker) == legacy
    GRConfig(beam_max_width=256).write_to_vllm_config(worker)
    assert GRConfig.from_vllm_config(worker).enable_short_suffix_attention is True


@pytest.mark.parametrize("value", [None, "1", 1])
def test_config_requires_boolean(value: Any) -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        GRConfig(beam_max_width=128, enable_short_suffix_attention=value)


@pytest.mark.parametrize("width", [None, 128])
def test_config_defaults_enable_short_suffix_but_no_pool_stays_dormant(
    width: int | None,
) -> None:
    config = GRConfig(beam_max_width=width)
    assert config.enable_short_suffix_attention is True
    assert GRConfig().enabled is False
    worker = SimpleNamespace(additional_config={"vllm_gr_beam_max_width": width})
    assert GRConfig.from_vllm_config(worker) == config
    canonical = gr_config_from_mapping(
        {
            "schema_version": 1,
            "beam": {"max_width": width},
            "constraint_table": {"max_top_k": 128},
        }
    )
    assert canonical.enable_short_suffix_attention is True
    assert gr_config_from_mapping(gr_config_to_mapping(canonical)) == canonical


def _short_suffix_engine_config(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    *,
    enabled: bool | None = None,
    width: int | None = 128,
    cuda: bool = True,
    fa_version: Literal[2, 3, 4] | None = None,
    custom: bool = True,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    from vllm.config import AttentionConfig

    from vllm_gr.config import (
        BeamExecutionMode,
        ConstraintTableConfig,
        GRAttentionBackend,
    )
    from vllm_gr.engine import arg_utils_gr

    options: dict[str, Any] = {} if enabled is None else {"enable_short_suffix_attention": enabled}
    if source == "canonical":
        options.update(schema_version=1, constraint_table=ConstraintTableConfig(max_top_k=128))
    gr_config = GRConfig(
        beam_max_width=width,
        beam_execution_mode=BeamExecutionMode.V1,
        attention_backend=GRAttentionBackend.CUSTOM if custom else None,
        **options,
    )
    config = SimpleNamespace(
        additional_config={},
        attention_config=AttentionConfig(flash_attn_version=fa_version),
        scheduler_config=SimpleNamespace(async_scheduling=True, enable_chunked_prefill=False),
    )
    args = SimpleNamespace()
    if source == "canonical":
        args._vllm_gr_config_input = gr_config_to_mapping(gr_config)
    else:
        gr_config.write_to_vllm_config(config)
    monkeypatch.setattr(arg_utils_gr, "origin_create_engine_config", lambda *args: config)
    monkeypatch.setattr(
        arg_utils_gr,
        "engine_config_service",
        lambda: legacy_config_service("cuda" if cuda else "npu"),
    )
    return config, args


@pytest.mark.parametrize("source", ["canonical", "legacy"])
@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, None),
        ({"enabled": True}, None),
        ({"fa_version": 2}, 2),
        ({"enabled": True, "fa_version": 2}, 2),
        ({"enabled": False}, None),
        ({"enabled": False, "fa_version": 3}, 3),
        ({"enabled": False, "fa_version": 4}, 4),
        ({"width": None}, None),
        ({"width": None, "fa_version": 3}, 3),
        ({"cuda": False}, None),
        ({"cuda": False, "fa_version": 3}, 3),
        ({"custom": False}, None),
        ({"custom": False, "fa_version": 3}, 3),
    ],
)
def test_engine_preserves_automatic_selection_and_explicit_version(
    monkeypatch: pytest.MonkeyPatch, source: str, options: dict[str, Any], expected: int | None
) -> None:
    from vllm_gr.engine import arg_utils_gr

    config, args = _short_suffix_engine_config(monkeypatch, source, **options)
    assert arg_utils_gr.create_engine_config(args) is config
    assert config.attention_config.flash_attn_version == expected
    assert GRConfig.from_vllm_config(config).enable_short_suffix_attention is options.get(
        "enabled", True
    )


@pytest.mark.parametrize("source", ["canonical", "legacy"])
@pytest.mark.parametrize("fa_version", [3, 4])
def test_fused_engine_accepts_explicit_fa(
    monkeypatch: pytest.MonkeyPatch, source: str, fa_version: Literal[3, 4]
) -> None:
    from vllm_gr.engine import arg_utils_gr

    config, args = _short_suffix_engine_config(monkeypatch, source, fa_version=fa_version)
    assert arg_utils_gr.create_engine_config(args) is config
    assert config.attention_config.flash_attn_version == fa_version


@pytest.mark.parametrize("source", ["canonical", "legacy"])
@pytest.mark.parametrize(
    ("enabled", "fa_version"),
    [(None, None), (False, None), (None, 2), (True, 2), (True, None), (True, 3)],
)
@pytest.mark.parametrize("capability,default_version", [(8, 2), (9, 3), (10, 4)])
def test_fusion_preserves_platform_auto_selection(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    enabled: bool | None,
    fa_version: Literal[2, 3] | None,
    capability: int,
    default_version: int,
    flash_backend: ModuleType,
) -> None:
    import vllm.config
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends import fa_utils

    from vllm_gr.engine import arg_utils_gr
    from vllm_gr.v1.attention.backends import beam_attn, beam_attn_gpu

    config, args = _short_suffix_engine_config(
        monkeypatch, source, enabled=enabled, fa_version=fa_version
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(
        fa_utils,
        "current_platform",
        SimpleNamespace(
            is_xpu=lambda: False,
            is_rocm=lambda: False,
            get_device_capability=lambda: DeviceCapability(capability, 0),
        ),
    )
    monkeypatch.setattr(
        flash_backend, "is_fa_version_supported", lambda version: version in (2, default_version)
    )
    bind_attention(monkeypatch, beam_attn, cuda=True, execution=lambda: beam_attn_gpu)
    # Exercise the installed vLLM selector, including device fallbacks.
    if fa_version == 3 and capability == 8:
        pytest.skip("explicit FA3 is not supported on SM8x")
    expected = default_version if fa_version in (None, 3) else fa_version
    assert beam_attn_gpu.get_flash_attn_version(head_size=128) == expected
    arg_utils_gr.create_engine_config(args)
    assert config.attention_config.flash_attn_version == fa_version
    assert GRConfig.from_vllm_config(config).enable_short_suffix_attention is (enabled is not False)
    kwargs: dict[str, Any] = {
        "num_heads": 16,
        "head_size": 128,
        "scale": 128**-0.5,
        "num_kv_heads": 8,
        "alibi_slopes": None,
        "sliding_window": None,
        "kv_cache_dtype": "auto",
    }
    impl = beam_attn.BeamAttentionImpl(**kwargs)
    assert impl.vllm_flash_attn_version == expected
    assert impl.supports_quant_query_input is (enabled is False)


@pytest.mark.parametrize("capacity", [0, -1])
@pytest.mark.parametrize("enabled", [False, True])
def test_config_rejects_nonpositive_capacity(capacity: int, enabled: bool) -> None:
    with pytest.raises(ValueError, match="beam_max_decode_steps"):
        GRConfig(
            beam_max_width=128,
            beam_max_decode_steps=capacity,
            enable_short_suffix_attention=enabled,
        )


@pytest.mark.parametrize("capacity", [0, -1])
def test_context_rejects_nonpositive_capacity_before_allocating(capacity: int) -> None:
    with pytest.raises(ValueError, match="positive pool capacity"):
        BeamAttentionContext(
            1,
            3,
            1,
            capacity,
            64,
            1,
            torch.device("cuda"),
            torch.float32,
            enable_short_suffix_attention=True,
        )


@pytest.mark.gpu_test
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("capacity", [4, 8, 16, 32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_context_accepts_positive_capacity(capacity: int, dtype: torch.dtype) -> None:
    context = BeamAttentionContext(
        1,
        3,
        1,
        capacity,
        64,
        1,
        torch.device("cuda"),
        dtype,
        enable_short_suffix_attention=True,
    )
    assert context.enable_short_suffix_attention is True
    assert context.max_decode_steps == capacity
    assert context.pools[0].unshared_key.shape[3] == capacity


@pytest.mark.parametrize(
    "changes",
    [
        {"num_heads": 15},
        {"head_size": 100},
        {"head_size": 512},
        {"fa_version": 5},
        {"sliding_window": (32, 0)},
        {"softcap": 10.0},
        {"alibi_slopes": torch.zeros(16)},
        {"sinks": torch.zeros(16)},
    ],
)
def test_static_layer_rejections(changes: dict[str, Any]) -> None:
    args: dict[str, Any] = {
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_size": 128,
        "fa_version": 2,
        "sliding_window": (-1, -1),
        "softcap": 0.0,
        "alibi_slopes": None,
        "sinks": None,
    }
    validate_short_suffix_attention(**args)
    with pytest.raises(ValueError, match="Short suffix attention"):
        validate_short_suffix_attention(**(args | changes))


@pytest.mark.parametrize("head_size", [32, 96])
def test_static_layer_accepts_generic_kernel_head_sizes(head_size: int) -> None:
    """Head sizes outside the FA2 MMA fast path fall back to the Triton merge."""
    validate_short_suffix_attention(
        num_heads=8,
        num_kv_heads=4,
        head_size=head_size,
        fa_version=2,
        sliding_window=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        sinks=None,
    )


def test_context_rejects_cpu_before_allocating() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        BeamAttentionContext(
            1,
            128,
            8,
            3,
            128,
            1,
            torch.device("cpu"),
            torch.float32,
            enable_short_suffix_attention=True,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"std_indices": torch.tensor([0])},
        {"suffix_max_q_len": 2},
        {"num_prefix_indices": 3},
        {"num_actual_tokens": 5},
        {"prefix_indices": None},
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_metadata_allows_batches_that_need_generic_attention(
    changes: dict[str, Any], enabled: bool
) -> None:
    args: dict[str, Any] = {
        "num_actual_tokens": 4,
        "max_query_len": 4,
        "query_start_loc": torch.tensor([0, 4]),
        "max_seq_len": 128,
        "seq_lens": torch.tensor([128]),
        "block_table": torch.zeros(1, 8),
        "slot_mapping": torch.arange(4),
        "is_beam_decode": True,
        "prefix_indices": torch.arange(4),
        "num_prefix_indices": 4,
        "suffix_max_q_len": 1,
        "beam_session": SimpleNamespace(
            beam_width=4, context=SimpleNamespace(enable_short_suffix_attention=enabled)
        ),
    }
    BeamAttentionMetadata(**args)
    metadata = BeamAttentionMetadata(**(args | changes))
    for name, value in changes.items():
        actual = getattr(metadata, name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual, value)
        else:
            assert actual == value


def _cascade_case(batch: int, enabled: bool, routing: str) -> SimpleNamespace:
    """Use CPU storage to exercise the real request-major cascade routing."""
    request_width, heads, kv_heads, dim, capacity, step = 4, 2, 1, 64, 3, 1
    width = batch * request_width

    physical = (
        torch.arange(batch * request_width * kv_heads * capacity * dim)
        .reshape(batch, request_width, kv_heads, capacity, dim)
        .float()
    )
    source_pool = SimpleNamespace(
        num_kv_heads=kv_heads,
        head_dim=dim,
        max_decode_steps=capacity,
        unshared_key=physical,
        unshared_value=physical + 1000,
    )

    context = SimpleNamespace(
        enable_short_suffix_attention=enabled,
        max_beams=request_width,
        max_decode_steps=capacity,
        decode_step_buf=torch.tensor(step, dtype=torch.int32),
        pools=[source_pool],
        suffix_k_buf=torch.full((width * capacity, kv_heads, dim), float("nan")),
        suffix_v_buf=torch.full((width * capacity, kv_heads, dim), float("nan")),
    )

    sessions = tuple(
        SimpleNamespace(
            context=context,
            slot=slot,
            beam_width=request_width,
            decode_step=step,
            is_active=True,
            session_id=slot,
            pools=context.pools,
            suffix_k_buf=context.suffix_k_buf[: request_width * capacity],
            suffix_v_buf=context.suffix_v_buf[: request_width * capacity],
        )
        for slot in range(batch)
    )
    session = sessions[0] if batch == 1 else GPUBeamAttentionBatch(sessions)
    pool = session.pools[0]

    actual_rows = beam_rows = width
    std_indices: torch.Tensor | None = None
    prefix_indices: torch.Tensor | None = torch.arange(width)
    suffix_max_q_len = 1
    if routing == "mixed":
        assert batch == 1
        actual_rows = 6
        std_indices = torch.tensor([0, 3])
        prefix_indices = torch.tensor([1, 2, 4, 5])
    elif routing == "many_queries":
        actual_rows = beam_rows = 2 * width
        prefix_indices = torch.arange(beam_rows)
        suffix_max_q_len = 2
    elif routing == "prefix_count":
        beam_rows = width - 1
        prefix_indices = torch.arange(beam_rows)
    elif routing == "total_count":
        actual_rows = width + 1
    elif routing == "std_only":
        std_indices = torch.arange(width)
        prefix_indices = None
        beam_rows = 0
    else:
        assert routing == "pure"

    query = (
        torch.arange((actual_rows + 2) * heads * dim).reshape(actual_rows + 2, heads, dim).float()
    )
    output = torch.full_like(query, -10000)
    key_cache = torch.zeros(1, 16, kv_heads, dim)
    value_cache = torch.ones_like(key_cache)

    suffix_offsets = torch.arange(width + 1) * suffix_max_q_len
    if routing == "prefix_count":
        suffix_offsets[-1] = beam_rows

    metadata = BeamAttentionMetadata(
        num_actual_tokens=actual_rows,
        max_query_len=actual_rows,
        query_start_loc=torch.tensor([0, actual_rows]),
        max_seq_len=16,
        seq_lens=torch.tensor([16]),
        block_table=torch.zeros(1, 1),
        slot_mapping=torch.arange(actual_rows),
        num_heads=heads,
        num_kv_heads=kv_heads,
        std_indices=std_indices,
        num_std_indices=0 if std_indices is None else len(std_indices),
        std_cu_seqlens_q_device=torch.tensor([0, 2]),
        std_seqlens_kv_device=torch.tensor([16]),
        std_max_q_len=2,
        std_max_seq_len=16,
        std_block_table=torch.zeros(1, 1),
        prefix_indices=prefix_indices,
        num_prefix_indices=beam_rows,
        prefix_cu_seqlens_q_device=torch.tensor([0, beam_rows]),
        prefix_seqlens_kv_device=torch.tensor([16]),
        prefix_max_q_len=beam_rows,
        prefix_max_seq_len=16,
        prefix_block_table=torch.zeros(1, 1),
        suffix_cu_seqlens_q_device=suffix_offsets,
        suffix_cu_seqlens_k_device=torch.arange(width + 1) * (step + 1),
        suffix_max_q_len=suffix_max_q_len,
        suffix_max_seq_len=step + 1,
        is_beam_decode=True,
        beam_session=session,
        beam_active_mask=torch.ones(width, dtype=torch.bool),
    )

    return SimpleNamespace(
        metadata=metadata,
        query=query,
        output=output,
        key_cache=key_cache,
        value_cache=value_cache,
        session=session,
        pool=pool,
        beam_rows=beam_rows,
        width=width,
        step=step,
    )


def _run_cascade(monkeypatch: pytest.MonkeyPatch, case: SimpleNamespace) -> list[str]:
    from vllm_gr.v1.attention.backends import beam_attn_gpu, beam_short_suffix

    calls: list[str] = []

    def flash_attention(**kwargs: Any) -> Any:
        query = kwargs["q"]
        if not kwargs.get("return_softmax_lse", False):
            calls.append("std")
            return query + 1000

        prefix = kwargs["k"] is case.key_cache
        calls.append("prefix" if prefix else "suffix")
        if not prefix:
            total = case.width * (case.step + 1)
            for buffer, source in (
                (kwargs["k"], case.pool.unshared_key),
                (kwargs["v"], case.pool.unshared_value),
            ):
                expected = source[case.session.slot, : case.width, :, : case.step + 1, :]
                expected = expected.permute(0, 2, 1, 3).reshape(total, 1, 64)
                torch.testing.assert_close(buffer[:total], expected)

        return query + (10 if prefix else 20), torch.zeros(query.shape[1], query.shape[0])

    def merge(
        output: torch.Tensor,
        prefix: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix: torch.Tensor,
        suffix_lse: torch.Tensor,
    ) -> None:
        calls.append("merge")
        assert prefix_lse.shape == suffix_lse.shape
        output.copy_((prefix + suffix) / 2)

    def fused(
        output: torch.Tensor,
        query: torch.Tensor,
        prefix: torch.Tensor,
        prefix_lse: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        slot: int,
        step: torch.Tensor,
        scale: float,
        active: torch.Tensor | None,
    ) -> None:
        calls.append("fused")
        assert output.data_ptr() == case.output.data_ptr()
        assert output.shape[0] == query.shape[0] == case.width
        assert keys is case.pool.unshared_key and values is case.pool.unshared_value
        assert slot == case.session.slot and step is case.session.context.decode_step_buf
        assert active is case.metadata.beam_active_mask
        assert scale == 64**-0.5
        assert prefix_lse.shape == (query.shape[1], query.shape[0])

        output.copy_(prefix + 50)

    # Routing is tested on CPU; numerical/device-length repair is exercised
    # with actual CUDA backends in test_beam_short_suffix.py.
    def clear_empty(state: Any, cu_q: torch.Tensor, lengths: torch.Tensor) -> Any:
        assert torch.all(lengths > 0)
        return state

    monkeypatch.setattr(beam_attn_gpu, "clear_empty_prefix_states", clear_empty)
    monkeypatch.setattr(beam_attn_gpu, "flash_attn_varlen_func", flash_attention)
    monkeypatch.setattr(beam_attn_gpu, "merge_attn_states", merge)
    monkeypatch.setattr(beam_short_suffix, "merge_short_suffix_attention", fused)

    beam_attn_gpu.cascade_attention(
        case.output,
        case.query,
        case.key_cache,
        case.value_cache,
        case.metadata,
        None,
        64**-0.5,
        0.0,
        2,
        beam_pool=case.pool,
    )

    return calls


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("enabled", [False, True])
def test_cascade_routes_pure_request_major_beams(
    monkeypatch: pytest.MonkeyPatch, batch: int, enabled: bool
) -> None:
    case = _cascade_case(batch, enabled, "pure")
    calls = _run_cascade(monkeypatch, case)

    assert calls == (["prefix", "fused"] if enabled else ["prefix", "suffix", "merge"])
    torch.testing.assert_close(
        case.output[: case.width], case.query[: case.width] + (60 if enabled else 15)
    )
    assert torch.all(case.output[case.width :] == -10000)

    if enabled:
        assert torch.isnan(case.session.suffix_k_buf).all()
        assert torch.isnan(case.session.suffix_v_buf).all()


@pytest.mark.parametrize("enabled", [False, True])
def test_cascade_mixed_batch_preserves_standard_and_scattered_beam_outputs(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    case = _cascade_case(1, enabled, "mixed")
    calls = _run_cascade(monkeypatch, case)

    assert calls == ["std", "prefix", "suffix", "merge"]

    expected = torch.full_like(case.output, -10000)
    expected[case.metadata.std_indices] = case.query[case.metadata.std_indices] + 1000
    expected[case.metadata.prefix_indices] = case.query[case.metadata.prefix_indices] + 15

    torch.testing.assert_close(case.output, expected)


@pytest.mark.parametrize("routing", ["prefix_count", "total_count", "std_only"])
@pytest.mark.parametrize("enabled", [False, True])
def test_cascade_nonpure_batches_use_generic_attention_without_overwriting_other_rows(
    monkeypatch: pytest.MonkeyPatch, routing: str, enabled: bool
) -> None:
    case = _cascade_case(1, enabled, routing)
    calls = _run_cascade(monkeypatch, case)

    expected = torch.full_like(case.output, -10000)
    if routing == "std_only":
        assert calls == ["std"]
        expected[: case.width] = case.query[: case.width] + 1000
    else:
        assert calls == ["prefix", "suffix", "merge"]
        expected[: case.beam_rows] = case.query[: case.beam_rows] + 15

    torch.testing.assert_close(case.output, expected)


@pytest.mark.l1
@pytest.mark.parametrize("enabled", [False, True])
def test_cascade_rejects_multi_query_beams_without_writing_outputs(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    case = _cascade_case(1, enabled, "many_queries")
    with pytest.raises(ValueError, match="exactly one query token per beam"):
        _run_cascade(monkeypatch, case)
    assert torch.all(case.output == -10000)
    assert torch.isnan(case.session.suffix_k_buf).all()
    assert torch.isnan(case.session.suffix_v_buf).all()


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("width", [None, 128])
@pytest.mark.parametrize("enabled", [None, False, True])
def test_layer_constructor_validates_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool | None,
    width: int | None,
    cuda: bool,
    flash_backend: ModuleType,
) -> None:
    import vllm.config

    from vllm_gr.v1.attention.backends import beam_attn

    additional_config: dict[str, Any] = {"vllm_gr_beam_max_width": width}
    if enabled is not None:
        additional_config["vllm_gr_enable_short_suffix_attention"] = enabled
    config = SimpleNamespace(additional_config=additional_config)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    bind_attention(
        monkeypatch,
        beam_attn,
        cuda=cuda,
        execution=lambda: SimpleNamespace(get_flash_attn_version=lambda **kwargs: 2),
    )
    kwargs: dict[str, Any] = {
        "num_heads": 16,
        "head_size": 128,
        "scale": 128**-0.5,
        "num_kv_heads": 8,
        "alibi_slopes": None,
        "sliding_window": None,
        "kv_cache_dtype": "auto",
        "logits_soft_cap": 10.0,
    }
    active = cuda and width is not None and enabled is not False
    if not cuda:
        with pytest.raises(NotImplementedError, match="logits_soft_cap"):
            beam_attn.BeamAttentionImpl(**kwargs)
    elif active:
        with pytest.raises(ValueError, match="softcap"):
            beam_attn.BeamAttentionImpl(**kwargs)
    else:
        beam_attn.BeamAttentionImpl(**kwargs)

    kwargs["logits_soft_cap"] = 0.0
    impl = beam_attn.BeamAttentionImpl(**kwargs)
    assert impl.supports_quant_query_input is (not active)


def _routing_tensor(shape: tuple[int, ...]) -> Any:
    """Metadata-only stand-in: routing reads shape/dtype/stride, never data."""

    class _Fake:
        def __init__(self) -> None:
            self.shape = shape
            self.dtype = torch.bfloat16
            self.is_cuda = True

        def stride(self) -> tuple[int, ...]:
            return (1,) * len(shape)

    return _Fake()


@pytest.mark.parametrize("fa_version", [2, 3, 4])
def test_fused_suffix_uses_the_fa2_emulating_kernel_for_every_prefix_backend(
    monkeypatch: pytest.MonkeyPatch, fa_version: int
) -> None:
    """The fused suffix kernel is chosen by shape/dtype, never by FA version.

    ``validate_short_suffix_attention`` accepts FA2/FA3/FA4 prefix states, so the
    suffix arithmetic must stay on the FA2-emulating kernel for all of them: the
    routing function has no version knob, which this test locks down because the
    GPU parity tests skip FA3/FA4 fused combinations on hosts without CuTe.
    """
    from vllm_gr.v1.attention.backends import beam_short_suffix as suffix

    assert "fa_version" not in inspect.signature(suffix.merge_short_suffix_attention).parameters
    validate_short_suffix_attention(
        num_heads=8,
        num_kv_heads=2,
        head_size=128,
        fa_version=fa_version,
        sliding_window=None,
        softcap=0.0,
        alibi_slopes=None,
        sinks=None,
    )
    # The Gluon-based fast kernel does not import on NPU Triton builds, so stub
    # the module instead of importing it; the routing decision is what matters.
    fast = ModuleType("vllm_gr.v1.attention.backends.beam_short_suffix_fa2")
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        fast, "merge_short_suffix_fa2", lambda *args: calls.append(args), raising=False
    )
    monkeypatch.setitem(sys.modules, fast.__name__, fast)
    suffix.merge_short_suffix_attention(
        output=_routing_tensor((4, 8, 128)),
        query=_routing_tensor((4, 8, 128)),
        prefix_output=_routing_tensor((4, 8, 128)),
        prefix_lse=_routing_tensor((8, 4)),
        pool_key=_routing_tensor((1, 1, 1, 3, 128)),
        pool_value=_routing_tensor((1, 1, 1, 3, 128)),
        slot=0,
        decode_step=_routing_tensor((4,)),
        scale=1.0,
    )
    assert len(calls) == 1


pytestmark = pytest.mark.cpu_test
