# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch

from vllm_gr.platform.npu.attention_backend import (
    GR_ATTENTION_BACKEND_KEY,
    GR_CUSTOM_ATTENTION_BACKEND,
)
from vllm_gr.platform.npu.capabilities import NPUAttentionCapabilities


def _config(*, backend: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={},
        attention_config=SimpleNamespace(backend=backend),
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
        ),
        model_config=SimpleNamespace(is_hybrid=False),
    )


def _affected_npu_capabilities() -> NPUAttentionCapabilities:
    return NPUAttentionCapabilities(
        is_npu=True,
        native_custom_backend=False,
        selector_api_compatible=True,
    )


class TestCustomAttentionBackendMarker:
    @mock.patch("vllm_gr.engine.arg_utils_gr.origin_create_engine_config")
    def test_npu_cli_custom_persists_marker(self, origin_create: mock.Mock) -> None:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_gr.engine.arg_utils_gr import create_engine_config

        origin_create.return_value = _config(backend=None)
        engine_args = SimpleNamespace(
            attention_backend="CUSTOM",
            attention_config=None,
            catalog_path=None,
        )
        with (
            mock.patch(
                "vllm_gr.platform.npu.attention_backend.is_npu_platform",
                return_value=True,
            ),
            mock.patch(
                "vllm_gr.platform.npu.attention_backend.detect_npu_attention_capabilities",
                return_value=_affected_npu_capabilities(),
            ),
        ):
            result = create_engine_config(engine_args)

        assert result.additional_config[GR_ATTENTION_BACKEND_KEY] == "CUSTOM"
        assert result.attention_config.backend == AttentionBackendEnum.CUSTOM

    @mock.patch("vllm_gr.engine.arg_utils_gr.origin_create_engine_config")
    def test_npu_attention_config_custom_persists_marker(self, origin_create: mock.Mock) -> None:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_gr.engine.arg_utils_gr import create_engine_config

        origin_create.return_value = _config(backend=None)
        engine_args = SimpleNamespace(
            attention_backend=None,
            attention_config=SimpleNamespace(backend=AttentionBackendEnum.CUSTOM),
            catalog_path=None,
        )
        with (
            mock.patch(
                "vllm_gr.platform.npu.attention_backend.is_npu_platform",
                return_value=True,
            ),
            mock.patch(
                "vllm_gr.platform.npu.attention_backend.detect_npu_attention_capabilities",
                return_value=_affected_npu_capabilities(),
            ),
        ):
            result = create_engine_config(engine_args)

        assert result.additional_config[GR_ATTENTION_BACKEND_KEY] == "CUSTOM"
        assert result.attention_config.backend == AttentionBackendEnum.CUSTOM

    @mock.patch("vllm_gr.engine.arg_utils_gr.origin_create_engine_config")
    def test_gpu_custom_does_not_write_npu_marker(self, origin_create: mock.Mock) -> None:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_gr.engine.arg_utils_gr import create_engine_config

        origin_create.return_value = _config(backend=AttentionBackendEnum.CUSTOM)
        engine_args = SimpleNamespace(
            attention_backend="CUSTOM",
            attention_config=None,
            catalog_path=None,
        )
        with mock.patch(
            "vllm_gr.platform.npu.attention_backend.is_npu_platform",
            return_value=False,
        ):
            result = create_engine_config(engine_args)

        assert GR_ATTENTION_BACKEND_KEY not in result.additional_config
        assert result.attention_config.backend == AttentionBackendEnum.CUSTOM

    @mock.patch("vllm_gr.engine.arg_utils_gr.origin_create_engine_config")
    def test_catalog_path_remains_platform_independent(self, origin_create: mock.Mock) -> None:
        from vllm_gr.engine.arg_utils_gr import create_engine_config

        origin_create.return_value = _config()
        engine_args = SimpleNamespace(
            attention_backend=None,
            attention_config=None,
            catalog_path="/tmp/catalog.json",
        )
        result = create_engine_config(engine_args)

        assert result.additional_config["vllm_gr_catalog_path"] == "/tmp/catalog.json"

    def test_marker_survives_additional_config_serialization(self) -> None:
        config = _config()
        config.additional_config[GR_ATTENTION_BACKEND_KEY] = GR_CUSTOM_ATTENTION_BACKEND

        restored = json.loads(json.dumps(config.additional_config))

        assert restored[GR_ATTENTION_BACKEND_KEY] == GR_CUSTOM_ATTENTION_BACKEND


class TestNpuDefaultBlockSize:
    def test_npu_uses_128_when_block_size_was_not_explicit(self) -> None:
        from vllm_gr.engine.arg_utils_gr import _apply_npu_default_block_size

        config = _config()
        with mock.patch(
            "vllm.platforms.current_platform",
            SimpleNamespace(device_type="npu"),
        ):
            _apply_npu_default_block_size(config)
        assert config.cache_config.block_size == 128

    def test_npu_preserves_explicit_block_size(self) -> None:
        from vllm_gr.engine.arg_utils_gr import _apply_npu_default_block_size

        config = _config()
        config.cache_config.user_specified_block_size = True
        with mock.patch(
            "vllm.platforms.current_platform",
            SimpleNamespace(device_type="npu"),
        ):
            _apply_npu_default_block_size(config)
        assert config.cache_config.block_size == 16

    def test_npu_preserves_hybrid_model_block_size(self) -> None:
        from vllm_gr.engine.arg_utils_gr import _apply_npu_default_block_size

        config = _config()
        config.model_config.is_hybrid = True
        with mock.patch(
            "vllm.platforms.current_platform",
            SimpleNamespace(device_type="npu"),
        ):
            _apply_npu_default_block_size(config)
        assert config.cache_config.block_size == 16

    def test_gpu_keeps_vllm_default(self) -> None:
        from vllm_gr.engine.arg_utils_gr import _apply_npu_default_block_size

        config = _config()
        with mock.patch(
            "vllm.platforms.current_platform",
            SimpleNamespace(device_type="cuda"),
        ):
            _apply_npu_default_block_size(config)
        assert config.cache_config.block_size == 16


class TestBeamAttentionVllmApiContract:
    def test_impl_accepts_vllm_sinks_keyword(self) -> None:
        import inspect

        from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionImpl

        parameters = inspect.signature(BeamAttentionImpl.__init__).parameters
        assert "sinks" in parameters
        assert "running_sinks" not in parameters

    def test_fa3_scheduler_uses_num_heads_kv_keyword(self) -> None:
        from vllm_gr.v1.attention.backends import beam_attn

        builder = object.__new__(beam_attn.BeamAttentionMetadataBuilder)
        builder.aot_schedule = True
        builder.cache_config = SimpleNamespace(cache_dtype="auto")
        builder.kv_cache_dtype = torch.bfloat16
        builder.num_heads_q = 16
        builder.num_kv_heads = 4
        builder.headdim = 128
        builder.block_size = 16
        builder.aot_sliding_window = (-1, -1)

        with mock.patch.object(beam_attn, "get_scheduler_metadata", return_value={}) as scheduler:
            builder._get_schedule(
                batch_size=2,
                cu_query_lens=torch.tensor([0, 1, 2], dtype=torch.int32),
                max_query_len=1,
                seqlens=torch.tensor([8, 8], dtype=torch.int32),
                max_seq_len=8,
                causal=True,
                max_num_splits=0,
            )

        assert scheduler.call_args.kwargs["num_heads_kv"] == 4
        assert "num_kv_heads" not in scheduler.call_args.kwargs


def test_non_npu_runtime_does_not_import_npu_worker() -> None:
    from vllm_gr.platform.npu.runtime import install_npu_runtime_compat

    before = set(sys.modules)
    with mock.patch("vllm_gr.platform.npu.runtime.is_npu_platform", return_value=False):
        assert install_npu_runtime_compat(force=True) == 0
    assert "vllm_ascend.worker.worker" not in set(sys.modules) - before
