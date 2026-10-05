# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm_gr.v1.attention.backends import beam_attn, beam_attn_gpu
from vllm_gr.v1.attention.backends.beam_attn_metadata import BeamAttentionMetadata


def test_direct_output_is_captured_at_backend_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = torch.empty(2, 1, 4)
    query = torch.zeros_like(output)
    received_outputs: list[torch.Tensor | None] = []

    def attention(**kwargs: Any) -> torch.Tensor:
        destination = kwargs["out"]
        received_outputs.append(destination)
        if destination is None:
            destination = torch.empty_like(kwargs["q"])
        return destination.fill_(7)

    monkeypatch.setattr(beam_attn_gpu, "flash_attn_varlen_func", attention)
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", True)
    monkeypatch.setattr(
        beam_attn,
        "_get_platform_execution_module",
        lambda: SimpleNamespace(
            get_flash_attn_version=lambda **kwargs: 2,
            cascade_attention=beam_attn_gpu.cascade_attention,
        ),
    )
    metadata = cast(
        BeamAttentionMetadata,
        SimpleNamespace(
            std_indices=torch.arange(2),
            prefix_indices=None,
            num_std_indices=2,
            std_cu_seqlens_q_device=None,
            std_max_q_len=1,
            std_seqlens_kv_device=None,
            std_max_seq_len=1,
            causal=True,
            std_block_table=None,
        ),
    )
    for enabled in (False, True, False):
        monkeypatch.setenv("VLLM_GR_BEAM_PROC_ENABLED", "1" if enabled else "0")
        impl = beam_attn.BeamAttentionImpl(
            num_heads=1,
            head_size=4,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            kv_sharing_target_layer_name="shared",
        )
        # Changing the environment after initialization must not change this engine.
        monkeypatch.setenv("VLLM_GR_BEAM_PROC_ENABLED", "0" if enabled else "1")
        output.fill_(float("nan"))
        impl.forward(
            torch.nn.Module(), query, query, query, torch.stack((query, query)), metadata, output
        )
        destination = received_outputs[-1]
        if enabled:
            assert destination is not None
            assert destination.data_ptr() == output.data_ptr()
        else:
            assert destination is None
        torch.testing.assert_close(output, torch.full_like(output, 7))
