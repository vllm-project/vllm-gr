# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Coverage for native Hybrid Beam admission and cache reparenting."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_gr.config.execution import validate_gr_execution_runtime
from vllm_gr.v1.beam.hybrid_cache import hybrid_block_budget
from vllm_gr.v1.worker.cuda.hybrid_beam_cache import (
    _attention_decode_blocks,
    _copy_blocks,
    _recurrent_decode_blocks,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_hybrid_runtime_requires_native_cuda_cache_modes() -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_hybrid=True,
            architectures=["UnlistedHybridForCausalLM"],
            enforce_eager=True,
            dtype=torch.bfloat16,
        ),
        device_config=SimpleNamespace(device="cuda"),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False, mamba_cache_mode="none", cache_dtype="auto"
        ),
        use_v2_model_runner=False,
    )
    validate_gr_execution_runtime(config)
    config.use_v2_model_runner = True
    validate_gr_execution_runtime(config)
    config.device_config.device = "npu"
    with pytest.raises(ValueError, match="Hybrid Beam requires CUDA"):
        validate_gr_execution_runtime(config)
    config.device_config.device = "cuda"
    config.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="mamba_cache_mode=align"):
        validate_gr_execution_runtime(config)
    config.cache_config.mamba_cache_mode = "align"
    config.scheduler_config.enable_chunked_prefill = True
    validate_gr_execution_runtime(config)


def test_hybrid_budget_includes_the_writable_tail_of_each_beam() -> None:
    groups = [
        KVCacheGroupSpec(
            ["mla"],
            MLAAttentionSpec(block_size=4, num_kv_heads=1, head_size=8, dtype=torch.bfloat16),
        ),
        KVCacheGroupSpec(
            ["kda"],
            MambaSpec(
                block_size=32,
                shapes=((3, 4), (1, 2, 2)),
                dtypes=(torch.bfloat16, torch.float32),
                mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            ),
        ),
    ]
    assert hybrid_block_budget(groups, prompt_tokens=7, decode_steps=2, beam_width=2) == (4, (4, 2))
    assert hybrid_block_budget(groups, prompt_tokens=8, decode_steps=2, beam_width=2) == (4, (2, 2))
    assert hybrid_block_budget(groups, prompt_tokens=7, decode_steps=0, beam_width=2) == (3, (0, 0))


@pytest.mark.parametrize(  # type: ignore[misc]
    "mamba_type",
    [
        MambaAttentionBackendEnum.MAMBA1,
        MambaAttentionBackendEnum.MAMBA2,
        MambaAttentionBackendEnum.SHORT_CONV,
        MambaAttentionBackendEnum.LINEAR,
        MambaAttentionBackendEnum.GDN_ATTN,
    ],
)
def test_hybrid_budget_accepts_native_recurrent_groups_with_paged_attention(
    mamba_type: MambaAttentionBackendEnum,
) -> None:
    spec = MambaSpec(
        block_size=4,
        shapes=((3, 4),),
        dtypes=(torch.bfloat16,),
        mamba_type=mamba_type,
        mamba_cache_mode="align",
    )
    groups = [
        KVCacheGroupSpec(
            ["attention"],
            FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=8, dtype=torch.bfloat16),
        ),
        KVCacheGroupSpec(
            ["recurrent"],
            UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs={"recurrent": spec}),
        ),
    ]
    assert hybrid_block_budget(groups, prompt_tokens=7, decode_steps=2, beam_width=2) == (6, (4, 2))
    cache = torch.arange(40).reshape(10, 4)
    private = torch.tensor([[6], [7]], dtype=torch.int32)
    table = _recurrent_decode_blocks(
        ((cache,),),
        [0, 3],
        private,
        torch.zeros(2, dtype=torch.long),
        prompt_length=7,
        step=0,
        spec=spec,
    )
    assert table.tolist() == [[6, 6], [7, 7]]
    torch.testing.assert_close(cache[6:8], cache[3].expand(2, -1))


def test_copy_preserves_recurrent_states_when_parents_repeat_or_swap() -> None:
    conv = torch.arange(24, dtype=torch.bfloat16).reshape(4, 3, 2)
    recurrent = torch.arange(32, dtype=torch.float32).reshape(4, 2, 2, 2)
    parents = torch.tensor([3, 1, 1])
    targets = torch.tensor([1, 2, 3])
    expected = [state[parents] for state in (conv, recurrent)]
    _copy_blocks((conv, recurrent), parents, targets)
    for state, copied in zip((conv, recurrent), expected, strict=True):
        torch.testing.assert_close(state[targets], copied)


def test_attention_decode_keeps_the_shared_page_and_forks_the_tail_across_boundary() -> None:
    cache = torch.arange(40).view(10, 4, 1)
    private = torch.tensor([[3, 4], [5, 6]], dtype=torch.int32)
    prompt_cache = cache[1:3].clone()
    parents = torch.tensor([0, 0])
    table, slots = _attention_decode_blocks(
        (cache,), [1, 2], private, parents, block_size=4, prompt_length=7, step=0
    )
    assert table.tolist() == [[1, 3, 4], [1, 5, 6]]
    assert slots.tolist() == [15, 23]
    torch.testing.assert_close(cache[3], prompt_cache[1])
    torch.testing.assert_close(cache[5], prompt_cache[1])
    cache[3, 3] = 80
    cache[5, 3] = 90

    _, slots = _attention_decode_blocks(
        (cache,), [1, 2], private, torch.tensor([1, 1]), block_size=4, prompt_length=7, step=1
    )
    assert slots.tolist() == [16, 24]
    assert cache[3, 3].item() == cache[5, 3].item() == 90
    assert not cache[[4, 6]].any()
    torch.testing.assert_close(cache[1:3], prompt_cache)
