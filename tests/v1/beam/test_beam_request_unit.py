# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Retained sampling and independent incremental-KV kernel contracts."""

from typing import Any

import pytest
from vllm.exceptions import VLLMValidationError

from vllm_gr.config import MAX_BEAM_WIDTH
from vllm_gr.sampling_params import BeamSearchParams


def test_beam_search_params_enforces_global_width_limit() -> None:
    assert BeamSearchParams(beam_width=256, max_tokens=1).beam_width == 256
    with pytest.raises(VLLMValidationError, match=rf"\[1, {MAX_BEAM_WIDTH}\]"):
        BeamSearchParams(beam_width=MAX_BEAM_WIDTH + 1, max_tokens=1)


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("strategy", ["fill", "null"])  # type: ignore[misc]
def test_beam_search_params_accepts_catalog_strategies(strategy: str) -> None:
    params = BeamSearchParams(beam_width=2, max_tokens=1, catalog_strategy=strategy)
    assert params.catalog_strategy == strategy


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("strategy", ["", "sparse", "FILL"])  # type: ignore[misc]
def test_beam_search_params_rejects_invalid_catalog_strategy(strategy: str) -> None:
    with pytest.raises(VLLMValidationError, match="catalog_strategy must be"):
        BeamSearchParams(beam_width=2, max_tokens=1, catalog_strategy=strategy)


def _sub_patch_preserves_greedy_validation() -> None:
    """Installing GR must not change ordinary vLLM sampling validation."""
    from vllm.sampling_params import SamplingParams

    from vllm_gr.patch import install_engine_session_hooks

    install_engine_session_hooks()

    try:
        SamplingParams(temperature=0.0, n=2)
    except VLLMValidationError as exc:
        assert "n must be 1 when using greedy sampling" in str(exc)
    else:
        raise AssertionError("GR disabled vLLM greedy sampling validation")

    assert SamplingParams(temperature=1.0, n=2).n == 2


def _run_subprocess(target: Any) -> None:
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=target)
    p.start()
    try:
        # A cold vLLM import plus patch initialization can approach 30 seconds.
        # This checks validation behavior, not startup performance.
        p.join(timeout=60)
        assert p.exitcode == 0, f"Subprocess {target.__name__} exited with code {p.exitcode}"
    finally:
        if p.is_alive():
            p.terminate()
            p.join()


class TestBeamKernelsDynamicBeamCount:
    """Incremental pool kernels must tolerate a shrinking live beam count.

    Serving beam search with catalog pruning (``catalog_strategy='null'``)
    drops dead-end beams step by step, so the number of beams written /
    reordered at a decode step can be smaller than the pool width fixed at
    session creation.  Both kernels slice to the actual beam count and leave
    the trailing pool slots stale-but-unreferenced.  Regression for the pool
    width mismatch caused by a partially exhausted beam group.
    """

    def test_cache_and_select_shrinking_beam_count(self) -> None:
        import torch

        from vllm_gr.v1.beam.beam_attention_pool import write_eager_step
        from vllm_gr.v1.beam.reorder import select_unshared_kv

        max_batch, W, H, steps, D = 2, 4, 2, 3, 2
        key_cache = torch.zeros(max_batch, W, H, steps, D)
        value_cache = torch.zeros(max_batch, W, H, steps, D)
        slot = 0

        # Step 0: 4 beams write their KV into pool slot 0.
        curr_key0 = torch.arange(W * H * D, dtype=torch.float32).reshape(W, H, D)
        curr_value0 = curr_key0 + 100
        write_eager_step(
            key_cache,
            value_cache,
            curr_key0,
            curr_value0,
            slot=slot,
            decode_step=0,
        )
        assert torch.equal(key_cache[slot, :, :, 0, :], curr_key0)

        # Step 0 decision: fork 4 beams -> 2 survivors (new 0 <- old 1,
        # new 1 <- old 3).  Only the front 2 slots are reordered.
        parent0 = torch.tensor([1, 3], dtype=torch.int32)
        select_unshared_kv(
            parent0,
            slot=slot,
            unshared_key_caches=[key_cache],
            unshared_value_caches=[value_cache],
            decode_step=0,
        )

        assert torch.equal(key_cache[slot, 0, :, 0, :], curr_key0[1])
        assert torch.equal(key_cache[slot, 1, :, 0, :], curr_key0[3])
        # Trailing slots are stale and untouched.
        assert torch.equal(key_cache[slot, 2, :, 0, :], curr_key0[2])
        assert torch.equal(key_cache[slot, 3, :, 0, :], curr_key0[3])

        # Step 1: only the 2 survivors write KV into pool slot 1.
        curr_key1 = curr_key0[[1, 3]] + 10
        curr_value1 = curr_value0[[1, 3]] + 10
        write_eager_step(
            key_cache,
            value_cache,
            curr_key1,
            curr_value1,
            slot=slot,
            decode_step=1,
        )
        assert torch.equal(key_cache[slot, :2, :, 1, :], curr_key1)
        # Trailing slots at step 1 were never written.
        assert torch.equal(key_cache[slot, 2:, :, 1, :], torch.zeros(2, H, D))

        # Step 1 decision: 2 -> 2, reorder both steps of history.
        parent1 = torch.tensor([1, 0], dtype=torch.int32)
        select_unshared_kv(
            parent1,
            slot=slot,
            unshared_key_caches=[key_cache],
            unshared_value_caches=[value_cache],
            decode_step=1,
        )
        assert torch.equal(key_cache[slot, 0, :, 0, :], curr_key0[3])
        assert torch.equal(key_cache[slot, 1, :, 0, :], curr_key0[1])
        assert torch.equal(key_cache[slot, 0, :, 1, :], curr_key1[1])
        assert torch.equal(key_cache[slot, 1, :, 1, :], curr_key1[0])

    def test_fixed_width_full_reorder_still_works(self) -> None:
        """Regression guard: n == W must behave exactly as before the fix."""
        import torch

        from vllm_gr.v1.beam.beam_attention_pool import write_eager_step
        from vllm_gr.v1.beam.reorder import select_unshared_kv

        max_batch, W, H, steps, D = 2, 4, 2, 3, 2
        key_cache = torch.zeros(max_batch, W, H, steps, D)
        value_cache = torch.zeros(max_batch, W, H, steps, D)
        slot = 0
        curr_key = torch.arange(W * H * D, dtype=torch.float32).reshape(W, H, D)
        curr_value = curr_key + 50
        write_eager_step(
            key_cache,
            value_cache,
            curr_key,
            curr_value,
            slot=slot,
            decode_step=0,
        )

        # Full-width reorder: new i <- old (i + 1) % W.
        parent = torch.tensor([1, 2, 3, 0], dtype=torch.int32)
        select_unshared_kv(
            parent,
            slot=slot,
            unshared_key_caches=[key_cache],
            unshared_value_caches=[value_cache],
            decode_step=0,
        )
        for i in range(W):
            assert torch.equal(key_cache[slot, i, :, 0, :], curr_key[(i + 1) % W])


def test_plugin_preserves_native_greedy_validation() -> None:
    _run_subprocess(_sub_patch_preserves_greedy_validation)


pytestmark = pytest.mark.cpu_test
