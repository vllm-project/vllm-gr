# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NPU full-vocabulary candidate/state/KV parity without catalog operators."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.v1.worker.full_vocab import full_vocab_topk
from vllm_gr.v1.worker.torch_beam_state import TorchBeamState, reorder_beam_kv

pytestmark = pytest.mark.npu_integration

try:
    import torch_npu  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"torch-npu is not installed: {exc}", allow_module_level=True)

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Ascend NPU is unavailable", allow_module_level=True)


@pytest.mark.parametrize(
    "batch,width", [(1, 1), (1, 128), (1, 256), (2, 8), (2, 128), (4, 8), (4, 128)]
)
def test_npu_candidates_and_persistent_eos_match_cpu(batch: int, width: int) -> None:
    def state(device: str) -> TorchBeamState:
        return TorchBeamState(
            beam_width=width,
            max_steps=3,
            prompt_lengths=tuple(127 + row for row in range(batch)),
            stop_token_ids=(0,),
            eos_token_id=0,
            length_penalty=1.0,
            device=device,
        )

    cpu, npu = state("cpu"), state("npu")
    for stage in range(3):
        rows = batch if stage == 0 else batch * width
        # Unique, well-separated scores keep this an exact candidate-ID test.
        # EOS leads every parent; ordinary tokens outside any SID vocabulary
        # still occupy the complete active frontier.
        logits = -torch.arange(1024, dtype=torch.float32)[None].expand(rows, -1) / 8
        logits = logits - torch.arange(rows, dtype=torch.float32)[:, None] / 16
        expected = full_vocab_topk(logits, 2 * width)
        actual = full_vocab_topk(logits.to("npu"), 2 * width)
        torch.testing.assert_close(actual.token_ids.cpu(), expected.token_ids, rtol=0, atol=0)
        torch.testing.assert_close(actual.logprobs.cpu(), expected.logprobs, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual.valid_mask.cpu(), expected.valid_mask)
        torch.testing.assert_close(actual.valid_counts.cpu(), expected.valid_counts)
        cpu.advance(expected.token_ids, expected.logprobs, expected.valid_mask, stage_index=stage)
        # Reuse exactly identical scores to isolate state selection from
        # device-specific log-softmax roundoff in this second comparison.
        npu.advance(
            expected.token_ids.to("npu"),
            expected.logprobs.to("npu"),
            expected.valid_mask.to("npu"),
            stage_index=stage,
        )
        torch.testing.assert_close(npu.token_ids.cpu(), cpu.token_ids, rtol=0, atol=0)
        torch.testing.assert_close(npu.local_parents.cpu(), cpu.local_parents, rtol=0, atol=0)
        torch.testing.assert_close(npu.scores.cpu(), cpu.scores)
    expected_output, actual_output = cpu.finalize(), npu.finalize()
    for name in ("sequences", "parents", "lengths", "stopped", "stop_token_ids"):
        torch.testing.assert_close(
            getattr(actual_output, name).cpu(), getattr(expected_output, name)
        )
    torch.testing.assert_close(actual_output.scores.cpu(), expected_output.scores)


@pytest.mark.parametrize("batch,width", [(1, 1), (1, 256), (2, 8), (4, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_kv_gather_preserves_cache_addresses_and_request_isolation(
    batch: int, width: int, dtype: torch.dtype
) -> None:
    source = torch.arange(batch * width * 2 * 3 * 2, dtype=torch.float32).reshape(
        batch, width, 2, 3, 2
    )
    source = source.to(dtype)
    # Includes unsorted duplicate parents, unlike a permutation-only gather.
    parents = ((torch.arange(width, dtype=torch.int64) // 2) * 3 % width).expand(batch, -1)
    pool = SimpleNamespace(unshared_key=source.to("npu"), unshared_value=(source + 100).to("npu"))
    addresses = (pool.unshared_key.data_ptr(), pool.unshared_value.data_ptr())
    reorder_beam_kv([pool], parents.to("npu"), decode_step=1)
    torch.npu.synchronize()
    expected = source.clone()
    indices = parents[:, :, None, None, None].expand_as(source[:, :, :, :2])
    expected[:, :, :, :2].copy_(source[:, :, :, :2].gather(1, indices))
    torch.testing.assert_close(pool.unshared_key.cpu(), expected, rtol=0, atol=0)
    expected_value = (source + 100).clone()
    expected_value[:, :, :, :2].copy_((source + 100)[:, :, :, :2].gather(1, indices))
    torch.testing.assert_close(pool.unshared_value.cpu(), expected_value, rtol=0, atol=0)
    assert addresses == (pool.unshared_key.data_ptr(), pool.unshared_value.data_ptr())
