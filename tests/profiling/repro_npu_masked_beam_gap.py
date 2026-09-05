# SPDX-License-Identifier: Apache-2.0
"""CPU reproduction of the unresolved persistent NPU candidate-mask ABI gap.

Run from the repository root:
    python tests/profiling/repro_npu_masked_beam_gap.py

Exit 1 means the repository's xllm-signature reference differs from the
mask-aware contract. This is a diagnostic, not a passing parity test or a
measurement of actual NPU kernel behavior. Remove/replace it with parity
coverage once the production decision boundary implements that contract.
"""

from __future__ import annotations

import argparse
import json

import torch

from vllm_gr.ops.onerec_decode import (
    _beam_search_group_torch,
    _beam_search_rec_final_select_torch,
)


def _cuda_group(cumulative, tokens, logprobs, sequence, mask) -> dict:
    """Optional actual GPU oracle; only this diagnostic materializes results."""
    from vllm_gr.ops.cuda.beam_selection import beam_search_group_cuda_worker_out

    width = tokens.shape[0]
    device = torch.device("cuda:0")
    cumulative, tokens, logprobs, sequence, mask = (
        tensor.to(device) for tensor in (cumulative, tokens, logprobs, sequence, mask)
    )
    out_tokens = torch.empty((1, width), dtype=torch.int32, device=device)
    out_parents = torch.empty_like(out_tokens)
    out_scores = torch.empty((1, width), dtype=torch.float32, device=device)
    status = torch.empty(1, dtype=torch.int32, device=device)
    beam_search_group_cuda_worker_out(
        cumulative,
        tokens,
        logprobs,
        sequence,
        1,
        out_tokens,
        out_parents,
        out_scores,
        torch.empty_like(sequence),
        torch.empty_like(out_tokens),
        torch.empty_like(out_scores),
        torch.empty_like(out_tokens),
        mask,
        mask.sum(dim=1, dtype=torch.int32),
        status,
        torch.empty(3 * width, dtype=torch.int32, device=device),
    )
    return {
        "device": torch.cuda.get_device_name(device),
        "tokens": out_tokens.cpu().flatten().tolist(),
        "parents": out_parents.cpu().flatten().tolist(),
        "status": status.cpu().tolist(),
        "all_scores_negative_infinity": bool(torch.isneginf(out_scores).all().cpu()),
    }


def reproduce(*, cuda: bool = False) -> dict:
    # W=32 satisfies the real final-select width and candidate-width gates.
    # Every row has one structurally legal candidate, so this is neither an
    # empty row nor global candidate exhaustion. Activity is not validity.
    width = 64 if cuda else 32
    cumulative = torch.zeros((width, 1), dtype=torch.float32)
    tokens = torch.zeros((width, width), dtype=torch.int32)
    tokens[:, 0] = torch.arange(10, 10 + width, dtype=torch.int32)
    logprobs = torch.full((width, width), -float("inf"), dtype=torch.float32)
    mask = torch.zeros((width, width), dtype=torch.bool)
    mask[:, 0] = True
    sequence = torch.zeros((1, width, 3), dtype=torch.int32)
    sequence[0, :, 0] = torch.arange(100, 100 + width, dtype=torch.int32)

    # All legal scores tie. The mask-aware merge must choose lower *legal*
    # flat indices: [0, W, 2W, ...], not [0, 1, 2, ...]. These winners are
    # already parent-major, so Group and FinalSelect have the same order.
    legal_flat = mask.flatten().nonzero().flatten()
    expected_tokens = tokens.flatten()[legal_flat].tolist()
    expected_parents = (legal_flat // width).tolist()
    expected_prefix = list(range(1, width + 1))
    group = _beam_search_group_torch(cumulative, tokens, logprobs, sequence, 1, width)
    final = _beam_search_rec_final_select_torch(cumulative, tokens, logprobs, sequence, 1, width)
    actual_group_tokens = group[0].flatten().tolist()
    actual_group_parents = group[1].flatten().tolist()
    actual_final_tokens = final[0].flatten().tolist()
    actual_final_parents = final[1].flatten().tolist()
    mismatch = (
        actual_group_tokens != expected_tokens
        or actual_group_parents != expected_parents
        or actual_final_tokens != expected_tokens
        or actual_final_parents != expected_parents
        or group[3].flatten().tolist() != expected_prefix
    )
    report = {
        "scope": "xllm-signature CPU reference; optional CUDA oracle; not NPU hardware",
        "beam_width": width,
        "generation_stage": 2,
        "candidate_counts": mask.sum(dim=1).tolist(),
        "legal_candidates": legal_flat.numel(),
        "mask_aware_expected_status": 0,
        "expected_tokens": expected_tokens,
        "expected_parents": expected_parents,
        "expected_group_prefix": expected_prefix,
        "reference_group_tokens": actual_group_tokens,
        "reference_group_parents": actual_group_parents,
        "reference_group_prefix": group[3].flatten().tolist(),
        "reference_final_tokens": actual_final_tokens,
        "reference_final_parents": actual_final_parents,
        "contract_mismatch": mismatch,
    }
    if cuda:
        actual = _cuda_group(cumulative, tokens, logprobs, sequence, mask)
        report["cuda_group"] = actual
        report["cuda_matches_mask_aware_oracle"] = (
            actual["tokens"] == expected_tokens
            and actual["parents"] == expected_parents
            and actual["status"] == [0]
            and actual["all_scores_negative_infinity"]
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="also run the GPU Worker W=64 kernel")
    report = reproduce(cuda=parser.parse_args().cuda)
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report["contract_mismatch"] else 0)
