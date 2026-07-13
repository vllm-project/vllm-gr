# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone latency benchmark for the cascade beam-attention GPU backend.

This is a benchmark, not a correctness test -- it has no pass/fail assertion,
just timing output -- so it lives outside ``test_*.py`` and is never collected
by pytest (the project's pre-commit hook runs `pytest --run-slow tests`,
which would otherwise execute it as an ordinary test on every commit).

Run directly, e.g.:
    python benchmark_cascade_beam_attn_gpu.py
    python benchmark_cascade_beam_attn_gpu.py --iter-num 200 --beam-widths 16 32 128
"""

from __future__ import annotations

import argparse
import itertools

import torch
from cascade_beam_attn_utils import (
    NUM_HEADS,
    prepare_inputs,
    run_cascade_beam_attention,
)
from vllm.utils.torch_utils import set_random_seed

try:
    from vllm.vllm_flash_attn import (
        fa_version_unsupported_reason,
        is_fa_version_supported,
    )
except ImportError as e:
    raise SystemExit(f"vllm_flash_attn is not supported: {e}")


def benchmark_one(
    batch_size: int,
    beam_width: int,
    seq_lens_and_common_prefix: tuple[int, int, int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    fa_version: int,
    iter_num: int,
    warmup_iters: int,
) -> float | None:
    if not is_fa_version_supported(fa_version):
        print(f"Skipping fa_version={fa_version}: {fa_version_unsupported_reason(fa_version)}")
        return None

    device = torch.device("cuda")
    set_random_seed(0)

    num_query_heads, num_kv_heads = num_heads
    q_seq_len, kv_seq_len, suffix_kv_len = seq_lens_and_common_prefix

    inputs = prepare_inputs(
        batch_size=batch_size,
        beam_width=beam_width,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        block_size=block_size,
        dtype=dtype,
        soft_cap=soft_cap,
        fa_version=fa_version,
        suffix_kv_len=suffix_kv_len,
        device=device,
    )

    def run_once() -> torch.Tensor:
        return run_cascade_beam_attention(
            inputs,
            device=device,
            vllm_flash_attn_version=inputs.fa_version,
            logits_soft_cap=inputs.soft_cap,
        )

    with torch.inference_mode():
        for _ in range(warmup_iters):
            run_once()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iter_num):
            run_once()
        end_event.record()
        torch.cuda.synchronize()

    elapsed_time_ms = float(start_event.elapsed_time(end_event) / iter_num)
    print(
        f"batch_size={batch_size}, beam_width={beam_width}, "
        f"seq_lens={seq_lens_and_common_prefix}, num_heads={num_heads}, "
        f"head_size={head_size}, dtype={dtype}, block_size={block_size}, "
        f"soft_cap={soft_cap}, fa_version={fa_version}, "
        f"avg_time={elapsed_time_ms:.3f} ms"
    )
    return elapsed_time_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8])
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[16, 32, 128])
    parser.add_argument(
        "--seq-len",
        type=int,
        nargs=3,
        default=[8, 2048, 8],
        metavar=("Q_SEQ_LEN", "KV_SEQ_LEN", "SUFFIX_KV_LEN"),
    )
    parser.add_argument("--head-sizes", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=["float16", "bfloat16"],
        choices=["float16", "bfloat16"],
    )
    parser.add_argument("--soft-caps", type=float, nargs="+", default=[0.0, 50.0])
    parser.add_argument("--fa-versions", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--iter-num", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    args = parse_args()
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtypes = [dtype_map[d] for d in args.dtypes]
    seq_lens_and_common_prefix = tuple(args.seq_len)

    combos = itertools.product(
        args.batch_sizes,
        args.beam_widths,
        NUM_HEADS,
        args.head_sizes,
        dtypes,
        args.block_sizes,
        args.soft_caps,
        args.fa_versions,
    )
    for (
        batch_size,
        beam_width,
        num_heads,
        head_size,
        dtype,
        block_size,
        soft_cap,
        fa_version,
    ) in combos:
        benchmark_one(
            batch_size=batch_size,
            beam_width=beam_width,
            seq_lens_and_common_prefix=seq_lens_and_common_prefix,
            num_heads=num_heads,
            head_size=head_size,
            dtype=dtype,
            block_size=block_size,
            soft_cap=soft_cap or None,
            fa_version=fa_version,
            iter_num=args.iter_num,
            warmup_iters=args.warmup_iters,
        )


if __name__ == "__main__":
    main()
