# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Strict active-KV and XAttention parity for device-length-masked gather.

Companion to check_npu_shared_kv_gather_contract.py; the original unmasked
contract remains required. A single captured graph changes actual per-row
prefix lengths short -> full -> short, with new block tables and KV values.
Unused output blocks are poisoned with NaNs before replay so accidental
downstream tail reads cannot silently pass. No performance claim is made.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def require(value: object, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="One representative interleaved B1 length-mask/graph case",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument(
        "--dtypes", nargs="+", choices=("bfloat16", "float16"), default=["bfloat16"]
    )
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite contract evidence")
    require(
        args.capacity >= 3 and all(width > 0 for width in args.widths), "Invalid capacity/width"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "state": "running",
        "cases": [],
        "attention_enabled": args.attention,
        "mask_default": "inherits_fused_request",
        "scope": "copy contract; no latency measurement",
        "invalid_device_length_policy": "conservative full-capacity copy, without host synchronization",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    code = 0
    try:
        os.environ["VLLM_GR_NPU_FUSED_SHARED_KV_GATHER"] = "1"
        os.environ["VLLM_GR_NPU_SHARED_KV_LENGTH_MASK"] = "0"
        import torch
        import torch_npu

        from vllm_gr.ops.npu import beam_ops
        from vllm_gr.v1.attention.backends.beam_attn_npu_xllm import (
            _gather_shared_blocks,
        )
        from vllm_gr.v1.attention.backends.npu_shared_kv_gather import (
            fused_shared_kv_gather_stats,
            try_fused_shared_kv_gather,
        )

        require(
            torch.npu.is_available() and torch.npu.device_count() == 1, "Allocate exactly one NPU"
        )
        if args.attention:
            require(beam_ops.is_available(), "Native x_attention required")
        torch.manual_seed(20261122)
        block, heads, dim = 128, 8, 128
        cap = args.capacity

        def equal_bits(left: torch.Tensor, right: torch.Tensor, label: str) -> None:
            require(tuple(left.shape) == tuple(right.shape), f"{label}: shape mismatch")
            require(
                torch.equal(
                    left.cpu().contiguous().view(torch.int16),
                    right.cpu().contiguous().view(torch.int16),
                ),
                f"{label}: not bitwise equal",
            )

        def reference(
            key: torch.Tensor, value: torch.Tensor, ids: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            cpu_ids = ids.cpu().long()
            return (
                key.cpu()[cpu_ids].reshape(-1, block, heads, dim).contiguous(),
                value.cpu()[cpu_ids].reshape(-1, block, heads, dim).contiguous(),
                torch.arange(cpu_ids.numel(), dtype=ids.dtype).reshape(cpu_ids.shape),
            )

        def check_copy(
            key: torch.Tensor,
            value: torch.Tensor,
            ids: torch.Tensor,
            lengths: torch.Tensor,
            outputs: Any,
            label: str,
            *,
            poisoned: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            require(outputs is not None, f"{label}: unexpected fallback")
            expected = reference(key, value, ids)
            require(torch.equal(outputs[2].cpu(), expected[2]), f"{label}: remap changed")
            ls = lengths.cpu().tolist()
            for row, length in enumerate(ls):
                active = (length + block - 1) // block if 0 < length <= cap * block else cap
                for name, actual, baseline in zip(
                    ("key", "value"), outputs[:2], expected[:2], strict=True
                ):
                    equal_bits(
                        actual[row * cap : row * cap + active],
                        baseline[row * cap : row * cap + active],
                        f"{label}:{name}:row{row}",
                    )
                    if poisoned and active < cap:
                        require(
                            bool(
                                torch.isnan(
                                    actual[row * cap + active : (row + 1) * cap].cpu()
                                ).all()
                            ),
                            f"{label}: unused {name} blocks were written",
                        )
            return expected

        for dtype_name in args.dtypes:
            dtype = getattr(torch, dtype_name)
            for layout in ("interleaved",) if args.quick else ("contiguous", "interleaved"):
                for batch in (1,) if args.quick else (1, 4):
                    for width in args.widths:
                        item: dict[str, Any] = {
                            "dtype": dtype_name,
                            "layout": layout,
                            "batch": batch,
                            "beam_width": width,
                            "capacity_columns": cap,
                            "state": "running",
                        }
                        report["cases"].append(item)
                        blocks = max(32, batch * cap)
                        if layout == "interleaved":
                            pool = torch.randn(
                                (blocks + 1, 2, block, heads, dim), dtype=dtype, device="npu"
                            )
                            pool = torch_npu.npu_format_cast(pool, 2)
                            require(torch_npu.get_npu_format(pool) == 2, "Fixture pool must be ND")
                            key, value = pool[1:, 0], pool[1:, 1]
                        else:
                            key = torch.randn(
                                (blocks, block, heads, dim), dtype=dtype, device="npu"
                            )
                            value = torch.randn_like(key)
                        ids = torch.randint(
                            0, blocks, (batch, cap), dtype=torch.int32, device="npu"
                        )
                        ids[:, 2] = ids[:, 0]
                        lengths = torch.full((batch,), 2 * block, dtype=torch.int32, device="npu")
                        item["input_metadata"] = {
                            name: {
                                "shape": list(tensor.shape),
                                "stride": list(tensor.stride()),
                                "format": int(torch_npu.get_npu_format(tensor)),
                                "storage_offset": int(tensor.storage_offset()),
                            }
                            for name, tensor in (
                                ("key", key),
                                ("value", value),
                                ("table", ids),
                                ("lengths", lengths),
                            )
                        }
                        # Explicitly retain the original unmasked kernel check.
                        unmasked = try_fused_shared_kv_gather(key, value, ids, 0)
                        baseline = reference(key, value, ids)
                        require(unmasked is not None, "Original unmasked fused path fell back")
                        for name, actual, expected in zip(
                            ("key", "value"), unmasked[:2], baseline[:2], strict=True
                        ):
                            equal_bits(actual, expected, f"unmasked:{name}")
                        os.environ["VLLM_GR_NPU_SHARED_KV_LENGTH_MASK"] = "1"
                        before_masked = fused_shared_kv_gather_stats()["host_launch_counts"].get(
                            "fused_masked", 0
                        )
                        eager = _gather_shared_blocks(key, value, ids, 0, prefix_seqlens=lengths)
                        require(
                            fused_shared_kv_gather_stats()["host_launch_counts"].get(
                                "fused_masked", 0
                            )
                            == before_masked + 1,
                            "Integration wrapper did not use masked fused copy",
                        )
                        check_copy(key, value, ids, lengths, eager, "masked eager")
                        graph = torch.npu.NPUGraph()
                        side = torch.npu.Stream()
                        side.wait_stream(torch.npu.current_stream())
                        with torch.npu.stream(side):
                            with torch.npu.graph(graph):
                                captured = try_fused_shared_kv_gather(key, value, ids, 0, lengths)
                        torch.npu.current_stream().wait_stream(side)
                        torch.npu.synchronize()
                        require(captured is not None, "Masked graph capture fell back")
                        addresses = [tensor.data_ptr() for tensor in captured]
                        q = torch.randn((batch * width, 16, dim), device="npu", dtype=dtype)
                        uk = torch.randn((batch * width, heads, 3, dim), device="npu", dtype=dtype)
                        uv = torch.randn_like(uk)
                        # Includes exact2 -> full8 ->2 and partial-block/mixed-row lengths.
                        sequence = [
                            [2 * block] * batch,
                            [cap * block] * batch,
                            [2 * block] * batch,
                            [129 + 128 * (row % 2) for row in range(batch)],
                            [cap * block - row for row in range(batch)],
                        ]
                        item["replays"] = []
                        for index, current in enumerate(sequence):
                            key.copy_(torch.randn(key.shape, device="npu", dtype=dtype))
                            value.copy_(torch.randn(value.shape, device="npu", dtype=dtype))
                            fresh_ids = torch.randint(
                                0, blocks, ids.shape, device="npu", dtype=ids.dtype
                            )
                            fresh_ids[:, 2] = fresh_ids[:, 0]
                            ids.copy_(fresh_ids)
                            lengths.copy_(torch.tensor(current, dtype=torch.int32))
                            captured[0].fill_(float("nan"))
                            captured[1].fill_(float("nan"))
                            graph.replay()
                            torch.npu.synchronize()
                            require(
                                [tensor.data_ptr() for tensor in captured] == addresses,
                                "Graph output addresses changed",
                            )
                            expected = check_copy(
                                key, value, ids, lengths, captured, f"replay{index}", poisoned=True
                            )
                            if args.attention:
                                full = tuple(tensor.to("npu") for tensor in expected)
                                left = beam_ops.x_attention(
                                    q, full[0], full[1], uk, uv, full[2], lengths, 1, 0.0
                                )
                                right = beam_ops.x_attention(
                                    q,
                                    captured[0],
                                    captured[1],
                                    uk,
                                    uv,
                                    captured[2],
                                    lengths,
                                    1,
                                    0.0,
                                )
                                equal_bits(right, left, f"replay{index}:x_attention")
                            item["replays"].append(
                                {
                                    "lengths": current,
                                    "active_copy_bitwise_equal": True,
                                    "unused_blocks_remain_nan": True,
                                    "attention_bitwise_equal": True if args.attention else None,
                                }
                            )
                        # Value errors retain complete old gather semantics. Do not invoke
                        # XAttention with invalid lengths or claim their model validity.
                        item["invalid_device_values"] = []
                        for invalid in (0, -1, cap * block + 1, 2147483647):
                            lengths.fill_(invalid)
                            graph.replay()
                            torch.npu.synchronize()
                            check_copy(key, value, ids, lengths, captured, f"invalid{invalid}")
                            item["invalid_device_values"].append(
                                {
                                    "value": invalid,
                                    "full_copy_bitwise_equal": True,
                                    "attention_executed": False,
                                }
                            )
                        # Device metadata guards are observable, never silent optimized success.
                        bad = [
                            (None, "prefix_lengths_missing"),
                            (lengths.cpu(), "prefix_lengths_device"),
                            (lengths.to(torch.int64), "prefix_lengths_dtype"),
                            (lengths.reshape(batch, 1), "prefix_lengths_shape"),
                            (
                                torch.empty((batch + 1,), dtype=torch.int32, device="npu"),
                                "prefix_lengths_shape",
                            ),
                            (
                                torch.empty((batch * 2,), dtype=torch.int32, device="npu")[::2],
                                "prefix_lengths_stride",
                            ),
                        ]
                        item["bad_metadata"] = []
                        for invalid, reason in bad:
                            before = fused_shared_kv_gather_stats()["host_launch_counts"].get(
                                f"fallback:{reason}", 0
                            )
                            require(
                                try_fused_shared_kv_gather(key, value, ids, 0, invalid) is None,
                                f"Invalid length metadata accepted: {reason}",
                            )
                            after = fused_shared_kv_gather_stats()["host_launch_counts"].get(
                                f"fallback:{reason}", 0
                            )
                            require(after == before + 1, f"Wrong/unobservable guard: {reason}")
                            item["bad_metadata"].append({"reason": reason, "rejected": True})
                        item.update(
                            state="passed",
                            output_addresses_stable=True,
                            fresh_kv_and_tables_each_replay=True,
                        )
                        del graph, captured, eager, unmasked
                        os.environ["VLLM_GR_NPU_SHARED_KV_LENGTH_MASK"] = "0"
        report["diagnostics"] = fused_shared_kv_gather_stats()
        report["state"] = "passed"
    except BaseException as error:
        code = 1
        report.update(
            state="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        if report["cases"] and report["cases"][-1]["state"] == "running":
            report["cases"][-1]["state"] = "failed"
        traceback.print_exc()
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "state": report["state"],
                "cases": len(report["cases"]),
                "output": str(args.output),
                "returncode": code,
            }
        )
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
