# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Standalone real-NPU copy/replay contract, with optional x_attention parity.

Run in the isolated candidate source environment. Writes only --output; does
not load a model, modify device allocation of other tasks, or change weights.
Every native kernel assertion is strict bitwise equality. A failed test exits
nonzero and retains the failure report; there are no skipped failing cases.
"""

from __future__ import annotations

import argparse
import json
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
        "--quick", action="store_true", help="One representative interleaved B1 copy/graph case"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--attention", action="store_true", help="Also require native x_attention bitwise parity"
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 128])
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("bfloat16", "float16", "float32"),
        default=["bfloat16", "float16"],
    )
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite contract evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "state": "running",
        "cases": [],
        "attention_enabled": args.attention,
        "timing_is_not_performance_evidence": True,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    code = 0
    try:
        import os

        os.environ["VLLM_GR_NPU_FUSED_SHARED_KV_GATHER"] = "1"
        os.environ["VLLM_GR_NPU_SHARED_KV_LENGTH_MASK"] = "0"
        import torch
        import torch_npu  # noqa: F401

        from vllm_gr.v1.attention.backends.beam_attn_npu_xllm import (
            _gather_shared_blocks,
        )
        from vllm_gr.v1.attention.backends.npu_shared_kv_gather import (
            fused_shared_kv_gather_stats,
            try_fused_shared_kv_gather,
        )

        require(
            torch.npu.is_available() and torch.npu.device_count() == 1,
            "Allocate exactly one NPU explicitly before running",
        )
        torch.manual_seed(20261122)

        def reference(
            key: torch.Tensor, value: torch.Tensor, table: torch.Tensor, bound: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            cols = table.shape[1]
            cols = bound if 0 < bound < cols else cols
            ids = table.cpu()[:, :cols].long()
            # CPU reference is deliberately independent of the NPU copy kernel.
            return (
                key.cpu()[ids].reshape(-1, *key.shape[1:]).contiguous(),
                value.cpu()[ids].reshape(-1, *value.shape[1:]).contiguous(),
                torch.arange(ids.numel(), dtype=table.dtype).reshape(ids.shape),
            )

        def equal_bits(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
            require(tuple(actual.shape) == tuple(expected.shape), f"{name}: shape changed")
            actual_cpu = actual.cpu().contiguous()
            expected_cpu = expected.cpu().contiguous()
            # Raw integer views also distinguish BF16/FP16 bit patterns.
            integer = torch.int16 if actual.element_size() == 2 else torch.int32
            require(
                torch.equal(actual_cpu.view(integer), expected_cpu.view(integer)),
                f"{name}: copy is not bitwise identical",
            )

        def assert_copy(
            key: torch.Tensor,
            value: torch.Tensor,
            table: torch.Tensor,
            bound: int,
            outputs: Any,
            label: str,
        ) -> None:
            require(outputs is not None, f"{label}: fused kernel silently fell back")
            expected = reference(key, value, table, bound)
            for name, actual, baseline in zip(
                ("key", "value", "table"), outputs, expected, strict=True
            ):
                require(actual.is_contiguous(), f"{label}: noncontiguous {name} output")
                if name == "table":
                    require(torch.equal(actual.cpu(), baseline), f"{label}: remapped table changed")
                else:
                    equal_bits(actual, baseline, f"{label}:{name}")

        def attention_parity(
            key: torch.Tensor,
            value: torch.Tensor,
            table: torch.Tensor,
            bound: int,
            outputs: Any,
            batch: int,
            width: int,
        ) -> dict[str, object]:
            from vllm_gr.ops.npu import beam_ops

            if key.dtype == torch.float32:
                return {
                    "executed": False,
                    "reason": "x_attention supports BF16/FP16; FP32 copy tested separately",
                }
            require(beam_ops.is_available(), "Native x_attention required by --attention")
            expected = reference(key, value, table, bound)
            baseline = tuple(tensor.to("npu") for tensor in expected)
            q = torch.randn((batch * width, 16, 128), device="npu", dtype=key.dtype)
            uk = torch.randn((batch * width, 8, 3, 128), device="npu", dtype=key.dtype)
            uv = torch.randn_like(uk)
            lengths = torch.full(
                (batch,), min(outputs[2].shape[1], 2) * 128, dtype=torch.int32, device="npu"
            )
            left = beam_ops.x_attention(
                q, baseline[0], baseline[1], uk, uv, baseline[2], lengths, 1, 0.0
            )
            right = beam_ops.x_attention(
                q, outputs[0], outputs[1], uk, uv, outputs[2], lengths, 1, 0.0
            )
            equal_bits(right, left, "native x_attention")
            return {"executed": True, "bitwise_equal": True}

        expected_last_stride_fallbacks = 0
        for dtype_name in args.dtypes:
            dtype = getattr(torch, dtype_name)
            for layout in (
                ("interleaved",) if args.quick else ("contiguous", "interleaved", "inner_strided")
            ):
                for batch in (1,) if args.quick else (1, 4):
                    for width in args.widths:
                        for bound in (0,) if args.quick else (0, 2):
                            item: dict[str, Any] = {
                                "dtype": dtype_name,
                                "layout": layout,
                                "batch": batch,
                                "beam_width": width,
                                "max_valid_cols": bound,
                                "state": "running",
                            }
                            report["cases"].append(item)
                            blocks, block_size, heads, dim = 16, 128, 8, 128
                            if layout == "contiguous":
                                key = torch.randn(
                                    (blocks, block_size, heads, dim), device="npu", dtype=dtype
                                )
                                value = torch.randn_like(key)
                            elif layout == "interleaved":
                                pool = torch.randn(
                                    (blocks + 1, 2, block_size, heads, dim),
                                    device="npu",
                                    dtype=dtype,
                                )
                                # Native 5D allocation may default to NCDHW30.
                                # Production paged KV is ND: set its base format
                                # explicitly before making noncontiguous halves.
                                pool = torch_npu.npu_format_cast(pool, 2)
                                require(
                                    torch_npu.get_npu_format(pool) == 2,
                                    "Fixture pool did not become ND",
                                )
                                # Nonzero storage offset plus per-block interleaving.
                                key, value = pool[1:, 0], pool[1:, 1]
                            else:
                                kbase = torch.randn(
                                    (blocks, block_size, heads, dim * 2), device="npu", dtype=dtype
                                )
                                vbase = torch.randn_like(kbase)
                                key, value = kbase[..., 1::2], vbase[..., ::2]
                            ids = torch.randint(
                                0, blocks, (batch, 8), device="npu", dtype=torch.int32
                            )
                            ids[:, 2] = ids[
                                :, 0
                            ]  # Duplicate physical block routes are intentional.
                            item["input_metadata"] = {
                                name: {
                                    "format": int(torch_npu.get_npu_format(tensor)),
                                    "shape": list(tensor.shape),
                                    "stride": list(tensor.stride()),
                                    "storage_offset": int(tensor.storage_offset()),
                                }
                                for name, tensor in (("key", key), ("value", value), ("table", ids))
                            }
                            outputs = try_fused_shared_kv_gather(key, value, ids, bound)
                            if layout == "inner_strided":
                                require(
                                    outputs is None,
                                    "Unsupported dtype-view stride must explicitly fall back",
                                )
                                integrated = _gather_shared_blocks(key, value, ids, bound)
                                torch.npu.synchronize()
                                assert_copy(
                                    key, value, ids, bound, integrated, "explicit Torch fallback"
                                )
                                expected_last_stride_fallbacks += 2
                                item.update(
                                    state="passed",
                                    execution="explicit_torch_fallback",
                                    fallback_reason="last_stride",
                                    bitwise_copy_equal=True,
                                    graph_replays_with_new_tables_and_kv=0,
                                    graph_scope_note="Nonunit innermost stride is outside fused/capture contract",
                                )
                                continue
                            torch.npu.synchronize()
                            assert_copy(key, value, ids, bound, outputs, "eager")
                            # Verify integration wrapper actually selects the fused path.
                            before_fused = fused_shared_kv_gather_stats()["host_launch_counts"].get(
                                "fused", 0
                            )
                            integrated = _gather_shared_blocks(key, value, ids, bound)
                            require(
                                fused_shared_kv_gather_stats()["host_launch_counts"].get("fused", 0)
                                == before_fused + 1,
                                "Attention wrapper did not use fused gather",
                            )
                            assert_copy(key, value, ids, bound, integrated, "integrated")

                            graph = torch.npu.NPUGraph()
                            side = torch.npu.Stream()
                            side.wait_stream(torch.npu.current_stream())
                            with torch.npu.stream(side):
                                with torch.npu.graph(graph):
                                    captured = try_fused_shared_kv_gather(key, value, ids, bound)
                            torch.npu.current_stream().wait_stream(side)
                            torch.npu.synchronize()
                            addresses = [tensor.data_ptr() for tensor in captured]
                            for replay in range(3):
                                # Same graph/pointers, fresh request block routing and fresh
                                # values (including formerly reused slots) every iteration.
                                key.copy_(torch.randn(key.shape, device="npu", dtype=dtype))
                                value.copy_(torch.randn(value.shape, device="npu", dtype=dtype))
                                fresh = torch.randint(
                                    0, blocks, ids.shape, device="npu", dtype=ids.dtype
                                )
                                fresh[:, 2] = fresh[:, 0]
                                ids.copy_(fresh)
                                graph.replay()
                                torch.npu.synchronize()
                                require(
                                    [tensor.data_ptr() for tensor in captured] == addresses,
                                    "Graph output addresses changed",
                                )
                                assert_copy(key, value, ids, bound, captured, f"replay-{replay}")
                            item["graph_replays_with_new_tables_and_kv"] = 3
                            item["execution"] = "fused_integer_view_kernel"
                            item["output_addresses_stable"] = True
                            item["bitwise_copy_equal"] = True
                            if args.attention:
                                item["native_attention"] = attention_parity(
                                    key, value, ids, bound, captured, batch, width
                                )
                            item["state"] = "passed"
                            del graph, captured, outputs, integrated
        report["diagnostics"] = fused_shared_kv_gather_stats()
        observed_fallbacks = {
            name: count
            for name, count in report["diagnostics"]["host_launch_counts"].items()
            if name.startswith("fallback:")
        }
        expected_fallbacks = (
            {"fallback:last_stride": expected_last_stride_fallbacks}
            if expected_last_stride_fallbacks
            else {}
        )
        require(
            observed_fallbacks == expected_fallbacks,
            "Unexpected fallback or missing explicit nonunit-stride fallback",
        )
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
