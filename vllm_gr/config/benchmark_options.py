# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit benchmark A/B switches for RecoGEM-inspired CUDA configuration."""

import argparse


def add_recogem_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--full-step-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture CUDA Decode forward, projection, candidate selection and Beam update.",
    )
    parser.add_argument(
        "--strict-bf16",
        action="store_true",
        help="Disable BF16 reduced-precision GEMM reduction for in-process V1 validation.",
    )


def beam_optimization_config(args: argparse.Namespace) -> dict[str, bool]:
    """Omit disabled defaults so existing canonical benchmark configs stay valid."""
    return {"enable_full_step_graph": True} if getattr(args, "full_step_graph", False) else {}


def optimization_settings(args: argparse.Namespace) -> dict[str, bool]:
    return {"enable_full_step_graph": bool(getattr(args, "full_step_graph", False))}


def configure_benchmark_precision(args: argparse.Namespace) -> dict[str, bool]:
    import os

    import torch

    strict = bool(getattr(args, "strict_bf16", False))
    if strict:
        if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
            raise ValueError("--strict-bf16 requires VLLM_ENABLE_V1_MULTIPROCESSING=0")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return {
        "strict_bf16": strict,
        "bf16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
    }
