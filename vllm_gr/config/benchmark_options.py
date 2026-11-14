# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit benchmark A/B switches for RecoGEM-inspired CUDA configuration."""

import argparse

_OPTIONS = {
    "enable_full_step_graph": (
        "full_step_graph",
        "Capture CUDA Decode forward, projection, selection and Beam update.",
    ),
    "enable_streaming_constrained_head": (
        "streaming_constrained_head",
        "Stream full-window LM-head logits and canonical candidate selection.",
    ),
}


def add_recogem_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--strict-bf16",
        action="store_true",
        help="Disable BF16 reduced-precision GEMM reduction for in-process V1 validation.",
    )
    for name, help_text in _OPTIONS.values():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            action=argparse.BooleanOptionalAction,
            default=False,
            help=help_text,
        )


def optimization_settings(args: argparse.Namespace) -> dict[str, bool]:
    return {field: bool(getattr(args, name, False)) for field, (name, _) in _OPTIONS.items()}


def beam_optimization_config(args: argparse.Namespace) -> dict[str, bool]:
    return {field: enabled for field, enabled in optimization_settings(args).items() if enabled}


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
