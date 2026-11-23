#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Evaluate offline GSM8K-mini accuracy for Pangu Embedded 1B via vllm-gr.

Uses ``GRLLM`` class without a dedicated adapter. Requests are routed through
the standard vLLM generate path. Pass recommended engine flags explicitly
(see ``_pangu_engine_kwargs``).

Example::

    python tests/st/pangu_generate.py \\
        --model /path/to/openPangu-Embedded-1B-V1.1 \\
        --dataset-path tests/resources/pangu1b/gsm8k_mini.json \\
        --num-prompts 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from . import pangu1b_st_common as pangu
else:
    import pangu1b_st_common as pangu  # Direct invocation via tests/st/pangu_generate.py.

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _REPO_ROOT / pangu.GSM8K_MINI


def _is_npu_platform() -> bool:
    """Return whether vLLM selected the Ascend platform."""
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    return getattr(current_platform, "device_type", None) == "npu"


def _pangu_engine_kwargs(block_size: int | None = None) -> dict[str, Any]:
    """Recommended vLLM engine flags for openPangu Embedded 1B."""
    kwargs: dict[str, Any] = {
        "config": _REPO_ROOT / pangu.PANGU_CONFIG,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_model_len": int(os.environ.get("PANGU1B_MAX_MODEL_LEN", "32768")),
        "tokenizer_mode": "slow",
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
    }
    # vLLM defaults to block_size=16. vLLM-Ascend's native attention
    # backends require a kernel block size of 128, but its platform hook only
    # promotes the default when chunked prefill or prefix caching is enabled.
    # Pangu intentionally disables both, so select the compatible value here.
    if block_size is None and _is_npu_platform():
        block_size = 128
    if block_size is not None:
        kwargs["block_size"] = block_size
        logger.info("Using Pangu KV cache block size %d", block_size)
    return kwargs


def _resolve_dataset_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate

    for base in (Path.cwd(), _REPO_ROOT, Path(__file__).resolve().parent):
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved
    return candidate


def run_offline_eval(
    *,
    model: str,
    dataset_path: Path,
    num_prompts: int | None,
    max_tokens: int,
    temperature: float,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    block_size: int | None = None,
) -> dict[str, Any]:
    try:
        from vllm import SamplingParams

        from vllm_gr.entrypoints.gr import GRLLM
    except ImportError as exc:
        raise RuntimeError("vLLM and vllm-gr are required for Pangu offline inference.") from exc

    records = pangu.load_dataset(dataset_path, num_prompts)
    if not records:
        raise ValueError(f"No records loaded from {dataset_path}")

    start = time.time()
    llm = GRLLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        **_pangu_engine_kwargs(block_size),
    )

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
    )

    prompts = [pangu.build_prompt(str(row["question"])) for row in records]
    outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)

    correct = 0
    details: list[dict[str, Any]] = []
    for row, request_output in zip(records, outputs):
        generated = request_output.outputs[0].text if request_output.outputs else ""
        score = pangu.score_answer(generated, str(row["answer"]))
        correct += score["correct"]
        details.append({"id": row.get("id"), **score, "generated": generated})

    elapsed = time.time() - start
    total = len(records)
    accuracy = correct / total if total else 0.0

    llm.shutdown()

    return {
        "model": model,
        "dataset_path": str(dataset_path),
        "num_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_time": elapsed,
        "details": details,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline Pangu Embedded 1B inference / GSM8K-mini eval"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("PANGU1B_MODEL_PATH") or os.environ.get("PANGU_MODEL_PATH", ""),
        help="Local path or HF id for openPangu-Embedded-1B-V1.1",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(_DEFAULT_DATASET),
        help="JSON list with question/answer fields",
    )
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--block-size",
        type=int,
        default=(
            int(os.environ["PANGU1B_BLOCK_SIZE"]) if os.environ.get("PANGU1B_BLOCK_SIZE") else None
        ),
        help="KV cache block size; defaults to 128 on Ascend NPU",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=float(os.environ.get("PANGU1B_MIN_ACCURACY", "0.5")),
        help="Exit with code 1 when accuracy is below this threshold",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to save results")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if not args.model:
        print(
            "Error: --model or PANGU1B_MODEL_PATH/PANGU_MODEL_PATH must point to openPangu-Embedded-1B-V1.1",
            file=sys.stderr,
        )
        return 2
    if args.block_size is not None and args.block_size <= 0:
        parser.error("--block-size must be greater than zero")

    dataset_path = _resolve_dataset_path(args.dataset_path)
    if not dataset_path.is_file():
        print(f"Error: dataset not found: {dataset_path}", file=sys.stderr)
        print(f"Hint: default fixture is {_DEFAULT_DATASET}", file=sys.stderr)
        return 2

    result = run_offline_eval(
        model=args.model,
        dataset_path=dataset_path,
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=args.block_size,
    )

    summary = {
        "accuracy": result["accuracy"],
        "correct": result["correct"],
        "num_samples": result["num_samples"],
        "elapsed_time": result["elapsed_time"],
        "model": result["model"],
        "dataset_path": result["dataset_path"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)

    if result["accuracy"] < args.min_accuracy:
        print(
            f"FAILURE: accuracy {result['accuracy']:.4f} < threshold {args.min_accuracy:.4f}",
            file=sys.stderr,
        )
        return 1

    print(
        f"SUCCESS: accuracy {result['accuracy']:.4f} >= threshold {args.min_accuracy:.4f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
