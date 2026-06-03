# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline HSTU GR accuracy benchmark runner for vllm-GR."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from vllm_gr.benchmarks.common.gr_accuracy_utils import (
    compute_metrics,
    extract_predictions_by_task,
    validate_accuracy_record,
)
from vllm_gr.benchmarks.models.hstu.datasets import (
    SUPPORTED_HSTU_DATASETS,
    get_samples,
)
from vllm_gr.entrypoints.gr import GRLLM


def run_gr_vllm(
    prompts: list[Any],
    args: argparse.Namespace,
    default_max_candidates: int,
) -> list[Any]:
    """Run benchmark prompts through GRLLM one request at a time."""
    engine_kwargs: dict[str, Any] = {
        "max_seq_len": args.max_seq_len,
        "max_candidates": args.max_candidates or default_max_candidates,
        "allow_missing_weights": args.allow_missing_weights,
    }
    if args.dtype is not None:
        engine_kwargs["dtype"] = args.dtype

    with GRLLM(model=args.model, **engine_kwargs) as llm:
        outputs: list[Any] = []
        prompt_iter = prompts
        if not args.disable_tqdm:
            prompt_iter = tqdm(prompts, desc="Running benchmark prompts")
        for prompt in prompt_iter:
            prompt_outputs = llm.recommend(prompt, use_tqdm=False)
            if len(prompt_outputs) != 1:
                raise ValueError(
                    "GRLLM.recommend() must return exactly one output for a single "
                    f"prompt, got {len(prompt_outputs)}"
                )
            outputs.extend(prompt_outputs)
    return outputs


def _parse_request_index(output_request_id: str) -> int:
    try:
        return int(str(output_request_id).rsplit("-", maxsplit=1)[-1])
    except ValueError:
        return 0


def build_accuracy_records(
    requests: list[Any],
    outputs: list[Any],
) -> list[dict[str, Any]]:
    """Join benchmark requests with model outputs into per-request records."""
    ordered_outputs = sorted(
        outputs,
        key=lambda output: _parse_request_index(output.request_id),
    )
    if len(ordered_outputs) != len(requests):
        raise ValueError(
            f"Output count mismatch: got {len(ordered_outputs)}, expected {len(requests)}"
        )

    records: list[dict[str, Any]] = []
    for request, output in zip(requests, ordered_outputs):
        predictions_by_task = extract_predictions_by_task(
            all_scores=output.all_scores,
            task_names=request.task_names,
            num_candidates=request.num_candidates,
        )
        records.append(
            {
                "request_id": request.request_id,
                "dataset_name": request.dataset_name,
                "task_names": request.task_names,
                "num_candidates": request.num_candidates,
                "predictions_by_task": predictions_by_task,
                "labels_by_task": request.labels_by_task,
                "weights_by_task": request.weights_by_task,
            }
        )
    return records


def summarize_gr_accuracy_records(
    records: list[dict[str, Any]],
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Aggregate per-request records into benchmark summary metrics."""
    if dataset_name is not None:
        records = [record for record in records if record["dataset_name"] == dataset_name]
    if not records:
        raise ValueError("No accuracy records found.")

    grouped_predictions_by_task: dict[str, list[list[float]]] = defaultdict(list)
    grouped_labels_by_task: dict[str, list[list[float]]] = defaultdict(list)
    grouped_weights_by_task: dict[str, list[list[float]]] = defaultdict(list)
    num_candidates: list[int] = []

    total_candidates = 0
    datasets_seen: set[str] = set()
    for record in records:
        validate_accuracy_record(record)
        datasets_seen.add(record["dataset_name"])
        current_num_candidates = int(record["num_candidates"])
        total_candidates += current_num_candidates
        num_candidates.append(current_num_candidates)
        for task_name in record["task_names"]:
            grouped_predictions_by_task[task_name].append(
                [float(value) for value in record["predictions_by_task"][task_name]]
            )
            grouped_labels_by_task[task_name].append(
                [float(value) for value in record["labels_by_task"][task_name]]
            )
            grouped_weights_by_task[task_name].append(
                [float(value) for value in record["weights_by_task"][task_name]]
            )

    metrics_by_task = compute_metrics(
        grouped_predictions_by_task=dict(grouped_predictions_by_task),
        grouped_labels_by_task=dict(grouped_labels_by_task),
        grouped_weights_by_task=dict(grouped_weights_by_task),
        num_candidates=num_candidates,
    )
    effective_dataset_name = dataset_name or (
        next(iter(datasets_seen)) if len(datasets_seen) == 1 else "mixed"
    )
    return {
        "dataset_name": effective_dataset_name,
        "num_requests": len(records),
        "num_candidates": total_candidates,
        "metrics_by_task": metrics_by_task,
    }


def write_records_jsonl(records: list[dict[str, Any]], output_path: str) -> None:
    """Persist per-request benchmark records as JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before benchmark execution."""
    if args.dataset_name not in SUPPORTED_HSTU_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{args.dataset_name}'. "
            f"Supported datasets: {', '.join(SUPPORTED_HSTU_DATASETS)}."
        )
    if args.num_prompts is not None and args.num_prompts <= 0:
        raise ValueError("--num-prompts must be > 0 when provided")
    if args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be > 0")
    if args.dataset_candidate_count <= 0:
        raise ValueError("--dataset-candidate-count must be > 0")
    if args.max_candidates is not None and args.max_candidates <= 0:
        raise ValueError("--max-candidates must be > 0 when provided")
    if args.max_candidates is not None and args.max_candidates < args.dataset_candidate_count:
        raise ValueError(
            "--max-candidates must be >= --dataset-candidate-count: "
            f"{args.max_candidates} < {args.dataset_candidate_count}"
        )


def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI arguments for the HSTU GR accuracy benchmark."""
    parser.add_argument("--model", required=True, help="Model name or local model path.")
    parser.add_argument(
        "--dataset-name",
        required=True,
        choices=sorted(SUPPORTED_HSTU_DATASETS),
        help="Benchmark dataset name.",
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to the prepared benchmark CSV file.",
    )
    parser.add_argument(
        "--dataset-candidate-count",
        type=int,
        default=5,
        help=(
            "Number of candidates to take from the tail of each dataset sequence "
            "when constructing benchmark requests."
        ),
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Optional maximum number of benchmark prompts to run.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--dtype", default=None, help="Optional GREngineArgs dtype override.")
    parser.add_argument(
        "--allow-missing-weights",
        action="store_true",
        help="Allow missing model weights and fall back to dummy loading.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum GR sequence length passed to GRLLM.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5,
        help=("Runtime GR candidate limit passed to GRLLM. Must be >= --dataset-candidate-count."),
    )
    parser.add_argument(
        "--disable-shuffle",
        action="store_true",
        help="Disable request shuffling during dataset sampling.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable benchmark progress bars.",
    )
    parser.add_argument(
        "--output-records-jsonl",
        default=None,
        help="Optional JSONL output path for per-request benchmark records.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the HSTU benchmark entrypoint."""
    return add_cli_args(argparse.ArgumentParser(description=__doc__))


def main(args: argparse.Namespace | None = None) -> None:
    """Run the HSTU GR accuracy benchmark and print a JSON summary."""
    if args is None:
        args = build_parser().parse_args()

    validate_args(args)
    requests, prompts, default_max_candidates = get_samples(
        dataset_name=args.dataset_name,
        input_csv=args.input_csv,
        num_prompts=args.num_prompts,
        dataset_candidate_count=args.dataset_candidate_count,
        random_seed=args.seed,
        disable_shuffle=args.disable_shuffle,
    )

    outputs = run_gr_vllm(
        prompts=prompts,
        args=args,
        default_max_candidates=default_max_candidates,
    )
    records = build_accuracy_records(requests=requests, outputs=outputs)
    if args.output_records_jsonl is not None:
        write_records_jsonl(records, args.output_records_jsonl)
    summary = summarize_gr_accuracy_records(records, dataset_name=args.dataset_name)
    sys.stdout.write(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
