#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate OneRec golden outputs with the official native-vLLM protocol.

OpenOneRec appends ``<|sid_begin|>`` to the prompt, asks the unmodified
``vllm.LLM.beam_search`` implementation for exactly the three semantic-ID
tokens, and does not score an end sentinel.  We wrap the returned IDs with the
two sentinels only so the local evaluator and decode endpoint share one output
contract; neither sentinel contributes to the cumulative beam score.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from benchmarks.open_one_rec.open_one_rec_loader import DataLoaderWrapper
from benchmarks.open_one_rec.regression import (
    RECOMMENDATION_TASKS,
    RecommendationMetricAccumulator,
    load_sample_manifest,
    prompt_digest,
    sha256_file,
    validate_generations,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)
SID_BEGIN_TOKEN = "<|sid_begin|>"
SID_END_TOKEN = "<|sid_end|>"
ONEREC_SEMANTIC_DEPTH = 3


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_prepared_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tasks"), dict):
        raise ValueError("prepared-data manifest is malformed")
    return manifest


def native_vllm_beam_search(
    *,
    llm: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    beam_width: int,
    temperature: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Call native ``LLM.beam_search`` using OpenOneRec's SID protocol."""
    from vllm.sampling_params import BeamSearchParams

    if max_tokens != ONEREC_SEMANTIC_DEPTH:
        raise ValueError(
            f"OpenOneRec native golden requires exactly {ONEREC_SEMANTIC_DEPTH} "
            f"semantic tokens, got {max_tokens}"
        )
    begin_token_id = tokenizer.convert_tokens_to_ids(SID_BEGIN_TOKEN)
    end_token_id = tokenizer.convert_tokens_to_ids(SID_END_TOKEN)
    if begin_token_id is None or begin_token_id < 0:
        raise ValueError(f"model tokenizer has no {SID_BEGIN_TOKEN} token")
    if end_token_id is None or end_token_id < 0:
        raise ValueError(f"model tokenizer has no {SID_END_TOKEN} token")

    params = BeamSearchParams(
        beam_width=beam_width,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    raw_outputs = llm.beam_search(
        [{"prompt": prompt + SID_BEGIN_TOKEN} for prompt in prompts],
        params,
        use_tqdm=False,
        concurrency_limit=len(prompts),
    )
    if len(raw_outputs) != len(prompts):
        raise RuntimeError(
            f"native vLLM returned {len(raw_outputs)} outputs for {len(prompts)} prompts"
        )

    results: list[dict[str, Any]] = []
    for raw_output in raw_outputs:
        generations: list[str] = []
        generation_token_ids: list[list[int]] = []
        cumulative_logprobs: list[float] = []
        for sequence in raw_output.sequences:
            semantic_token_ids = list(sequence.tokens[-max_tokens:])
            if len(semantic_token_ids) != ONEREC_SEMANTIC_DEPTH:
                raise RuntimeError(
                    f"native vLLM returned an incomplete OneRec SID: {semantic_token_ids}"
                )
            token_ids = [begin_token_id, *semantic_token_ids, end_token_id]
            generation_token_ids.append(token_ids)
            generations.append(
                tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            cumulative_logprobs.append(float(sequence.cum_logprob))
        results.append(
            {
                "generations": generations,
                "generation_token_ids": generation_token_ids,
                "cumulative_logprobs": cumulative_logprobs,
            }
        )
    return results


def _selected_tasks(tasks_csv: str) -> list[str]:
    tasks = [task.strip() for task in tasks_csv.split(",") if task.strip()]
    if not tasks:
        raise ValueError("--tasks must select at least one task")
    unsupported = sorted(set(tasks).difference(RECOMMENDATION_TASKS))
    if unsupported:
        raise ValueError(f"unsupported recommendation tasks: {unsupported}")
    if len(tasks) != len(set(tasks)):
        raise ValueError("--tasks contains duplicates")
    return tasks


def run(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if os.environ.get("VLLM_PLUGINS") != "":
        raise ValueError("native golden requires VLLM_PLUGINS='' so vllm-gr patches cannot load")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise ValueError(
            "native golden requires VLLM_USE_V2_MODEL_RUNNER=0 to pin the same "
            "vLLM BeamSearch implementation used by the regression baseline"
        )
    if not 1 <= args.beam_width <= 128:
        raise ValueError("--beam-width must be in [1, 128]")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    tasks = _selected_tasks(args.tasks)
    prepared = _load_prepared_manifest(args.prepared_manifest)
    sample_ids_by_task = load_sample_manifest(
        args.sample_manifest,
        dataset_path=args.dataset_path,
    )
    missing = [task for task in tasks if task not in sample_ids_by_task]
    if missing:
        raise ValueError(f"sample manifest has no IDs for tasks: {missing}")

    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM

    native_implementation = f"vllm-{vllm.__version__}-LLM.beam_search"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    loader = DataLoaderWrapper(tokenizer=tokenizer, data_dir=args.dataset_path)
    datasets = {
        task: loader.load_data(
            task_name=task,
            split="test",
            sample_ids=sample_ids_by_task[task],
            cache_sample=False,
        )
        for task in tasks
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        records_path = output_dir / f"{task}.jsonl"
        summary_path = output_dir / f"{task}.summary.json"
        if args.overwrite:
            records_path.unlink(missing_ok=True)
            summary_path.unlink(missing_ok=True)
        elif records_path.exists() or summary_path.exists():
            raise ValueError(f"golden output already exists for {task}; use --overwrite")

    llm = LLM(
        model=args.model,
        revision=args.revision,
        trust_remote_code=True,
        max_logprobs=2 * args.beam_width,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    runner_revision = _git_revision()
    summaries: dict[str, dict[str, Any]] = {}
    try:
        for task in tasks:
            domain = prepared["tasks"].get(task)
            if not isinstance(domain, str):
                raise ValueError(f"prepared-data manifest has no domain for {task}")
            try:
                domain_info = prepared["domains"][domain]
                logical_universe_digest = domain_info["logical_universe_sha256"]
            except KeyError as error:
                raise ValueError(f"prepared-data manifest is missing {error.args[0]!r}") from error
            try:
                dataset = datasets[task]
                accumulator = RecommendationMetricAccumulator(
                    task,
                    args.dataset_path,
                    evaluation_mode=args.evaluation_mode,
                )
                started = time.perf_counter()
                records_path = output_dir / f"{task}.jsonl"
                source_parquet = (Path(args.dataset_path) / task / f"{task}_test.parquet").resolve()
                source_digest = sha256_file(source_parquet)
                invalid_samples = 0
                sample_ids = list(dataset)
                completed = 0
                for start in range(0, len(sample_ids), args.batch_size):
                    batch_ids = sample_ids[start : start + args.batch_size]
                    batch_items = [dataset[sample_id] for sample_id in batch_ids]
                    batch_results = native_vllm_beam_search(
                        llm=llm,
                        tokenizer=tokenizer,
                        prompts=[item["prompt"] for item in batch_items],
                        beam_width=args.beam_width,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    records = []
                    for sample_id, item, result in zip(batch_ids, batch_items, batch_results):
                        contract = validate_generations(result["generations"], args.beam_width)
                        invalid = (
                            not contract["beam_count_ok"]
                            or contract["empty_count"] > 0
                            or contract["invalid_sid_count"] > 0
                        )
                        invalid_samples += int(invalid)
                        metrics = accumulator.add(
                            result["generations"],
                            item["ground_truth"],
                            item["metadata"],
                        )
                        records.append(
                            {
                                "format_version": 1,
                                "task": task,
                                "sample_id": sample_id,
                                "prompt_sha256": prompt_digest(item["prompt"]),
                                "run_identity": {
                                    "model": args.model,
                                    "revision": args.revision,
                                    "implementation_revision": native_implementation,
                                    "execution_variant": "native-llm-beam-search",
                                    "source_sha256": source_digest,
                                    "execution": "offline",
                                    "constraint": {
                                        "domain": domain,
                                        "mode": "none",
                                        "logical_universe_digest": logical_universe_digest,
                                    },
                                    "beam_width": args.beam_width,
                                    "max_tokens": args.max_tokens,
                                    "batch_size": args.batch_size,
                                    "temperature": args.temperature,
                                    "seed": args.seed,
                                    "evaluation_mode": args.evaluation_mode,
                                },
                                **result,
                                "contract": contract,
                                "metrics": metrics,
                            }
                        )
                    write_jsonl(records_path, records)
                    completed += len(batch_ids)
                    if completed == len(dataset) or completed % args.log_every == 0:
                        LOGGER.info(
                            "Completed native golden %s %d/%d",
                            task,
                            completed,
                            len(dataset),
                        )

                summary = {
                    "format_version": 1,
                    "task": task,
                    "model": args.model,
                    "revision": args.revision,
                    "git_revision": runner_revision,
                    "implementation_revision": native_implementation,
                    "execution_variant": "native-llm-beam-search",
                    "python": sys.version,
                    "platform": platform.platform(),
                    "execution": "offline",
                    "dataset_path": str(Path(args.dataset_path).resolve()),
                    "source_parquet": str(source_parquet),
                    "source_sha256": source_digest,
                    "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
                    "sample_manifest_sha256": sha256_file(args.sample_manifest),
                    "constraint": {
                        "domain": domain,
                        "mode": "none",
                        "logical_universe_digest": logical_universe_digest,
                    },
                    "beam_width": args.beam_width,
                    "max_tokens": args.max_tokens,
                    "batch_size": args.batch_size,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "selected_samples": len(dataset),
                    "invalid_samples": invalid_samples,
                    "elapsed_time": time.perf_counter() - started,
                    "metrics": accumulator.summary(),
                }
                summary_path = output_dir / f"{task}.summary.json"
                with summary_path.open("w", encoding="utf-8") as file_obj:
                    json.dump(summary, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
                    file_obj.write("\n")
                summaries[task] = summary
                if args.strict and invalid_samples:
                    raise RuntimeError(
                        f"{invalid_samples} native golden samples violated the output contract"
                    )
            finally:
                gc.collect()
    finally:
        llm.llm_engine.engine_core.shutdown()

    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=",".join(RECOMMENDATION_TASKS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=ONEREC_SEMANTIC_DEPTH)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--evaluation-mode", choices=["sid", "pid", "both"], default="both")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    try:
        run(args)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
