#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a resumable, per-sample OneRec recommendation precision regression."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmarks.open_one_rec.open_one_rec_loader import DataLoaderWrapper
from benchmarks.open_one_rec.regression import (
    RECOMMENDATION_TASKS,
    RecommendationMetricAccumulator,
    iter_jsonl,
    load_sample_manifest,
    prompt_digest,
    sha256_file,
    validate_generations,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)

# Environment knobs that change engine behaviour. They are recorded with the
# run so a golden can be traced back to its capture configuration; they are
# deliberately NOT part of the hard-gated configuration contract, so one golden stays
# reusable across a wide range of serving parameters.
CONFIG_CONTRACT_ENV_VARS = (
    "VLLM_USE_V2_MODEL_RUNNER",
    "VLLM_ENABLE_PREFILL_CUDAGRAPH",
    "VLLM_GR_PREFILL_GRAPH_KV_BOUND",
    "VLLM_PLUGINS",
    "PYTHONHASHSEED",
)


def _config_env_snapshot() -> dict[str, str | None]:
    return {name: os.getenv(name) for name in CONFIG_CONTRACT_ENV_VARS}


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


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


def _resolve_constraint(
    manifest: dict[str, Any],
    task: str,
    mode: str,
    artifact_backend: str,
    beam_width: int = 128,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        domain_name = manifest["tasks"][task]
        domain = manifest["domains"][domain_name]
    except KeyError as error:
        raise ValueError(f"prepared-data manifest is missing {error.args[0]!r}") from error

    if mode == "trie":
        return (
            {"catalog_path": domain["catalog_path"], "constraint_backend": "trie"},
            {"domain": domain_name, "mode": mode, "digest": domain["catalog_sha256"]},
        )
    if mode == "triples":
        return (
            {
                "catalog_path": domain["triples_path"],
                "constraint_backend": "constraint_table",
            },
            {"domain": domain_name, "mode": mode, "digest": domain["triples_sha256"]},
        )
    if mode == "artifact":
        artifact = domain.get("artifact")
        if not isinstance(artifact, dict):
            raise ValueError(
                f"prepared domain {domain_name!r} has no artifact; rerun preparation "
                "with --build-artifacts"
            )
        canonical = {
            "schema_version": 1,
            "beam": {
                "execution_mode": "v1",
                "graph_enabled": True,
                "worker_decision": True,
                "max_width": beam_width,
                "max_decode_steps": 3,
            },
            "constraint_table": {
                "enabled": True,
                "backend": artifact_backend,
                "path": artifact["path"],
                "format": "constraint_table_v1",
                "artifact_digest": artifact["artifact_digest"],
                "tokenizer_digest": artifact["tokenizer_digest"],
                "max_top_k": 256,
            },
            "attention_backend": "CUSTOM",
        }
        return (
            {"vllm_gr_config": canonical},
            {"domain": domain_name, "mode": mode, "digest": artifact["artifact_digest"]},
        )
    raise ValueError(f"unknown constraint mode: {mode}")


def _engine_is_beam_v1(llm: Any) -> bool:
    """True when the engine was built with the canonical Beam V1 config.

    The artifact constraint mode boots GRLLM with the canonical
    ``vllm_gr_config`` (``beam.execution_mode=v1`` plus startup graph
    capture); that engine must be driven through ``beam_search_v1``. Any
    other engine (legacy beam config, or a test double without an engine)
    keeps the legacy frontend.
    """
    try:
        from vllm_gr.config import GRConfig
        from vllm_gr.config.execution import BeamExecutionMode

        vllm_config = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
        if vllm_config is None:
            return False
        return GRConfig.from_vllm_config(vllm_config).beam_execution_mode is BeamExecutionMode.V1
    except Exception:
        return False


def _offline_batch(
    llm: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    beam_width: int,
    temperature: float,
) -> list[dict[str, Any]]:
    from vllm_gr.sampling_params import BeamSearchParams

    if _engine_is_beam_v1(llm):
        # Canonical Beam V1 offline path (the maintained GPU path, identical
        # to the ABBA harness): the engine owns execution_mode=v1 with the
        # startup Decode graph and padded Prefill buckets, and requests go
        # through the V1 frontend. ignore_eos keeps beams alive until
        # <|sid_end|> regardless of the model's EOS token.  The V1 offline
        # frontend drives one session per call, so prompts are submitted
        # sequentially.
        params = BeamSearchParams(
            beam_width=beam_width,
            max_tokens=5,
            temperature=temperature,
            begin_token="<|sid_begin|>",
            end_token="<|sid_end|>",
            ignore_eos=True,
            include_stop_str_in_output=True,
        )
        raw_outputs = [
            llm.beam_search_v1([{"prompt": prompt}], params, concurrency_limit=1)[0]
            for prompt in prompts
        ]
    else:
        params = BeamSearchParams(
            beam_width=beam_width,
            max_tokens=5,
            temperature=temperature,
        )
        if hasattr(params, "begin_token"):
            params.begin_token = "<|sid_begin|>"
            params.end_token = "<|sid_end|>"
        raw_outputs = llm.beam_search(
            [{"prompt": prompt} for prompt in prompts],
            params,
            concurrency_limit=1,
        )
    if len(raw_outputs) != len(prompts):
        raise RuntimeError(
            f"beam search returned {len(raw_outputs)} outputs for {len(prompts)} prompts"
        )

    normalized = []
    for prompt, raw_output in zip(prompts, raw_outputs):
        generation_texts: list[str] = []
        generation_token_ids: list[list[int]] = []
        cumulative_logprobs: list[float] = []
        for sequence in raw_output.sequences:
            # GRLLM appends SID-begin to the prompt, generates the three SID
            # levels, then appends SID-end.  Taking the final five tokens is
            # independent of tokenizer add_special_tokens policy and avoids
            # confusing itemic tokens already present in the user history.
            token_ids = list(sequence.tokens[-5:])
            generation_token_ids.append(token_ids)
            generation_texts.append(
                tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            cumulative_logprobs.append(float(sequence.cum_logprob))
        normalized.append(
            {
                "generations": generation_texts,
                "generation_token_ids": generation_token_ids,
                "cumulative_logprobs": cumulative_logprobs,
            }
        )
    return normalized


def _online_batch(
    *,
    prompts: Sequence[str],
    model: str,
    host: str,
    port: int,
    endpoint: str,
    api_key: str,
    beam_width: int,
    temperature: float,
    timeout: float,
) -> list[dict[str, Any]]:
    import requests

    url = f"http://{host}:{port}{endpoint}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    results = []
    for prompt in prompts:
        payload = {
            "model": model,
            # DataLoaderWrapper has already rendered the dataset messages with
            # the model's chat template.  Send that text through the Completion
            # API verbatim; wrapping it in a new Chat user message applies the
            # template twice and changes the model input.
            "prompt": prompt,
            "use_beam_search": True,
            "max_tokens": 5,
            "temperature": temperature,
            "top_p": 0.95,
            "n": beam_width,
            "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"},
        }
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        document = response.json()
        choices = document.get("choices")
        if not isinstance(choices, list) or len(choices) != beam_width:
            raise RuntimeError(
                f"online response returned {len(choices) if isinstance(choices, list) else 0} "
                f"choices, expected {beam_width}"
            )
        texts = [str(choice.get("text", "")) for choice in choices]
        results.append(
            {
                "generations": texts,
                "generation_token_ids": [],
                "cumulative_logprobs": [],
            }
        )
    return results


_ENGINE_CACHE: dict[str, Any] = {}


def _shutdown_engine(llm: Any) -> None:
    shutdown = getattr(llm, "shutdown", None)
    if callable(shutdown):
        shutdown()
    else:
        # Upstream vLLM.LLM exposes the engine shutdown one level below the
        # wrapper, while GRLLM provides shutdown directly.
        llm.llm_engine.engine_core.shutdown()


def _cached_engine(engine_args: dict[str, Any]) -> Any:
    """Reuse one Beam V1 engine across tasks that share the same config.

    The expensive part of an offline run is the engine startup (graph capture
    and compilation, ~90s); loading another task's dataset against the SAME
    engine costs almost nothing. The cache is keyed by the full engine config
    so tasks from different constraint domains still get their own engine.
    """
    from vllm_gr.entrypoints.gr import GRLLM

    key = json.dumps(engine_args, sort_keys=True, default=str)
    llm = _ENGINE_CACHE.get(key)
    if llm is None:
        # Keep at most one engine alive: a second constraint domain needs its
        # own KV cache, and two resident engines overflow the device. Tasks of
        # one domain are contiguous in the CI, so this only bites when a caller
        # mixes domains in a single --tasks list.
        for stale_key, stale_llm in list(_ENGINE_CACHE.items()):
            _shutdown_engine(stale_llm)
            del _ENGINE_CACHE[stale_key]
        llm = GRLLM(**engine_args)
        _ENGINE_CACHE[key] = llm
    return llm


def _selected_tasks(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "tasks", None)
    if raw:
        tasks = [task.strip() for task in raw.split(",") if task.strip()]
    elif getattr(args, "task", None):
        tasks = [args.task]
    else:
        raise ValueError("one of --task or --tasks is required")
    unknown = [task for task in tasks if task not in RECOMMENDATION_TASKS]
    if unknown:
        raise ValueError(f"tasks must be a subset of {RECOMMENDATION_TASKS}: {unknown}")
    if len(set(tasks)) != len(tasks):
        raise ValueError("--tasks contains duplicate task names")
    return tasks


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.task not in RECOMMENDATION_TASKS:
        raise ValueError(f"task must be one of {RECOMMENDATION_TASKS}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.beam_width <= 0 or args.beam_width > 256:
        raise ValueError("--beam-width must be in [1, 256]")
    prepared = _load_prepared_manifest(args.prepared_manifest)
    constraint_args, constraint_identity = _resolve_constraint(
        prepared,
        args.task,
        args.constraint_mode,
        args.artifact_backend,
        args.beam_width,
    )
    runner_git_revision = _git_revision()
    implementation_revision = args.implementation_revision or runner_git_revision
    source_parquet = (Path(args.dataset_path) / args.task / f"{args.task}_test.parquet").resolve()
    source_sha256 = sha256_file(source_parquet)
    run_identity = {
        "model": args.model,
        "revision": args.revision,
        "implementation_revision": implementation_revision,
        "execution_variant": args.execution_variant,
        "source_sha256": source_sha256,
        "execution": args.execution,
        "constraint": constraint_identity,
        "beam_width": args.beam_width,
        "temperature": args.temperature,
        "seed": args.seed,
        "evaluation_mode": args.evaluation_mode,
        "max_model_len": args.max_model_len,
        "config_env": _config_env_snapshot(),
    }

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    loader = DataLoaderWrapper(tokenizer=tokenizer, data_dir=args.dataset_path)
    sample_ids = None
    if args.sample_manifest:
        task_samples = load_sample_manifest(
            args.sample_manifest,
            dataset_path=args.dataset_path,
        )
        if args.task not in task_samples:
            raise ValueError(f"sample manifest has no IDs for task {args.task!r}")
        sample_ids = task_samples[args.task]
    sample_size: int | str | None = None
    if sample_ids is None:
        sample_size = args.num_prompts if args.num_prompts is not None else "full"
    dataset = loader.load_data(
        task_name=args.task,
        split="test",
        sample_size=sample_size,
        sample_ids=sample_ids,
        cache_sample=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"{args.task}.jsonl"
    summary_path = output_dir / f"{args.task}.summary.json"
    if args.overwrite:
        records_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    elif records_path.exists() and not args.resume:
        raise ValueError(f"result already exists: {records_path}; use --resume or --overwrite")

    accumulator = RecommendationMetricAccumulator(
        args.task,
        args.dataset_path,
        evaluation_mode=args.evaluation_mode,
        # Record every k so the gate can run a per-sample paired test over the
        # full pass@k / position1_pass@k / recall@k grid.
        k_values=tuple(range(1, 33)),
    )
    completed: set[str] = set()
    invalid_samples = 0
    if records_path.exists():
        for record in iter_jsonl(records_path):
            if record.get("task") != args.task:
                raise ValueError(f"resume file contains a different task: {record.get('task')}")
            if record.get("run_identity") != run_identity:
                raise ValueError("resume file was produced by a different model/run configuration")
            sample_id = str(record["sample_id"])
            if sample_id in completed:
                raise ValueError(f"resume file contains duplicate sample ID {sample_id}")
            if sample_id not in dataset:
                raise ValueError(f"resume file sample ID {sample_id} is absent from selected data")
            if record.get("prompt_sha256") != prompt_digest(dataset[sample_id]["prompt"]):
                raise ValueError(f"resume prompt fingerprint mismatch for sample {sample_id}")
            completed.add(sample_id)
            accumulator.add_metrics(record["metrics"])
            contract = record.get("contract", {})
            invalid_samples += int(
                not contract.get("beam_count_ok", False)
                or contract.get("empty_count", 0) > 0
                or contract.get("invalid_sid_count", 0) > 0
            )

    pending_ids = [sample_id for sample_id in dataset if sample_id not in completed]
    LOGGER.info(
        "OneRec %s: selected=%d completed=%d pending=%d mode=%s",
        args.task,
        len(dataset),
        len(completed),
        len(pending_ids),
        args.constraint_mode,
    )

    llm = None
    started = time.perf_counter()
    try:
        if pending_ids and args.execution == "offline":
            engine_args = {
                "model": args.model,
                "revision": args.revision,
                "trust_remote_code": True,
                "max_logprobs": args.beam_width * 2,
                "beam_max_width": args.beam_width,
                "attention_config": {"backend": "CUSTOM"},
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "seed": args.seed,
                **constraint_args,
            }
            # Canonical config owns beam sizing and cannot be combined with
            # legacy beam arguments.  The canonical Beam V1 engine also
            # requires the async-scheduling pipeline and one session per
            # beam_search_v1 call (the ABBA harness boots the same shape).
            if args.constraint_mode == "artifact":
                engine_args.pop("beam_max_width")
                engine_args.pop("attention_config")
                engine_args.update(
                    max_logprobs=args.beam_width,
                    async_scheduling=True,
                    max_num_seqs=1,
                    enable_prefix_caching=True,
                    enforce_eager=False,
                )
                if args.max_model_len is not None:
                    engine_args["max_model_len"] = args.max_model_len
            llm = _cached_engine(engine_args)

        for id_batch in _chunks(pending_ids, args.batch_size):
            prompts = [dataset[sample_id]["prompt"] for sample_id in id_batch]
            if args.execution == "offline":
                normalized = _offline_batch(
                    llm,
                    tokenizer,
                    prompts,
                    args.beam_width,
                    args.temperature,
                )
            else:
                normalized = _online_batch(
                    prompts=prompts,
                    model=args.model,
                    host=args.host,
                    port=args.port,
                    endpoint=args.endpoint,
                    api_key=args.api_key,
                    beam_width=args.beam_width,
                    temperature=args.temperature,
                    timeout=args.request_timeout,
                )

            records = []
            for sample_id, output in zip(id_batch, normalized):
                item = dataset[sample_id]
                generations = output["generations"]
                contract = validate_generations(generations, args.beam_width)
                invalid = (
                    not contract["beam_count_ok"]
                    or contract["empty_count"] > 0
                    or contract["invalid_sid_count"] > 0
                )
                invalid_samples += int(invalid)
                metrics = accumulator.add(generations, item["ground_truth"], item["metadata"])
                records.append(
                    {
                        "format_version": 1,
                        "task": args.task,
                        "sample_id": sample_id,
                        "prompt_sha256": prompt_digest(item["prompt"]),
                        "run_identity": run_identity,
                        **output,
                        "contract": contract,
                        "metrics": metrics,
                    }
                )
            write_jsonl(records_path, records)
            completed_count = accumulator.summary()["total_samples"]
            if completed_count == len(dataset) or completed_count % args.log_every == 0:
                LOGGER.info("Completed %d/%d samples", completed_count, len(dataset))
    finally:
        if llm is not None and not getattr(args, "_keep_engine", False):
            _shutdown_engine(llm)

    elapsed = time.perf_counter() - started
    summary = {
        "format_version": 1,
        "task": args.task,
        "model": args.model,
        "revision": args.revision,
        "git_revision": runner_git_revision,
        "implementation_revision": implementation_revision,
        "execution_variant": args.execution_variant,
        "python": sys.version,
        "platform": platform.platform(),
        "execution": args.execution,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "source_parquet": str(source_parquet),
        "source_sha256": source_sha256,
        "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
        "sample_manifest_sha256": (
            sha256_file(args.sample_manifest) if args.sample_manifest else None
        ),
        "constraint": constraint_identity,
        "beam_width": args.beam_width,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "config_env": _config_env_snapshot(),
        "selected_samples": len(dataset),
        "invalid_samples": invalid_samples,
        "elapsed_time": elapsed,
        "metrics": accumulator.summary(),
    }
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.strict and invalid_samples:
        raise RuntimeError(f"{invalid_samples} samples violated the OneRec output contract")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=RECOMMENDATION_TASKS)
    parser.add_argument(
        "--tasks",
        help="Comma-separated tasks to run in one process (one engine per constraint domain).",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--implementation-revision",
        help="Git revision of the inference implementation (not the model revision).",
    )
    parser.add_argument(
        "--execution-variant",
        default="default",
        help="Stable label for the server execution path, such as graph or eager-worker.",
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--constraint-mode",
        choices=["trie", "triples", "artifact"],
        required=True,
    )
    parser.add_argument("--artifact-backend", choices=["cuda", "npu", "cpu"], default="cuda")
    parser.add_argument("--execution", choices=["offline", "online"], default="offline")
    parser.add_argument("--sample-manifest")
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--evaluation-mode", choices=["sid", "pid", "both"], default="both")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Serving context length for offline engines (long OneRec prompts "
        "exceed the model config's default context)",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    try:
        tasks = _selected_tasks(args)
        if len(tasks) == 1:
            args.task = tasks[0]
            run(args)
        else:
            # Keep the engine alive between tasks, then shut every domain's
            # engine down once at the end.
            args._keep_engine = True
            try:
                for task in tasks:
                    args.task = task
                    run(args)
            finally:
                for llm in list(_ENGINE_CACHE.values()):
                    _shutdown_engine(llm)
                _ENGINE_CACHE.clear()
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
