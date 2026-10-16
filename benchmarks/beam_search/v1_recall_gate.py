# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure fixed-sample OneRec Recall@32 with native GPU Beam V1 batches."""

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
logger = logging.getLogger(__name__)
TASKS = ("video", "ad", "product", "label_cond", "interactive")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_utilization_for_budget(free_bytes, total_bytes, requested=None):
    """vLLM's utilization is a fraction of total VRAM, not currently free VRAM."""
    if not 0 < free_bytes <= total_bytes:
        raise ValueError("invalid or exhausted CUDA memory snapshot")
    if requested is not None:
        if not 0 < requested < 1:
            raise ValueError("GPU memory utilization must be in (0, 1)")
        if math.ceil(total_bytes * requested) > free_bytes:
            raise ValueError(
                f"Requested GPU budget {total_bytes * requested / 2**30:.2f} GiB "
                f"exceeds free memory {free_bytes / 2**30:.2f} GiB. "
                "Unset GPU_MEMORY_UTILIZATION to use the automatic budget, "
                "or choose a lower value."
            )
        return requested
    # OneRec-1.7B does not need most of a 140 GiB card for B<=4. Cap at 30%
    # and leave room for CUDA/NCCL initialization and changes in other jobs.
    headroom = max(4 * 2**30, free_bytes * 0.1)
    utilization = math.floor(min(0.3, (free_bytes - headroom) / total_bytes) * 1000) / 1000
    if utilization <= 0:
        raise ValueError("Not enough free GPU memory after reserving startup headroom")
    return utilization


def resolve_memory_utilization(requested):
    import torch

    # Logical cuda:0 follows CUDA_VISIBLE_DEVICES, including UUID selections.
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    utilization = memory_utilization_for_budget(free_bytes, total_bytes, requested)
    logger.info(
        "CUDA_VISIBLE_DEVICES=%s cuda:0 (%s): free=%.2f GiB total=%.2f GiB; "
        "gpu_memory_utilization=%.3f budget=%.2f GiB (%s)",
        os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        torch.cuda.get_device_name(0),
        free_bytes / 2**30,
        total_bytes / 2**30,
        utilization,
        total_bytes * utilization / 2**30,
        "auto" if requested is None else "explicit",
    )
    return utilization


def batch_ranges(count, batch_size, supported_sizes=(1, 2, 4)):
    """Cover every sample once, using smaller supported batches for the tail."""
    if batch_size not in supported_sizes:
        raise ValueError(f"unsupported batch size {batch_size}: {supported_sizes}")
    start = 0
    while start < count:
        size = max(size for size in supported_sizes if size <= min(batch_size, count - start))
        yield start, start + size
        start += size


def infer_batch(llm, tokenizer, token_rows, params):
    """Submit one native batch and decode only generated SID tokens."""
    outputs = llm.beam_search_v1([{"prompt_token_ids": row} for row in token_rows], params)
    if len(outputs) != len(token_rows):
        raise RuntimeError(f"expected {len(token_rows)} outputs, got {len(outputs)}")
    predictions = []
    for tokens, output in zip(token_rows, outputs, strict=True):
        beams = output.sequences
        if len(beams) != params.beam_width:
            raise RuntimeError(f"expected {params.beam_width} beams, got {len(beams)}")
        suffixes = []
        for beam in beams:
            full_tokens = list(beam.tokens)
            if full_tokens[: len(tokens)] != tokens:
                raise RuntimeError("output prompt/order mismatch within the batch")
            suffix = full_tokens[len(tokens) :]
            if len(suffix) != params.max_tokens:
                raise RuntimeError(f"unexpected SID output length: {len(suffix)}")
            suffixes.append(suffix)
        predictions.append(
            tokenizer.batch_decode(
                suffixes, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        )
    return predictions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--catalog-path", type=Path, required=True, help="Constraint triples JSON")
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task-type", choices=TASKS)
    task_group.add_argument("--task-types", choices=TASKS, nargs="+")
    parser.add_argument("--num-prompts", type=int, help="Omit to run the full test Parquet")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--n", type=int, default=128, help="Beam width")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of total VRAM; default auto: cap at 0.3 and reserve free-memory headroom",
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--validate-only", action="store_true", help="Load/tokenize data without an engine"
    )
    args = parser.parse_args()
    if args.task_types is not None and len(set(args.task_types)) != len(args.task_types):
        parser.error("--task-types must not contain duplicates")
    if args.num_prompts is not None and args.num_prompts <= 0:
        parser.error("--num-prompts must be positive")
    if not 1 <= args.n <= 256:
        parser.error("--n must be in [1, 256]")
    if args.max_model_len <= 5 or (
        args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization < 1
    ):
        parser.error("invalid model length or GPU memory utilization")
    for path in (args.model, args.dataset_path, args.catalog_path):
        if not path.exists():
            parser.error(f"missing path: {path}")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import pandas as pd
    import pyarrow.parquet as pq
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from benchmarks.open_one_rec.open_one_rec_loader import (
        TASK_REGISTRY,
        DataLoaderWrapper,
    )

    tasks = args.task_types or [args.task_type]
    grouped = args.task_types is not None
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    loader = DataLoaderWrapper(tokenizer=tokenizer, data_dir=str(args.dataset_path))
    prepared = []
    for task in tasks:
        task_reg = TASK_REGISTRY[task]
        task_config = copy.deepcopy(task_reg.config)
        gen_config = task_config["generation_config"]
        gen_config.update(
            num_beams=args.n, num_return_sequences=args.n, temperature=args.temperature
        )
        split = task_config.get("splits", ["test"])[0]
        parquet = args.dataset_path / task / f"{task}_{split}.parquet"
        full_count = pq.read_metadata(parquet).num_rows
        expected_count = args.num_prompts if args.num_prompts is not None else full_count
        if expected_count > full_count:
            raise ValueError(f"requested {expected_count} samples but {task} has {full_count}")
        # Select original row IDs; sampled parquet files discard their source indices.
        source_ids = (
            pd.DataFrame(index=range(full_count))
            .sample(n=expected_count, random_state=42)
            .index.tolist()
            if expected_count < full_count
            else list(range(full_count))
        )
        dataset = loader.load_data(task, split=split, sample_ids=source_ids)
        if len(dataset) != expected_count:
            raise RuntimeError(
                f"loaded {len(dataset)} valid {task} samples, expected {expected_count}"
            )
        sample_ids = list(dataset)
        token_rows = [
            tokenizer.encode(dataset[sid]["prompt"], add_special_tokens=False)
            for sid in tqdm(sample_ids, desc=f"Tokenize {task}")
        ]
        max_tokens = gen_config["max_new_tokens"]
        if max_tokens != 5:
            raise ValueError("this recommendation runner expects three SID tokens plus begin/end")
        tailcut_count = 0
        for index, (sid, tokens) in enumerate(zip(sample_ids, token_rows, strict=True)):
            if task == "interactive" and len(tokens) + max_tokens > args.max_model_len:
                token_rows[index] = tokens[-(args.max_model_len - max_tokens) :]
                tokens = token_rows[index]
                tailcut_count += 1
            if not tokens or len(tokens) + max_tokens > args.max_model_len:
                raise ValueError(
                    f"{task} sample {sid}: {len(tokens)} prompt tokens plus {max_tokens} "
                    f"output tokens exceed model limit {args.max_model_len}"
                )
        batches = list(batch_ranges(len(sample_ids), args.batch_size))
        histogram = dict(sorted(Counter(end - start for start, end in batches).items()))
        # The scheduled prefill holds all B prompt rows and one SID begin per row.
        prefill_budget = max(
            sum(len(row) + 1 for row in token_rows[start:end]) for start, end in batches
        )
        result_dir = args.result_dir / task if grouped else args.result_dir
        prepared.append(
            {
                "task": task,
                "registry": task_reg,
                "config": task_config,
                "parquet": parquet,
                "source_ids": source_ids,
                "dataset": dataset,
                "sample_ids": sample_ids,
                "token_rows": token_rows,
                "batches": batches,
                "histogram": histogram,
                "prefill_budget": prefill_budget,
                "tailcut_count": tailcut_count,
                "result_dir": result_dir,
            }
        )
        logger.info(
            "task=%s samples=%d batch_size=%d batch_counts=%s max_prompt_tokens=%d "
            "prefill_budget=%d",
            task,
            len(sample_ids),
            args.batch_size,
            histogram,
            max(map(len, token_rows)),
            prefill_budget,
        )
    max_prefill_budget = max(item["prefill_budget"] for item in prepared)
    custom_buckets = os.getenv("VLLM_GR_PREFILL_GRAPH_BUCKETS")
    if custom_buckets:
        configured = [int(value) for value in custom_buckets.split(",")]
        used = [bucket for bucket in configured if bucket < max_prefill_budget]
        covering = next((bucket for bucket in configured if bucket >= max_prefill_budget), None)
        if covering is None:
            raise ValueError(f"prefill graph buckets do not cover {max_prefill_budget} tokens")
        used.append(covering)
        os.environ["VLLM_GR_PREFILL_GRAPH_BUCKETS"] = ",".join(map(str, used))
        logger.info("CI prefill graph buckets: %s", used)
    engine_config = {
        "async_scheduling": True,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": False,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": max(args.max_model_len, max_prefill_budget),
        "max_num_seqs": max(args.n, args.batch_size),
        "max_logprobs": args.n * 2,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": False,
        "seed": 0,
    }
    logger.info("engine_config=%s", engine_config)
    if args.validate_only:
        print("Data, prompt lengths and batch plan validated; no GPU inference started.")
        return

    from examples.offline_inference.beam_search.offline_beam_search_v1 import (
        build_artifact,
    )
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams
    from vllm_gr.v1.beam.batch_contracts import SUPPORTED_BEAM_BATCH_SIZES

    for item in prepared:
        if any(size not in SUPPORTED_BEAM_BATCH_SIZES for size in item["histogram"]):
            raise ValueError(
                f"runtime supports {SUPPORTED_BEAM_BATCH_SIZES}, planned {item['histogram']}"
            )
        item["result_dir"].mkdir(parents=True, exist_ok=True)
    params = BeamSearchParams(
        beam_width=args.n,
        max_tokens=max_tokens,
        temperature=args.temperature,
        ignore_eos=True,
        begin_token="<|sid_begin|>",
        end_token="<|sid_end|>",
        include_stop_str_in_output=True,
        length_penalty=1.0,
    )
    with tempfile.TemporaryDirectory(prefix="onerec-multibatch-") as directory:
        artifact = Path(directory) / "constraints.safetensors"
        artifact_digest, tokenizer_digest = build_artifact(
            args.model.resolve(), args.catalog_path.resolve(), artifact
        )
        config = {
            "schema_version": 1,
            "attention_backend": "CUSTOM",
            "beam": {
                "execution_mode": "v1",
                "graph_enabled": True,
                "worker_decision": True,
                "max_width": args.n,
                "max_decode_steps": 3,
            },
            "constraint_table": {
                "enabled": True,
                "backend": "cuda",
                "max_top_k": args.n,
                "path": str(artifact),
                "format": "constraint_table_v1",
                "artifact_digest": artifact_digest,
                "tokenizer_digest": tokenizer_digest,
            },
        }
        engine_config["gpu_memory_utilization"] = resolve_memory_utilization(
            args.gpu_memory_utilization
        )
        init_start = time.perf_counter()
        with GRLLM(
            model=str(args.model), trust_remote_code=True, vllm_gr_config=config, **engine_config
        ) as llm:
            logger.info(
                "Initialized one GRLLM for %s in %.1f s", tasks, time.perf_counter() - init_start
            )
            for item in prepared:
                task = item["task"]
                task_start = time.perf_counter()
                samples = {}
                elapsed = 0.0
                for start, end in tqdm(item["batches"], desc=f"{task} B={args.batch_size}"):
                    begin = time.perf_counter()
                    predictions = infer_batch(llm, tokenizer, item["token_rows"][start:end], params)
                    elapsed += time.perf_counter() - begin
                    for sid, generations in zip(
                        item["sample_ids"][start:end], predictions, strict=True
                    ):
                        original = item["dataset"][sid]
                        samples[sid] = {
                            "prompt": original["prompt"],
                            "generations": generations,
                            "ground_truth": original["ground_truth"],
                            "metadata": original.get("metadata", {}),
                        }

                # Preserve the existing per-task evaluator and SID/PID definitions.
                evaluator = item["registry"].evaluator_class(
                    samples=samples,
                    task_name=task,
                    task_config=item["config"],
                    data_dir=str(args.dataset_path),
                    predictions_dir=str(item["result_dir"]),
                    debug=False,
                )
                metrics, _ = evaluator.evaluate()
                repo = Path(__file__).resolve().parents[2]
                metrics.update(
                    model=str(args.model),
                    task=task,
                    beam_width=args.n,
                    temperature=args.temperature,
                    batch_size=args.batch_size,
                    sample_ids=item["source_ids"],
                    source_sha256=sha256_file(item["parquet"]),
                    interactive_tailcut_count=item["tailcut_count"],
                    batch_counts=item["histogram"],
                    api="GRLLM.beam_search_v1",
                    commit=subprocess.check_output(
                        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
                    ).strip(),
                    dataset_path=str(args.dataset_path),
                    catalog_path=str(args.catalog_path),
                    engine_config=engine_config,
                    prefill_graph_buckets=os.getenv("VLLM_GR_PREFILL_GRAPH_BUCKETS"),
                    prefill_graph_capture_num_reqs=os.getenv(
                        "VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS"
                    ),
                    v1_startup_capture_batch_size=os.getenv(
                        "VLLM_GR_V1_STARTUP_CAPTURE_BATCH_SIZE"
                    ),
                    elapsed_time=elapsed,
                    time_per_sample=elapsed / len(samples),
                    artifact_digest=artifact_digest,
                    tokenizer_digest=tokenizer_digest,
                )
                output = item["result_dir"] / (
                    f"{args.model.name}_{task}_bw{args.n}_bs{args.batch_size}_{len(samples)}.json"
                )
                output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
                (item["result_dir"] / "wall_seconds.txt").write_text(
                    str(round(time.perf_counter() - task_start)) + "\n"
                )
                logger.info("Saved %s metrics to %s", task, output)


if __name__ == "__main__":
    main()
