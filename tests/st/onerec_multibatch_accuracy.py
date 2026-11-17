# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Real video accuracy for beam_search_v1, using the system benchmark evaluator.

Run as a module in a fresh process. The runner flag selects ModelRunner;
both runners use the same public Beam V1 protocol and CUDA constraint table.
All batch sizes use the same ordered samples, without prompt resizing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def save(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def fingerprint(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    return {
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def run(args: argparse.Namespace) -> int:
    # Set runner selection before importing vLLM or GR in this process.
    os.environ.update(
        USE_TF="0",
        VLLM_USE_V2_MODEL_RUNNER=str(int(args.runner == "v2")),
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )

    from transformers import AutoTokenizer

    from benchmarks.open_one_rec.open_one_rec_loader import (
        TASK_REGISTRY,
        DataLoaderWrapper,
    )
    from examples.offline_inference.beam_search.offline_beam_search_v1 import (
        build_artifact,
        load_triples,
    )
    from tests.st.model_runner_policy_isolation import check_idle, startup_expectations
    from tests.st.multibatch_accuracy_checks import compare_output, validate_output
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "started_utc": datetime.now(UTC).isoformat(),
        "runner": args.runner,
        "command": sys.argv,
        "status": "preparing",
        "cases": {},
    }

    def persist() -> None:
        save(out / "results.json", result)

    persist()
    model = Path(args.model).resolve()
    triples = Path(args.triples).resolve()
    task = TASK_REGISTRY["video"]
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    dataset = DataLoaderWrapper(tokenizer=tokenizer, data_dir=args.dataset_path).load_data(
        task_name="video",
        split=task.config.get("splits", ["test"])[0],
        sample_size=args.num_prompts,
    )
    samples = []
    for sample_id, item in dataset.items():
        samples.append({"sample_id": str(sample_id), **item})
    ids = [tokenizer.encode(s["prompt"], add_special_tokens=False) for s in samples]
    lengths = [len(row) + 1 for row in ids]
    assert len(samples) == args.num_prompts
    assert max(lengths) + 3 <= 8192
    assert all(len(samples) % b == 0 for b in args.batches), "use complete batches"
    max_total = max(
        sum(lengths[i : i + b]) for b in args.batches for i in range(0, len(samples), b)
    )
    # Keep B=1/2/4 comparable, even when running only a control subset.
    capture_max = max(
        8192, ((max(sum(lengths[i : i + 4]) for i in range(0, len(samples), 4)) + 127) // 128) * 128
    )
    capture_sizes = [1, 2, 4, 8, 16, 32, 64, *range(128, capture_max + 1, 128)]
    save(out / "samples.json", samples)
    save(out / "prompt_tokens.json", ids)
    result["inputs"] = {
        str(p): fingerprint(p)
        for p in (
            Path(args.catalog),
            triples,
            Path(args.dataset_path) / "video" / f"video_test_sample_{args.num_prompts}.parquet",
            model / "config.json",
            model / "tokenizer.json",
            out / "samples.json",
        )
    }
    result["prompt_lengths"] = {
        "includes_sid_begin": True,
        "min": min(lengths),
        "max": max(lengths),
        "mean": statistics.mean(lengths),
        "max_batch_total": max_total,
        "all": lengths,
    }
    result["task_config"] = task.config
    result["cache_policy"] = (
        "reset once before each B sweep; retain normal prefix caching within sweep"
    )
    result["environment"] = {
        k: os.environ.get(k)
        for k in (
            "CUDA_VISIBLE_DEVICES",
            "HF_ENDPOINT",
            "VLLM_USE_V2_MODEL_RUNNER",
            "VLLM_ENABLE_V1_MULTIPROCESSING",
            "VLLM_WORKER_MULTIPROC_METHOD",
        )
    }
    result["packages"] = {
        k: importlib.metadata.version(k)
        for k in ("torch", "vllm", "vllm-gr", "transformers", "datasets")
    }
    result["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_paths = sorted(
        set(Path("vllm_gr").rglob("*.py"))
        | set(Path("benchmarks/open_one_rec").rglob("*.py"))
        | {
            Path(__file__),
            Path("tests/st/model_runner_policy_isolation.py"),
            Path("tests/st/multibatch_accuracy_checks.py"),
            Path(".github/benchmark-gate.yaml"),
        }
    )
    save(out / "source.json", {str(p): fingerprint(p) for p in source_paths})
    (out / "working_tree.diff").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD", "--", "vllm_gr", "benchmarks/open_one_rec"])
    )
    persist()

    artifact = args.artifact.resolve()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    manifest = artifact.with_suffix(".manifest.json")
    artifact_inputs = {
        "triples": fingerprint(triples),
        "tokenizer": fingerprint(model / "tokenizer.json"),
        "config": fingerprint(model / "config.json"),
    }
    if artifact.exists() and manifest.exists():
        artifact_info = json.loads(manifest.read_text())
        assert artifact_info["inputs"] == artifact_inputs
        assert artifact_info["file"] == fingerprint(artifact)
    else:
        print("Compiling full real catalog constraint artifact", flush=True)
        digest, tokenizer_digest = build_artifact(model, triples, artifact)
        artifact_info = {
            "inputs": artifact_inputs,
            "file": fingerprint(artifact),
            "artifact_digest": digest,
            "tokenizer_digest": tokenizer_digest,
        }
        save(manifest, artifact_info)
    legal = {tuple(tokenizer.convert_tokens_to_ids(row)) for row in load_triples(triples)}
    result["catalog_legal_triples"] = len(legal)
    config = {
        "prefill_kv_bound": 8192,
        "beam_graph_enabled": True,
        "worker_decision": True,
        "beam_width": [128] * 3,
        "max_decode_steps": 3,
        "constraint_mode": "table",
        "constraint_table": {
            "backend": "cuda",
            "max_top_k": 128,
            "path": str(artifact),
            "format": "constraint_table_v1",
            "digests": {
                "artifact": artifact_info["artifact_digest"],
                "tokenizer": artifact_info["tokenizer_digest"],
            },
        },
        "candidate_width": [128] * 3,
    }
    kwargs: dict[str, Any] = {
        "model": str(model),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "seed": 0,
        "async_scheduling": True,
        "max_num_seqs": 4,
        "max_model_len": 8192,
        "max_num_batched_tokens": capture_max,
        "max_logprobs": 256,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.7,
        "enforce_eager": False,
        "compilation_config": {"cudagraph_capture_sizes": capture_sizes},
        "worker_extension_cls": "tests.st.model_runner_policy_isolation.PolicyWorkerExtension",
    }
    expected_rows, expected_batches = startup_expectations(
        runner=args.runner,
        config=config,
        max_num_seqs=kwargs["max_num_seqs"],
        enforce_eager=kwargs["enforce_eager"],
    )
    gen = task.config["generation_config"]
    params = BeamSearchParams(
        beam_width=gen["num_beams"],
        max_tokens=gen["max_new_tokens"],
        temperature=gen["temperature"],
        ignore_eos=True,
        begin_token="<|sid_begin|>",
        end_token="<|sid_end|>",
        include_stop_str_in_output=True,
        length_penalty=1.0,
    )
    result.update(
        engine_kwargs=kwargs,
        gr_config=config,
        beam_params={
            name: getattr(params, name)
            for name in (
                "beam_width",
                "max_tokens",
                "temperature",
                "ignore_eos",
                "begin_token",
                "end_token",
                "include_stop_str_in_output",
                "length_penalty",
            )
        },
        status="initializing",
    )
    thresholds = {
        metric: float(os.environ[name])
        for metric, name in (
            ("pass@1", "PASS_AT_1_THRESHOLD"),
            ("pass@32", "PASS_AT_32_THRESHOLD"),
            ("recall@32", "RECALL_AT_32_THRESHOLD"),
        )
    }
    result["thresholds"] = thresholds
    persist()
    engine_start = time.perf_counter()
    llm = GRLLM(**kwargs, attention_config={"backend": "CUSTOM"}, gr=config)
    result["startup_seconds"] = time.perf_counter() - engine_start
    begin, end = tokenizer.convert_tokens_to_ids(["<|sid_begin|>", "<|sid_end|>"])
    addresses = None
    references = None

    def snapshot() -> dict[str, Any]:
        nonlocal addresses
        value: dict[str, Any] = llm.collective_rpc("policy_snapshot")[0]
        check_idle(value)
        assert value["backend_rows"] == expected_rows
        if args.runner == "v2":
            if addresses is None:
                addresses = value["addresses"]
            assert value["addresses"] == addresses
            assert value["counts"].get("capture", 0) == 0
        return value

    try:
        llm.collective_rpc("policy_observe")
        result["initial"] = snapshot()
        actual_class = result["initial"]["runner_class"]
        assert (".gpu.model_runner." in actual_class) == (args.runner == "v2"), actual_class
        assert result["initial"]["decode_batches"] == expected_batches
        for b in args.batches:
            assert llm.reset_prefix_cache()
            before = snapshot()
            rows, batch_records, evaluator_samples = [], [], {}
            result["status"] = f"running B={b}"
            persist()
            for offset in range(0, len(samples), b):
                tick = time.perf_counter()
                outputs = llm.beam_search_v1(
                    [{"prompt": s["prompt"]} for s in samples[offset : offset + b]],
                    params,
                    use_tqdm=False,
                    output_mode="full",
                )
                assert len(outputs) == b
                for index, output in enumerate(outputs, offset):
                    sample = samples[index]
                    candidates, generations = [], []
                    for sequence in output.sequences:
                        tokens = sequence.tokens
                        assert tokens[: len(ids[index])] == ids[index]
                        tail = tokens[len(ids[index]) :]
                        assert len(tail) == 5 and tail[0] == begin and tail[-1] == end
                        assert sequence.text.startswith(sample["prompt"])
                        text = sequence.text[len(sample["prompt"]) :]
                        assert text == tokenizer.decode(tail, skip_special_tokens=False)
                        candidates.append([tail[1:-1], sequence.cum_logprob])
                        generations.append(text)
                    validate_output(candidates, width=128, steps=3, legal=legal)
                    rows.append(
                        {
                            "sample_id": sample["sample_id"],
                            "candidates": candidates,
                            "generations": generations,
                        }
                    )
                    evaluator_samples[sample["sample_id"]] = {**sample, "generations": generations}
                elapsed = time.perf_counter() - tick
                after = snapshot()
                queries = after["queries"][len(before["queries"]) :]
                assert len(queries) == 1 and len(queries[0]) == b, queries
                counts = {k: v - before["counts"].get(k, 0) for k, v in after["counts"].items()}
                assert counts["replay"] == 2
                assert counts["eager"] == 0
                full = (
                    counts["prefill_replay"]
                    if args.runner == "v1"
                    else after["graphs"]["prefill_graph_full_dispatches"]
                    - before["graphs"]["prefill_graph_full_dispatches"]
                )
                batch_records.append(
                    {
                        "offset": offset,
                        "batch_size": b,
                        "queries": queries,
                        "prefill_full": full,
                        "counts": counts,
                        "api_and_validation_seconds": elapsed,
                    }
                )
                before = after
                if (offset + b) % 20 == 0 or offset + b == len(samples):
                    print(f"{args.runner} B={b}: {offset + b}/{len(samples)} samples", flush=True)
            save(out / f"b{b}_outputs.json", rows)
            save(out / f"b{b}_batches.json", batch_records)
            evaluator = task.evaluator_class(
                samples=evaluator_samples,
                task_name="video",
                task_config=task.config,
                data_dir=args.dataset_path,
                predictions_dir=None,
                debug=False,
            )
            metrics, per_sample = evaluator.evaluate()
            save(out / f"b{b}_per_sample_metrics.json", per_sample)
            gates = {
                k: {"value": metrics[k], "minimum": v, "passed": metrics[k] >= v}
                for k, v in thresholds.items()
            }
            case = {
                "samples": len(rows),
                "batches": len(batch_records),
                "metrics": metrics,
                "business_gates": gates,
                "business_passed": all(g["passed"] for g in gates.values()),
                "prefill_full_batches": sum(r["prefill_full"] for r in batch_records),
                "final_snapshot": before,
            }
            if b == 1:
                references = rows
            elif references is not None:
                comparisons = []
                for ref, row in zip(references, rows, strict=True):
                    assert ref["sample_id"] == row["sample_id"]
                    comparisons.append(
                        {
                            "sample_id": row["sample_id"],
                            **compare_output(ref["candidates"], row["candidates"]),
                        }
                    )
                save(out / f"b{b}_vs_b1.json", comparisons)
                case["vs_b1"] = {
                    **{
                        key: sum(c[key] for c in comparisons)
                        for key in (
                            "exact",
                            "ordered_candidates_equal",
                            "top1_equal",
                            "candidate_gate_passed",
                        )
                    },
                    "mean_candidate_overlap": statistics.mean(
                        c["candidate_overlap"] for c in comparisons
                    ),
                    "min_candidate_overlap": min(c["candidate_overlap"] for c in comparisons),
                    "max_score_abs_error": max(
                        (
                            c["max_score_abs_error"]
                            for c in comparisons
                            if c["max_score_abs_error"] is not None
                        ),
                        default=None,
                    ),
                }
            result["cases"][str(b)] = case
            persist()
            print(
                json.dumps(
                    {
                        "runner": args.runner,
                        "batch": b,
                        **case["business_gates"],
                        "vs_b1": case.get("vs_b1"),
                    }
                ),
                flush=True,
            )
        result["status"] = "completed"
        result["all_business_gates_passed"] = all(
            c["business_passed"] for c in result["cases"].values()
        )
        llm.collective_rpc("policy_stop")
    finally:
        llm.shutdown()
        result["finished_utc"] = datetime.now(UTC).isoformat()
        result["total_seconds"] = time.perf_counter() - started
        persist()
    return 0 if result["all_business_gates_passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("v1", "v2"), default="v2")
    parser.add_argument("--batches", type=int, nargs="+", choices=(1, 2, 4), default=[1, 2, 4])
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_GR_TEST_MODEL"),
        required="VLLM_GR_TEST_MODEL" not in os.environ,
    )
    parser.add_argument(
        "--dataset-path",
        default=os.environ.get("DATASET_PATH"),
        required="DATASET_PATH" not in os.environ,
    )
    parser.add_argument(
        "--catalog",
        default=os.environ.get("CATALOG_PATH"),
        required="CATALOG_PATH" not in os.environ,
    )
    parser.add_argument("--triples", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
