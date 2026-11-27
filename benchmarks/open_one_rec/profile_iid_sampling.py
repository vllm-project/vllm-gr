# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Measure completed prefill to materialized SID bags on RecIF Video (one CUDA GPU)."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, GenerationConfig

from benchmarks.beam_search.measure_onerec_recall import sha256_file
from benchmarks.open_one_rec.open_one_rec_loader import DataLoaderWrapper
from vllm_gr import GRLLM
from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest


class IIDProfileWorker:
    """Worker extension for the benchmark engine, including its unmerged control."""

    def iid_profile_arm(self, shared: bool) -> list[int]:
        from vllm_gr.v1.worker.cuda import gpu_iid_sampling as iid
        from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import GPUBeamStageRunner

        if not hasattr(self, "_iid_marks"):
            self._iid_marks = []
            execute = GPUBeamStageRunner.execute
            sample = iid.GPUIIDSampling.sample
            group = iid._group_draws

            def execute_with_boundary(runner, *args, **kwargs):
                output = execute(runner, *args, **kwargs)
                if runner.previous.stage_index == 0:
                    torch.cuda.current_stream().synchronize()
                    self._iid_marks.append(time.monotonic_ns())
                return output

            def keep_every_draw(parents, columns, degree):
                return parents, columns, torch.ones_like(parents)

            def select(runner, logits, step, temperature):
                # Keep the same CDF, draws, KV and result packing in both arms.
                # Only coalescing before the following forward changes.
                iid._group_draws = group if self._iid_shared or step == 2 else keep_every_draw
                try:
                    return sample(runner, logits, step, temperature)
                finally:
                    iid._group_draws = group

            GPUBeamStageRunner.execute = execute_with_boundary
            iid.GPUIIDSampling.sample = select
        marks, self._iid_marks = self._iid_marks, []
        self._iid_shared = shared
        return marks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True, help="RecIF benchmark_data/")
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--cache", choices=("cold", "hot"), default="hot")
    parser.add_argument("--kv-gib", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "beam_search/recall_baseline_batch4.json").read_text())["tasks"][
        "video"
    ]
    tok = AutoTokenizer.from_pretrained(args.model)
    generation = GenerationConfig.from_pretrained(args.model)
    table_digest = "sha256:" + sha256_file(args.table)
    if (
        sha256_file(args.dataset_path / "video/video_test.parquet") != reference["source_sha256"]
        or compute_tokenizer_digest(tok) != reference["tokenizer_digest"]
        or table_digest != reference["artifact_digest"]
    ):
        raise ValueError(
            "expected the tokenizer, full catalog and test split of the Video baseline"
        )
    row_ids = reference["sample_ids"][:24]
    loaded = DataLoaderWrapper(tokenizer=tok, data_dir=str(args.dataset_path)).load_data(
        "video", split="test", sample_ids=row_ids
    )
    prompts = [tok.encode(loaded[str(i)]["prompt"], add_special_tokens=False) for i in row_ids]
    batch = args.batch_size
    groups = [prompts[i : i + batch] for i in range(0, len(prompts), batch)]
    draws = 128
    gr = {
        "beam_width": [draws] * 3,
        "candidate_width": [draws] * 3,
        "beam_batch_sizes": [batch],
        "max_decode_steps": 3,
        "worker_decision": True,
        "beam_graph_enabled": False,
        "constraint_mode": "table",
        "constraint_table": {
            "backend": "cuda",
            "max_top_k": draws,
            "format": "constraint_table_v1",
            "path": str(args.table.resolve()),
            "digests": {"artifact": table_digest, "tokenizer": reference["tokenizer_digest"]},
        },
    }
    policy = dict(
        temperature=generation.temperature, top_k=generation.top_k, top_p=generation.top_p
    )
    rows = []
    with GRLLM(
        model=args.model,
        gr=gr,
        dtype="bfloat16",
        attention_config={"backend": "CUSTOM"},
        async_scheduling=True,
        enforce_eager=True,
        generation_config="vllm",
        enable_prefix_caching=args.cache == "hot",
        kv_cache_memory_bytes=args.kv_gib * 2**30,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=22000,
        max_model_len=max(map(len, prompts)) + 6,
        max_num_seqs=128,
        max_logprobs=draws,
        worker_extension_cls="benchmarks.open_one_rec.profile_iid_sampling.IIDProfileWorker",
    ) as llm:
        for shared in (False, True):
            llm.collective_rpc("iid_profile_arm", args=(shared,))
            for j, group in enumerate(groups):
                llm.sample_sids_batch(group, draws, seed=1089 + j, **policy)
        for phase, shared in enumerate((False, True, True, False)):
            llm.collective_rpc("iid_profile_arm", args=(shared,))
            starts, ends = [], []
            for i in range(max(12, len(groups))):
                j = i % len(groups)
                starts.append(time.monotonic_ns())
                results = llm.sample_sids_batch(groups[j], draws, seed=1089 + j, **policy)
                ends.append(time.monotonic_ns())
                if len(results) != batch or any(
                    sum(result.counts.values()) != draws for result in results
                ):
                    raise RuntimeError("incomplete IID result")
            (marks,) = llm.collective_rpc("iid_profile_arm", args=(shared,))
            for start, boundary, end in zip(starts, marks, ends, strict=True):
                if not start < boundary < end:
                    raise RuntimeError(
                        "expected the client and worker to share a host monotonic clock"
                    )
                rows.append(
                    {
                        "phase": phase,
                        "shared": shared,
                        "post_prefill_ms": (end - boundary) / 1e6,
                        "request_ms": (end - start) / 1e6,
                    }
                )
    summary = {
        str(arm): statistics.mean(r["post_prefill_ms"] for r in rows if r["shared"] == arm)
        for arm in (False, True)
    }
    args.output.write_text(
        json.dumps(
            dict(
                batch=batch,
                cache=args.cache,
                num_samples=draws,
                sample_ids=row_ids,
                sampling=policy,
                kv_gib=args.kv_gib,
                mean_ms=summary,
                observations=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Post-prefill mean: {summary['False']:.2f} ms unmerged, {summary['True']:.2f} ms shared")


if __name__ == "__main__":
    main()
