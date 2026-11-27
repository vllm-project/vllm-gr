# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Single-XPU OneRec persistent Beam smoke test (eager, unconstrained)."""

import argparse
import json
import math
from pathlib import Path

import torch
import vllm
from vllm import SamplingParams
from vllm.inputs import tokens_input

from vllm_gr import GRLLM
from vllm_gr.sampling_params import BeamSearchParams


def resolve_model(path: Path) -> Path:
    if (path / "config.json").is_file():
        return path.resolve()
    snapshots = path / "snapshots"
    revision_file = path / "refs" / "main"
    if revision_file.is_file():
        selected = snapshots / revision_file.read_text().strip()
        if (selected / "config.json").is_file():
            return selected.resolve()
    candidates = list(snapshots.glob("*/config.json"))
    if len(candidates) != 1:
        raise ValueError("Pass a model snapshot directory or a cache with one resolved snapshot")
    return candidates[0].parent.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = resolve_model(args.model)
    params = BeamSearchParams(
        beam_width=args.beam_width,
        max_tokens=5,
        ignore_eos=True,
        begin_token="<|sid_begin|>",
        end_token="<|sid_end|>",
    )
    with GRLLM(
        model=str(model),
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.25,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=max(4, args.beam_width),
        max_logprobs=max(4, args.beam_width),
        attention_config={"backend": "CUSTOM"},
        gr={
            "beam_width": [args.beam_width] * 3,
            "candidate_width": [args.beam_width] * 3,
            "max_decode_steps": 3,
            "beam_graph_enabled": False,
            "prefill_graph_enabled": False,
            "constraint_mode": "none",
            "worker_decision": True,
        },
    ) as llm:
        prompt = {
            "prompt": "Video history: "
            + ("<|sid_begin|><s_a_346><s_b_6566><s_c_5603><|sid_end|>. " * 16)
            + "Recommend the next video."
        }
        runs = []
        # A repeated prompt exercises prefix-cache reuse and owner retirement.
        for _ in range(2):
            output = llm.beam_search([prompt], params)[0]
            if len(output.sequences) != args.beam_width:
                raise RuntimeError("XPU Beam returned the wrong candidate count")
            tokenizer = llm.get_tokenizer()
            runs.append(
                [
                    {
                        "tokens": sequence.tokens,
                        "text": sequence.text,
                        "score": sequence.cum_logprob,
                    }
                    for sequence in output.sequences
                ]
            )
            begin = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
            end = tokenizer.convert_tokens_to_ids("<|sid_end|>")
            for sequence in output.sequences:
                if sequence.tokens[-5] != begin or sequence.tokens[-1] != end:
                    raise RuntimeError("XPU Beam SID boundary tokens are incorrect")
        report = {
            "model": str(model),
            "backend": "xpu",
            "runner": "native-v2",
            "graph_mode": "eager",
            "constraint_mode": "none",
            "beam_width": args.beam_width,
            "generated_steps": 3,
            "runs": runs,
            "torch_version": torch.__version__,
            "vllm_version": vllm.__version__,
            "device": torch.xpu.get_device_name(),
            "dtype": "bfloat16",
        }

        # Teacher-force each returned tail through native XPU Prefill. Its
        # prompt logprobs are independent of the Beam suffix-cache path.
        reference = llm.generate(
            [tokens_input(row["tokens"][:-1]) for row in runs[-1]],
            SamplingParams(max_tokens=1, ignore_eos=True, prompt_logprobs=0),
            use_tqdm=False,
        )
        reference_scores = []
        for row, result in zip(runs[-1], reference, strict=True):
            scores = result.prompt_logprobs
            if scores is None:
                raise RuntimeError("Native XPU reference did not return prompt logprobs")
            score = sum(
                scores[i][row["tokens"][i]].logprob for i in range(len(scores) - 3, len(scores))
            )
            if not math.isclose(row["score"], score, abs_tol=0.15, rel_tol=0):
                raise RuntimeError(f"Beam/native score mismatch: {row['score']} vs {score}")
            reference_scores.append(score)
        report["native_teacher_forced_scores"] = reference_scores
        rendered = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)


if __name__ == "__main__":
    main()
