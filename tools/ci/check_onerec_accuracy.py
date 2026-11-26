#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Check OneRec offline Beam results against an FP32 Hugging Face reference.

Reads the per-beam JSON lines that
``examples/offline_inference/beam_search/quickstart.py --print-beams N`` prints,
then:

1. teacher-forces every beam through the FP32 HF model and compares the summed
   log-probability of the searched tokens with ``cumulative_logprob``;
2. runs an exact FP32 full-vocabulary Beam search of the same width and counts
   how many returned beams it also finds.

BF16 execution reorders near-tie beams, so the checks use tolerances rather
than exact equality. The defaults leave margin over the AMD ROCm validation
(mean 0.045-0.055 nats, max 0.15-0.20 nats, top-128 overlap 115-122) while
catching attention errors (an LSE layout bug gave mean 9.0, max 22, overlap 3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = "Recommend the next video."


def parse_beams(log: Path) -> list[dict[str, Any]]:
    beams = []
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith('{"request"'):
            beams.append(json.loads(line))
    beams.sort(key=lambda beam: (beam["request"], beam["rank"]))
    return [beam for beam in beams if beam["request"] == 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True, help="quickstart stdout")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--generated-steps", type=int, default=3)
    parser.add_argument("--max-mean-diff", type=float, default=0.10)
    parser.add_argument("--max-abs-diff", type=float, default=0.50)
    parser.add_argument("--min-overlap", type=int, default=100)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    beams = parse_beams(args.log)
    if not beams:
        print(f"No beam JSON lines in {args.log}", file=sys.stderr)
        return 1
    width = len(beams)
    steps = args.generated_steps
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = tokenizer.encode(args.prompt, add_special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa", device_map="cuda"
    ).eval()

    with torch.no_grad():
        diffs = []
        for beam in beams:
            tokens = beam["beam_token_ids"]
            ids = torch.tensor([prompt + tokens], device="cuda")
            logprobs = torch.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)
            start = len(prompt)  # Predicts tokens[1]; tokens[0] is the forced SID start.
            score = sum(logprobs[start + i - 1, tokens[i]].item() for i in range(1, steps + 1))
            diffs.append(abs(score - beam["cumulative_logprob"]))

        sequences = [prompt + beams[0]["beam_token_ids"][:1]]
        scores = torch.zeros(1, device="cuda")
        for _ in range(steps):
            ids = torch.tensor(sequences, device="cuda")
            step = torch.log_softmax(model(input_ids=ids).logits[:, -1].float(), dim=-1)
            vocab = step.shape[1]
            top = torch.topk((scores[:, None] + step).flatten(), width)
            sequences = [sequences[i // vocab] + [i % vocab] for i in top.indices.tolist()]
            scores = top.values

    reference = {tuple(sequence[len(prompt) :]) for sequence in sequences}
    returned = {tuple(beam["beam_token_ids"][: steps + 1]) for beam in beams}
    result = {
        "beams": width,
        "mean_abs_diff": sum(diffs) / width,
        "max_abs_diff": max(diffs),
        "reference_overlap": len(reference & returned),
    }
    failures = []
    if result["mean_abs_diff"] > args.max_mean_diff:
        failures.append(f"mean |diff| {result['mean_abs_diff']:.4f} > {args.max_mean_diff}")
    if result["max_abs_diff"] > args.max_abs_diff:
        failures.append(f"max |diff| {result['max_abs_diff']:.4f} > {args.max_abs_diff}")
    if result["reference_overlap"] < args.min_overlap:
        failures.append(f"overlap {result['reference_overlap']} < {args.min_overlap}")
    result["passed"] = not failures
    print(json.dumps(result))
    for failure in failures:
        print(f"FAILED: {failure}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
