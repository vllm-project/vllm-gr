#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one offline Beam Search V1 request with a CUDA Constraint Table."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")


BEGIN_TOKEN = "<|sid_begin|>"
END_TOKEN = "<|sid_end|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="Constraint triples JSON produced by tools/build_constraint_triples.py",
    )
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--generated-steps", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--print-beams", type=int, default=5)
    return parser.parse_args()


def load_triples(path: Path) -> list[list[str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("triples") if isinstance(document, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} is not a nonempty Constraint Table triples document")
    triples = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not all(key in row for key in ("a", "b", "c")):
            raise ValueError(f"invalid triples row {index}: {row!r}")
        triples.append([str(row["a"]), str(row["b"]), str(row["c"])])
    return sorted(triples)


def build_artifact(
    model: Path,
    catalog: Path,
    artifact_path: Path,
) -> tuple[str, str]:
    from transformers import AutoConfig, AutoTokenizer

    from vllm_gr.constraints.artifact import save_constraint_table_artifact
    from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
    from vllm_gr.constraints.json_compiler import compile_relations_json
    from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    tokenizer_digest = compute_tokenizer_digest(tokenizer)
    table = ConstraintTableArtifactCompiler(
        tokenizer,
        int(model_config.vocab_size),
        tokenizer_digest=tokenizer_digest,
    ).compile(compile_relations_json(load_triples(catalog)))
    artifact_digest = save_constraint_table_artifact(table, artifact_path)
    return artifact_digest, tokenizer_digest


def make_prompt(tokenizer: Any, path: Path, input_len: int) -> list[int]:
    # input_len includes the begin token appended by beam_search_v1.
    submitted_len = input_len - 1
    if submitted_len <= 0:
        raise ValueError("--input-len must be at least 2")
    source = tokenizer.encode(path.read_text(encoding="utf-8"), add_special_tokens=False)
    if len(source) >= submitted_len:
        return [int(token) for token in source[-submitted_len:]]
    newline = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline) != 1:
        raise RuntimeError(f"newline must map to exactly one token, got {newline}")
    return [int(newline[0])] * (submitted_len - len(source)) + [int(token) for token in source]


def run(args: argparse.Namespace, artifact_path: Path) -> None:
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    artifact_digest, tokenizer_digest = build_artifact(
        args.model.resolve(),
        args.catalog.resolve(),
        artifact_path,
    )
    vllm_gr_config = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "beam": {
            "execution_mode": "v1",
            "graph_enabled": True,
            "worker_decision": True,
            "max_width": args.beam_width,
            "max_decode_steps": args.generated_steps,
        },
        "constraint_table": {
            "enabled": True,
            "backend": "cuda",
            "max_top_k": args.beam_width,
            "path": str(artifact_path),
            "format": "constraint_table_v1",
            "artifact_digest": artifact_digest,
            "tokenizer_digest": tokenizer_digest,
        },
    }
    params = BeamSearchParams(
        beam_width=args.beam_width,
        # Public max_tokens includes the begin and end boundary positions.
        max_tokens=args.generated_steps + 2,
        temperature=0.0,
        ignore_eos=True,
        include_stop_str_in_output=True,
        begin_token=BEGIN_TOKEN,
        end_token=END_TOKEN,
        length_penalty=args.length_penalty,
    )

    with GRLLM(
        model=str(args.model.resolve()),
        trust_remote_code=True,
        vllm_gr_config=vllm_gr_config,
        async_scheduling=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=args.beam_width,
        max_logprobs=args.beam_width,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    ) as llm:
        tokenizer = llm.get_tokenizer()
        prompt_token_ids = make_prompt(tokenizer, args.prompt.resolve(), args.input_len)
        outputs = llm.beam_search_v1(
            [{"prompt_token_ids": prompt_token_ids}],
            params,
        )

        sequences = outputs[0].sequences
        print(f"returned_beams={len(sequences)}")
        for rank, sequence in enumerate(sequences[: args.print_beams]):
            token_ids = [int(token) for token in sequence.tokens]
            sid_token_ids = token_ids[-(args.generated_steps + 2) :]
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "cumulative_logprob": float(sequence.cum_logprob),
                        "finish_reason": sequence.finish_reason,
                        "stop_reason": sequence.stop_reason,
                        "sid_token_ids": sid_token_ids,
                        "sid_tokens": tokenizer.convert_ids_to_tokens(sid_token_ids),
                    },
                    ensure_ascii=False,
                )
            )


def main() -> None:
    args = parse_args()
    for path in (args.model, args.catalog, args.prompt):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.input_len + args.generated_steps + 1 > args.max_model_len:
        raise ValueError("prompt and generated tokens exceed --max-model-len")
    if args.print_beams <= 0:
        raise ValueError("--print-beams must be positive")

    with tempfile.TemporaryDirectory(prefix="vllm-gr-beam-v1-") as directory:
        artifact_path = Path(directory).resolve() / "constraint_table.safetensors"
        run(args, artifact_path)


if __name__ == "__main__":
    main()
