#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Produce native/GR raw-token goldens in separate processes, then compare.

Input is a JSON list of nonempty token-ID lists, with <|sid_begin|> already
included. No tokenizer preprocessing, SID filtering or catalog repair is done.
Use separate invocations (and optionally separate environments/machines) for
native and GR to keep monkey patches out of the native baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any


def load_prompts(path: Path) -> list[list[int]]:
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(
            not isinstance(row, list)
            or not row
            or any(type(token) is not int or token < 0 for token in row)
            for row in prompts
        )
    ):
        raise ValueError("prompts must be a nonempty JSON list of nonempty token-ID lists")
    return prompts


def native_engine_kwargs(common: dict[str, Any], engine_args_type: type) -> dict[str, Any]:
    """Bind explicit options to native EngineArgs without guessing its version.

    Official OpenOneRec used vLLM 0.7.3, whose EngineArgs predates
    async_scheduling. Preserve every shared baseline option and only add the
    async override when that field actually exists. Unknown shared options
    are errors, not silently dropped changes to the oracle configuration.
    """
    parameters = inspect.signature(engine_args_type).parameters
    accepted = {
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unsupported = set(common) - accepted
    if unsupported:
        raise ValueError(f"native EngineArgs does not support options: {sorted(unsupported)}")
    kwargs = dict(common)
    if "async_scheduling" in accepted:
        kwargs["async_scheduling"] = False
    return kwargs


def native_prompts(prompts: list[list[int]]) -> list[dict[str, list[int]]]:
    """Use the TokensPrompt schema shared by native vLLM 0.7.3 and 0.22.1."""
    # Do not import the newer tokens_input helper or add cache_salt/type keys:
    # those fields are not part of the old public token-prompt API.
    return [{"prompt_token_ids": list(row)} for row in prompts]


def compare_reports(native: dict[str, Any], gr: dict[str, Any], atol: float) -> None:
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    if native["implementation"] != "native" or gr["implementation"] != "gr":
        raise ValueError("compare expects native baseline followed by GR report")
    for key in ("prompt_sha256", "beam_width", "steps", "dtype", "eos_token_id"):
        if native[key] != gr[key]:
            raise AssertionError(f"incompatible report field {key}")
    if len(native["outputs"]) != len(gr["outputs"]):
        raise AssertionError("request count differs")
    for request_index, (expected, actual) in enumerate(zip(native["outputs"], gr["outputs"])):
        if len(expected) != len(actual):
            raise AssertionError(f"request {request_index}: returned Beam count differs")
        for rank, (left, right) in enumerate(zip(expected, actual)):
            label = f"request {request_index}, rank {rank}"
            if left["token_ids"] != right["token_ids"]:
                raise AssertionError(f"{label}: raw generated tokens differ: {left} != {right}")
            if left["finish_reason"] != right["finish_reason"]:
                raise AssertionError(f"{label}: finish reason differs")
            if not math.isclose(
                left["cumulative_logprob"], right["cumulative_logprob"], rel_tol=0, abs_tol=atol
            ):
                raise AssertionError(f"{label}: cumulative logprob differs: {left} != {right}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Must precede all vLLM imports, including its platform-plugin discovery.
    if args.implementation == "native":
        plugins = args.native_plugins.split(",")
        if "vllm_gr" in plugins:
            raise ValueError("native baseline must not enable the vllm_gr plugin")
        os.environ["VLLM_PLUGINS"] = args.native_plugins
    os.environ.setdefault("USE_TF", "0")
    prompts = load_prompts(args.prompts)

    common = dict(
        model=str(args.model),
        trust_remote_code=True,
        dtype=args.dtype,
        enforce_eager=args.execution == "eager",
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(args.max_model_len, sum(map(len, prompts))),
        max_num_seqs=max(len(prompts) * args.beam_width, args.beam_width),
        max_logprobs=2 * args.beam_width,
        gpu_memory_utilization=args.memory_utilization,
    )
    if args.implementation == "native":
        from vllm import LLM
        from vllm.engine.arg_utils import EngineArgs
        from vllm.sampling_params import BeamSearchParams

        llm = LLM(**native_engine_kwargs(common, EngineArgs))
        generate = llm.beam_search
        inputs = native_prompts(prompts)
    else:
        from vllm.inputs import tokens_input

        from vllm_gr.entrypoints.gr import GRLLM
        from vllm_gr.sampling_params import BeamSearchParams

        llm = GRLLM(
            **common,
            async_scheduling=True,
            vllm_gr_config={
                "schema_version": 1,
                "attention_backend": "CUSTOM",
                "beam": {
                    "execution_mode": "v1",
                    "worker_decision": True,
                    "max_width": args.beam_width,
                    "max_decode_steps": args.steps,
                    "graph_enabled": args.execution == "graph",
                    "graph_batch_size": len(prompts),
                },
                "constraint_table": {"enabled": False},
            },
        )
        generate = llm.beam_search_v1
        inputs = [tokens_input(row, cache_salt=f"alignment-{i}") for i, row in enumerate(prompts)]
    try:
        eos = llm.get_tokenizer().eos_token_id
        params = BeamSearchParams(
            beam_width=args.beam_width,
            max_tokens=args.steps,
            temperature=0.0,
            ignore_eos=False,
            length_penalty=1.0,
        )
        all_runs = []
        for _ in range(args.repeat):
            outputs = generate(inputs, params)
            if len(outputs) != len(prompts):
                raise AssertionError("engine returned the wrong number of requests")
            packed = []
            for prompt, output in zip(prompts, outputs):
                if not 1 <= len(output.sequences) <= args.beam_width:
                    raise AssertionError("engine returned an invalid number of Beams")
                beams = []
                for sequence in output.sequences:
                    tokens = list(sequence.tokens)
                    if tokens[: len(prompt)] != prompt:
                        raise AssertionError("engine changed the prompt prefix")
                    suffix = tokens[len(prompt) :]
                    if not 1 <= len(suffix) <= args.steps:
                        raise AssertionError("generated suffix violates the step budget")
                    reason = "stop" if suffix[-1] == eos else "length"
                    # Upstream versions may not populate finish_reason; infer
                    # it from the unmodified raw suffix for portable goldens.
                    declared = getattr(sequence, "finish_reason", None)
                    if declared is not None and declared != reason:
                        raise AssertionError("declared finish reason disagrees with raw tokens")
                    if not math.isfinite(sequence.cum_logprob):
                        raise AssertionError("engine returned a non-finite Beam score")
                    beams.append(
                        dict(
                            token_ids=suffix,
                            cumulative_logprob=float(sequence.cum_logprob),
                            finish_reason=reason,
                        )
                    )
                packed.append(beams)
            all_runs.append(packed)
        if any(run != all_runs[0] for run in all_runs[1:]):
            raise AssertionError("repeated requests on the same engine are not deterministic")
        stats = None
        if args.implementation == "gr":
            stats = llm.collective_rpc("get_gr_beam_graph_runtime_stats")
            needs_decode = any(len(beam["token_ids"]) > 1 for row in all_runs[0] for beam in row)
            if (
                args.execution == "graph"
                and needs_decode
                and any(not worker_stats["full_dispatches"] for worker_stats in stats)
            ):
                raise AssertionError("graph requested but no Beam Decode replay was observed")
        import vllm

        return dict(
            implementation=args.implementation,
            model=str(args.model),
            vllm_version=vllm.__version__,
            dtype=args.dtype,
            execution=args.execution,
            beam_width=args.beam_width,
            steps=args.steps,
            repeat=args.repeat,
            prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
            eos_token_id=eos,
            graph_stats=stats,
            outputs=all_runs[0],
        )
    finally:
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    produce = subparsers.add_parser("run")
    produce.add_argument("--implementation", choices=("native", "gr"), required=True)
    produce.add_argument("--model", type=Path, required=True)
    produce.add_argument("--prompts", type=Path, required=True)
    produce.add_argument("--output", type=Path, required=True)
    produce.add_argument(
        "--native-plugins", default="", help="NPU: retain platform plugin, e.g. ascend"
    )
    produce.add_argument("--execution", choices=("eager", "graph"), default="eager")
    produce.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    produce.add_argument("--beam-width", type=int, choices=range(1, 257), default=128)
    produce.add_argument("--steps", type=int, choices=(1, 2, 3), default=3)
    produce.add_argument("--repeat", type=int, default=2)
    produce.add_argument("--max-model-len", type=int, default=8192)
    produce.add_argument("--memory-utilization", type=float, default=0.5)
    compare = subparsers.add_parser("compare")
    compare.add_argument("native", type=Path)
    compare.add_argument("gr", type=Path)
    compare.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.command == "compare":
        compare_reports(
            json.loads(args.native.read_text()), json.loads(args.gr.read_text()), args.atol
        )
        print("PASS: ordered raw generated tokens, finish reasons and cumulative logprobs match")
    else:
        if args.repeat < 1:
            parser.error("--repeat must be positive")
        report = run(args)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
