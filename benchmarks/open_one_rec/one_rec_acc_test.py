# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import requests

try:
    from vllm import SamplingParams

    from vllm_gr.entrypoints.gr import GRLLM
except ImportError:
    print("Error: vLLM is not installed. Please install it to run this benchmark.")
    sys.exit(1)

# Import patch to register onerec dataset and GR beam-search backends.
# Must be imported before any vllm.benchmarks module usage.
import benchmarks.open_one_rec.one_rec_patch  # noqa: F401
from benchmarks.open_one_rec.base_evaluator import EvaluationError
from benchmarks.open_one_rec.chat import messages_for_prompt, text_from_message
from benchmarks.open_one_rec.open_one_rec_loader import TASK_REGISTRY, DataLoaderWrapper

try:
    from vllm_gr.sampling_params import BeamSearchParams
except ImportError:
    from vllm.sampling_params import BeamSearchParams

logger = logging.getLogger(__name__)
DEFAULT_MODEL = "OpenOneRec/OneRec-1.7B"


def get_sampling_params(gen_config: dict[str, Any]) -> SamplingParams:
    """
    Construct vLLM SamplingParams from the task generation configuration.
    """
    # Extract standard vLLM parameters
    n = gen_config.get("num_return_sequences", 1)
    max_tokens = gen_config.get("max_new_tokens", 5)
    temperature = gen_config.get("temperature", 1.0)
    top_p = gen_config.get("top_p", 1.0)
    top_k = gen_config.get("top_k", -1)
    frequency_penalty = gen_config.get("frequency_penalty", 0.0)
    presence_penalty = gen_config.get("presence_penalty", 0.0)

    # Handle logprobs
    logprobs = None
    if gen_config.get("return_logprobs", False):
        # vLLM expects an integer for logprobs (number of top logprobs to return)
        # The config has 'logprobs': 10000, which is quite high, we clamp it for safety in this example
        logprobs = gen_config.get("logprobs", 1)

    # Handle Beam Search
    num_beams = gen_config.get("num_beams", 1)

    # Extract custom parameters (not passed to vLLM, but read as requested)
    num_return_thinking = gen_config.get("num_return_thinking_sequences", 0)
    max_thinking_tokens = gen_config.get("max_new_thinking_tokens", 0)

    logger.info(f"  [Config Read] num_return_sequences: {n}")
    logger.info(f"  [Config Read] num_beams: {num_beams}")
    logger.info(
        f"  [Config Read] num_return_thinking_sequences: {num_return_thinking} (Custom param)"
    )
    logger.info(f"  [Config Read] max_new_thinking_tokens: {max_thinking_tokens} (Custom param)")

    return SamplingParams(
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=logprobs,
        stop_token_ids=None,  # Could extract stop tokens if present in config
    )


class MockCompletionOutput:
    def __init__(self, text: str, logprobs: Any = None) -> None:
        self.text = text
        self.logprobs = logprobs


class MockRequestOutput:
    def __init__(self, prompt: str, outputs: list[MockCompletionOutput]) -> None:
        self.prompt = prompt
        self.outputs = outputs


def run_offline_v1_inference(
    llm: Any,
    prompts: list[str],
    params: Any,
    batch_size: int,
    *,
    sample_ids: list[str] | None = None,
    expected_beams: int | None = None,
) -> list[MockRequestOutput]:
    """Preserve sample order and decode only the generated SID token suffix."""
    if batch_size not in (1, 2, 4):
        raise ValueError("V1 batch_size must be 1, 2 or 4")
    outputs = []
    if sample_ids is None:
        sample_ids = [str(index) for index in range(len(prompts))]
    tokenizer = llm.get_tokenizer()
    offset = 0
    while offset < len(prompts):
        remaining = min(batch_size, len(prompts) - offset)
        size = 4 if remaining >= 4 else 2 if remaining >= 2 else 1
        batch = prompts[offset : offset + size]
        raw_outputs = llm.beam_search(
            [{"prompt": prompt} for prompt in batch], params, output_mode="compact"
        )
        batch_ids = sample_ids[offset : offset + size]
        if len(raw_outputs) != size:
            reason = f"V1 returned an unexpected number of items: {len(raw_outputs)} of {size}"
            raise InferenceError(dict.fromkeys(batch_ids, reason))
        for sample_id, prompt, output in zip(batch_ids, batch, raw_outputs, strict=True):
            expected = params.beam_width if expected_beams is None else expected_beams
            if len(output.sequences) != expected:
                reason = (
                    f"V1 returned an unexpected number of beams: "
                    f"{len(output.sequences)} of {expected}"
                )
                raise InferenceError({sample_id: reason})
            # Compact output excludes the original prompt. Keep the public SID
            # budget and decode each short candidate once, without materializing
            # full prompt-plus-tail tokens or normalizing the input text.
            completions = [
                MockCompletionOutput(
                    tokenizer.decode(
                        sequence.token_ids[-params.max_tokens :], skip_special_tokens=False
                    ),
                    sequence.cumulative_logprob,
                )
                for sequence in output.sequences
            ]
            outputs.append(MockRequestOutput(prompt, completions))
        offset += size
    return outputs


class InferenceError(RuntimeError):
    def __init__(self, failures: dict[str, str]) -> None:
        self.failures = failures
        sample_id, error = next(iter(failures.items()))
        super().__init__(f"Inference failed for {len(failures)} sample(s); {sample_id}: {error}")


def run_api_inference(
    prompts: list[str],
    model: str,
    api_url: str,
    api_key: str,
    gen_config: dict[str, Any],
    max_concurrency: int | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    *,
    samples: list[tuple[str, dict[str, Any]]] | None = None,
    request_timeout: float = 180.0,
) -> list[MockRequestOutput]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Map generation config to API parameters
    api_params = {
        "model": model,
        "use_beam_search": True if (gen_config.get("num_beams", 1) > 1) else False,
        "max_tokens": gen_config.get("max_new_tokens", 1024),
        "temperature": gen_config.get("temperature", 0.0),
        "top_p": gen_config.get("top_p", 1.0),
        "n": gen_config.get("num_return_sequences", 1),
        "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"},
    }
    if not prompts:
        raise InferenceError({"dataset": "no samples to evaluate"})

    def send_request(
        entry: tuple[str, dict[str, Any], str],
    ) -> MockRequestOutput | str:
        _, sample, prompt = entry
        payload = {
            **api_params,
            "messages": messages_for_prompt(prompt, sample.get("chat_messages")),
        }
        template_options = sample.get("chat_template_kwargs")
        if template_options is not None or chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = {
                **(template_options or {}),
                **(chat_template_kwargs or {}),
            }

        try:
            response = requests.post(
                api_url, headers=headers, json=payload, timeout=request_timeout
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") if isinstance(data, dict) else None
            if not isinstance(choices, list) or not choices:
                return "response contains no choices"
            if len(choices) != api_params["n"]:
                return f"expected {api_params['n']} choices, received {len(choices)}"

            outputs = []
            for choice in choices:
                message = choice.get("message") if isinstance(choice, dict) else None
                if not isinstance(message, dict):
                    return "choice has no text message"
                parts = [
                    message.get(field) for field in ("reasoning", "reasoning_content", "content")
                ]
                if any(part is not None and not isinstance(part, str) for part in parts):
                    return "choice contains non-text message content"
                text = text_from_message(message)
                if not text.strip():
                    return "choice has no text message"
                outputs.append(MockCompletionOutput(text))
            return MockRequestOutput(prompt, outputs)
        except (requests.RequestException, ValueError) as exc:
            return f"{type(exc).__name__}: {exc}"

    worker_limit = max_concurrency if max_concurrency is not None else 16
    if samples is None:
        samples = [(str(index), {}) for index in range(len(prompts))]
    entries = [
        (key, sample, prompt) for (key, sample), prompt in zip(samples, prompts, strict=True)
    ]
    logger.info("Sending %s requests to %s", len(entries), api_url)
    with ThreadPoolExecutor(max_workers=min(len(entries), worker_limit)) as executor:
        results = list(executor.map(send_request, entries))
    failures = {
        sample_id: result
        for (sample_id, _, _), result in zip(entries, results, strict=True)
        if isinstance(result, str)
    }
    if failures:
        raise InferenceError(failures)
    return [result for result in results if isinstance(result, MockRequestOutput)]


def _report_failure(args: argparse.Namespace, reason: str, **diagnostics: Any) -> int:
    report = {"status": "failed", "task": args.task_type, "reason": reason, **diagnostics}
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if args.result_dir is not None:
        path = Path(args.result_dir) / f"{args.task_type}-failed.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1


def _offline_settings(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    from vllm_gr.config import gr_config_to_mapping
    from vllm_gr.config.parser import (
        common_engine_defaults,
        load_engine_gr_config,
        read_settings,
    )
    from vllm_gr.engine.startup import native_settings

    document = read_settings(args.config) if args.config else {}
    native = native_settings(document)
    gr = load_engine_gr_config(None, None, args.gr, startup_document=document)
    width = gr.beam_max_width
    if width is None or args.n not in (None, width) or args.beam_max_width not in (None, width):
        raise ValueError(
            "Offline Beam requires --n and --beam-max-width to match max(gr.beam_width)"
        )
    settings = {
        "model": DEFAULT_MODEL,
        "trust_remote_code": True,
        "attention_config": {"backend": "CUSTOM"},
        "gpu_memory_utilization": 0.7,
        "max_num_seqs": max(width, args.batch_size),
        "max_logprobs": max(gr.deployment.candidate_width or (width,)),
        **native,
        "gr": gr_config_to_mapping(gr)["gr"],
    }
    # With a fixed context, reserve one full prompt per member unless YAML
    # specifies the scheduler budget. Native auto length is resolved at startup.
    model_len = native.get("max_model_len", common_engine_defaults()["max_model_len"])
    if isinstance(model_len, int) and model_len > 0 and "max_num_batched_tokens" not in native:
        settings["max_num_batched_tokens"] = model_len * args.batch_size
    if args.model is not None:
        settings["model"] = args.model
    if args.backend is not None:
        settings["attention_config"] = {
            **(settings["attention_config"] or {}),
            "backend": args.backend,
        }
    if args.enforce_eager:
        settings["enforce_eager"] = True
    return settings, width


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenOneRec Benchmark")
    parser.add_argument(
        "--model", help="Override the deployment model (default: OpenOneRec/OneRec-1.7B)"
    )
    parser.add_argument("--dataset-name", default="onerec", help="Name of the dataset to use.")
    parser.add_argument("--dataset-path", type=str, default="./data", help="Path to data directory")
    parser.add_argument(
        "--task-type", type=str, required=True, choices=TASK_REGISTRY.keys(), help="Task to run"
    )
    parser.add_argument("--num-prompts", type=int, help="Number of samples to run")
    parser.add_argument("--temperature", type=float, help="Temperature for generation")
    parser.add_argument(
        "--endpoint", type=str, default=None, help="Endpoint URL for OpenAI-compatible API"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host for OpenAI-compatible API"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", default=None, help="Name of the attention backend to use.")
    parser.add_argument(
        "--api-key", type=str, default="EMPTY", help="API Key for OpenAI-compatible API"
    )
    parser.add_argument(
        "--use-beam-search", action="store_true", help="Whether to use beam search for generation."
    )
    parser.add_argument("--n", type=int, help="Beam width to use if beam search is enabled.")
    parser.add_argument("--config", help="Optional explicit GR configuration file")
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable offline engine graphs; for online testing set this on the server",
    )
    parser.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        help='Online chat template JSON object, e.g. {"enable_thinking": false}',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="Offline V1 logical batch size; partial tails split into 4/2/1",
    )
    parser.add_argument(
        "--gr-config",
        dest="gr",
        type=json.loads,
        help="Override YAML gr settings with a JSON object",
    )
    parser.add_argument("--gr", nargs="?", const={}, type=json.loads, help=argparse.SUPPRESS)

    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Directory to save debug evaluation files.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for evaluators.")
    parser.add_argument(
        "--catalog-path", type=str, default=None, help="Path to item validation (catalog) json file"
    )
    parser.add_argument(
        "--constraint-backend",
        type=str,
        default=None,
        choices=["trie", "constraint_table"],
        help=(
            "Constraint backend for constrained beam search. 'constraint_table' "
            "requires --catalog-path to point at a tools/build_constraint_triples.py "
            "triples document."
        ),
    )
    parser.add_argument(
        "--no-constrained-generation",
        action="store_true",
        help="Run offline V1 over the full vocabulary; requires a compatible V1 config.",
    )
    parser.add_argument(
        "--beam-max-width",
        type=int,
        default=None,
        help="Optional compatibility check; must equal both --n and the V1 config max_width.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Online request concurrency; offline V1 uses --batch-size",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="Online HTTP connect/read timeout in seconds",
    )
    args = parser.parse_args()
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("--request-timeout must be positive and finite")
    if args.n is not None and args.n <= 0:
        parser.error("--n must be positive")
    if args.max_concurrency is not None and args.max_concurrency <= 0:
        parser.error("--max-concurrency must be positive")
    if args.enforce_eager and args.endpoint:
        parser.error("Online Eager mode requires --enforce-eager on the server")
    if args.chat_template_kwargs is not None:
        if not isinstance(args.chat_template_kwargs, dict):
            parser.error("--chat-template-kwargs must be a JSON object")
        if not args.endpoint:
            parser.error("--chat-template-kwargs applies only to online requests")

    # 1. Load Configuration
    if args.gr is not None and not isinstance(args.gr, dict):
        parser.error("--gr-config must be a JSON object")
    if args.endpoint and (args.config or args.gr):
        parser.error("Set --config and --gr-config on the server for online accuracy testing")

    if args.task_type not in TASK_REGISTRY:
        print(f"Task {args.task_type} not found in configuration.")
        print(f"Available tasks: {list(TASK_REGISTRY.keys())}")
        sys.exit(1)

    task_reg = TASK_REGISTRY[args.task_type]
    task_config = task_reg.config
    gen_config = task_config.get("generation_config", {}).copy()
    if task_config.get("task_type") == "logprobs_classification":
        return _report_failure(
            args,
            "This accuracy entry point does not provide label token logprobs",
            auc=None,
            auc_unavailable_reason="classification_logprobs_unavailable",
            total_samples=0,
            valid_samples=0,
            invalid_samples=0,
        )

    logger.info(f"Loaded configuration for task: {task_config['name']}")
    logger.info(f"Description: {task_config.get('description')}")

    if args.temperature is not None:
        gen_config["temperature"] = args.temperature
        logger.info(f"Using temperature from CLI parameter: {args.temperature}")
    if args.n is not None:
        gen_config["num_beams"] = args.n
        gen_config["num_return_sequences"] = args.n  # Ensure num_return_sequences matches num_beams
        logger.info(f"Using beam search with beam width from CLI parameter: {args.n}")
    else:
        logger.info(
            f"Using beam search according to the config with beam width: {gen_config.get('num_beams', 1)}"
        )

    # 2. Initialize Engine
    engine_kwargs = {}
    if args.endpoint:
        args.model = args.model or DEFAULT_MODEL
        logger.info(f"\nUsing OpenAI-compatible API at: {args.endpoint}")
        logger.info(f"Model: {args.model}")
    else:
        if not args.config and args.gr is None:
            parser.error("Supply the offline deployment with --config or --gr-config")
        steps = gen_config.get("max_new_tokens", 5) - 2
        if steps not in (1, 2, 3):
            parser.error(
                "V1 supports only 1/2/3 generated SID tokens; long reasoning is unsupported"
            )
        if args.catalog_path or args.constraint_backend:
            parser.error("Supply the compiled artifact in gr.constraint_table")
        if args.backend not in (None, "CUSTOM"):
            parser.error("The offline Beam benchmark uses the CUSTOM attention backend")
        if args.max_concurrency is not None:
            parser.error("Use --batch-size 1/2/4 for offline V1 batching")
        if args.no_constrained_generation:
            parser.error("Select constraint_mode in --gr-config")
        try:
            engine_kwargs, width = _offline_settings(args)
        except ValueError as error:
            parser.error(str(error))
        args.model = engine_kwargs["model"]
        gen_config["num_beams"] = width

    with nullcontext() if args.endpoint else GRLLM(**engine_kwargs) as llm:
        if llm is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        else:
            tokenizer = llm.get_tokenizer()
        loader = DataLoaderWrapper(tokenizer=tokenizer, data_dir=args.dataset_path)
        dataset = loader.load_data(
            task_name=args.task_type,
            split=task_config.get("splits", ["test"])[0],
            sample_size=args.num_prompts or None,
        )
        if not dataset:
            return _report_failure(args, "no samples to evaluate")
        sample_ids = list(dataset)
        prompts = [item["prompt"] for item in dataset.values()]
        try:
            if llm is None:
                gen_start = time.perf_counter()
                outputs = run_api_inference(
                    prompts,
                    args.model,
                    f"http://{args.host}:{args.port}{args.endpoint}",
                    args.api_key,
                    gen_config,
                    max_concurrency=args.max_concurrency,
                    chat_template_kwargs=args.chat_template_kwargs,
                    samples=list(dataset.items()),
                    request_timeout=args.request_timeout,
                )
            else:
                from vllm_gr.entrypoints.beam_search.request_params import (
                    apply_beam_request_defaults,
                )

                params = apply_beam_request_defaults(
                    llm.llm_engine.vllm_config,
                    BeamSearchParams(
                        beam_width=gen_config["num_beams"],
                        max_tokens=gen_config["max_new_tokens"],
                        temperature=gen_config["temperature"],
                        ignore_eos=True,
                        begin_token="<|sid_begin|>",
                        end_token="<|sid_end|>",
                        include_stop_str_in_output=True,
                    ),
                )
                expected = (
                    params.result_width or (params.dynamic_beam_widths or [params.beam_width])[-1]
                )
                gen_start = time.perf_counter()
                outputs = run_offline_v1_inference(
                    llm,
                    prompts,
                    params,
                    args.batch_size,
                    sample_ids=sample_ids,
                    expected_beams=expected,
                )
            gen_end = time.perf_counter()
        except InferenceError as exc:
            return _report_failure(args, str(exc), sample_errors=exc.failures)

    # Print a sample output
    if outputs:
        logger.info("\nSample Output [0]:")
        logger.info(f"Prompt: {outputs[0].prompt}")
        logger.info(f"Generated Text: {outputs[0].outputs[0].text}")
        if outputs[0].outputs[0].logprobs:
            logger.info(f"Logprobs available: {outputs[0].outputs[0].logprobs}")

    # 5. Run Evaluation
    logger.info("\n--- Running Evaluation ---")

    # Prepare samples dictionary in the format expected by the evaluator.
    # This mimics the structure of a test_generated.json file.
    samples_dict = {}
    for i, output in enumerate(outputs):
        sample_id = sample_ids[i]

        # Get original data from the loaded dataset
        original_sample = dataset.get(sample_id)
        if not original_sample:
            logger.warning(
                f"Warning: Could not find original sample data for sample_id {sample_id}. Skipping."
            )
            continue

        # Get generations from the model output
        if len(output.outputs) > 1:
            generations = [o.text for o in output.outputs]
        else:
            generations = [output.outputs[0].text]

        samples_dict[sample_id] = {
            "prompt": original_sample["prompt"],
            "generations": generations,
            "ground_truth": original_sample["ground_truth"],
            "metadata": original_sample.get("metadata", {}),
        }

    # Instantiate evaluator and compute metrics
    evaluator_class = task_reg.evaluator_class
    evaluator = evaluator_class(
        samples=samples_dict,
        task_name=args.task_type,
        task_config=task_config,
        data_dir=args.dataset_path,
        predictions_dir=args.result_dir,
        debug=args.debug,
    )
    try:
        metrics, per_sample_metrics = evaluator.evaluate()
    except EvaluationError as exc:
        return _report_failure(args, str(exc))
    if "auc" in metrics and metrics["auc"] is None:
        return _report_failure(args, "AUC is undefined", **metrics)
    metrics["elapsed_time"] = gen_end - gen_start

    print(f"\n--- Evaluation Results for {args.task_type} on model {args.model} ---")

    model_params = {}
    model_params["model"] = args.model
    model_params["task"] = args.task_type
    model_params["beam_width"] = gen_config.get("num_beams", 1)
    model_params["temperature"] = gen_config.get("temperature", 0.0)
    model_params["time_per_sample"] = (gen_end - gen_start) / len(prompts) if prompts else 0.0
    metrics = {**model_params, **metrics}
    print(json.dumps(metrics, indent=2))
    if args.result_dir is not None:
        os.makedirs(args.result_dir, exist_ok=True)
        bw = model_params["beam_width"]
        filename = f"{args.model.split('/')[-1]}_{args.task_type}_bw{bw}_{len(prompts)}.json"
        output_path = os.path.join(args.result_dir, filename)
        # Save metrics (a Python dict) as JSON
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
