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
from pathlib import Path
from typing import Any

import requests

try:
    from vllm_gr.entrypoints.gr import GRLLM
except ImportError:
    print("Error: vLLM is not installed. Please install it to run this benchmark.")
    sys.exit(1)

# Import patch to register onerec dataset and GR beam-search backends.
# Must be imported before any vllm.benchmarks module usage.
import benchmarks.open_one_rec.one_rec_patch  # noqa: F401
from benchmarks.open_one_rec.base_evaluator import EvaluationError
from benchmarks.open_one_rec.chat import completion_prompt
from benchmarks.open_one_rec.open_one_rec_loader import (
    DEFAULT_DATASET_PATH,
    TASK_REGISTRY,
    DataLoaderWrapper,
)

try:
    from vllm_gr.sampling_params import BeamSearchParams
except ImportError:
    from vllm.sampling_params import BeamSearchParams

logger = logging.getLogger(__name__)


class MockCompletionOutput:
    def __init__(self, content: str, logprobs: Any = None, *, reasoning: str = "") -> None:
        self.content = content
        self.reasoning = reasoning
        self.logprobs = logprobs

    @property
    def text(self) -> str:
        return self.reasoning + self.content


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
            if len(output.sequences) != params.beam_width:
                reason = (
                    f"V1 returned an unexpected number of beams: "
                    f"{len(output.sequences)} of {params.beam_width}"
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
    tokenizer: Any = None,
) -> list[MockRequestOutput]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Map generation config to API parameters
    api_params = {
        "model": model,
        "add_special_tokens": False,
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
            "prompt": prompt,
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
                content = choice.get("text") if isinstance(choice, dict) else None
                if not isinstance(content, str) or not content.strip():
                    return "choice has no text completion"
                reasoning = ""
                raw_content = content.lstrip()
                if raw_content.startswith("<think>"):
                    # Keep truncated reasoning separate from the scored final answer.
                    reasoning, _, content = raw_content.removeprefix("<think>").partition(
                        "</think>"
                    )
                outputs.append(MockCompletionOutput(content, reasoning=reasoning))
            return MockRequestOutput(prompt, outputs)
        except (requests.RequestException, ValueError) as exc:
            return f"{type(exc).__name__}: {exc}"

    worker_limit = max_concurrency if max_concurrency is not None else 16
    if samples is None:
        samples = [(str(index), {}) for index in range(len(prompts))]
    entries = [
        (key, sample, completion_prompt(prompt, sample, tokenizer, chat_template_kwargs))
        for (key, sample), prompt in zip(samples, prompts, strict=True)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenOneRec Benchmark")
    parser.add_argument(
        "--model", type=str, default="OpenOneRec/OneRec-1.7B", help="Path to model or HF hub name"
    )
    parser.add_argument("--dataset-name", default="onerec", help="Name of the dataset to use.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Directory containing the task folders and catalog mappings. Default: %(default)s.",
    )
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
        help='Client-side prompt template options, e.g. {"enable_thinking": false}',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="Offline V1 logical batch size; partial tails split into 4/2/1",
    )
    parser.add_argument(
        "--gr", nargs="?", const={}, type=json.loads, help="Explicit GR settings as a JSON object"
    )

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
            "requires --catalog-path to point at a tools/constraints/build_constraint_triples.py "
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
    if args.num_prompts is not None and args.num_prompts <= 0:
        parser.error("--num-prompts must be positive")
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("--request-timeout must be positive and finite")
    if args.n is not None and args.n <= 0:
        parser.error("--n must be positive")
    if args.max_concurrency is not None and args.max_concurrency <= 0:
        parser.error("--max-concurrency must be positive")
    if args.enforce_eager and args.endpoint:
        parser.error("Online Eager mode requires --enforce-eager on the server")
    if args.endpoint and args.endpoint != "/v1/completions":
        parser.error("Online accuracy requires --endpoint /v1/completions")
    if args.chat_template_kwargs is not None:
        if not isinstance(args.chat_template_kwargs, dict):
            parser.error("--chat-template-kwargs must be a JSON object")
        if not args.endpoint:
            parser.error("--chat-template-kwargs applies only to online requests")

    # 1. Load Configuration
    if args.gr is not None and not isinstance(args.gr, dict):
        parser.error("--gr must be a JSON object")
    if args.gr is not None:
        logger.info("Detected --gr flag, enabling vllm-gr mode for accuracy test.")

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
        logger.info(f"\nUsing OpenAI-compatible API at: {args.endpoint}")
        logger.info(f"Model: {args.model}")
    else:
        if not args.config and not args.gr:
            parser.error(
                "Offline accuracy testing requires a V1 constraint artifact in --gr or --config"
            )
        width = gen_config.get("num_beams", 1)
        steps = gen_config.get("max_new_tokens", 5) - 2
        if steps not in (1, 2, 3):
            parser.error(
                "V1 supports only 1/2/3 generated SID tokens; long reasoning is unsupported"
            )
        if args.beam_max_width not in (None, width):
            parser.error("Request beam width must equal the canonical V1 max_width")
        if args.catalog_path or args.constraint_backend:
            parser.error("Supply the compiled artifact in gr.constraint_table")
        if args.backend not in (None, "CUSTOM"):
            parser.error("The offline Beam benchmark uses the CUSTOM attention backend")
        if args.max_concurrency is not None:
            parser.error("Use --batch-size 1/2/4 for offline V1 batching")
        if args.no_constrained_generation:
            parser.error("Select constraint_mode in --gr")
        engine_kwargs = {
            "model": args.model,
            "trust_remote_code": True,
            "config": args.config,
            "gr": args.gr,
            **({"attention_config": {"backend": "CUSTOM"}} if args.gr and not args.config else {}),
            "async_scheduling": True,
            "enforce_eager": args.enforce_eager,
            "enable_chunked_prefill": False,
            "max_model_len": 8192,
            "max_num_batched_tokens": 8192 * args.batch_size,
            "max_num_seqs": max(width, args.batch_size),
            "max_logprobs": width,
            "gpu_memory_utilization": 0.7,
        }

    # 3. Prepare Data
    logger.info(f"Loading real data from {args.dataset_path}...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    loader = DataLoaderWrapper(tokenizer=tokenizer, data_dir=args.dataset_path)
    split = task_config.get("splits", ["test"])[0]
    dataset = loader.load_data(
        task_name=args.task_type,
        split=split,
        sample_size=args.num_prompts,
    )

    prompts = []
    sample_ids = []

    for sample_id, item in dataset.items():
        prompt = item["prompt"]
        prompts.append(prompt)
        sample_ids.append(sample_id)

    if not prompts:
        return _report_failure(args, "no samples to evaluate")

    # 4. Run Inference
    logger.info("\nStarting Inference...")
    gen_start = 0.0
    gen_end = 0.0
    if args.endpoint:
        gen_start = time.perf_counter()
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        try:
            outputs = run_api_inference(
                prompts,
                args.model,
                api_url,
                args.api_key,
                gen_config,
                max_concurrency=args.max_concurrency,
                chat_template_kwargs=args.chat_template_kwargs,
                samples=list(dataset.items()),
                request_timeout=args.request_timeout,
                tokenizer=tokenizer,
            )
        except InferenceError as exc:
            return _report_failure(args, str(exc), sample_errors=exc.failures)
        gen_end = time.perf_counter()
    else:
        params = BeamSearchParams(
            beam_width=gen_config["num_beams"],
            max_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            ignore_eos=True,
            begin_token="<|sid_begin|>",
            end_token="<|sid_end|>",
            include_stop_str_in_output=True,
        )
        from vllm_gr.config import GRConfig

        try:
            with GRLLM(**engine_kwargs) as llm:
                config = GRConfig.from_vllm_config(llm.llm_engine.vllm_config)
                if config.beam_max_width is None or width != config.beam_max_width:
                    parser.error(
                        "Offline accuracy testing requires V1 and the configured beam width"
                    )
                gen_start = time.perf_counter()
                outputs = run_offline_v1_inference(
                    llm, prompts, params, args.batch_size, sample_ids=sample_ids
                )
                gen_end = time.perf_counter()
        except InferenceError as exc:
            return _report_failure(args, str(exc), sample_errors=exc.failures)

    # Print a sample output
    if outputs:
        completion = outputs[0].outputs[0]
        logger.info("\nSample Output [0]:")
        logger.info(f"Prompt: {outputs[0].prompt}")
        logger.info(f"Generated Text: {completion.content}")
        if completion.reasoning:
            logger.info(f"Reasoning: {completion.reasoning}")
        if completion.logprobs:
            logger.info(f"Logprobs available: {completion.logprobs}")

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

        samples_dict[sample_id] = {
            "prompt": original_sample["prompt"],
            "generations": [completion.content for completion in output.outputs],
            "ground_truth": original_sample["ground_truth"],
            "metadata": original_sample.get("metadata", {}),
        }

    # Instantiate evaluator and compute metrics
    evaluator = task_reg.evaluator_class(
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
