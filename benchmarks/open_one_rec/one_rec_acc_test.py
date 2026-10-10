import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import regex as re
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
from benchmarks.open_one_rec.open_one_rec_loader import TASK_REGISTRY, DataLoaderWrapper
from benchmarks.open_one_rec.regression import (
    RECOMMENDATION_TASKS,
    validate_generations,
)

try:
    from vllm_gr.sampling_params import BeamSearchParams
except ImportError:
    from vllm.sampling_params import BeamSearchParams

logger = logging.getLogger(__name__)


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
    llm: Any, prompts: list[str], params: Any, batch_size: int
) -> list[MockRequestOutput]:
    """Preserve sample order and decode only the generated SID token suffix."""
    if batch_size not in (1, 2, 4):
        raise ValueError("V1 batch_size must be 1, 2 or 4")
    outputs = []
    tokenizer = llm.get_tokenizer()
    offset = 0
    while offset < len(prompts):
        remaining = min(batch_size, len(prompts) - offset)
        size = 4 if remaining >= 4 else 2 if remaining >= 2 else 1
        batch = prompts[offset : offset + size]
        raw_outputs = llm.beam_search_v1([{"prompt": prompt} for prompt in batch], params)
        if len(raw_outputs) != size:
            raise RuntimeError("V1 returned an unexpected number of items")
        for prompt, output in zip(batch, raw_outputs, strict=True):
            if len(output.sequences) != params.beam_width:
                raise RuntimeError("V1 returned an unexpected number of beams")
            # V1 tokens contain the original prompt plus the public SID budget.
            # Decoding token suffixes avoids character slicing after a tokenizer
            # round-trip (which can normalize the original prompt).
            completions = [
                MockCompletionOutput(
                    tokenizer.decode(
                        sequence.tokens[-params.max_tokens :], skip_special_tokens=False
                    ),
                    sequence.cum_logprob,
                )
                for sequence in output.sequences
            ]
            outputs.append(MockRequestOutput(prompt, completions))
        offset += size
    return outputs


# OneRec dataset prompts are stored as fully rendered chat transcripts
# (Qwen format: "<|im_start|>role\ncontent<|im_end|>" turns). Sending such a
# transcript as one user message makes the server's chat template wrap the
# already-rendered turns a second time (a user turn embedding the original
# system turn), which desynchronizes the online prompt from the offline
# beam-search input. Parse the transcript back into structured messages so
# the template is applied exactly once, server-side.
_CHAT_TURN_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.DOTALL)


def _messages_for_prompt(prompt: str) -> list[dict[str, str]]:
    """Return structured chat messages for a dataset prompt.

    Falls back to a single user turn when the prompt is not a recognizable
    chat transcript (coverage guard: parsed content must explain most of
    the original text before structured form is trusted).
    """
    turns = _CHAT_TURN_RE.findall(prompt)
    parsed_chars = sum(len(role) + len(content) for role, content in turns)
    if turns and parsed_chars >= 0.5 * len(prompt):
        return [{"role": role, "content": content.strip("\n")} for role, content in turns]
    return [{"role": "user", "content": prompt}]


def run_api_inference(
    prompts: list[str],
    model: str,
    api_url: str,
    api_key: str,
    gen_config: dict[str, Any],
    max_concurrency: int | None = None,
    request_timeout: float = 600.0,
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

    def _send_request(prompt):
        response = None
        # Rendered dataset transcripts sent to the Completion API must pass
        # through verbatim: wrapping them in a chat message would apply the
        # model's chat template a second time. Chat-API calls keep structured
        # messages so the server-side template is applied exactly once.
        if api_url.rstrip("/").endswith("/v1/completions"):
            payload = {"prompt": prompt, **api_params}
        else:
            payload = {"messages": _messages_for_prompt(prompt), **api_params}
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=request_timeout,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("API response JSON must be an object")

            choices = data.get("choices")
            expected_choices = int(api_params["n"])
            if not isinstance(choices, list):
                raise ValueError("API response 'choices' must be a list")
            if len(choices) != expected_choices:
                raise ValueError(
                    f"API response returned {len(choices)} choices, expected {expected_choices}"
                )

            mock_outputs = []
            for choice in choices:
                if not isinstance(choice, dict):
                    raise ValueError("API response choices must be objects")
                if api_url.rstrip("/").endswith("/v1/completions"):
                    full_text = choice.get("text")
                    if not isinstance(full_text, str):
                        raise ValueError("Completion choice 'text' must be a string")
                else:
                    message = choice.get("message")
                    if not isinstance(message, dict):
                        raise ValueError("Chat choice 'message' must be an object")
                    content = message.get("content", "")
                    reasoning = message.get("reasoning_content", "")  # Support for reasoning models
                    if content is None:
                        content = ""
                    if reasoning is None:
                        reasoning = ""
                    if not isinstance(content, str) or not isinstance(reasoning, str):
                        raise ValueError("Chat choice content fields must be strings")
                    full_text = reasoning + content
                mock_outputs.append(MockCompletionOutput(full_text))
            return MockRequestOutput(prompt, mock_outputs), None
        except Exception as e:  # noqa: BLE001 - record per-sample failures, keep the run going
            status_code = getattr(response, "status_code", "N/A")
            response_text = getattr(response, "text", "N/A")
            logger.error(f"Request failed. Status code: {status_code}")
            logger.error(f"[DEBUG] Response text: {response_text}")
            logger.error(f"Request failed for prompt: {prompt[:50]}... Error: {e}")
            return None, f"status={status_code}: {e}"

    # Run in parallel. The BeamAttentionPool is max_batch=1: only one beam
    # session may be active at a time, so cap workers at max_concurrency
    # (callers running beam-search accuracy tests pass --max-concurrency 1).
    worker_limit = max_concurrency if max_concurrency is not None else 16
    logger.info(f"Sending {len(prompts)} requests to {api_url}...")
    with ThreadPoolExecutor(max_workers=min(len(prompts), worker_limit)) as executor:
        request_results = list(executor.map(_send_request, prompts))

    outputs: list[MockRequestOutput] = []
    failures: list[str] = []
    for output, failure in request_results:
        if failure is not None:
            failures.append(failure)
        elif output is not None:
            outputs.append(output)
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(prompts)} API requests failed; first failure: {failures[0]}"
        )

    return outputs


def _validate_recommendation_outputs(
    outputs: list[MockRequestOutput],
    *,
    expected_requests: int,
    beam_width: int,
) -> None:
    """Reject incomplete or structurally invalid OneRec recommendation output."""
    if len(outputs) != expected_requests:
        raise RuntimeError(
            f"inference returned {len(outputs)} requests, expected {expected_requests}"
        )

    invalid: list[tuple[int, dict[str, Any]]] = []
    for index, output in enumerate(outputs):
        contract = validate_generations([choice.text for choice in output.outputs], beam_width)
        if (
            not contract["beam_count_ok"]
            or contract["empty_count"] > 0
            or contract["invalid_sid_count"] > 0
        ):
            invalid.append((index, contract))
    if invalid:
        first_index, first_contract = invalid[0]
        raise RuntimeError(
            f"{len(invalid)} of {expected_requests} requests violated the OneRec output "
            f"contract; first invalid request index={first_index}: {first_contract}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run OpenOneRec Benchmark")
    parser.add_argument(
        "--model", type=str, default="OpenOneRec/OneRec-1.7B", help="Path to model or HF hub name"
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
    parser.add_argument("--vllm-gr-config", help="Canonical V1 config JSON; required offline")
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="Offline V1 logical batch size; partial tails split into 4/2/1",
    )
    parser.add_argument("--gr", action="store_true", help="Whether to use vllm-gr.")

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
    args = parser.parse_args()

    # 1. Load Configuration
    if args.gr:
        logger.info("Detected --gr flag, enabling vllm-gr mode for accuracy test.")

    if args.task_type not in TASK_REGISTRY:
        print(f"Task {args.task_type} not found in configuration.")
        print(f"Available tasks: {list(TASK_REGISTRY.keys())}")
        sys.exit(1)

    task_reg = TASK_REGISTRY[args.task_type]
    task_config = task_reg.config
    gen_config = task_config.get("generation_config", {})

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
        from vllm_gr.config import BeamExecutionMode
        from vllm_gr.config.parser import load_gr_config

        if not args.vllm_gr_config:
            parser.error("Offline accuracy testing requires --vllm-gr-config with a V1 artifact")
        config = load_gr_config(args.vllm_gr_config)
        width = gen_config.get("num_beams", 1)
        steps = gen_config.get("max_new_tokens", 5) - 2
        if config.beam_execution_mode is not BeamExecutionMode.V1:
            parser.error("Offline accuracy testing requires beam.execution_mode='v1'")
        if steps not in (1, 2, 3):
            parser.error(
                "V1 supports only 1/2/3 generated SID tokens; long reasoning is unsupported"
            )
        if width != config.beam_max_width or args.beam_max_width not in (None, width):
            parser.error("Request beam width must equal the canonical V1 max_width")
        if args.catalog_path or args.constraint_backend:
            parser.error(
                "Use the canonical artifact in --vllm-gr-config instead of legacy catalog options"
            )
        if args.backend not in (None, "CUSTOM"):
            parser.error("Offline attention backend is selected by --vllm-gr-config")
        if args.max_concurrency is not None:
            parser.error("Use --batch-size 1/2/4 for offline V1 batching")
        if args.no_constrained_generation:
            try:
                config = replace(config, constrained_generation=False)
            except ValueError as exc:
                parser.error(f"--no-constrained-generation conflicts with V1 config: {exc}")
        engine_kwargs = {
            "model": args.model,
            "trust_remote_code": True,
            "vllm_gr_config": config,
            "async_scheduling": True,
            "enable_chunked_prefill": False,
            "max_model_len": 8192,
            "max_num_batched_tokens": 8192 * args.batch_size,
            "max_num_seqs": max(width, args.batch_size),
            "max_logprobs": width,
            "gpu_memory_utilization": 0.7,
        }

    # 3. Prepare Data
    # prompts = mock_data_loader(task_config, args.num_prompts)  # Replace with real data loading
    logger.info(f"Loading real data from {args.dataset_path}...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    loader = DataLoaderWrapper(tokenizer=tokenizer, data_dir=args.dataset_path)
    split = task_config.get("splits", ["test"])[0]
    dataset = loader.load_data(
        task_name=args.task_type,
        split=split,
        sample_size=args.num_prompts if args.num_prompts else None,
    )

    prompts = []
    references = {}
    sample_ids = []

    for sample_id, item in dataset.items():
        prompt = item["prompt"]
        prompts.append(prompt)
        references[sample_id] = item["ground_truth"]
        sample_ids.append(sample_id)

    # 4. Run Inference
    logger.info("\nStarting Inference...")
    gen_start = 0.0
    gen_end = 0.0
    if args.endpoint:
        gen_start = time.perf_counter()
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        outputs = run_api_inference(
            prompts,
            args.model,
            api_url,
            args.api_key,
            gen_config,
            max_concurrency=args.max_concurrency,
        )
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
        with GRLLM(**engine_kwargs) as llm:
            gen_start = time.perf_counter()
            outputs = run_offline_v1_inference(llm, prompts, params, args.batch_size)
            gen_end = time.perf_counter()

    # Enforce the standalone-SID output contract only for the five SID
    # recommendation tasks; the other registered tasks (label_pred,
    # item_understand, rec_reason) emit yes/no, captions or explanations and
    # must keep their own output contracts.
    if args.task_type in RECOMMENDATION_TASKS:
        _validate_recommendation_outputs(
            outputs,
            expected_requests=len(prompts),
            beam_width=gen_config["num_beams"],
        )

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
    metrics, per_sample_metrics = evaluator.evaluate()
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


if __name__ == "__main__":
    main()
