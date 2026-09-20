import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
    def __init__(self, text, logprobs=None):
        self.text = text
        self.logprobs = logprobs


class MockRequestOutput:
    def __init__(self, prompt, outputs: list[MockCompletionOutput]):
        self.prompt = prompt
        self.outputs = outputs


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
        payload = {"messages": _messages_for_prompt(prompt), **api_params}
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract content
            if data.get("choices"):
                mock_outputs = []
                for choice in data["choices"]:
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    reasoning = message.get("reasoning_content", "")  # Support for reasoning models

                    full_text = ""
                    if reasoning:
                        full_text += reasoning
                    if content:
                        full_text += content
                    mock_outputs.append(MockCompletionOutput(full_text))
                return MockRequestOutput(prompt, mock_outputs)
            return MockRequestOutput(prompt, [MockCompletionOutput("")])
        except Exception as e:
            status_code = response.status_code if response is not None else "N/A"
            response_text = response.text if response is not None else "N/A"
            logger.error(f"Request failed. Status code: {status_code}")
            logger.error(f"[DEBUG] Response text: {response_text}")
            logger.error(f"Request failed for prompt: {prompt[:50]}... Error: {e}")
            return MockRequestOutput(prompt, [MockCompletionOutput("")])

    # Run in parallel. The BeamAttentionPool is max_batch=1: only one beam
    # session may be active at a time, so cap workers at max_concurrency
    # (callers running beam-search accuracy tests pass --max-concurrency 1).
    worker_limit = max_concurrency if max_concurrency is not None else 16
    logger.info(f"Sending {len(prompts)} requests to {api_url}...")
    with ThreadPoolExecutor(max_workers=min(len(prompts), worker_limit)) as executor:
        results = list(executor.map(_send_request, prompts))

    return results


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
        "--beam-max-width",
        type=int,
        default=None,
        help="Upper bound on beam width: sizes the beam KV pool and the "
        "startup-captured FULL CUDA graph shape (num_tokens=width, runtime "
        "is replay-only); must be >= the request beam width (--n). Unset "
        "derives it from the effective beam width (--n / task config).",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=None, help="maximum prompts to run concurrently"
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
    llm = None
    if args.endpoint:
        logger.info(f"\nUsing OpenAI-compatible API at: {args.endpoint}")
        logger.info(f"Model: {args.model}")
    else:
        # The beam KV pool must cover every request's beam width.  This
        # benchmark drives exactly gen_config["num_beams"], so derive the
        # pool/graph upper bound from it unless the operator pinned an
        # explicit --beam-max-width override.
        beam_max_width = args.beam_max_width
        if beam_max_width is None:
            beam_max_width = gen_config.get("num_beams", 1)
        logger.info(f"Beam KV pool upper bound (beam_max_width): {beam_max_width}")
        logger.info(f"\nInitializing vLLM with model: {args.model}")
        # Select the V1 execution path for the GPU accuracy gate. Legacy mode
        # retains its existing engine configuration.
        beam_v1_mode = os.environ.get("VLLM_GR_BENCHMARK_V1") == "1"
        beam_v1_kwargs: dict = {}
        if beam_v1_mode:
            # Compile the canonical Constraint Table before EngineCore startup
            # so graph capture can use it. Workers load it through vllm_gr_config.
            from tempfile import TemporaryDirectory

            from transformers import AutoConfig, AutoTokenizer

            from vllm_gr.constraints.artifact import save_constraint_table_artifact
            from vllm_gr.constraints.artifact_compiler import (
                ConstraintTableArtifactCompiler,
            )
            from vllm_gr.constraints.json_compiler import compile_relations_json
            from vllm_gr.constraints.tokenizer_digest import compute_tokenizer_digest

            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            vocab_size = AutoConfig.from_pretrained(args.model, trust_remote_code=True).vocab_size
            document = json.loads(Path(args.catalog_path).read_text())
            catalog = [[t["a"], t["b"], t["c"]] for t in document["triples"]]
            tokenizer_digest = compute_tokenizer_digest(tokenizer)
            table = ConstraintTableArtifactCompiler(
                tokenizer, vocab_size, tokenizer_digest=tokenizer_digest
            ).compile(compile_relations_json(catalog))
            artifact_dir = TemporaryDirectory(prefix="issue398_v1_table_")
            artifact = Path(artifact_dir.name) / "video_constraints.safetensors"
            digest = save_constraint_table_artifact(table, artifact)
            effective_steps = int(gen_config["max_new_tokens"]) - 2
            v1_config = {
                "schema_version": 1,
                "attention_backend": "CUSTOM",
                "beam": {
                    "execution_mode": "v1",
                    "graph_enabled": True,
                    "worker_decision": True,
                    "max_width": beam_max_width,
                    "max_decode_steps": effective_steps,
                },
                "constraint_table": {
                    "enabled": True,
                    "backend": "cuda",
                    "max_top_k": beam_max_width,
                    "path": str(artifact),
                    "format": "constraint_table_v1",
                    "artifact_digest": digest,
                    "tokenizer_digest": tokenizer_digest,
                },
            }
            beam_v1_kwargs = dict(
                vllm_gr_config=v1_config,
                async_scheduling=True,
                max_num_seqs=1,
                enable_chunked_prefill=False,
                enable_prefix_caching=True,
            )
            # The canonical table rides vllm_gr_config; keep the driver-side
            # convenience installers off in V1 mode.
            logger.info("VLLM_GR_BENCHMARK_V1=1: using beam_search_v1 execution mode")
        gr_llm_kwargs: dict = dict(
            model=args.model,
            trust_remote_code=True,
            max_logprobs=gen_config.get("num_beams", 1) * 2,
            gpu_memory_utilization=0.7,
        )
        if beam_v1_mode:
            gr_llm_kwargs.update(beam_v1_kwargs)
        else:
            gr_llm_kwargs.update(
                catalog_path=args.catalog_path if args.catalog_path is not None else None,
                constraint_backend=args.constraint_backend,
                beam_max_width=beam_max_width,
                attention_config={
                    "backend": args.backend if args.backend is not None else "CUSTOM"
                },
            )
        try:
            llm = GRLLM(**gr_llm_kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            sys.exit(1)

    # 3. Prepare Data
    # prompts = mock_data_loader(task_config, args.num_prompts)  # Replace with real data loading
    logger.info(f"Loading real data from {args.dataset_path}...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    custom_input_len = os.environ.get("VLLM_GR_ACC_CUSTOM_INPUT_LEN")
    loader = DataLoaderWrapper(
        tokenizer=tokenizer,
        data_dir=args.dataset_path,
        custom_input_len=int(custom_input_len) if custom_input_len else None,
        truncate_side=os.environ.get("ONEREC_TRUNCATE_SIDE", "head"),
    )
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
        )
        # Add vLLM-GR specific parameters if available
        if hasattr(BeamSearchParams, "begin_token"):
            params.begin_token = "<|sid_begin|>"
            params.end_token = "<|sid_end|>"
        prompts_dict = [{"prompt": prompt} for prompt in prompts]
        gen_start = time.perf_counter()
        if beam_v1_mode:
            # beam_search_v1's public contract is one prompt per call (B=1).
            params.ignore_eos = True
            raw_offline_outputs = [llm.beam_search_v1([item], params)[0] for item in prompts_dict]
        else:
            raw_offline_outputs = llm.beam_search(
                prompts_dict, params, concurrency_limit=args.max_concurrency
            )
        assert len(raw_offline_outputs) == len(prompts)
        gen_end = time.perf_counter()

        # Normalize beam_search outputs to match MockRequestOutput format
        outputs = []
        for i, raw_output in enumerate(raw_offline_outputs):
            # beam_search returns BeamSearchOutput with .outputs containing BeamSearchSequence objects
            # Each BeamSearchSequence has .text attribute
            mock_completions = []
            prompt_len = len(prompts[i])
            if hasattr(raw_output, "sequences"):
                for beam_seq in raw_output.sequences:
                    text = (
                        beam_seq.text[prompt_len:] if hasattr(beam_seq, "text") else str(beam_seq)
                    )
                    logprobs = getattr(beam_seq, "cum_logprob", None)
                    mock_completions.append(MockCompletionOutput(text, logprobs))
            else:
                # Fallback if format is different
                mock_completions.append(MockCompletionOutput(str(raw_output)))

            outputs.append(MockRequestOutput(prompts[i], mock_completions))

        # Optionally retain ranked output text and scores for contract comparisons.
        dump_path = os.environ.get("VLLM_GR_DUMP_SEQUENCES")
        if dump_path:
            import hashlib

            dump = []
            for i, raw_output in enumerate(raw_offline_outputs):
                dump.append(
                    {
                        "prompt_sha256": hashlib.sha256(
                            prompts[i].encode("utf-8", "replace")
                        ).hexdigest(),
                        "sequences": [
                            {
                                "cum_logprob": getattr(s, "cum_logprob", None),
                                "rank_text": (
                                    s.text[len(prompts[i]) :] if hasattr(s, "text") else str(s)
                                ),
                            }
                            for s in getattr(raw_output, "sequences", [])
                        ],
                    }
                )
            Path(dump_path).write_text(json.dumps(dump, ensure_ascii=False))
            logger.info(f"Dumped beam sequences to {dump_path}")

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
