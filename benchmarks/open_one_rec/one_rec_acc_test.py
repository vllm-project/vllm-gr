import argparse
import os
import sys
import json
from typing import List, Dict, Any
import requests
from concurrent.futures import ThreadPoolExecutor

# Ensure we can import from the local project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: vLLM is not installed. Please install it to run this benchmark.")
    sys.exit(1)

from benchmarks.open_one_rec.open_one_rec_loader import DataLoaderWrapper, TASK_REGISTRY


def get_sampling_params(gen_config: Dict[str, Any]) -> SamplingParams:
    """
    Construct vLLM SamplingParams from the task generation configuration.
    """
    # Extract standard vLLM parameters
    n = gen_config.get("num_return_sequences", 1)
    max_tokens = gen_config.get("max_new_tokens", 16)
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
    # use_beam_search = False
    # best_of = None

    # if num_beams > 1:
    #     use_beam_search = True
    #     best_of = num_beams
    #     # vLLM Constraint: best_of must be >= n
    #     if best_of < n:
    #         print(f"Warning: num_beams ({best_of}) < num_return_sequences ({n}). Adjusting n to {best_of}.")
    #         n = best_of

    # Extract custom parameters (not passed to vLLM, but read as requested)
    num_return_thinking = gen_config.get("num_return_thinking_sequences", 0)
    max_thinking_tokens = gen_config.get("max_new_thinking_tokens", 0)

    print(f"  [Config Read] num_return_sequences: {n}")
    print(f"  [Config Read] num_beams: {num_beams}")
    print(f"  [Config Read] num_return_thinking_sequences: {num_return_thinking} (Custom param)")
    print(f"  [Config Read] max_new_thinking_tokens: {max_thinking_tokens} (Custom param)")

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
    def __init__(self, prompt, outputs: List[MockCompletionOutput]):
        self.prompt = prompt
        self.outputs = outputs


def run_api_inference(
    prompts: List[str], model: str, api_url: str, api_key: str, gen_config: Dict[str, Any]
) -> List[MockRequestOutput]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Map generation config to API parameters
    api_params = {
        "model": model,
        "max_tokens": gen_config.get("max_new_tokens", 1024),
        "temperature": gen_config.get("temperature", 0.0),
        "top_p": gen_config.get("top_p", 1.0),
        "n": gen_config.get("num_return_sequences", 1),
    }
    if gen_config.get("num_beams", 1) > 1:
        api_params["use_beam_search"] = True

    def _send_request(prompt):
        payload = {"messages": [{"role": "user", "content": prompt}], **api_params}
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract content
            if "choices" in data and data["choices"]:
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
            print(f"Request failed for prompt: {prompt[:50]}... Error: {e}")
            return MockRequestOutput(prompt, [MockCompletionOutput("")])

    # Run in parallel
    print(f"Sending {len(prompts)} requests to {api_url}...")
    with ThreadPoolExecutor(max_workers=min(len(prompts), 16)) as executor:
        results = list(executor.map(_send_request, prompts))

    return results


def main():
    parser = argparse.ArgumentParser(description="Run OpenOneRec Benchmark")
    parser.add_argument(
        "--task", type=str, required=True, choices=TASK_REGISTRY.keys(), help="Task to run"
    )
    parser.add_argument(
        "--model", type=str, default="facebook/opt-125m", help="Path to model or HF hub name"
    )
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to run")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="TP size")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument(
        "--api_url", type=str, default=None, help="Endpoint URL for OpenAI-compatible API"
    )
    parser.add_argument(
        "--api_key", type=str, default="EMPTY", help="API Key for OpenAI-compatible API"
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default=None,
        help="Directory to save debug evaluation files.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for evaluators.")

    args = parser.parse_args()

    # 1. Load Configuration
    if args.task not in TASK_REGISTRY:
        print(f"Task {args.task} not found in configuration.")
        print(f"Available tasks: {list(TASK_REGISTRY.keys())}")
        sys.exit(1)

    task_reg = TASK_REGISTRY[args.task]
    task_config = task_reg.config
    gen_config = task_config.get("generation_config", {})
    prompt_config = task_config.get("prompt_config", {})
    eval_config = task_config.get("evaluation_config", {})

    print(f"Loaded configuration for task: {task_config['name']}")
    print(f"Description: {task_config.get('description')}")

    # 2. Prepare Sampling Params
    sampling_params = None
    if not args.api_url:
        try:
            sampling_params = get_sampling_params(gen_config)
        except Exception as e:
            print(f"Failed to create SamplingParams: {e}")
            sys.exit(1)

    # 3. Initialize Engine
    llm = None
    if args.api_url:
        print(f"\nUsing OpenAI-compatible API at: {args.api_url}")
        print(f"Model: {args.model}")
    else:
        print(f"\nInitializing vLLM with model: {args.model}")
        try:
            llm = LLM(
                model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Failed to initialize vLLM: {e}")
            sys.exit(1)

    # 4. Prepare Data
    # prompts = mock_data_loader(task_config, args.num_samples)
    print(f"Loading real data from {args.data_dir}...")
    loader = DataLoaderWrapper(
        model_path=args.model, benchmark_version="v1.0", data_dir=args.data_dir
    )
    split = task_config.get("splits", ["test"])[0]
    dataset = loader.load_data(task_name=args.task, split=split, sample_size=args.num_samples)

    prompts = []
    references = {}
    sample_ids = []
    prompt_token = ""
    if not prompt_config.get("enable_thinking", False):
        prompt_token = gen_config.get("prompt_token", "")

    for sample_id, item in dataset.items():
        prompt = item[
            "prompt"
        ]  # + prompt_token ## prompt token is being filtered out in chat completion API
        prompts.append(prompt)
        references[sample_id] = item["ground_truth"]
        sample_ids.append(sample_id)

    # 5. Run Inference
    print("\nStarting Inference...")
    if args.api_url:
        outputs = run_api_inference(prompts, args.model, args.api_url, args.api_key, gen_config)
    else:
        outputs = llm.generate(prompts, sampling_params)

    # Print a sample output
    if outputs:
        print("\nSample Output [0]:")
        print(f"Prompt: {outputs[0].prompt}")
        print(f"Generated Text: {outputs[0].outputs[0].text}")
        if outputs[0].outputs[0].logprobs:
            print("Logprobs available: Yes")

    # 6. Run Evaluation
    print("\n--- Running Evaluation ---")

    # Prepare samples dictionary in the format expected by the evaluator.
    # This mimics the structure of a test_generated.json file.
    samples_dict = {}
    for i, output in enumerate(outputs):
        sample_id = sample_ids[i]

        # Get original data from the loaded dataset
        original_sample = dataset.get(sample_id)
        if not original_sample:
            print(
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

    # Build code_to_pid by inverting pid2sid
    with open(args.data_dir + "/sid2pid.json", "r") as f:
        code_to_pid = json.load(f)  # maps SID string -> PID

    # Instantiate evaluator and compute metrics
    evaluator_class = task_reg.evaluator_class
    evaluator = evaluator_class(
        samples=samples_dict,
        task_name=args.task,
        task_config=task_config,
        data_dir=args.data_dir,
        predictions_dir=args.predictions_dir,
        debug=args.debug,
    )
    metrics, per_sample_metrics = evaluator.evaluate()

    print("\n--- Evaluation Results ---")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
