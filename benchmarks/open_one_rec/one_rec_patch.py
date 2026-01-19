import vllm.benchmarks.datasets
import vllm.benchmarks.throughput

from benchmarks.open_one_rec.open_one_rec_dataset import OneRecDataset
from vllm.benchmarks.datasets import get_samples as original_get_samples
from vllm.benchmarks.datasets import add_dataset_parser as original_add_parser
from vllm.benchmarks.throughput import add_cli_args as original_add_cli_args
from vllm.benchmarks.throughput import get_requests as original_get_requests
from vllm.benchmarks.throughput import validate_args as original_validate_args
from vllm.benchmarks.throughput import main as original_main
import atexit
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput
from vllm.benchmarks.lib.endpoint_request_func import async_request_openai_chat_completions
import vllm.benchmarks.lib.endpoint_request_func

atexit.register(cleanup_dist_env_and_memory)

####################### for bench serve with beam search #######################
# Monkey patch to add beam search fields
original_init = RequestFuncInput.__init__


def __init__(self, *args, use_beam_search=True, n=8, **kwargs):
    original_init(self, *args, **kwargs)
    self.use_beam_search = use_beam_search
    self.n = n


RequestFuncInput.__init__ = __init__


async def patched_async_request_openai_chat_completions(request_func_input, pbar=None):
    # Add beam search to JSON payload
    if request_func_input.use_beam_search:
        if request_func_input.extra_body is None:
            request_func_input.extra_body = {}
        request_func_input.extra_body["use_beam_search"] = True
        request_func_input.extra_body["n"] = request_func_input.n

    return await async_request_openai_chat_completions(request_func_input, pbar)


###################################################################


def patched_get_samples(args, tokenizer):
    """Patch get_samples to handle OneRecDataset"""
    # Handle default request_id_prefix
    if not hasattr(args, "request_id_prefix"):
        args.request_id_prefix = ""

    # Check if dataset is onerec
    if args.dataset_name == "onerec":
        dataset = OneRecDataset(
            task_types=args.task_types.split(","),
            model_path=args.model,
            dataset_path=args.data_dir,
            disable_shuffle=args.disable_shuffle,
            tokenizer=tokenizer,
        )
        input_requests = dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
        )
        return input_requests

    # For all other datasets, call original function

    return original_get_samples(args, tokenizer)


def patched_add_parser(parser):
    """Patch add_dataset_parser to include onerec in choices"""
    # Call original parser first
    result = original_add_parser(parser)

    # Add onerec to dataset_name choices
    for action in parser._actions:
        if hasattr(action, "dest") and action.dest == "dataset_name":
            if "onerec" not in action.choices:
                action.choices = list(action.choices) + ["onerec"]
            break

    # Add onerec-specific argument group
    onerec_group = parser.add_argument_group("onerec dataset options")
    onerec_group.add_argument(
        "--task-types",
        type=str,
        default="rec_reason,item_understand,ad,product,label_cond,video,interactive,label_pred",
        help="Comma-separated list of task types",
    )
    onerec_group.add_argument(
        "--data-dir", type=str, default=None, help="Directory containing benchmark data"
    )
    onerec_group.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save results"
    )
    return result


def patched_add_cli_args(parser):
    """Patch add_cli_args to include onerec arguments"""
    # Call original parser first
    result = original_add_cli_args(parser)

    # Add onerec to dataset_name choices
    for action in parser._actions:
        if hasattr(action, "dest") and action.dest == "dataset_name":
            if "onerec" not in action.choices:
                action.choices = list(action.choices) + ["onerec"]
            break

    # Add onerec argument group
    onerec_group = parser.add_argument_group("onerec dataset options")
    onerec_group.add_argument(
        "--task-types",
        type=str,
        default="rec_reason,item_understand,ad,product,label_cond,video,interactive,label_pred",
        help="Comma-separated list of task types",
    )

    return result


def patched_get_requests(args, tokenizer):
    """Patch get_requests to handle OneRecDataset"""
    # Import your dataset
    from benchmarks.open_one_rec.open_one_rec_dataset import OneRecDataset

    # Handle onerec dataset
    if args.dataset_name == "onerec":
        dataset = OneRecDataset(
            dataset_path=args.dataset_path,
            task_types=args.task_types.split(","),
            model_path=args.model,
            tokenizer=tokenizer,
        )
        return dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
        )

    # Call original function for other datasets
    return original_get_requests(args, tokenizer)


def patched_validate_args(args):
    """Patch validate_args to handle onerec dataset"""
    # Handle onerec dataset BEFORE calling original validation
    if args.dataset_name == "onerec":
        # Skip the random dataset validation for onerec
        return

    # Call original validation for other datasets
    original_validate_args(args)


def patched_main(args):
    """Patch main to fix tokenizer initialization"""
    # Fix tokenizer argument
    if args.tokenizer is None and hasattr(args, "model"):
        args.tokenizer = args.model

    # Call original main
    original_main(args)


print("patching vllm.benchmarks.datasets.get_samples and add_dataset_parser")

# Apply the patches
vllm.benchmarks.datasets.get_samples = patched_get_samples
vllm.benchmarks.datasets.add_dataset_parser = patched_add_parser
vllm.benchmarks.throughput.add_cli_args = patched_add_cli_args
vllm.benchmarks.throughput.get_requests = patched_get_requests
vllm.benchmarks.throughput.validate_args = patched_validate_args
vllm.benchmarks.throughput.main = patched_main
vllm.benchmarks.lib.endpoint_request_func.async_request_openai_chat_completions = (
    patched_async_request_openai_chat_completions
)
