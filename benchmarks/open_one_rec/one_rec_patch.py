import json
import logging
import os
import sys
import time
import traceback
from typing import Literal

import aiohttp
import requests
import vllm
import vllm.benchmarks.datasets
import vllm.benchmarks.serve
import vllm.benchmarks.throughput
from tqdm import tqdm
from vllm.benchmarks.datasets import add_dataset_parser as original_add_parser
from vllm.benchmarks.datasets import get_samples as original_get_samples
from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
    StreamedResponseHandler,
    _get_chat_content,
    _update_headers_common,
    _update_payload_common,
    _validate_api_url,
)
from vllm.benchmarks.serve import add_cli_args as original_serve_add_cli_args
from vllm.benchmarks.serve import main as original_serve_main
from vllm.benchmarks.throughput import add_cli_args as original_add_cli_args
from vllm.benchmarks.throughput import get_requests as original_get_requests
from vllm.benchmarks.throughput import main as original_main
from vllm.benchmarks.throughput import validate_args as original_validate_args

from benchmarks.open_one_rec.open_one_rec_dataset import OneRecDataset

# ============================================================================
# Constants
# ============================================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
HTTP_OK = 200
DATASET_NAME = "onerec"
DEFAULT_TASK_TYPES = "ad,product,label_cond,video,interactive,label_pred"


# ============================================================================
# Prometheus Metrics Collection
# ============================================================================
def _fetch_prometheus_metrics(metrics_url: str) -> dict:
    """Fetch beam search metrics from the /metrics endpoint using Prometheus exposition format."""
    try:
        response = requests.get(f"{metrics_url}/metrics", timeout=5)
        if response.status_code != HTTP_OK:
            return {}

        metrics = {
            "execution_time": 0.0,
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "sort_time": 0.0,
            "requests": 0.0,
            "p2d_requests": 0.0,
            "beams": 0.0,
        }

        metric_names = {
            "execution_time": "vllm_gr_beam_search_execution_time_seconds_total",
            "prefill_time": "vllm_gr_beam_search_prefill_time_seconds_total",
            "decode_time": "vllm_gr_beam_search_decode_time_seconds_total",
            "sort_time": "vllm_gr_beam_search_sort_time_seconds_total",
            "requests": "vllm_gr_beam_search_requests_total",
            "p2d_requests": "vllm_gr_beam_search_p2d_requests_total",
            "p2d_requests_legacy": "vllm_gr_beam_search_execution_duration_seconds_count",
            "beams": "vllm_gr_beam_search_total_beams_generated_total",
        }

        # Track which metrics were actually exported by the server so we can
        # detect servers running an older vllm_gr without the expected
        # counters (in that case the fields would silently report 0).
        seen: set[str] = set()

        # Parse exposition format line-by-line
        for line in response.text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse: metric_name{labels} value
            for key, metric_name in metric_names.items():
                if line.startswith(metric_name):
                    try:
                        value = float(line.split()[-1])
                        metrics[key] = value
                        seen.add(key)
                    except (ValueError, IndexError):
                        pass

        if "p2d_requests" not in seen and "p2d_requests_legacy" in seen:
            metrics["p2d_requests"] = metrics.get("p2d_requests_legacy", 0.0)
            seen.add("p2d_requests")

        metrics["_seen"] = seen
        return metrics
    except Exception:  # noqa: BLE001
        return {}


def _get_metrics_url(args) -> str | None:
    """Build metrics base URL from serve args."""
    if hasattr(args, "base_url") and args.base_url:
        return args.base_url.rstrip("/")
    elif hasattr(args, "port") and args.port:
        host = getattr(args, "host", "127.0.0.1")
        return f"http://{host}:{args.port}"
    return None


def _print_beam_search_delta(before: dict, after: dict):
    """Print beam search metrics delta."""
    if not before or not after:
        return
    num = after.get("requests", 0) - before.get("requests", 0)
    if num <= 0:
        return
    avg_exec = (after.get("execution_time", 0) - before.get("execution_time", 0)) / num
    avg_prefill = (after.get("prefill_time", 0) - before.get("prefill_time", 0)) / num
    avg_decode = (after.get("decode_time", 0) - before.get("decode_time", 0)) / num
    avg_sort = (after.get("sort_time", 0) - before.get("sort_time", 0)) / num
    p2d_reqs = after.get("p2d_requests", 0) - before.get("p2d_requests", 0)
    seen_after = after.get("_seen", set())
    total = avg_exec + avg_prefill + avg_decode + avg_sort
    if "p2d_requests" in seen_after or "p2d_requests_legacy" in seen_after:
        p2d_reqs = max(0.0, min(float(num), p2d_reqs))
    elif avg_exec > 0 and avg_prefill == 0 and avg_decode == 0:
        p2d_reqs = float(num)
    else:
        p2d_reqs = 0.0

    print("---------------Beam Search Metrics----------------")
    print(f"Total Beam Time (ms):                    {total * 1000:.2f}")
    print(f"P2D Requests:                            {int(p2d_reqs)}")
    print(f"Beam Search Requests (total):            {int(num)}")
    print("=" * 50)


def _consume_use_beam_search_flag():
    """Check for `--use-beam-search` in sys.argv, remove it, and return True if present.

    This avoids argparse errors for unknown options while allowing users to pass
    `--use-beam-search` on the command line. We intentionally consume the token
    at import time so that the existing vLLM parsers won't see it."""
    if "--use-beam-search" in sys.argv:
        logger.info(
            "Detected --use-beam-search flag in command. Enabling beam search for this benchmark run."
        )
        sys.argv.remove("--use-beam-search")
        return True
    return False


def _get_n_beams_from_argv():
    """
    Parses sys.argv to find the value for --n.
    NOTE: This is a workaround. Reading directly from sys.argv at module import
    is fragile. A more robust solution would involve passing the parsed 'n'
    argument from the benchmark's main function down to where RequestFuncInput
    is created.
    """
    try:
        if "--n" in sys.argv:
            idx = sys.argv.index("--n")
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
                n_value = sys.argv[idx + 1]
                return int(n_value)
    except (ValueError, IndexError):
        pass  # Fallback to 1 if parsing fails
    return 1


def _get_output_len_from_argv():
    """
    Parses sys.argv to find the value for --output-len.
    NOTE: This is a workaround. Reading directly from sys.argv at module import
    is fragile. A more robust solution would involve passing the parsed 'output_len'
    argument from the benchmark's main function down to where RequestFuncInput
    is created.
    """
    try:
        if "--output-len" in sys.argv:
            idx = sys.argv.index("--output-len")
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
                output_len_value = sys.argv[idx + 1]
                return int(output_len_value)
    except (ValueError, IndexError):
        pass  # Fallback to 1 if parsing fails
    return 0


def _get_beam_star_from_argv():
    """
    Parses sys.argv to find the value for --beam-star.
    NOTE: Reading directly from sys.argv at module import is a fallback
    for when RequestFuncInput is initialized directly.
    """
    try:
        if "--beam-star" in sys.argv:
            idx = sys.argv.index("--beam-star")
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
                return float(sys.argv[idx + 1])
            return 3.3
    except (ValueError, IndexError):
        pass
    return None


def _get_p2d_mode_from_argv():
    """Parses sys.argv to find the value for --p2d-mode."""
    try:
        if "--p2d-mode" in sys.argv:
            idx = sys.argv.index("--p2d-mode")
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
                return sys.argv[idx + 1].lower()
    except (ValueError, IndexError):
        pass
    return None


def _parse_step_beam_widths_value(raw_value: str | None) -> list[int] | None:
    """Validates a step-beam-widths string: 3 positive integers, else None."""
    if raw_value is None:
        return None
    try:
        widths = [int(x.strip()) for x in raw_value.split(",") if x.strip()]
    except (TypeError, ValueError) as exc:
        logger.warning("Invalid --step-beam-widths=%r: %r; ignoring.", raw_value, exc)
        return None
    if len(widths) != 3 or any(w <= 0 for w in widths):
        logger.warning(
            "Invalid --step-beam-widths=%r: expected 3 positive integers; ignoring.",
            raw_value,
        )
        return None
    return widths


def _get_step_beam_widths_from_argv():
    """Parses sys.argv to find --step-beam-widths (space- or equals-separated)."""
    try:
        for idx, arg in enumerate(sys.argv):
            if arg.startswith("--step-beam-widths="):
                return _parse_step_beam_widths_value(arg.split("=", 1)[1])
            if arg == "--step-beam-widths" and idx + 1 < len(sys.argv):
                next_arg = sys.argv[idx + 1]
                if not next_arg.startswith("-"):
                    return _parse_step_beam_widths_value(next_arg)
    except (ValueError, IndexError):
        pass
    return None


USE_BEAM_SEARCH = _consume_use_beam_search_flag()
ARGS_N_BEAMS = _get_n_beams_from_argv()
ARGS_OUTPUT_LEN = _get_output_len_from_argv()
ARGS_BEAM_STAR = _get_beam_star_from_argv()
ARGS_P2D_MODE = _get_p2d_mode_from_argv()
ARGS_STEP_BEAM_WIDTHS = _get_step_beam_widths_from_argv()

# ============================================================================
# Beam Search Patching
# ============================================================================
original_init = RequestFuncInput.__init__


def __init__(
    self,
    *args,
    use_beam_search=USE_BEAM_SEARCH,
    n=ARGS_N_BEAMS,
    args_output_len=ARGS_OUTPUT_LEN,
    beam_star=ARGS_BEAM_STAR,
    p2d_mode=ARGS_P2D_MODE,
    step_beam_widths=ARGS_STEP_BEAM_WIDTHS,
    **kwargs,
):
    """Patched init to support beam search parameters."""
    original_init(self, *args, **kwargs)
    self.use_beam_search = use_beam_search
    self.n = n
    self.beam_star = beam_star
    self.p2d_mode = p2d_mode
    self.step_beam_widths = step_beam_widths
    if args_output_len > 0:
        logger.info(f"Setting output_len to {args_output_len} based on command line argument")
        self.output_len = args_output_len


RequestFuncInput.__init__ = __init__


async def patched_async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
    mm_position: Literal["first", "last"] = "last",
):
    """Add beam search parameters to request payload."""
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Chat Completions API", "chat/completions")

    content = _get_chat_content(request_func_input, mm_position=mm_position)
    temperature = 0.0
    if hasattr(request_func_input, "temperature") and request_func_input.temperature is not None:
        temperature = request_func_input.temperature
    elif (
        hasattr(request_func_input, "extra_body")
        and request_func_input.extra_body is not None
        and hasattr(request_func_input.extra_body, "temperature")
        and request_func_input.extra_body.temperature is not None
    ):
        temperature = request_func_input.extra_body.temperature
    vllm_xargs = {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"}
    beam_star_val = getattr(request_func_input, "beam_star", None)
    if beam_star_val is None:
        beam_star_val = ARGS_BEAM_STAR
    if beam_star_val is not None:
        vllm_xargs["beam_star"] = True
        vllm_xargs["beam_star_coef"] = float(beam_star_val)

    step_bw = getattr(request_func_input, "step_beam_widths", None) or ARGS_STEP_BEAM_WIDTHS
    if step_bw:
        vllm_xargs["step_beam_widths"] = [int(w) for w in step_bw]
    # When --n was not passed (defaults to 1) but step widths are active, size
    # the request's beam count to the final width so the response carries w2
    # beams and the engine reserves enough logprob space.
    request_n = request_func_input.n
    if step_bw and request_n in (None, 1):
        request_n = step_bw[-1]

    payload = {
        "model": (
            request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model
        ),
        "messages": [
            {"role": "user", "content": content},
        ],
        "use_beam_search": bool(request_func_input.use_beam_search) or (step_bw is not None),
        "n": request_n,
        "temperature": temperature,
        "max_tokens": request_func_input.output_len,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
        "vllm_xargs": vllm_xargs,
    }
    if getattr(request_func_input, "p2d_mode", None):
        payload["extra_body"] = {"p2d_mode": request_func_input.p2d_mode}
    _update_payload_common(payload, request_func_input)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    _update_headers_common(headers, request_func_input)

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == HTTP_OK:
                handler = StreamedResponseHandler()
                async for chunk_bytes in response.content.iter_any():
                    if not chunk_bytes:
                        continue

                    messages = handler.add_chunk(chunk_bytes)
                    for message in messages:
                        message = message.strip()
                        if not message:
                            continue

                        # NOTE: SSE comments (often used as pings) start with
                        # a colon. These are not JSON data payload and should
                        # be skipped.
                        if message.startswith((":", "event:", "id:")):
                            continue

                        if not message.startswith("data:"):
                            continue

                        chunk = message.removeprefix("data:").strip()
                        if not chunk or chunk == "[DONE]":
                            continue

                        timestamp = time.perf_counter()
                        try:
                            data = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue

                        if choices := data.get("choices"):
                            delta = choices[0]["delta"]
                            content = delta.get("content")
                            reasoning_content = delta.get("reasoning_content")
                            # First token
                            if ttft == 0.0:
                                ttft = timestamp - st
                                output.ttft = ttft

                            # Decoding phase
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)

                            if reasoning_content:
                                generated_text += reasoning_content
                            if content:
                                generated_text += content
                        elif usage := data.get("usage"):
                            output.output_tokens = usage.get("completion_tokens")

                        most_recent_timestamp = timestamp

                output.generated_text = generated_text
                output.success = True
                output.latency = most_recent_timestamp - st
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:  # noqa: BLE001
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


# ============================================================================
# Dataset Patching
# ============================================================================
def _add_dataset_to_parser_choices(parser, dataset_name):
    """Helper to add a dataset to parser choices."""
    for action in parser._actions:
        if hasattr(action, "dest") and action.dest == "dataset_name":
            if dataset_name not in action.choices:
                action.choices = list(action.choices) + [dataset_name]
            return True
    return False


def _has_option(parser, opt: str) -> bool:
    """Return True if the parser already knows about option string ``opt``.

    Used to avoid argparse's "conflicting option string" error when vLLM's
    own benchmark parser has already registered the same flag.
    """
    for action in parser._actions:
        if hasattr(action, "option_strings") and opt in action.option_strings:
            return True
    return False


def _add_onerec_arguments(parser):
    """Helper to add OneRec-specific arguments to parser."""
    onerec_group = parser.add_argument_group("onerec dataset options")

    onerec_group.add_argument(
        "--task-types",
        type=str,
        default=DEFAULT_TASK_TYPES,
        help="Comma-separated list of task types",
    )
    if not _has_option(parser, "--custom-input-len"):
        onerec_group.add_argument(
            "--custom-input-len",
            type=int,
            default=None,
            help="Target prompt token length; prompts are trimmed or repeated to match.",
        )
    # enable_thinking is normally controlled by each task's config, but we
    # expose it on the CLI so users can override it for ablation runs.
    if not _has_option(parser, "--enable-thinking"):
        onerec_group.add_argument(
            "--enable-thinking",
            action="store_true",
            default=None,
            help="Force-enable thinking mode (overrides task config).",
        )
    if not _has_option(parser, "--disable-thinking"):
        onerec_group.add_argument(
            "--disable-thinking",
            action="store_false",
            dest="enable_thinking",
            help="Force-disable thinking mode (overrides task config).",
        )


def _add_beams_arguments(parser):
    """Helper to add beam search arguments to parser."""
    beams_group = parser.add_argument_group("beam search options")

    if not _has_option(parser, "--n"):
        beams_group.add_argument(
            "--n",
            type=int,
            default=ARGS_N_BEAMS,
            help="Number of beams for beam search",
        )
    if not _has_option(parser, "--beam-star"):
        beams_group.add_argument(
            "--beam-star",
            type=float,
            nargs="?",
            const=3.3,
            default=None,
            help="Enable beam* candidate path masking with given coefficient (default: 3.3)",
        )
    if not _has_option(parser, "--p2d-mode"):
        beams_group.add_argument(
            "--p2d-mode",
            type=str,
            default=None,
            choices=["auto", "monolithic", "eager_p2d"],
            help="P2D execution mode override: auto, monolithic, or eager_p2d",
        )
    if not _has_option(parser, "--step-beam-widths"):
        beams_group.add_argument(
            "--step-beam-widths",
            type=str,
            default=None,
            help="Comma-separated per-step beam widths (W0,W1,W2) for P2D.",
        )


def patched_get_samples(args, tokenizer):
    """Patch get_samples to handle OneRecDataset."""
    if not hasattr(args, "request_id_prefix"):
        args.request_id_prefix = ""

    if args.dataset_name == DATASET_NAME:
        dataset = OneRecDataset(
            task_types=args.task_types.split(","),
            dataset_path=args.dataset_path,
            disable_shuffle=args.disable_shuffle,
            tokenizer=tokenizer,
            custom_input_len=args.custom_input_len,
            sample_size=args.num_prompts,
            enable_thinking=getattr(args, "enable_thinking", None),
        )
        return dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            request_id_prefix=args.request_id_prefix,
            # OneRec intentionally disables oversample: duplicated requests
            # would inflate throughput via spurious KV-cache hits.
            no_oversample=True,
        )

    return original_get_samples(args, tokenizer)


def patched_add_parser(parser):
    """Patch add_dataset_parser to include onerec in choices."""
    original_add_parser(parser)
    _add_dataset_to_parser_choices(parser, DATASET_NAME)
    _add_onerec_arguments(parser)
    _add_beams_arguments(parser)
    return parser


def patched_add_cli_args(parser):
    """Patch add_cli_args to include onerec arguments."""
    original_add_cli_args(parser)
    parser.add_argument(
        "--beam-search-v1",
        action="store_true",
        help="Use canonical V1 and measure sequential text-to-text cache misses",
    )
    parser.add_argument("--constraint-triples", help="Canonical three-step constraint JSON for V1")
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=10,
        help="Warmup requests excluded from V1 latency statistics",
    )
    _add_dataset_to_parser_choices(parser, DATASET_NAME)
    _add_onerec_arguments(parser)
    return parser


def patched_get_requests(args, tokenizer):
    """Patch get_requests to handle OneRecDataset."""
    if args.dataset_name == DATASET_NAME:
        dataset = OneRecDataset(
            dataset_path=args.dataset_path,
            task_types=args.task_types.split(","),
            tokenizer=tokenizer,
            custom_input_len=args.custom_input_len,
            sample_size=args.num_prompts,
            # throughput benchmark doesn't register --disable-shuffle, so it
            # may be absent from args; default to False (shuffle enabled).
            disable_shuffle=getattr(args, "disable_shuffle", False),
            enable_thinking=getattr(args, "enable_thinking", None),
        )
        return dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            request_id_prefix=getattr(args, "request_id_prefix", ""),
            # OneRec intentionally disables oversample (see patched_get_samples).
            no_oversample=True,
        )

    return original_get_requests(args, tokenizer)


def patched_validate_args(args):
    """Patch validate_args to handle onerec dataset.

    For onerec, run a minimal self-contained validation instead — the
    upstream validator has dataset-specific branches (random/hf/sonnet)
    that would misroute onerec (e.g. rewrite dataset_name to "random"
    when dataset_path is empty).
    """
    if args.dataset_name != DATASET_NAME:
        original_validate_args(args)
        return

    # onerec-specific minimal validation
    if not getattr(args, "tokenizer", None):
        args.tokenizer = args.model

    if not getattr(args, "dataset_path", None):
        raise ValueError(
            "--dataset-path is required for onerec dataset. Example: --dataset-path ./data"
        )


def _run_offline_beam_search_benchmark(args):
    """
    Run offline beam search benchmark for OneRec dataset.

    This is called when --use-beam-search is specified for offline throughput mode.
    It replaces the standard throughput benchmark for beam search scenarios.
    """

    try:
        from vllm_gr.sampling_params import BeamSearchParams
    except ImportError:
        from vllm.sampling_params import BeamSearchParams
    from transformers import AutoTokenizer

    from benchmarks.open_one_rec.open_one_rec_dataset import OneRecDataset
    from vllm_gr.entrypoints.gr import GRLLM

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    use_v1 = getattr(args, "beam_search_v1", False)
    sample_size = max(args.num_prompts, args.warmup_requests) if use_v1 else args.num_prompts
    # Load dataset
    logger.info(f"Loading OneRec dataset from {args.dataset_path}")
    dataset = OneRecDataset(
        dataset_path=args.dataset_path,
        task_types=args.task_types.split(","),
        tokenizer=tokenizer,
        custom_input_len=getattr(args, "custom_input_len", None),
        sample_size=sample_size,
        enable_thinking=getattr(args, "enable_thinking", None),
    )

    # Sample requests
    requests = dataset.sample(
        num_requests=sample_size,
        no_oversample=use_v1,
        tokenizer=tokenizer,
    )

    if use_v1:
        from benchmarks.open_one_rec.one_rec_v1_benchmark import run_v1_benchmark

        run_v1_benchmark(args, tokenizer, requests)
        return

    prompts = [{"prompt": req.prompt} for req in requests]

    # Create beam search parameters
    beam_width = getattr(args, "n", 8)
    max_tokens = requests[0].expected_output_len if requests else 128
    if hasattr(args, "output_len") and args.output_len is not None:
        max_tokens = args.output_len

    params = BeamSearchParams(
        beam_width=beam_width,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    # Add vLLM-GR specific parameters if available
    if hasattr(BeamSearchParams, "begin_token"):
        params.begin_token = "<|sid_begin|>"
        params.end_token = "<|sid_end|>"
    beam_star_val = getattr(args, "beam_star", ARGS_BEAM_STAR)
    if beam_star_val is not None:
        params.beam_star = True
        params.beam_star_coef = float(beam_star_val)
    step_bw = getattr(args, "step_beam_widths", None) or ARGS_STEP_BEAM_WIDTHS
    widths = None
    if step_bw:
        if isinstance(step_bw, str):
            widths = _parse_step_beam_widths_value(step_bw)
        elif isinstance(step_bw, (list, tuple)):
            widths = [int(w) for w in step_bw]
    if widths and hasattr(BeamSearchParams, "step_beam_widths"):
        params.step_beam_widths = widths

    logger.info(f"Running beam search benchmark with {beam_width} beams on {len(prompts)} prompts")
    logger.info(f"Max tokens per request: {max_tokens}")

    # The throughput CLI registers --max-logprobs through
    # AsyncEngineArgs.add_cli_args with a default of 20, so the attribute is
    # always present even when the operator never set it.  Beam search samples
    # beam_width logprobs per step, so raise the floor to beam_width + 1 (one
    # extra row for the sampled token) unless an explicit larger value was
    # passed on the command line.
    max_logprobs = max(getattr(args, "max_logprobs", None) or 0, beam_width + 1)
    logger.info(f"Offline beam search max_logprobs: {max_logprobs}")

    # Beam decode on the CUSTOM backend requires a persistent beam KV pool
    # sized to the maximum served beam width; the throughput CLI has no
    # --beam-max-width flag, so derive it from the request beam width.
    beam_max_width = getattr(args, "beam_max_width", None) or beam_width

    # Keep engine cleanup scoped to the process that owns the offline engine.
    # Online benchmarks are clients of a separate server and must not register
    # process-wide vLLM distributed cleanup callbacks.
    logger.info(f"Initializing GRLLM with model: {args.model}")
    with GRLLM(
        model=args.model,
        trust_remote_code=True,
        max_logprobs=max_logprobs,
        beam_max_width=beam_max_width,
        attention_config={"backend": "CUSTOM"},
        # Add other relevant vLLM arguments if available in args
        **(
            {"max_num_batched_tokens": getattr(args, "max_num_batched_tokens", None)}
            if hasattr(args, "max_num_batched_tokens") and args.max_num_batched_tokens
            else {}
        ),
    ) as llm:
        start_time = time.perf_counter()
        outputs = llm.beam_search(prompts, params)
        end_time = time.perf_counter()

    # Calculate metrics
    total_tokens = sum(len(output.sequences) * max_tokens for output in outputs)
    elapsed_time = end_time - start_time

    # Prepare results
    results = {
        "elapsed_time": elapsed_time,
        "num_requests": len(prompts),
        "total_num_tokens": total_tokens,
        "requests_per_second": len(prompts) / elapsed_time,
        "tokens_per_second": total_tokens / elapsed_time,
        "beam_width": beam_width,
    }

    logger.info("\n" + "=" * 60)
    logger.info("Offline Beam Search Benchmark Results")
    logger.info("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.2f}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=" * 60)

    # Save results if requested
    if hasattr(args, "output_json") and args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {args.output_json}")


def patched_main(args):
    """Patch main to handle beam search and fix tokenizer initialization."""
    # Check if this is an offline beam search benchmark
    is_beam_search = (
        getattr(args, "beam_search_v1", False)
        or USE_BEAM_SEARCH
        or (hasattr(args, "n") and args.n and args.n > 1)
        or bool(getattr(args, "step_beam_widths", None))
    )
    is_offline = not hasattr(args, "endpoint") or not args.endpoint

    if is_beam_search and is_offline and args.dataset_name == DATASET_NAME:
        # Use custom offline beam search benchmark
        _run_offline_beam_search_benchmark(args)
    else:
        # Use standard vLLM benchmark
        if args.tokenizer is None and hasattr(args, "model"):
            args.tokenizer = args.model
        original_main(args)


def patched_serve_add_cli_args(parser):
    """Patch serve.add_cli_args to include onerec in choices."""
    original_serve_add_cli_args(parser)
    _add_dataset_to_parser_choices(parser, DATASET_NAME)
    _add_onerec_arguments(parser)
    _add_beams_arguments(parser)
    return parser


def patched_serve_main(args):
    """Patch serve.main to collect beam search metrics before/after."""
    metrics_url = _get_metrics_url(args)
    is_beam_search = (
        USE_BEAM_SEARCH
        or (hasattr(args, "n") and getattr(args, "n", 1) > 1)
        or bool(getattr(args, "step_beam_widths", None))
    )

    metrics_before = None
    if is_beam_search and metrics_url:
        metrics_before = _fetch_prometheus_metrics(metrics_url)

    result = original_serve_main(args)

    if is_beam_search and metrics_url:
        metrics_after = _fetch_prometheus_metrics(metrics_url)
        _print_beam_search_delta(metrics_before, metrics_after)

    return result


# ============================================================================
# Apply Patches
# ============================================================================
logger.info("Patching vllm.benchmarks for OneRec support")

vllm.benchmarks.datasets.get_samples = patched_get_samples
vllm.benchmarks.datasets.add_dataset_parser = patched_add_parser
vllm.benchmarks.throughput.add_cli_args = patched_add_cli_args
vllm.benchmarks.throughput.get_requests = patched_get_requests
vllm.benchmarks.throughput.validate_args = patched_validate_args
vllm.benchmarks.throughput.main = patched_main
vllm.benchmarks.serve.add_cli_args = patched_serve_add_cli_args
vllm.benchmarks.serve.get_samples = patched_get_samples
vllm.benchmarks.serve.main = patched_serve_main

# Also patch the CLI module if it already imported the original reference
if "vllm.entrypoints.cli.benchmark.serve" in sys.modules:
    sys.modules["vllm.entrypoints.cli.benchmark.serve"].add_cli_args = patched_serve_add_cli_args
    sys.modules["vllm.entrypoints.cli.benchmark.serve"].main = patched_serve_main

ASYNC_REQUEST_FUNCS["openai-chat"] = patched_async_request_openai_chat_completions
