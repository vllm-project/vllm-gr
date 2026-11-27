# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Adapted from https://github.com/vllm-project/vllm

import json
import logging
import math
import os
import sys
import time
import traceback
from codecs import BOM_UTF8
from collections.abc import AsyncIterable, AsyncIterator
from importlib import import_module
from typing import Literal

import aiohttp
import requests
from tqdm import tqdm
from vllm.benchmarks.datasets import add_dataset_parser as original_add_parser
from vllm.benchmarks.datasets import get_samples as original_get_samples
from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
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

from benchmarks.open_one_rec.chat import messages_for_prompt, text_from_message
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
_BeamMetricSnapshot = dict[str, dict[tuple[tuple[str, str], ...], float]]


def _fetch_prometheus_metrics(metrics_url: str) -> _BeamMetricSnapshot:
    """Fetch beam search metrics from the /metrics endpoint using Prometheus exposition format."""
    try:
        from prometheus_client.parser import text_string_to_metric_families

        response = requests.get(f"{metrics_url}/metrics", timeout=5)
        if response.status_code != HTTP_OK:
            return {}

        # This is compatibility with older servers. Missing metrics must stay
        # absent: V1 has no producer for the retired frontend phase counters.
        metrics: _BeamMetricSnapshot = {}

        metric_names = {
            "vllm_gr_beam_search_prefill_time_seconds_total": "prefill_time",
            "vllm_gr_beam_search_decode_time_seconds_total": "decode_time",
            "vllm_gr_beam_search_sort_time_seconds_total": "sort_time",
            "vllm_gr_beam_search_requests_total": "requests",
            "vllm_gr_beam_search_total_beams_generated_total": "beams",
        }

        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                # Match exactly: *_created is not the counter. Preserve labels
                # until deltas are validated so another engine cannot hide a reset.
                key = metric_names.get(sample.name)
                if key is None:
                    continue
                labels = tuple(sorted(sample.labels.items()))
                series = metrics.setdefault(key, {})
                if labels in series or not math.isfinite(sample.value) or sample.value < 0:
                    return {}
                series[labels] = sample.value

        return metrics
    except Exception:
        return {}


def _get_metrics_url(args) -> str | None:
    """Build metrics base URL from serve args."""
    if hasattr(args, "base_url") and args.base_url:
        return args.base_url.rstrip("/")
    elif hasattr(args, "port") and args.port:
        host = getattr(args, "host", "127.0.0.1")
        return f"http://{host}:{args.port}"
    return None


def _print_beam_search_delta(
    before: _BeamMetricSnapshot | None, after: _BeamMetricSnapshot | None
) -> None:
    """Print beam search metrics delta."""
    required = {"prefill_time", "decode_time", "sort_time", "requests"}
    # Some older servers omit beams. Once observed, it must obey the same
    # snapshot consistency checks even though it is not used in phase averages.
    if (before and "beams" in before) or (after and "beams" in after):
        required.add("beams")
    if not before or not after or not required <= before.keys() & after.keys():
        logger.info("Legacy Beam phase metrics unavailable; use benchmark E2E statistics.")
        return
    labels = before["requests"].keys()
    if not labels or any(
        snapshot[key].keys() != labels for snapshot in (before, after) for key in required
    ):
        logger.info("Legacy Beam phase series changed or incomplete; use benchmark E2E statistics.")
        return
    series_deltas = {
        key: [after[key][label] - before[key][label] for label in labels] for key in required
    }
    # Reject the whole window before aggregation, including resets hidden by
    # growth in other series. A stable series with no new requests is valid.
    if any(
        not math.isfinite(value) or value < 0
        for values in series_deltas.values()
        for value in values
    ):
        logger.info("Legacy Beam phase counter reset; use benchmark E2E statistics.")
        return
    delta = {key: sum(values) for key, values in series_deltas.items()}
    if delta["requests"] <= 0 or not all(math.isfinite(value) for value in delta.values()):
        return
    num = delta["requests"]
    avg_prefill = delta["prefill_time"] / num
    avg_decode = delta["decode_time"] / num
    avg_sort = delta["sort_time"] / num
    total = avg_prefill + avg_decode + avg_sort
    print("---------------Beam Search Metrics----------------")
    print(f"Avg Prefill Time (ms):                   {avg_prefill * 1000:.2f}")
    print(f"Avg Decode Time (ms):                    {avg_decode * 1000:.2f}")
    print(f"Avg Sort Time (ms):                      {avg_sort * 1000:.3f}")
    print(f"Total Beam Time (ms):                    {total * 1000:.2f}")
    print(f"Beam Search Requests:                    {int(num)}")
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


USE_BEAM_SEARCH = _consume_use_beam_search_flag()
ARGS_N_BEAMS = _get_n_beams_from_argv()
ARGS_OUTPUT_LEN = _get_output_len_from_argv()

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
    **kwargs,
):
    """Patched init to support beam search parameters."""
    original_init(self, *args, **kwargs)
    self.use_beam_search = use_beam_search
    self.n = n
    if args_output_len > 0:
        logger.info(f"Setting output_len to {args_output_len} based on command line argument")
        self.output_len = args_output_len


RequestFuncInput.__init__ = __init__


async def _iter_sse_data(chunks: AsyncIterable[bytes]) -> AsyncIterator[str]:
    buffer = b""
    stream_start = True
    previous_chunk_ended_in_cr = False
    async for chunk in chunks:
        if not chunk:
            continue
        # A CRLF may straddle network chunks.
        if previous_chunk_ended_in_cr:
            chunk = chunk.removeprefix(b"\n")
        previous_chunk_ended_in_cr = chunk.endswith(b"\r")
        buffer += chunk.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

        while b"\n\n" in buffer:
            event, buffer = buffer.split(b"\n\n", 1)
            if stream_start:
                event = event.removeprefix(BOM_UTF8)
                stream_start = False
            data_lines = []
            for line in event.split(b"\n"):
                field, _, value = line.partition(b":")
                if field == b"data":
                    data_lines.append(value.removeprefix(b" "))
            if payload := b"\n".join(data_lines):
                yield payload.decode("utf-8")


async def patched_async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
    mm_position: Literal["first", "last"] = "last",
):
    """Add beam search parameters to request payload."""
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Chat Completions API", "chat/completions")

    chat_messages = request_func_input.chat_messages
    content = (
        _get_chat_content(request_func_input, mm_position=mm_position)
        if chat_messages is None
        else None
    )
    temperature = 0.0
    if hasattr(request_func_input, "temperature") and request_func_input.temperature is not None:
        temperature = request_func_input.temperature
    elif hasattr(request_func_input, "extra_body") and request_func_input.extra_body is not None:
        if (
            hasattr(request_func_input.extra_body, "temperature")
            and request_func_input.extra_body.temperature is not None
        ):
            temperature = request_func_input.extra_body.temperature
    payload = {
        "model": request_func_input.model_name
        if request_func_input.model_name
        else request_func_input.model,
        "messages": messages_for_prompt(content, chat_messages, parse_transcript=False),
        "use_beam_search": request_func_input.use_beam_search,
        "n": request_func_input.n,
        "temperature": temperature,
        "max_tokens": request_func_input.output_len,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
        "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"},
    }
    _update_payload_common(payload, request_func_input)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    _update_headers_common(headers, request_func_input)

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    last_text_timestamp: float | None = None
    received_done = False
    stream_error: str | None = None
    st = time.perf_counter()
    output.start_time = st
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == HTTP_OK:
                async for chunk in _iter_sse_data(response.content.iter_any()):
                    if chunk == "[DONE]":
                        received_done = True
                        break

                    data = json.loads(chunk)
                    if "error" in data:
                        if stream_error is None:
                            error = data["error"]
                            error_message = (
                                error.get("message") if isinstance(error, dict) else error
                            )
                            stream_error = str(error_message or chunk)
                        continue

                    if choices := data.get("choices"):
                        delta = choices[0]["delta"]
                        text = text_from_message(delta)
                        if text:
                            timestamp = time.perf_counter()
                            if last_text_timestamp is None:
                                output.ttft = timestamp - st
                            else:
                                output.itl.append(timestamp - last_text_timestamp)
                            last_text_timestamp = timestamp
                            generated_text += text
                    if usage := data.get("usage"):
                        output.output_tokens = usage.get("completion_tokens", output.output_tokens)
                        output.prompt_len = usage.get("prompt_tokens", output.prompt_len)

                output.success = (
                    received_done and last_text_timestamp is not None and stream_error is None
                )
                if stream_error is not None:
                    output.error = stream_error
                elif not received_done:
                    output.error = "response ended without [DONE]"
                elif last_text_timestamp is None:
                    output.error = "response contains no text output"
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = stream_error or "".join(traceback.format_exception(*exc_info))

    output.generated_text = generated_text
    output.latency = time.perf_counter() - st
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
            chat_template_kwargs=(getattr(args, "extra_body", None) or {}).get(
                "chat_template_kwargs"
            ),
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
    from benchmarks.streaming_head import add_streaming_head_argument

    original_add_cli_args(parser)
    add_streaming_head_argument(parser)
    parser.add_argument(
        "--beam-search",
        action="store_true",
        help="Select Beam benchmarking (all GR Beam benchmarks use V1)",
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
            "--dataset-path is required for onerec dataset. "
            "Example: --dataset-path ./data/benchmark_data"
        )


def _run_offline_beam_search_benchmark(args):
    """Run the supported OneRec Beam benchmark through persistent V1."""
    from transformers import AutoTokenizer

    from benchmarks.open_one_rec.one_rec_v1_benchmark import run_v1_benchmark
    from benchmarks.open_one_rec.open_one_rec_dataset import OneRecDataset

    if not getattr(args, "constraint_triples", None) or not getattr(args, "output_json", None):
        raise ValueError("Offline Beam V1 requires --constraint-triples and --output-json")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sample_size = max(args.num_prompts, args.warmup_requests)
    dataset = OneRecDataset(
        dataset_path=args.dataset_path,
        task_types=args.task_types.split(","),
        tokenizer=tokenizer,
        custom_input_len=getattr(args, "custom_input_len", None),
        sample_size=sample_size,
        enable_thinking=getattr(args, "enable_thinking", None),
    )
    requests = dataset.sample(
        num_requests=sample_size,
        no_oversample=True,
        tokenizer=tokenizer,
    )
    run_v1_benchmark(args, tokenizer, requests)


def patched_main(args):
    """Patch main to handle beam search and fix tokenizer initialization."""
    # Check if this is an offline beam search benchmark
    is_beam_search = (
        getattr(args, "beam_search", False)
        or USE_BEAM_SEARCH
        or (hasattr(args, "n") and args.n and args.n > 1)
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
    """Run the benchmark, optionally reading phase counters from older servers."""
    metrics_url = _get_metrics_url(args)
    is_beam_search = USE_BEAM_SEARCH or (hasattr(args, "n") and getattr(args, "n", 1) > 1)

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

# Cached submodules may outlive their parent-package attribute bindings.
# Resolve the actual modules rather than traversing vllm.benchmarks attributes.
_datasets_module = import_module("vllm.benchmarks.datasets")
_throughput_module = import_module("vllm.benchmarks.throughput")
_serve_module = import_module("vllm.benchmarks.serve")

_datasets_module.get_samples = patched_get_samples
_datasets_module.add_dataset_parser = patched_add_parser
_throughput_module.add_cli_args = patched_add_cli_args
_throughput_module.get_requests = patched_get_requests
_throughput_module.validate_args = patched_validate_args
_throughput_module.main = patched_main
_serve_module.add_cli_args = patched_serve_add_cli_args
_serve_module.get_samples = patched_get_samples
_serve_module.main = patched_serve_main

# Also patch the CLI module if it already imported the original reference
if "vllm.entrypoints.cli.benchmark.serve" in sys.modules:
    sys.modules["vllm.entrypoints.cli.benchmark.serve"].add_cli_args = patched_serve_add_cli_args
    sys.modules["vllm.entrypoints.cli.benchmark.serve"].main = patched_serve_main

ASYNC_REQUEST_FUNCS["openai-chat"] = patched_async_request_openai_chat_completions
