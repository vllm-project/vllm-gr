# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native startup settings shared by Python and CLI entrypoints."""

import argparse
import logging
import os
from collections.abc import Iterator, Mapping
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


def configure_model_runner_default() -> None:
    """Default vllm-gr processes to the V1 model runner.

    The GR beam-search worker patches target the V1 GPU/NPU model runner
    lifecycle by default. vLLM can enable ModelRunner V2 automatically for
    supported models such as Qwen3, so set the environment default before
    vLLM constructs its configuration.

    An explicit ``VLLM_USE_V2_MODEL_RUNNER=1`` (or any truthy spelling) is
    preserved as an opt-in experimental path: the V2 runner is
    beam_search-only (``execution_mode=v1`` persistent GR stage path).  The
    LEGACY ``beam_data`` frame stack is not supported under V2 and fails fast.
    """
    value = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    if value is None:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        logger.info(
            "VLLM_USE_V2_MODEL_RUNNER was not set; defaulting to ModelRunner V1 "
            "for vllm-gr compatibility."
        )
        return

    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        # Normalize truthy spellings to "1": vLLM's env parser converts the
        # value with int(), which rejects "true"/"yes"/"on" (ValueError).
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        logger.warning(
            "VLLM_USE_V2_MODEL_RUNNER=%s is an experimental vllm-gr path; "
            "the V2 runner is beam_search-only (legacy beam_data frames "
            "fail fast).",
            value,
        )
    elif normalized in ("0", "false", "no", "off", ""):
        # Likewise normalize falsy spellings (and empty string) to "0".
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    # Any other value is left untouched so vLLM surfaces its own parse error.


def argument_parsers(parser: argparse.ArgumentParser) -> Iterator[argparse.ArgumentParser]:
    yield parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                yield from argument_parsers(child)


@lru_cache(maxsize=1)
def _native_parser() -> argparse.ArgumentParser:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from vllm_gr.engine.arg_utils import origin_add_cli_args

    return origin_add_cli_args(FlexibleArgumentParser(add_help=False))


def native_settings(
    document: Mapping, *, parser: argparse.ArgumentParser | None = None
) -> dict[str, Any]:
    """Use upstream CLI converters for strings, retaining typed YAML nulls/lists."""
    actions = {
        action.dest: action
        for owner in argument_parsers(parser or _native_parser())
        for action in owner._actions
    }
    result = {}
    for key, value in document.items():
        if key == "gr":
            continue
        name = key.replace("-", "_")
        if name not in actions:
            raise ValueError(f"Unknown engine configuration setting: {key}")
        action = actions[name]
        try:
            if action.type is not None:
                if isinstance(value, str):
                    value = action.type(value)
                elif isinstance(value, list) and action.nargs is not None:
                    value = [action.type(item) if isinstance(item, str) else item for item in value]
            if value is not None and action.choices is not None:
                choices = value if isinstance(value, list) else [value]
                if any(item not in action.choices for item in choices):
                    raise ValueError(f"expected one of {action.choices}")
        except (ValueError, TypeError, argparse.ArgumentTypeError) as error:
            raise ValueError(f"Invalid engine configuration setting {key}: {error}") from error
        result[name] = value
    return result


def resolve_native_settings(
    document: Mapping | None,
    overrides: Mapping[str, Any],
    *,
    explicit_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Merge defaults < YAML/imports < explicit arguments, then derive defaults.

    CLI namespaces carry parser defaults too, so their explicit field set must
    be captured before parsing. An explicit value equal to a default is still
    explicit; null, false and empty lists are never dropped.
    """
    from vllm_gr.config.parser import common_engine_defaults

    supplied = native_settings(document) if document is not None else {}
    explicit = set(supplied) | (set(overrides) if explicit_fields is None else explicit_fields)
    defaults = common_engine_defaults()
    selected_overrides = {
        key: value
        for key, value in overrides.items()
        if explicit_fields is None
        or key in explicit_fields
        or key not in defaults.keys() | supplied.keys()
    }
    native_names = {action.dest for action in _native_parser()._actions}
    converted = native_settings(
        {
            key: value
            for key, value in selected_overrides.items()
            if key in native_names and isinstance(value, str)
        }
    )
    values = {**defaults, **supplied, **selected_overrides, **converted}
    model_len = values.get("max_model_len")
    budget = values.get("max_num_batched_tokens")
    if (
        "max_num_batched_tokens" not in explicit
        and values.get("enable_chunked_prefill") is False
        and type(model_len) is int
        and type(budget) is int
    ):
        values["max_num_batched_tokens"] = max(budget, model_len)
    return values
