# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Engine-arg defaults for Pangu Embedded and other causal LM checkpoints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_gr.engine.arg_utils import GRCommonEngineArgsMixin

logger = logging.getLogger(__name__)

_PANGU_DEFAULT_MAX_MODEL_LEN = 32768


def _cli_explicitly_set(engine_args: Any, flag_name: str) -> bool:
    """Return True when the user passed the CLI flag on the command line."""
    return flag_name in getattr(engine_args, "_explicit_cli_args", set())


def apply_pangu_llm_defaults(engine_args: "GRCommonEngineArgsMixin") -> None:
    """Apply causal-LM defaults; do not route through GR worker/pooling."""
    engine_args.trust_remote_code = True
    engine_args.skip_tokenizer_init = False
    engine_args.runner = "generate"

    # Use vLLM / vllm-ascend default worker — clear any prior GR worker_cls.
    engine_args.worker_cls = None

    if engine_args.dtype == "auto":
        engine_args.dtype = "bfloat16"

    if engine_args.max_model_len is None:
        engine_args.max_model_len = _PANGU_DEFAULT_MAX_MODEL_LEN

    tokenizer_mode = getattr(engine_args, "tokenizer_mode", None)
    if tokenizer_mode is None or tokenizer_mode == "auto":
        engine_args.tokenizer_mode = "slow"

    if not _cli_explicitly_set(engine_args, "enable_chunked_prefill"):
        if getattr(engine_args, "enable_chunked_prefill", None) is not False:
            engine_args.enable_chunked_prefill = False

    if not _cli_explicitly_set(engine_args, "enable_prefix_caching"):
        if getattr(engine_args, "enable_prefix_caching", None) is not False:
            engine_args.enable_prefix_caching = False

    logger.info(
        "Applied LLM passthrough defaults: runner=generate, "
        "max_model_len=%s, tokenizer_mode=%s",
        engine_args.max_model_len,
        getattr(engine_args, "tokenizer_mode", None),
    )
