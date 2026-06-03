# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FuXi family defaults for engine args."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_gr.engine.arg_utils import GRCommonEngineArgsMixin

logger = logging.getLogger(__name__)


def apply_fuxi_defaults(engine_args: "GRCommonEngineArgsMixin") -> None:
    """Apply FuXi-specific runtime defaults to *engine_args*."""
    from vllm_gr.adapters.registry import get_worker_cls_path

    engine_args.trust_remote_code = True
    engine_args.skip_tokenizer_init = True
    engine_args.runner = "pooling"
    engine_args.enforce_eager = True
    engine_args.disable_hybrid_kv_cache_manager = True

    if engine_args.kv_cache_dtype == "auto":
        engine_args.kv_cache_dtype = "bfloat16"
    engine_args.enable_prefix_caching = False

    if engine_args.max_model_len is None:
        engine_args.max_model_len = engine_args.max_seq_len
    elif engine_args.max_model_len < engine_args.max_seq_len:
        raise ValueError(
            "For FuXi, max_model_len must cover history length. "
            f"Got max_model_len={engine_args.max_model_len}, but "
            f"max_seq_len={engine_args.max_seq_len}."
        )

    worker_cls = get_worker_cls_path(
        "fuxi",
        is_npu=engine_args._is_ascend_platform(),
    )
    if worker_cls is not None:
        engine_args.worker_cls = worker_cls

    if engine_args.dtype == "auto":
        engine_args.dtype = "bfloat16"

    _validate_fuxi_config_fields(engine_args)

    if (
        engine_args.model
        and os.path.isdir(engine_args.model)
        and not engine_args._has_model_weights()
    ):
        if engine_args.allow_missing_weights or engine_args.load_format == "dummy":
            engine_args.load_format = "dummy"
            logger.info(
                "No model weights found in local model directory %r; "
                "using dummy loader because allow_missing_weights=True",
                engine_args.model,
            )
            return

        raise FileNotFoundError(
            "No model weight files found in local model directory "
            f"{engine_args.model!r}. Expected one of *.safetensors, *.bin, *.pt, *.pth. "
            "If this is an intentional demo/random-init run, pass "
            "allow_missing_weights=True explicitly."
        )


def _validate_fuxi_config_fields(engine_args: "GRCommonEngineArgsMixin") -> None:
    local_config = engine_args._read_local_config_dict()
    if local_config is None:
        return

    dqk = int(local_config.get("dqk", 0))
    dv = int(local_config.get("dv", 0))
    if dqk <= 0 or dv <= 0:
        raise ValueError(f"FuXi config requires positive dqk and dv values; got dqk={dqk}, dv={dv}")

    if bool(local_config.get("enable_relative_attention_bias", False)):
        raise ValueError("FuXi-RAB-off adapter only supports enable_relative_attention_bias=False")
