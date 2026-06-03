# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
HSTU family defaults for engine args.

Pure Python — reads/writes attributes on the engine_args dataclass.
No torch / vLLM heavy imports required.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_gr.engine.arg_utils import GRCommonEngineArgsMixin

logger = logging.getLogger(__name__)


def apply_hstu_defaults(engine_args: "GRCommonEngineArgsMixin") -> None:
    """Apply HSTU-specific runtime defaults to *engine_args*."""
    from vllm_gr.adapters.registry import get_worker_cls_path

    engine_args.trust_remote_code = True
    required_max_model_len = _get_hstu_runtime_max_model_len(engine_args)
    if engine_args.max_model_len is None:
        engine_args.max_model_len = required_max_model_len
    elif engine_args.max_model_len < required_max_model_len:
        raise ValueError(
            "For HSTU, max_model_len must cover the scheduler-visible full "
            "sequence: contextual prefix + UIH/history + candidates. "
            f"Got max_model_len={engine_args.max_model_len}, but the current "
            f"runtime settings require at least {required_max_model_len} "
            f"(max_seq_len={engine_args.max_seq_len}, "
            f"max_candidates={engine_args.max_candidates})."
        )
    engine_args.skip_tokenizer_init = True
    engine_args.runner = "pooling"
    engine_args.enforce_eager = True
    # HSTU currently exposes a single full-attention KV cache spec.
    # Hybrid KV cache coordination is for mixed attention-group models and
    # provides no benefit here; forcing it off also avoids NPU runtimes
    # incorrectly routing single-group HSTU models through the hybrid path.
    engine_args.disable_hybrid_kv_cache_manager = True
    _ensure_hstu_model_max_length_override(engine_args, required_max_model_len)

    worker_cls = get_worker_cls_path(
        "hstu",
        is_npu=engine_args._is_ascend_platform(),
    )
    if worker_cls is not None:
        engine_args.worker_cls = worker_cls

    if engine_args.dtype == "auto":
        engine_args.dtype = "bfloat16"

    if engine_args.tensor_parallel_size > 1:
        logger.warning(
            "HSTU models use full model replication (data parallel), "
            "not tensor sharding. Each worker loads the full model. "
            "tensor_parallel_size=%d creates %d full replicas.",
            engine_args.tensor_parallel_size,
            engine_args.tensor_parallel_size,
        )

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


def _get_hstu_runtime_max_model_len(engine_args: "GRCommonEngineArgsMixin") -> int:
    from vllm_gr.config.hstu_config import get_hstu_full_sequence_len

    return get_hstu_full_sequence_len(
        max_seq_len=engine_args.max_seq_len,
        max_candidates=engine_args.max_candidates,
        contextual_seq_len=_resolve_hstu_contextual_seq_len(engine_args),
    )


def _resolve_hstu_contextual_seq_len(engine_args: "GRCommonEngineArgsMixin") -> int:
    from vllm_gr.config.hstu_config import get_hstu_contextual_seq_len

    if isinstance(engine_args.hf_overrides, dict):
        contextual_seq_len = get_hstu_contextual_seq_len(engine_args.hf_overrides)
        if contextual_seq_len > 0:
            return contextual_seq_len

    local_config = engine_args._read_local_config_dict()
    if local_config is not None:
        return get_hstu_contextual_seq_len(local_config)
    return 0


def _ensure_hstu_model_max_length_override(
    engine_args: "GRCommonEngineArgsMixin",
    max_model_len: int,
) -> None:
    if isinstance(engine_args.hf_overrides, dict):
        current_value = engine_args.hf_overrides.get("model_max_length")
        if current_value is None or int(current_value) < max_model_len:
            engine_args.hf_overrides["model_max_length"] = max_model_len
        return

    if engine_args.hf_overrides is None:
        engine_args.hf_overrides = {"model_max_length": max_model_len}
        return

    if callable(engine_args.hf_overrides):
        original_overrides = engine_args.hf_overrides

        def _wrapped_hf_overrides(config):
            updated = original_overrides(config)
            target = updated if updated is not None else config
            current_value = getattr(target, "model_max_length", None)
            if current_value is None or int(current_value) < max_model_len:
                setattr(target, "model_max_length", max_model_len)
            return updated

        engine_args.hf_overrides = _wrapped_hf_overrides
