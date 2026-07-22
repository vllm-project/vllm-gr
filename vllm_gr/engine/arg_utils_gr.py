# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import logging

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

origin_add_cli_args = EngineArgs.add_cli_args
origin_create_engine_config = EngineArgs.create_engine_config
origin_from_cli_args = EngineArgs.from_cli_args.__func__
logger = logging.getLogger("vllm_gr.engine.arg_utils_gr")


def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser = origin_add_cli_args(parser)
    model_group = parser.add_argument_group(title="vLLM-GR")
    model_group.add_argument(
        "--catalog-path",
        type=str,
        default=None,
        help="Optional JSON catalog used to constrain generation results.",
    )
    return parser


def from_cli_args(cls, args: argparse.Namespace) -> EngineArgs:
    engine_args = origin_from_cli_args(cls, args)
    engine_args.catalog_path = None
    if hasattr(args, "catalog_path"):
        engine_args.catalog_path = args.catalog_path
    return engine_args


def _user_requested_custom_backend(self) -> bool:
    """Return True if the user explicitly requested CUSTOM attention backend."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if self.attention_backend is not None:
        if isinstance(self.attention_backend, str):
            if self.attention_backend.upper() == "CUSTOM":
                return True
        elif self.attention_backend == AttentionBackendEnum.CUSTOM:
            return True
    attention_config = getattr(self, "attention_config", None)
    if (
        attention_config is not None
        and getattr(attention_config, "backend", None) == AttentionBackendEnum.CUSTOM
    ):
        return True
    return False


def _apply_npu_default_block_size(config) -> None:
    """Use an Ascend-compatible default without overriding user intent.

    vLLM 0.22 resolves an unspecified block size to 16 before the Ascend
    platform hook runs. vLLM-Ascend 0.22.1rc1 keeps that value when both
    chunked prefill and prefix caching are disabled, although its standard
    attention backends require a kernel block size of 128.
    """
    try:
        from vllm.platforms import current_platform
    except Exception:
        return
    if getattr(current_platform, "device_type", None) != "npu":
        return

    cache_config = getattr(config, "cache_config", None)
    model_config = getattr(config, "model_config", None)
    if cache_config is None:
        return
    if getattr(cache_config, "user_specified_block_size", False):
        return
    # Hybrid attention/state models have model-specific cache sizing rules in
    # vLLM-Ascend and must not be forced onto the dense-attention default.
    if model_config is not None and getattr(model_config, "is_hybrid", False):
        return
    if cache_config.block_size != 128:
        logger.info(
            "Using Ascend-compatible default KV cache block size 128 "
            "(vLLM resolved the unspecified value to %s).",
            cache_config.block_size,
        )
        cache_config.block_size = 128


def create_engine_config(self, usage_context=None, headless=False):
    custom_backend = _user_requested_custom_backend(self)
    if custom_backend:
        # Register before vLLM validates the requested backend. CUDA can then
        # use its native selection path; NPU intent is restored below if the
        # Ascend platform hook erased it while creating the config.
        from vllm_gr.v1.attention.backends import register_beam_attention_backend

        register_beam_attention_backend()

    config = origin_create_engine_config(self, usage_context, headless)
    _apply_npu_default_block_size(config)
    catalog_path = getattr(self, "catalog_path", None)
    if catalog_path is not None:
        # vLLM 0.22 config classes are Pydantic dataclasses with extra fields
        # forbidden. Keep GR-only values in VllmConfig.additional_config so
        # that they also survive process serialization.
        additional_config = config.additional_config
        if not isinstance(additional_config, dict):
            raise ValueError("vLLM-GR options require --additional-config to be a JSON object")
        additional_config["vllm_gr_catalog_path"] = catalog_path

    if custom_backend:
        from vllm_gr.platform.npu.attention_backend import (
            preserve_custom_backend_intent,
        )

        preserve_custom_backend_intent(config, requested=True)
    return config


def patch_add_cli_args() -> None:
    EngineArgs.add_cli_args = add_cli_args
    EngineArgs.create_engine_config = create_engine_config
    EngineArgs.from_cli_args = classmethod(from_cli_args)
