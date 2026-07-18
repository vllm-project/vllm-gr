# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

origin_add_cli_args = EngineArgs.add_cli_args
origin_create_engine_config = EngineArgs.create_engine_config
origin_from_cli_args = EngineArgs.from_cli_args.__func__


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


def create_engine_config(self, usage_context=None, headless=False):
    config = origin_create_engine_config(self, usage_context, headless)
    catalog_path = getattr(self, "catalog_path", None)
    custom_backend = _user_requested_custom_backend(self)
    if catalog_path is not None or custom_backend:
        # vLLM 0.22 config classes are Pydantic dataclasses with extra fields
        # forbidden. Keep GR-only values in VllmConfig.additional_config so
        # that they also survive process serialization.
        additional_config = config.additional_config
        if not isinstance(additional_config, dict):
            raise ValueError("vLLM-GR options require --additional-config to be a JSON object")
        if catalog_path is not None:
            additional_config["vllm_gr_catalog_path"] = catalog_path
        if custom_backend:
            additional_config["vllm_gr_attention_backend"] = "CUSTOM"

    if custom_backend:
        attention_config = getattr(config, "attention_config", None)
        if attention_config is not None and getattr(attention_config, "backend", None) is None:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum

            attention_config.backend = AttentionBackendEnum.CUSTOM
    return config


def patch_add_cli_args() -> None:
    EngineArgs.add_cli_args = add_cli_args
    EngineArgs.create_engine_config = create_engine_config
    EngineArgs.from_cli_args = classmethod(from_cli_args)
