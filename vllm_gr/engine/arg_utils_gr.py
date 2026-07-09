# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
from dataclasses import dataclass

from vllm.config import ModelConfig
from vllm.engine.arg_utils import EngineArgs, get_kwargs
from vllm.utils.argparse_utils import FlexibleArgumentParser


@dataclass
class ModelConfigGr(ModelConfig):
    catalog_path: str | None = None
    """ Optionally define a path for valid results file"""


origin_add_cli_args = EngineArgs.add_cli_args
origin_create_model_config = EngineArgs.create_model_config
origin_create_engine_config = EngineArgs.create_engine_config
origin_from_cli_args = EngineArgs.from_cli_args.__func__


def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser = origin_add_cli_args(parser)
    model_kwargs = get_kwargs(ModelConfigGr)
    model_group = parser.add_argument_group(title="ModelConfig", description=ModelConfigGr.__doc__)
    model_group.add_argument("--catalog-path", **model_kwargs["catalog_path"])
    return parser


def create_model_config(self) -> ModelConfig:
    config = origin_create_model_config(self)
    config.__class__ = ModelConfigGr
    config.catalog_path = getattr(self, "catalog_path", None)
    return config


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
    if _user_requested_custom_backend(self):
        # Persist the user intent in a field that survives platform
        # check_and_update_config resets and process serialization.
        additional_config = getattr(config, "additional_config", None)
        if additional_config is None:
            config.additional_config = {}
            additional_config = config.additional_config
        if isinstance(additional_config, dict):
            additional_config["vllm_gr_attention_backend"] = "CUSTOM"
        attention_config = getattr(config, "attention_config", None)
        if attention_config is not None and getattr(attention_config, "backend", None) is None:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum

            attention_config.backend = AttentionBackendEnum.CUSTOM
    return config


def patch_add_cli_args() -> None:
    EngineArgs.add_cli_args = add_cli_args
    EngineArgs.create_model_config = create_model_config
    EngineArgs.create_engine_config = create_engine_config
    EngineArgs.from_cli_args = classmethod(from_cli_args)
