# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
from dataclasses import dataclass
from typing import Any

from vllm.config import ModelConfig
from vllm.engine.arg_utils import EngineArgs, get_kwargs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_gr.v1.attention.backend_selector import (
    VLLM_GR_ATTENTION_BACKEND_KEY,
    VLLM_GR_BEAM_ATTENTION_BACKEND,
)


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
    gr_group = parser.add_argument_group(title="vLLM-GR")
    gr_group.add_argument(
        "--vllm-gr-beam-attn",
        action="store_true",
        help="Use vllm-gr's BeamAttentionBackend instead of the vLLM CUSTOM registry.",
    )
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
    engine_args.vllm_gr_beam_attn = bool(getattr(args, "vllm_gr_beam_attn", False))
    return engine_args


def create_engine_config(self, *args: Any, **kwargs: Any):
    """Add the explicit Beam marker without changing generic CUSTOM usage."""
    if not getattr(self, "vllm_gr_beam_attn", False):
        return origin_create_engine_config(self, *args, **kwargs)

    if not isinstance(self.additional_config, dict):
        raise ValueError("--vllm-gr-beam-attn requires --additional-config to be a JSON object.")

    original_additional_config = self.additional_config
    self.additional_config = {
        **original_additional_config,
        VLLM_GR_ATTENTION_BACKEND_KEY: VLLM_GR_BEAM_ATTENTION_BACKEND,
    }
    try:
        return origin_create_engine_config(self, *args, **kwargs)
    finally:
        self.additional_config = original_additional_config


def patch_add_cli_args() -> None:
    EngineArgs.add_cli_args = add_cli_args
    EngineArgs.create_model_config = create_model_config
    EngineArgs.create_engine_config = create_engine_config
    EngineArgs.from_cli_args = classmethod(from_cli_args)
