# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import weakref
from dataclasses import dataclass

from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs, get_kwargs
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)


@dataclass
class ModelConfigGr(ModelConfig):
    catalog_path: str | None = None
    """ Optionally define a path for valid results file"""


origin_add_cli_args = EngineArgs.add_cli_args
origin_async_add_cli_args = AsyncEngineArgs.add_cli_args
origin_create_model_config = EngineArgs.create_model_config
origin_create_engine_config = EngineArgs.create_engine_config
origin_from_cli_args = EngineArgs.from_cli_args.__func__
_engine_args_catalog_paths: dict[int, str | None] = {}
_engine_args_catalog_finalizers: dict[int, weakref.finalize] = {}


def _install_catalog_path_property() -> None:
    existing = getattr(EngineArgs, "catalog_path", None)
    if isinstance(existing, property):
        return

    def _get_catalog_path(self) -> str | None:
        return _engine_args_catalog_paths.get(id(self))

    def _set_catalog_path(self, value: str | None) -> None:
        key = id(self)
        _engine_args_catalog_paths[key] = value
        if key not in _engine_args_catalog_finalizers:
            _engine_args_catalog_finalizers[key] = weakref.finalize(
                self,
                lambda object_id=key: (
                    _engine_args_catalog_paths.pop(object_id, None),
                    _engine_args_catalog_finalizers.pop(object_id, None),
                ),
            )

    EngineArgs.catalog_path = property(_get_catalog_path, _set_catalog_path)


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
    _install_catalog_path_property()
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
    custom_backend_requested = _user_requested_custom_backend(self)
    config = origin_create_engine_config(self, usage_context, headless)
    if custom_backend_requested:
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


def async_add_cli_args(
    cls,
    parser: FlexibleArgumentParser,
    async_args_only: bool = False,
) -> FlexibleArgumentParser:
    parser = origin_async_add_cli_args(parser, async_args_only=async_args_only)
    # AsyncEngineArgs.add_cli_args may call current_platform.pre_register_and_update()
    # after general plugins load, which can overwrite GR engine-core patches on NPU.
    from vllm_gr.patch import re_apply_patches

    re_apply_patches()
    return parser


def patch_add_cli_args() -> None:
    _install_catalog_path_property()
    EngineArgs.add_cli_args = add_cli_args
    AsyncEngineArgs.add_cli_args = classmethod(async_add_cli_args)
    EngineArgs.create_model_config = create_model_config
    EngineArgs.create_engine_config = create_engine_config
    EngineArgs.from_cli_args = classmethod(from_cli_args)
