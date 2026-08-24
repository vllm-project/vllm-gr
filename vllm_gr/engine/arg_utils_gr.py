# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import logging
from collections.abc import Mapping
from typing import Final

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_gr.config import GRAttentionBackend, GRConfig, GRConfigInput
from vllm_gr.config.parser import normalize_gr_config_input

origin_add_cli_args = EngineArgs.add_cli_args
origin_create_engine_config = EngineArgs.create_engine_config
origin_from_cli_args = EngineArgs.from_cli_args.__func__
origin_engine_args_init = EngineArgs.__init__
logger = logging.getLogger("vllm_gr.engine.arg_utils_gr")
_NESTED_ATTENTION_BACKEND_EXPLICIT_ATTR: Final = "_vllm_gr_nested_attention_backend_explicit"


def _init_engine_args(
    self: EngineArgs,
    *args: object,
    vllm_gr_config: GRConfigInput | None = None,
    **kwargs: object,
) -> None:
    """Accept the sole vLLM-GR configuration entry on the Python API path."""
    origin_engine_args_init(self, *args, **kwargs)
    self._vllm_gr_config_input = vllm_gr_config


def _nested_attention_backend_is_explicit(value: object) -> bool:
    if isinstance(value, Mapping):
        return "backend" in value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return False
        return isinstance(decoded, Mapping) and "backend" in decoded
    return bool(getattr(value, _NESTED_ATTENTION_BACKEND_EXPLICIT_ATTR, False))


def _preserve_nested_attention_backend_source(
    parser: FlexibleArgumentParser,
) -> None:
    """Keep source information that upstream AttentionConfig normalizes away."""
    action = next(
        (action for action in parser._actions if action.dest == "attention_config"),
        None,
    )
    if action is None or not callable(action.type):
        return

    parse_attention_config = action.type

    def parse_with_source(value: str) -> object:
        parsed = parse_attention_config(value)
        setattr(
            parsed,
            _NESTED_ATTENTION_BACKEND_EXPLICIT_ATTR,
            _nested_attention_backend_is_explicit(value),
        )
        return parsed

    action.type = parse_with_source


def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser = origin_add_cli_args(parser)
    _preserve_nested_attention_backend_source(parser)
    model_group = parser.add_argument_group(title="vLLM-GR")
    model_group.add_argument(
        "--vllm-gr-config",
        type=str,
        default=None,
        help="Path to the vLLM-GR schema-versioned YAML configuration.",
    )
    return parser


def from_cli_args(cls, args: argparse.Namespace) -> EngineArgs:
    raw_attention_config = getattr(args, "attention_config", None)
    nested_backend_is_explicit = _nested_attention_backend_is_explicit(raw_attention_config)
    engine_args = origin_from_cli_args(cls, args)
    engine_args._vllm_gr_config_input = getattr(args, "vllm_gr_config", None)
    # AttentionConfig normalizes explicit "auto" and JSON null to None. Keep
    # the CLI source information so both values can override a stale CUSTOM
    # marker rather than becoming indistinguishable from an omitted backend.
    setattr(
        engine_args,
        _NESTED_ATTENTION_BACKEND_EXPLICIT_ATTR,
        nested_backend_is_explicit,
    )
    return engine_args


def _explicit_gr_attention_backend(
    self,
) -> tuple[bool, GRAttentionBackend | None]:
    """Return whether a backend was explicit and its GR representation.

    ``None`` with ``explicit=True`` means the user selected a non-GR backend,
    which intentionally clears a stale GR CUSTOM marker.
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    requested_backend = getattr(self, "attention_backend", None)
    if requested_backend is not None:
        is_custom = (
            isinstance(requested_backend, str) and requested_backend.upper() == "CUSTOM"
        ) or requested_backend == AttentionBackendEnum.CUSTOM
        return True, GRAttentionBackend.CUSTOM if is_custom else None

    attention_config = getattr(self, "attention_config", None)
    configured_backend = getattr(attention_config, "backend", None)
    nested_backend_is_explicit = getattr(
        self,
        _NESTED_ATTENTION_BACKEND_EXPLICIT_ATTR,
        False,
    )
    if nested_backend_is_explicit or configured_backend is not None:
        return (
            True,
            GRAttentionBackend.CUSTOM
            if configured_backend == AttentionBackendEnum.CUSTOM
            else None,
        )
    return False, None


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
    config = origin_create_engine_config(self, usage_context, headless)
    _apply_npu_default_block_size(config)
    # Parse the serialized value unconditionally. The YAML file is Driver-only;
    # Workers reconstruct exactly this normalized value from VllmConfig.
    existing_gr_config = GRConfig.from_vllm_config(config)
    config_input = getattr(self, "_vllm_gr_config_input", None)
    input_gr_config = (
        normalize_gr_config_input(config_input) if config_input is not None else existing_gr_config
    )
    backend_is_explicit, backend_override = _explicit_gr_attention_backend(self)
    effective_gr_config = GRConfig(
        schema_version=input_gr_config.schema_version,
        beam=input_gr_config.beam,
        constraint_table=input_gr_config.constraint_table,
        attention_backend=(
            backend_override if backend_is_explicit else input_gr_config.attention_backend
        ),
    )
    if effective_gr_config != existing_gr_config:
        # vLLM 0.22 config classes are Pydantic dataclasses with extra fields
        # forbidden. Keep GR-only values in VllmConfig.additional_config so
        # that they also survive process serialization.
        effective_gr_config.write_to_vllm_config(config)

    if effective_gr_config.attention_backend is GRAttentionBackend.CUSTOM:
        attention_config = getattr(config, "attention_config", None)
        if attention_config is not None and getattr(attention_config, "backend", None) is None:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum

            attention_config.backend = AttentionBackendEnum.CUSTOM
    return config


def engine_args_accepts_gr_values() -> bool:
    """Return whether :func:`patch_add_cli_args` is mounted.

    The offline ``GRLLM`` Python API threads its configuration input through
    ``LLM.__init__`` only when this patch is active.
    """
    return EngineArgs.__init__ is _init_engine_args


def patch_add_cli_args() -> None:
    EngineArgs.add_cli_args = add_cli_args
    EngineArgs.create_engine_config = create_engine_config
    EngineArgs.from_cli_args = classmethod(from_cli_args)
    EngineArgs.__init__ = _init_engine_args
