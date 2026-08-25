# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import logging
from collections.abc import Mapping
from typing import Final

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_gr.config import (
    ConstraintBackend,
    GRAttentionBackend,
    GRConfig,
    GRConfigInput,
    write_canonical_gr_config,
)
from vllm_gr.config.parser import (
    GR_CONFIG_KEY,
    KNOWN_GR_CONFIG_KEYS,
    normalize_gr_config_input,
)

origin_add_cli_args = EngineArgs.add_cli_args
origin_create_engine_config = EngineArgs.create_engine_config
origin_from_cli_args = EngineArgs.from_cli_args.__func__
origin_engine_args_init = EngineArgs.__init__
logger = logging.getLogger("vllm_gr.engine.arg_utils_gr")
_NESTED_ATTENTION_BACKEND_EXPLICIT_ATTR: Final = "_vllm_gr_nested_attention_backend_explicit"


def _init_engine_args(
    self: EngineArgs,
    *args: object,
    catalog_path: str | None = None,
    constraint_backend: ConstraintBackend | str | None = None,
    vllm_gr_config: GRConfigInput | None = None,
    **kwargs: object,
) -> None:
    """Accept the vllm-gr engine args on the Python-API construction path.

    The CLI path (``--catalog-path`` / ``--constraint-backend``) stashes these
    values as attributes in :func:`from_cli_args` after construction.  The
    offline ``GRLLM`` Python API builds its ``EngineArgs`` indirectly through
    ``LLM.__init__`` -> ``EngineArgs(model=..., **kwargs)``, so it cannot reach
    the instance to set the attributes afterwards.  Threading the two values as
    ``**kwargs`` here stashes them identically, and the patched
    :func:`create_engine_config` reads them exactly like the CLI path —
    otherwise ``EngineCore`` would not learn that the backend is
    ``constraint_table`` and would never run the constrained hooks.
    """
    origin_engine_args_init(self, *args, **kwargs)
    self.catalog_path = catalog_path
    self.constraint_backend = constraint_backend
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
        "--catalog-path",
        type=str,
        default=None,
        help="Optional JSON catalog used to constrain generation results.",
    )
    model_group.add_argument(
        "--constraint-backend",
        type=str,
        choices=[backend.value for backend in ConstraintBackend],
        default=None,
        help="Constraint implementation: trie (default) or constraint_table.",
    )
    model_group.add_argument(
        "--vllm-gr-config",
        type=str,
        default=None,
        help="Path to a schema-versioned vLLM-GR YAML config.",
    )
    model_group.add_argument(
        "--beam-max-width",
        type=int,
        default=None,
        help=(
            "Upper bound on beam width for the persistent beam KV pool and "
            "the captured CUDA graph shape (num_tokens=width). Must be >= "
            "any served request's beam_width. When set, the engine allocates "
            "the beam KV pool and (unless --no-beam-graph) captures a FULL "
            "CUDA graph for this width at startup. Unset (default) disables "
            "beam decode — no pool allocated, no graph captured."
        ),
    )
    model_group.add_argument(
        "--no-beam-graph",
        action="store_true",
        default=False,
        help="Disable the beam-decode CUDA graph (beam decode runs eager).",
    )
    model_group.add_argument(
        "--beam-max-decode-steps",
        type=int,
        default=None,
        help="Upper bound on beam decode steps for the persistent beam KV "
        "pool (default 3, matching OneRec SID: max_tokens=5 minus begin/end).",
    )
    return parser


def from_cli_args(cls, args: argparse.Namespace) -> EngineArgs:
    raw_attention_config = getattr(args, "attention_config", None)
    nested_backend_is_explicit = _nested_attention_backend_is_explicit(raw_attention_config)
    engine_args = origin_from_cli_args(cls, args)
    engine_args.catalog_path = None
    if hasattr(args, "catalog_path"):
        engine_args.catalog_path = args.catalog_path
    engine_args.constraint_backend = None
    if hasattr(args, "constraint_backend"):
        engine_args.constraint_backend = args.constraint_backend
    engine_args._vllm_gr_config_input = getattr(args, "vllm_gr_config", None)
    for attr in ("beam_max_width", "beam_max_decode_steps"):
        setattr(engine_args, attr, getattr(args, attr, None))
    engine_args.no_beam_graph = getattr(args, "no_beam_graph", False)
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
    # Parse unconditionally so invalid raw vllm_gr_* options fail before
    # the EngineConfig is returned and before runtime execution begins.
    additional_config = getattr(config, "additional_config", None)
    canonical_in_additional_config = (
        isinstance(additional_config, Mapping) and GR_CONFIG_KEY in additional_config
    )
    legacy_keys_in_additional_config = (
        sorted(key for key in KNOWN_GR_CONFIG_KEYS if key in additional_config)
        if isinstance(additional_config, Mapping)
        else []
    )
    existing_gr_config = GRConfig.from_vllm_config(config)
    canonical_input = getattr(self, "_vllm_gr_config_input", None)
    catalog_path = getattr(self, "catalog_path", None)
    constraint_backend = getattr(self, "constraint_backend", None)
    backend_is_explicit, backend_override = _explicit_gr_attention_backend(self)

    # GRLLM historically forwards its default ConstraintBackend.TRIE value.
    # Treat that exact enum default as omitted so a canonical config can pass
    # through GRLLM's **kwargs path without changing the legacy signature.
    legacy_argument_present = (
        catalog_path is not None
        or (constraint_backend is not None and constraint_backend is not ConstraintBackend.TRIE)
        or getattr(self, "beam_max_width", None) is not None
        or getattr(self, "beam_max_decode_steps", None) is not None
        or bool(getattr(self, "no_beam_graph", False))
        or backend_is_explicit
    )

    if canonical_input is not None:
        if canonical_in_additional_config:
            raise ValueError(
                "vllm_gr_config cannot be combined with an existing canonical "
                "additional_config['vllm_gr']"
            )
        if legacy_argument_present or legacy_keys_in_additional_config:
            raise ValueError(
                "canonical vllm_gr_config cannot be combined with legacy "
                "vLLM-GR CLI, Python, or additional_config inputs"
            )
        input_gr_config = normalize_gr_config_input(canonical_input)
        effective_gr_config = input_gr_config
        canonical_source = True
    elif canonical_in_additional_config:
        if legacy_argument_present:
            raise ValueError(
                "canonical additional_config['vllm_gr'] cannot be combined "
                "with legacy vLLM-GR CLI or Python inputs"
            )
        effective_gr_config = existing_gr_config
        canonical_source = True
    else:
        canonical_source = False

        effective_catalog_path = (
            catalog_path if catalog_path is not None else existing_gr_config.catalog_path
        )
        effective_constraint_backend = (
            ConstraintBackend(constraint_backend)
            if constraint_backend is not None
            else existing_gr_config.constraint_backend
        )
        offset, slice_size = None, None
        if effective_catalog_path is not None:
            model_name = getattr(self, "model", None)
            if model_name is not None:
                from vllm_gr.utils.catalog import (
                    compute_catalog_slice_config,
                )

                trust_remote_code = bool(getattr(self, "trust_remote_code", False))
                offset, slice_size = compute_catalog_slice_config(
                    effective_catalog_path,
                    model_name,
                    trust_remote_code=trust_remote_code,
                )
                if offset is not None and slice_size is not None:
                    from vllm_gr.ops.sliced_logits import set_vocab_slice_config

                    set_vocab_slice_config(offset, slice_size)

        effective_gr_config = GRConfig(
            catalog_path=effective_catalog_path,
            constraint_backend=effective_constraint_backend,
            attention_backend=(
                backend_override if backend_is_explicit else existing_gr_config.attention_backend
            ),
            vocab_offset=(offset if offset is not None else existing_gr_config.vocab_offset),
            vocab_slice_size=(
                slice_size if slice_size is not None else existing_gr_config.vocab_slice_size
            ),
            beam_graph_enabled=(
                existing_gr_config.beam_graph_enabled and not getattr(self, "no_beam_graph", False)
            ),
            beam_max_width=(
                getattr(self, "beam_max_width", None)
                if getattr(self, "beam_max_width", None) is not None
                else existing_gr_config.beam_max_width
            ),
            beam_max_decode_steps=(
                getattr(self, "beam_max_decode_steps", None)
                if getattr(self, "beam_max_decode_steps", None) is not None
                else existing_gr_config.beam_max_decode_steps
            ),
            schema_version=existing_gr_config.schema_version,
            constraint_table=existing_gr_config.constraint_table,
        )

    if (
        canonical_source
        and effective_gr_config.constraint_table is not None
        and effective_gr_config.constraint_table.enabled
    ):
        raise RuntimeError(
            "canonical constraint_table.enabled=true is not supported until "
            "the Worker integration PR; refusing to fall back to Trie or "
            "unconstrained execution"
        )

    if effective_gr_config != existing_gr_config or canonical_input is not None:
        # vLLM 0.22 config classes are Pydantic dataclasses with extra fields
        # forbidden. Keep GR-only values in VllmConfig.additional_config so
        # that they also survive process serialization.
        if canonical_source:
            write_canonical_gr_config(config, effective_gr_config)
        else:
            effective_gr_config.write_to_vllm_config(config)

    if effective_gr_config.attention_backend is GRAttentionBackend.CUSTOM:
        attention_config = getattr(config, "attention_config", None)
        if attention_config is not None and getattr(attention_config, "backend", None) is None:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum

            attention_config.backend = AttentionBackendEnum.CUSTOM
    return config


def engine_args_accepts_gr_values() -> bool:
    """Return whether :func:`patch_add_cli_args` is mounted.

    The offline ``GRLLM`` Python API only threads ``catalog_path`` /
    ``constraint_backend`` through ``LLM.__init__`` kwargs when this patch is
    active; without it, ``EngineArgs.__init__`` would reject the extra keyword
    arguments.
    """
    return EngineArgs.__init__ is _init_engine_args


def patch_add_cli_args() -> None:
    EngineArgs.add_cli_args = add_cli_args
    EngineArgs.create_engine_config = create_engine_config
    EngineArgs.from_cli_args = classmethod(from_cli_args)
    EngineArgs.__init__ = _init_engine_args
