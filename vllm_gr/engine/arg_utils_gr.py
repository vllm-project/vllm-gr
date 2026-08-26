# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import logging
from collections.abc import Mapping
from dataclasses import replace
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
from vllm_gr.config.models import AUTO_CONSTRAINT_TABLE_PATH
from vllm_gr.config.parser import (
    GR_CONFIG_KEY,
    KNOWN_GR_CONFIG_KEYS,
    normalize_gr_config_input,
)


def _catalog_is_triples_document(catalog_path: str) -> bool:
    """Return whether ``catalog_path`` holds a Constraint Tables triples doc.

    The triples document produced by ``tools/build_constraint_triples.py`` is a
    JSON object keyed by ``format_version`` / ``triples`` / ``sentinels`` /
    ``relations``.  The Trie frontend catalog (``convert_catalog.py`` output,
    and the inline serving-catalog tests) is a JSON *list* of token sequences.
    The NPU auto-promotion must only fire on a triples document — promoting a
    Trie-list catalog to the Worker-side Constraint Tables backend would feed a
    list to ``compile_constraint_table_resource`` and fail fast.
    """
    try:
        with open(catalog_path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        # Unreadable / malformed: let the downstream compiler surface the
        # precise error rather than guessing here.
        return False
    return isinstance(document, dict)


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
            "any served request's beam_width and <= 256. When set, the engine allocates "
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
    model_group.add_argument(
        "--beam-worker-decision",
        action="store_true",
        default=None,
        help="Move the beam decision (beam_search_group / rec_final_select + "
        "select_unshared_kv) into the accelerator worker and return results via "
        "ModelRunnerOutput.beam_search_result (ACS-aligned save flow). CUDA "
        "beam deployments enable this by default; NPU requires xllm decode "
        "execution and its operator library. Without it the frontend NumPy "
        "decision path is kept.",
    )
    model_group.add_argument(
        "--no-beam-worker-decision",
        action="store_false",
        dest="beam_worker_decision",
        default=None,
        help="Explicitly keep the frontend NumPy beam decision path (opt-out "
        "of worker-side beam decision).",
    )
    model_group.add_argument(
        "--beam-self-advance",
        action="store_true",
        default=None,
        help="Let EngineCore drive intermediate beam steps itself: the "
        "worker's beam decision result is applied in-process to enqueue the "
        "next beam-decode step, eliminating the per-step frontend RPC "
        "round-trip. Requires --beam-worker-decision. Synchronously scheduled "
        "accelerator Constraint Table beam deployments enable it by default; "
        "async scheduling keeps frontend step submission unless explicitly enabled.",
    )
    model_group.add_argument(
        "--no-beam-self-advance",
        action="store_false",
        dest="beam_self_advance",
        default=None,
        help="Explicitly keep the frontend RPC beam step-update loop (opt-out "
        "of EngineCore self-advance).",
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
    engine_args.beam_worker_decision = getattr(args, "beam_worker_decision", None)
    engine_args.beam_self_advance = getattr(args, "beam_self_advance", None)
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


def _is_npu_platform() -> bool:
    """Return whether the active platform is an Ascend NPU."""
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    return getattr(current_platform, "device_type", None) == "npu"


def _is_cuda_platform() -> bool:
    """Return whether the active platform is CUDA."""
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    return getattr(current_platform, "device_type", None) == "cuda"


def _constraint_explicitly_configured(
    engine_args: object,
    existing_gr_config: GRConfig,
) -> bool:
    """Whether the user explicitly chose a constraint backend.

    The NPU platform default (Constraint Tables + worker decision) must not
    override an explicit user choice.  Any explicit backend (including an
    explicit ``TRIE`` opt-out) suppresses the promotion to Constraint
    Tables.  A catalog path alone does NOT suppress the promotion: on NPU
    it is the constraint data source for the promoted backend (a Triples
    document when Constraint Tables are promoted, the Trie catalog when
    ``trie`` is explicit).
    """
    explicit_backend = (
        getattr(engine_args, "constraint_backend", None) is not None
        or existing_gr_config.constraint_backend is not ConstraintBackend.TRIE
    )
    return explicit_backend


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
        or getattr(self, "beam_worker_decision", None) is not None
        or getattr(self, "beam_self_advance", None) is not None
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
        effective_beam_worker_decision = (
            getattr(self, "beam_worker_decision", None)
            if getattr(self, "beam_worker_decision", None) is not None
            else existing_gr_config.beam_worker_decision
        )
        effective_beam_self_advance = (
            getattr(self, "beam_self_advance", None)
            if getattr(self, "beam_self_advance", None) is not None
            else existing_gr_config.beam_self_advance
        )
        # NPU platform default for beam-serve deployments: when the user
        # enables beam search (beam_max_width) with a constraint catalog
        # (a Triples document) and no explicit backend choice, run Constraint
        # Tables with worker-side beam decision by default.  Promotion requires
        # a catalog: the tokenizer auto-build sentinel would fail fast on
        # models whose s_a/s_b/s_c groups exceed max_valid_items (e.g. OneRec
        # 8192 > 500), so a catalog-free beam deployment stays unconstrained
        # (legacy) instead.  This only fires on a real NPU device with an
        # active beam pool, so non-GR / non-beam configs and non-NPU platforms
        # are untouched.  beam_max_width is read from the parsed config (GRLLM
        # rides it in additional_config) or the EngineArgs attribute (serve
        # CLI).
        beam_width_configured = (
            getattr(self, "beam_max_width", None) is not None
            or existing_gr_config.beam_max_width is not None
        )
        worker_decision_explicit = (
            getattr(self, "beam_worker_decision", None) is not None
            or existing_gr_config.beam_worker_decision is not None
        )
        self_advance_explicit = (
            getattr(self, "beam_self_advance", None) is not None
            or existing_gr_config.beam_self_advance is not None
        )
        if (
            _is_npu_platform()
            and beam_width_configured
            and not _constraint_explicitly_configured(self, existing_gr_config)
            and effective_catalog_path is not None
            and _catalog_is_triples_document(effective_catalog_path)
        ):
            # Promote to Constraint Tables on NPU beam deployments.  A user
            # catalog_path is kept as-is: it is expected to be a
            # tools/build_constraint_triples.py Triples document (the
            # constraint data source for the promoted backend); only a triples
            # document (JSON object) triggers the promotion, so a Trie-list
            # catalog keeps its Trie semantics.  The sentinel auto-build path is
            # NOT used here: a tokenizer whose s_a/s_b/s_c groups exceed
            # max_valid_items fails fast (e.g. OneRec 8192>500), and a
            # catalog-free GRLLM (no constraint data at all) must stay
            # unconstrained (legacy) instead of force-building tables.  A
            # catalog document is therefore required for the promotion.
            effective_constraint_backend = ConstraintBackend.CONSTRAINT_TABLE
        # Worker-side beam decision is enabled on accelerator backends by default for any beam
        # deployment unless the user explicitly disabled it.  Independent of
        # the Constraint Table promotion above (which may already be explicit
        # via GRLLM passing the auto-build sentinel).
        if (
            (_is_npu_platform() or _is_cuda_platform())
            and beam_width_configured
            and not worker_decision_explicit
        ):
            if effective_beam_worker_decision is None:
                effective_beam_worker_decision = True
        # The canonical Worker decision otherwise paid one frontend
        # step-update round-trip per SID decode.  On accelerator Constraint
        # Tables, make the already-supported EngineCore self-advance path the
        # default so the canonical path retains the legacy device-wire
        # latency.  Explicit False remains a full opt-out. NPU cascade (no
        # xllm decision result), async scheduling, and PP>1 keep their existing
        # fallback behavior in the runtime gates.
        if (
            (_is_npu_platform() or _is_cuda_platform())
            and beam_width_configured
            and effective_constraint_backend is ConstraintBackend.CONSTRAINT_TABLE
            and effective_beam_worker_decision
            and not getattr(self, "async_scheduling", False)
            and not self_advance_explicit
            and effective_beam_self_advance is None
        ):
            effective_beam_self_advance = True
        # Self-advance only applies when the worker decision path is active
        # (the self-advance step consumes the worker's beam_search_result).
        # Validate after platform defaults so an explicit self-advance can use
        # the accelerator's default worker decision.
        if effective_beam_self_advance and not effective_beam_worker_decision:
            raise ValueError(
                "beam_self_advance=True requires beam_worker_decision=True "
                "(CLI --beam-worker-decision or additional_config "
                "'vllm_gr_beam_worker_decision')"
            )
        offset, slice_size = None, None
        # CUDA Constraint Tables support sliced logits while keeping global
        # table token ids (CUDAConstrainedTopKExecution._vocab_window).  The
        # slice helper accepts both Trie lists and triples documents, so keep
        # the old lm_head fast path on CUDA. NPU retains its full-vocabulary
        # ABI and validate_constraint_table_platform rejects slicing there.
        constraint_slice_supported = (
            effective_constraint_backend is not ConstraintBackend.CONSTRAINT_TABLE
            or _is_cuda_platform()
        )
        if (
            effective_catalog_path is not None
            and effective_catalog_path != AUTO_CONSTRAINT_TABLE_PATH
            and constraint_slice_supported
        ):
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
            beam_worker_decision=effective_beam_worker_decision,
            beam_self_advance=effective_beam_self_advance,
            schema_version=existing_gr_config.schema_version,
            constraint_table=existing_gr_config.constraint_table,
        )

    # Canonical configs bypass the legacy merge above. Apply accelerator
    # runtime defaults here as well so omitted decision controls use the same
    # Worker + EngineCore path as legacy arguments.
    if (
        canonical_source
        and (_is_cuda_platform() or _is_npu_platform())
        and effective_gr_config.beam_max_width is not None
        and effective_gr_config.beam_worker_decision is None
    ):
        effective_gr_config = replace(effective_gr_config, beam_worker_decision=True)
    if (
        canonical_source
        and (_is_cuda_platform() or _is_npu_platform())
        and effective_gr_config.beam_max_width is not None
        and (
            effective_gr_config.constraint_backend is ConstraintBackend.CONSTRAINT_TABLE
            or (
                effective_gr_config.constraint_table is not None
                and effective_gr_config.constraint_table.enabled
            )
        )
        and effective_gr_config.beam_worker_decision
        and not getattr(self, "async_scheduling", False)
        and effective_gr_config.beam_self_advance is None
    ):
        effective_gr_config = replace(effective_gr_config, beam_self_advance=True)

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
