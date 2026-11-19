# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Driver-side Constraint Tables startup wiring.

Connects the already-built M1-M3 primitives (compiler, digest resource,
platform adapters, collective-RPC injection) into the engine startup path so a
``constraint_backend=constraint_table`` request can actually run.  Every
platform-unavailable / misconfiguration case fails explicitly (issue #246 §7);
nothing ever falls back to Trie.

Flow:

- :func:`prepare_constraint_table` — activate the platform backend, reject
  unsupported configurations, and compile the triples document into a
  :class:`~vllm_gr.constraints.compiler.ConstraintTableResource`.
- :func:`install_constraint_table` — ``prepare`` plus a synchronous broadcast
  of the resource into every worker (offline ``GRLLM`` path).  The OpenAI
  serving path defers the async broadcast to the first ``beam_search``.
- :func:`create_constraint_execution` — platform adapter factory, shared with
  the Worker raw-logits hook (N3).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from vllm_gr.config import GRConfig
from vllm_gr.constraints.compiler import (
    FORMAT_VERSION,
    ConstraintTableCompiler,
    ConstraintTableResource,
)
from vllm_gr.platforms.services import constraint_execution_override

logger = logging.getLogger(__name__)


def _backend_label(provider: Any) -> str:
    """Describe the active provider for a startup diagnostic.

    The backend ID is the GR implementation; tensor placement stays available
    through the platform identity when the two differ.
    """
    backend_id = getattr(provider, "backend_id", None)
    return repr(backend_id) if isinstance(backend_id, str) else repr(provider)


def create_constraint_execution(
    *,
    device: str | Any | None = None,
    plan: Any = None,
) -> Any:
    """Create the platform constrained-top-k adapter, failing explicitly.

    Honors the test-only ``VLLM_GR_CONSTRAINT_EXECUTION=torch`` override
    (CI/dev only).  Production resolves the active platform provider; a ``None``
    adapter (platform without an implementation) or an unavailable adapter
    (NPU custom op missing) is a hard failure — Constraint Tables never silently
    degrades to Trie.
    """
    override = constraint_execution_override(device=device)
    if override is not None:
        return override

    from vllm_gr.platforms.loader import activate_current_backend

    provider = activate_current_backend()
    if plan is None:
        adapter = provider.create_constrained_topk_execution(device=device)
    else:
        from vllm_gr.platforms.services import planning_provider

        adapter = planning_provider(provider).create_constrained_topk_execution(
            plan=plan, device=device
        )
    if adapter is None:
        raise RuntimeError(
            "constraint_backend='constraint_table' cannot start: the active "
            f"backend {_backend_label(provider)} does not implement Constraint Table "
            "execution. Constraint Tables are not available on this platform; "
            "starting must fail explicitly rather than fall back to Trie."
        )
    if not adapter.is_available():
        raise RuntimeError(
            "constraint_backend='constraint_table' cannot start: the active "
            f"backend {_backend_label(provider)} provides a Constraint Table adapter but "
            "its custom operator is unavailable (is_available()=False). "
            "Verify the operator binary / environment; starting must fail "
            "explicitly rather than fall back to Trie."
        )
    return adapter


def validate_constraint_table_platform(
    gr_config: GRConfig,
    *,
    device_type: str | None = None,
) -> None:
    """Reject configurations Constraint Tables cannot support in this release.

    CUDA carries full-vocabulary table ids alongside a sliced logits window
    and translates ids to local logits columns inside the fused kernel. Other
    platforms still use the original full-vocabulary ABI and reject slicing.
    """
    slicing_enabled = gr_config.vocab_offset is not None or gr_config.vocab_slice_size is not None
    if slicing_enabled and device_type != "cuda":
        raise ValueError(
            "constraint_backend='constraint_table' cannot be combined with "
            f"vocab slicing (vocab_offset={gr_config.vocab_offset!r}, "
            f"vocab_slice_size={gr_config.vocab_slice_size!r}) on platform "
            f"{device_type or 'unknown'!r}"
        )


def compile_constraint_table_resource(
    catalog_path: str,
    tokenizer: Any,
    vocab_size: int,
) -> ConstraintTableResource:
    """Load the triples document and compile it into a validated resource.

    Args:
        catalog_path: path to the task-5 triples JSON document
            (``tools/build_constraint_triples.py`` output).
        tokenizer: model tokenizer used to map token strings to IDs.
        vocab_size: ``model_config.vocab_size``.

    Raises:
        ValueError: malformed document, wrong ``format_version``, unknown /
            out-of-range token, non-three-token catalog, or oversized resource.
    """
    with open(catalog_path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        detail = ""
        if document is None or isinstance(document, (str, int, float, bool)):
            detail = f" ({document!r})"
        raise ValueError(
            "Constraint triples document must be a JSON object with "
            f"{FORMAT_VERSION!r} top-level keys; got {type(document).__name__}"
            f"{detail}. Did you point --catalog-path at the "
            "tools/build_constraint_triples.py output?"
        )
    resource = ConstraintTableCompiler(tokenizer, vocab_size).compile(document)
    logger.info(
        "vllm-gr: compiled Constraint Tables digest=%s vocab_size=%d "
        "total_elements=%d total_bytes=%d from %s",
        resource.digest,
        vocab_size,
        resource.total_elements,
        resource.total_bytes,
        catalog_path,
    )
    return resource


def compile_constraint_table_resource_from_tokenizer(
    tokenizer: Any,
    vocab_size: int,
    max_valid_items: int = 500,
    valid_item_ids: list[int] | None = None,
    prefix1_file: str | None = None,
    prefix2_file: str | None = None,
) -> ConstraintTableResource:
    """Build Constraint Tables directly from the tokenizer's s_a/s_b/s_c groups.

    When the tokenizer carries ``<s_a_*>`` / ``<s_b_*>`` / ``<s_c_*>`` special tokens
    (OneRec-style), the six CSR tables are constructed without a catalog JSON
    document, so ``constraint_backend=constraint_table`` can run by default on
    models that declare the three token groups.

    Returns:
        A validated :class:`ConstraintTableResource` ready for worker injection.

    Raises:
        ValueError: when the tokenizer has no s_a/s_b/s_c groups (caller must
            not silently fall back — see :func:`prepare_constraint_table`).
    """
    from vllm_gr.constraints.tokenizer_builder import (
        build_constraint_tables_from_tokenizer,
    )

    resource = build_constraint_tables_from_tokenizer(
        tokenizer,
        vocab_size=vocab_size,
        max_valid_items=max_valid_items,
        valid_item_ids=valid_item_ids,
        prefix1_file=prefix1_file,
        prefix2_file=prefix2_file,
    )
    if resource is None:
        raise ValueError(
            "constraint_backend='constraint_table' without a catalog document "
            "requires a tokenizer with <s_a_*>/<s_b_*>/<s_c_*> special tokens "
            "to auto-build the tables"
        )
    return resource


def prepare_constraint_table(
    *,
    catalog_path: str | None,
    tokenizer: Any,
    vocab_size: int,
    gr_config: GRConfig | None = None,
) -> ConstraintTableResource:
    """Activate the platform backend, validate, and compile the resource.

    Does not inject into workers. Callers of the retained resource-injection
    compatibility API must broadcast the returned resource separately.
    Canonical V1 artifacts are loaded independently by each worker.

    ``catalog_path`` may be ``None`` — the tables are then built directly from
    the tokenizer's ``<s_a_*>`` / ``<s_b_*>`` / ``<s_c_*>`` groups (the default
    for OneRec-style models). The sentinel
    :data:`~vllm_gr.config.models.AUTO_CONSTRAINT_TABLE_PATH` selects the same
    tokenizer path explicitly (used by the NPU platform default).

    The platform adapter is probed *at startup* via
    :func:`create_constraint_execution` — a CUDA host (no adapter) or an NPU
    host whose custom operator is unavailable must fail here, before any
    constrained request runs, never at the first constrained step.
    """
    from vllm_gr.config.models import AUTO_CONSTRAINT_TABLE_PATH
    from vllm_gr.platforms.identity import current_platform_identity
    from vllm_gr.platforms.loader import activate_current_backend

    # Activate the provider before validation so an unsupported platform fails
    # here rather than at the first constrained step.
    activate_current_backend()
    if gr_config is not None:
        validate_constraint_table_platform(
            gr_config,
            device_type=current_platform_identity().device_type,
        )
    # Startup-time fail-fast (issue #246 §7): the probe adapter is discarded;
    # the Worker lazily creates its own with the runner device (N3).
    create_constraint_execution()
    if catalog_path is None or catalog_path == AUTO_CONSTRAINT_TABLE_PATH:
        return compile_constraint_table_resource_from_tokenizer(tokenizer, vocab_size)
    return compile_constraint_table_resource(catalog_path, tokenizer, vocab_size)


def install_constraint_table(
    engine: Any,
    *,
    catalog_path: str | None,
    tokenizer: Any,
    vocab_size: int,
    gr_config: GRConfig | None = None,
) -> ConstraintTableResource:
    """Wire Constraint Tables into an offline engine: prepare + inject.

    Broadcasts the compiled resource to every worker via
    ``EngineCore.collective_rpc`` (see
    :func:`vllm_gr.constraints.driver_inject.inject_constraint_resources`),
    which verifies each worker acks the same digest (TP/DP consistency).

    Args:
        engine: offline ``LLM``/``LLMEngine`` exposing ``engine_core``.
        catalog_path / tokenizer / vocab_size: see
            :func:`compile_constraint_table_resource`.  ``catalog_path`` may be
            ``None`` to auto-build from the tokenizer's s_a/s_b/s_c groups.
        gr_config: optional; derived from ``engine.vllm_config`` when omitted.

    Returns:
        The compiled resource (digest available for observability).
    """
    if gr_config is None:
        gr_config = GRConfig.from_vllm_config(getattr(engine, "vllm_config", None))
    resource = prepare_constraint_table(
        catalog_path=catalog_path,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        gr_config=gr_config,
    )
    from vllm_gr.constraints.driver_inject import inject_constraint_resources

    inject_constraint_resources(engine, resource)
    return resource
