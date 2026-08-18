# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
import os
from typing import Any

from vllm_gr.config import GRConfig
from vllm_gr.constraints.compiler import (
    FORMAT_VERSION,
    ConstraintTableCompiler,
    ConstraintTableResource,
)

logger = logging.getLogger(__name__)

# Test-only override that forces the Torch oracle adapter regardless of the
# active platform.  Used by CI / development only; production always goes
# through the platform provider and fails fast when it cannot run the operator.
_CONSTRAINT_EXECUTION_ENV = "VLLM_GR_CONSTRAINT_EXECUTION"
_TORCH_OVERRIDE = "torch"


def create_constraint_execution(
    *,
    device: str | Any | None = None,
) -> Any:
    """Create the platform constrained-top-k adapter, failing explicitly.

    Honors the test-only ``VLLM_GR_CONSTRAINT_EXECUTION=torch`` override
    (CI/dev only).  Production resolves the active platform provider; a ``None``
    adapter (platform without an implementation) or an unavailable adapter
    (NPU custom op missing) is a hard failure — Constraint Tables never silently
    degrades to Trie.
    """
    override = os.environ.get(_CONSTRAINT_EXECUTION_ENV)
    if override is not None:
        if override != _TORCH_OVERRIDE:
            raise ValueError(
                f"unsupported {_CONSTRAINT_EXECUTION_ENV}={override!r}; "
                f"expected {_TORCH_OVERRIDE!r} or unset"
            )
        from vllm_gr.constraints.execution import TorchConstrainedTopKExecution

        logger.info(
            "vllm-gr: constraint execution overridden to Torch oracle (%s=%s); CI/dev only",
            _CONSTRAINT_EXECUTION_ENV,
            override,
        )
        return TorchConstrainedTopKExecution(device=device)

    from vllm_gr.platforms.loader import activate_current_backend

    provider = activate_current_backend()
    adapter = provider.create_constrained_topk_execution(device=device)
    if adapter is None:
        platform = provider.device_type
        raise RuntimeError(
            "constraint_backend='constraint_table' cannot start: the active "
            f"platform {platform!r} does not implement Constraint Table "
            "execution. Constraint Tables are not available on this platform; "
            "starting must fail explicitly rather than fall back to Trie."
        )
    if not adapter.is_available():
        platform = provider.device_type
        raise RuntimeError(
            "constraint_backend='constraint_table' cannot start: the active "
            f"platform {platform!r} provides a Constraint Table adapter but "
            "its custom operator is unavailable (is_available()=False). "
            "Verify the operator binary / environment; starting must fail "
            "explicitly rather than fall back to Trie."
        )
    return adapter


def validate_constraint_table_platform(gr_config: GRConfig) -> None:
    """Reject configurations Constraint Tables cannot support in this release.

    Vocabulary slicing changes the logits vocab while the compiled tables are
    keyed on the full ``model_config.vocab_size``; the two cannot be combined
    yet (issue #246 §7).
    """
    if gr_config.vocab_offset is not None or gr_config.vocab_slice_size is not None:
        raise ValueError(
            "constraint_backend='constraint_table' cannot be combined with "
            f"vocab slicing (vocab_offset={gr_config.vocab_offset!r}, "
            f"vocab_slice_size={gr_config.vocab_slice_size!r}) in the initial "
            "release"
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


def prepare_constraint_table(
    *,
    catalog_path: str,
    tokenizer: Any,
    vocab_size: int,
    gr_config: GRConfig | None = None,
) -> ConstraintTableResource:
    """Activate the platform backend, validate, and compile the resource.

    Does not inject into workers; used by the OpenAI serving path where the
    broadcast is deferred to the first ``beam_search`` (see
    :func:`~vllm_gr.entrypoints.openai.serving_models.OpenAIServingModels._ensure_constraint_tables_injected`).

    The platform adapter is probed *at startup* via
    :func:`create_constraint_execution` — a CUDA host (no adapter) or an NPU
    host whose custom operator is unavailable must fail here, before any
    constrained request runs, never at the first constrained step.
    """
    from vllm_gr.platforms.loader import activate_current_backend

    activate_current_backend()
    if gr_config is not None:
        validate_constraint_table_platform(gr_config)
    # Startup-time fail-fast (issue #246 §7): the probe adapter is discarded;
    # the Worker lazily creates its own with the runner device (N3).
    create_constraint_execution()
    return compile_constraint_table_resource(catalog_path, tokenizer, vocab_size)


def install_constraint_table(
    engine: Any,
    *,
    catalog_path: str,
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
            :func:`compile_constraint_table_resource`.
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
