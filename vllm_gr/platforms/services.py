# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Backend-owned execution services, so shared code never selects by device name.

The provider layer already answers "which backend am I". This module answers the
next question: "which implementation performs this piece of execution". Every
answer is returned by the selected provider, so a call site asks for a service
instead of comparing a device string.

Each service is intentionally narrow, and its hardware module imports live inside
method bodies. Shared code therefore never imports a CUDA or Ascend module merely
to decide what to do, and ``vllm_gr.platforms`` keeps its import isolation.

The service set mirrors the RFC #706 execution table. Stage execution, native
retirement, startup graph capture and Beam attention ownership are the
responsibilities that still carried device-name branches in shared code at
provider API version 2. Constraint and ModelRunner patch selection already flow
through provider factories on their production paths.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StageExecutionService(Protocol):
    """Execute, retire and pre-capture one GR Beam stage for a backend.

    ``capture_startup_graph`` carries the per-backend startup capture policy. A
    backend returns ``True`` when the capture it owns succeeded, ``False`` when
    capture does not apply to this runner, and raises when the capture was
    attempted and failed. Shared startup code therefore never inspects a device
    type to decide which of those three outcomes is expected.

    ``admit_unconstrained_beam`` carries the per-backend admission rules for an
    unconstrained session. A backend that cannot serve that configuration raises;
    a backend that can returns normally.

    The ``*_beam_*`` methods carry the operator-path policy that RFC #706 places
    with the runner adapter or stage service. Shared Beam decision code keeps one
    implementation of the semantics and asks the backend which execution shape
    applies, instead of comparing a device name.
    """

    def execute(
        self,
        worker: Any,
        scheduled: Any,
        original: Callable[..., Any],
        *args: Any,
        hooks_type: Any = None,
        **kwargs: Any,
    ) -> Any: ...

    def retire(self, worker: Any, scheduled: Any) -> Any: ...

    def capture_startup_graph(
        self,
        worker: Any,
        *,
        hooks_type: Any = None,
        allow_eager_fallback: bool = False,
    ) -> bool: ...

    def admit_unconstrained_beam(self, worker: Any) -> None: ...

    def supports_worker_beam_decision(self) -> bool:
        """Whether worker-side Beam decision is this backend's execution path."""

    def uses_fused_beam_decision(self) -> bool:
        """Whether the fused decision operators are this backend's choices.

        A backend without them leaves shared code on the portable reference
        path rather than failing at the first decoded step.
        """

    def requires_beam_candidate_width_match(self) -> bool:
        """Whether this backend's decision operators require ``K == W``.

        The fused operators hard-code one survivor per Beam row; the reference
        path accepts a narrower candidate set.
        """

    def batches_beam_kv_reorder(self) -> bool:
        """Whether this backend reorders every request's KV pool in one call."""

    def requires_aligned_final_select(self) -> bool:
        """Whether final selection needs an aligned result width on this backend."""


@runtime_checkable
class BeamAttentionService(Protocol):
    """Own Beam attention execution selection and configuration admissibility.

    Shared Beam attention asks this service instead of testing a platform
    constant. The service answers both questions that used to be device-name
    branches: which execution module runs a given path, and whether a
    configuration is admissible at all.

    Selection is split by path because a backend may use different kernels for
    Prefill and Decode while registering one CUSTOM backend, and because a
    decode-only fused path must not be asked to serve Prefill.

    Every admissibility answer is a property of the backend, not of a device
    name. A backend that cannot serve a feature returns ``False`` and the shared
    caller rejects the configuration with the same message it used before.
    """

    backend_id: str

    def execution_module(self) -> Any:
        """Return the module serving Decode and ordinary attention."""

    def prefill_module(self) -> Any:
        """Return the module serving the Prefill path.

        Always a module that implements ``prefill_attention``. A backends whose
        fused Decode path does not implement Prefill returns its cascade module
        here.
        """

    def decode_mode(self) -> str:
        """Return the resolved Decode execution mode name for diagnostics."""

    def supports_sinks(self) -> bool: ...

    def supports_fp8_kv_cache(self) -> bool: ...

    def supports_batch_invariance(self) -> bool: ...

    def requires_flash_attn_version(self) -> bool:
        """Whether Decode attention requires a detected FlashAttention version."""

    def supports_short_suffix_attention(self) -> bool:
        """Whether this backend implements the fused short-suffix prefix path."""

    def supports_prefill_version_selection(self) -> bool:
        """Whether this backend separates the Prefill FlashAttention version."""

    def supports_compute_capability(self, capability: Any) -> bool: ...

    def requires_layer_kv_scale(self) -> bool:
        """Whether the cache write consumes the layer's ``_k_scale``/``_v_scale``."""

    def uses_bsnd_prefill(self, attn_metadata: Any) -> bool:
        """Whether this Prefill metadata describes a BSND static-shape layout."""

    def beam_row_selector(self) -> Any:
        """Return a dense-row reuse hook, or ``None`` when this backend has none.

        A backend that can prove its Beam rows are already contiguous may avoid
        materializing a gather; one that cannot returns ``None`` and shared code
        performs the ordinary index gather.
        """

    def validate_impl_features(
        self,
        *,
        sliding_window: Any,
        alibi_slopes: Any,
        logits_soft_cap: Any,
        has_sinks: bool,
    ) -> None:
        """Reject an attention configuration this backend cannot execute.

        Runs before any per-instance state is stored, so a rejected
        configuration leaves no partially initialized implementation behind.
        """

    def resolve_prefill_version(
        self,
        requested: Any,
        decode_version: Any,
        **kwargs: Any,
    ) -> Any:
        """Resolve the Prefill FlashAttention version from the configured request."""

    def update_standard_graph_params(
        self,
        update_stream: Any,
        forward_context: Any,
        num_tokens: Any,
    ) -> None:
        """Refresh backend-native attention tasks during a graph parameter update."""

    def capture_standard_prefill(
        self,
        output: Any,
        query: Any,
        key_cache: Any,
        value_cache: Any,
        attn_metadata: Any,
        scale: float,
        layer_name: str,
        max_model_len: int,
    ) -> bool:
        """Capture a backend-native standard Prefill task group, if this path does.

        Returns ``True`` when the capture ran and the caller must return
        ``output``; ``False`` when ordinary Prefill attention applies.
        """

    def has_standard_prefill_capture(self) -> bool:
        """Whether this backend captures native Prefill tasks at all.

        Answering ``False`` keeps the caller from consulting the forward
        context, which a backend without native task groups never needs.
        """


@runtime_checkable
class BackendConfigService(Protocol):
    """Answer the configuration-time policy that a backend owns.

    These questions used to be platform predicates in shared configuration code.
    RFC #706 requires that adding a backend does not need edits to shared default
    pinning, so each answer moves behind the provider.
    """

    backend_id: str

    def apply_config_defaults(self, config: Any) -> None:
        """Pin backend-specific configuration defaults on a resolved config."""

    def supports_worker_decision_by_default(self) -> bool:
        """Whether an omitted Beam worker-decision default resolves to enabled."""

    def promotes_constraint_tables(self) -> bool:
        """Whether a triples catalog promotes an unset backend to Constraint Tables."""

    def supports_constraint_vocab_slicing(self) -> bool:
        """Whether this backend's Constraint Table ABI accepts a sliced logits window."""

    def validate_unconstrained_runner(self, config: Any) -> None:
        """Apply this backend's admission rules for an unconstrained Beam runner."""

    def requires_persistent_v2_protocol(self) -> bool:
        """Whether a V2 ModelRunner Beam engine requires the persistent protocol."""

    def install_engine_core_config_sanitizer(self) -> None:
        """Install any pre-native EngineCore configuration filter this backend needs."""

    def default_constraint_table_backend(self) -> str | None:
        """Return this backend's Constraint Table placement, or ``None``."""


__all__ = [
    "BackendConfigService",
    "BeamAttentionService",
    "StageExecutionService",
]
