# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Attention binding during native layer construction, metadata setup and serving."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from weakref import WeakSet

from .plan import BackendPlan, read_backend_plan

# vLLM creates AttentionImpl without passing a runner. Keep its owner in the
# native model-load scope only. Forward holds the executor directly on the layer.
_loading: ContextVar[Any] = ContextVar("gr_attention_loading", default=None)
_selector: ContextVar[Any] = ContextVar("gr_attention_selector", default=None)


@dataclass(frozen=True)
class MetadataProfile:
    """Physical tensor packing requirements for a backend."""

    allocates_fixed_buffers: bool = False
    uses_v2_prefill_capture: bool = False
    forces_single_reduction_split: bool = False
    packs_seqlens_non_blocking: bool = False
    allows_identity_block_table_slice: bool = False
    marks_identity_prefix_rows: bool = False
    writes_seqlens_per_tensor: bool = False

    def make_causal_mask(self, device: Any) -> Any:
        return None

    def scheduler_metadata(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("This backend does not support ahead-of-time attention scheduling")


class AttentionExecutor:
    """Legacy execution primitives plus the portable Beam layout and admission."""

    uses_pool_pointers = False
    block_multiple = 16
    metadata_profile = MetadataProfile()

    def __init__(self, primitives: Any = None, *, kv_layout: str | None = None) -> None:
        self.primitives = primitives
        self.kv_layout = kv_layout
        self._builders: WeakSet[Any] = WeakSet()

    def track_builder(self, builder: Any) -> None:
        self._builders.add(builder)

    def invalidate(self) -> None:
        """Forget cached geometry before a graph or device layout is replaced."""
        for builder in self._builders:
            builder.invalidate_geometry()

    def close(self) -> None:
        self.invalidate()
        self._builders.clear()

    @property
    def implementation_id(self) -> str:
        module = self.execution_module()
        return str(getattr(module, "__name__", type(module).__qualname__))

    def execution_module(self) -> Any:
        return self.primitives

    def prefill_module(self) -> Any:
        return self.execution_module()

    def decode_mode(self) -> Any:
        return "cascade"

    def supports_sinks(self) -> bool:
        return bool(self.execution_module().supports_sinks())

    def supports_sliding_window(self) -> bool:
        return False

    def supports_fp8_kv_cache(self) -> bool:
        return False

    def supports_batch_invariance(self) -> bool:
        return False

    def requires_flash_attn_version(self) -> bool:
        return False

    def supports_short_suffix_device(self, device: Any) -> bool:
        return self.supports_short_suffix_attention()

    def supports_short_suffix_attention(self) -> bool:
        return False

    def supports_prefill_version_selection(self) -> bool:
        return False

    def supports_compute_capability(self, capability: Any) -> bool:
        return True

    def requires_layer_kv_scale(self) -> bool:
        return True

    def beam_row_selector(self) -> Any:
        return None

    def validate_impl_features(
        self, *, sliding_window: Any, alibi_slopes: Any, logits_soft_cap: Any, has_sinks: bool
    ) -> None:
        pass

    def has_standard_prefill_capture(self) -> bool:
        return False

    def update_standard_graph_params(self, *args: Any) -> None:
        pass

    def validate_prefill(self, impl: Any, metadata: Any, query: Any, key: Any, value: Any) -> bool:
        return False

    def short_suffix_head_size_supported(self, head_size: int) -> bool:
        return False

    def dense_prefill_mode(self, config: Any) -> str | None:
        return None

    def resolve_prefill_version(self, requested: Any, decode_version: Any, **kwargs: Any) -> Any:
        if requested not in (None, "auto"):
            raise ValueError("Backend does not support independent Prefill version selection")
        return decode_version

    def supports_head_size(self, size: int) -> bool:
        return size > 0 and size % 8 == 0 and size <= 256

    def supports_kv_cache_dtype(self, dtype: Any) -> bool:
        return dtype in (None, "auto", "float16", "bfloat16") or (
            dtype in ("fp8", "fp8_e4m3") and self.supports_fp8_kv_cache()
        )

    def supports_combination(
        self,
        head_size: int,
        dtype: Any,
        kv_cache_dtype: Any,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: Any,
    ) -> str | None:
        # vLLM's validate_configuration calls the individual capability hooks
        # first, including our narrower head-size, block-size and KV dtype limits.
        if not self.execution_module().is_available():
            return "No FlashAttention varlen support."
        return None

    def reorder_kv(self, *args: Any, **kwargs: Any) -> None:
        from vllm_gr.v1.beam.reorder import reorder_unshared_kv

        reorder_unshared_kv(*args, **kwargs)

    def kv_cache_views(self, cache: Any, head_size: int) -> tuple[Any, Any]:
        if isinstance(cache, tuple):
            return cache
        key, value = cache.transpose(1, 2).split(head_size, dim=-1)
        return key, value

    def kv_layouts(self) -> tuple[Any, ...]:
        from vllm.v1.kv_cache_layout import KVCacheLayout

        return (
            KVCacheLayout.LBNHC,
            KVCacheLayout.LBHNC,
            KVCacheLayout.BLNHC,
            KVCacheLayout.BLHNC,
            KVCacheLayout.BHLNC,
            KVCacheLayout.LHBNC,
        )

    def kv_cache_shape(
        self, num_blocks: int, block_size: int, num_kv_heads: int, head_size: int
    ) -> tuple[int, ...]:
        if self.kv_layout in ("LBHNC", "HND"):
            return (2, num_blocks, num_kv_heads, block_size, head_size)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    def write_suffix_kv(self, pool: Any, key: Any, value: Any, session: Any, metadata: Any) -> None:
        self.execution_module().cache_unshared_kv(
            pool.unshared_key,
            pool.unshared_value,
            key,
            value,
            session.slot,
            session.decode_step,
            beam_session=session,
        )


def _surface(executor: Any) -> Any:
    """Keep v1/v2 primitive providers working without assuming fused features."""
    if callable(getattr(executor, "execution_module", None)):
        return executor
    return AttentionExecutor(executor)


def validate_beam_attention(executor: Any, plan: BackendPlan | None = None) -> None:
    """Check the public BeamAttentionExecution primitives on the selected worker."""
    surface = _surface(executor)
    primitives = surface.execution_module()
    for method in (
        "get_flash_attn_version",
        "supports_sinks",
        "reshape_and_cache",
        "cascade_attention",
        "is_available",
    ):
        if not callable(getattr(primitives, method, None)):
            raise RuntimeError(f"Backend CUSTOM attention does not implement {method}()")
    if not primitives.is_available():
        raise RuntimeError("Backend CUSTOM attention execution is unavailable")
    if plan is None or plan.request.uses_beam_v1:
        writer = getattr(surface, "write_suffix_kv", None)
        if not callable(writer):
            raise RuntimeError("Backend Beam Decode attention does not implement write_suffix_kv()")
        if getattr(writer, "__func__", None) is AttentionExecutor.write_suffix_kv and not callable(
            getattr(primitives, "cache_unshared_kv", None)
        ):
            raise RuntimeError(
                "Backend Beam Decode attention does not implement cache_unshared_kv()"
            )
    if plan is not None and plan.choices.to_dict().get("prefill_graph") in (True, "shared"):
        get_prefill = getattr(surface, "prefill_module", None)
        if not callable(get_prefill) or not callable(
            getattr(get_prefill(), "prefill_attention", None)
        ):
            raise RuntimeError(
                "Backend Prefill graph attention does not implement prefill_attention()"
            )


def validate_selected_attention(runner: Any, executor: Any, plan: BackendPlan) -> None:
    """Validate the execution paths vLLM built, including explicit native backend classes."""
    config = getattr(runner.vllm_config, "compilation_config", None)
    impls = {
        type(layer.impl)
        for layer in getattr(config, "static_forward_context", {}).values()
        if getattr(layer, "attn_backend", None) is not None
        and getattr(layer, "impl", None) is not None
    }
    if not impls:
        # Standalone runners can use the default Beam implementation directly.
        validate_beam_attention(executor, plan)
        return
    for impl in impls:
        validate = getattr(impl, "validate_gr_attention_executor", None)
        if validate is not None:
            validate(executor, plan)


@contextmanager
def loading_attention(runner: Any) -> Iterator[None]:
    """Let layers and the eventual binding share one provisionally owned executor."""
    token = _loading.set(runner)
    try:
        yield
    finally:
        _loading.reset(token)
        pending = getattr(runner, "_gr_loading_attention", None)
        if pending is not None:
            del runner._gr_loading_attention
            from .binding import _close_services

            _close_services([pending[1], pending[2]])


def bind_attention(runner: Any, provider: Any, plan: BackendPlan) -> tuple[Any, Any]:
    """Transfer the provider executor and the exact surface used by its native layers."""
    pending = getattr(runner, "_gr_loading_attention", None)
    if pending is not None:
        if pending[0] != plan:
            raise RuntimeError("Layer attention belongs to another admitted plan")
        del runner._gr_loading_attention
        return pending[1], pending[2]
    executor = provider.create_attention_executor(plan=plan, device=getattr(runner, "device", None))
    return executor, _surface(executor) if executor is not None else None


def construction_config() -> Any:
    runner = _loading.get()
    if runner is not None:
        return getattr(runner, "vllm_config", None)
    from vllm.config import get_current_vllm_config_or_none

    return get_current_vllm_config_or_none()


def admission_for_config() -> Any:
    """Return the cold capability policy even when vLLM is constructing layers."""
    selected = _selector.get()
    return selected if selected is not None else _admission(construction_config())


@contextmanager
def selector_admission(selector: Any = None, *, num_heads: int | None = None) -> Iterator[Any]:
    """Use one stateless policy across native class-level capability queries."""
    config = construction_config()
    policy = _admission(config)
    token = _selector.set(policy)
    try:
        normalize = getattr(policy, "normalize_bootstrap_selector", None)
        if selector is not None and _loading.get() is None and callable(normalize):
            selector = normalize(selector, config=config, num_heads=num_heads)
        yield selector
    finally:
        _selector.reset(token)


def for_config(config: Any = None, *, layer_names: Any = None) -> Any:
    """Resolve the native selector or builder at setup; forward uses the layer field."""
    selected = _selector.get()
    if selected is not None:
        return selected
    runner = _loading.get()
    if config is None:
        config = construction_config()
    if runner is not None and config is getattr(runner, "vllm_config", None):
        pending = getattr(runner, "_gr_loading_attention", None)
        if pending is not None:
            return pending[2]
        native = getattr(runner, "_gr_native_attention", None)
        if native is not None:
            return native
    if runner is not None and config is runner.vllm_config:
        from .loader import activate_current_backend
        from .services import planning_provider

        plan = read_backend_plan(config)
        if plan is not None and "attention" in plan.required_services:
            executor, surface = bind_attention(
                runner, planning_provider(activate_current_backend()), plan
            )
            if executor is None:
                raise RuntimeError("Backend did not bind the required attention executor")
            runner._gr_loading_attention = (plan, executor, surface)
            return surface

    # The builder supplies the layer names from its native KV group. Keep
    # separate runners' resources independent even when a config is reused.
    layers = getattr(getattr(config, "compilation_config", None), "static_forward_context", {})
    if layer_names is not None:
        executors = {}
        for name in layer_names:
            impl = getattr(layers.get(name), "impl", None)
            executor = getattr(impl, "_execution", None)
            if executor is not None:
                executors[id(executor)] = executor
        if len(executors) > 1 and read_backend_plan(config) is not None:
            raise RuntimeError("Native attention group spans multiple runner bindings")
        if executors:
            return next(iter(executors.values()))
    service = _admission(config)
    if runner is not None and config is runner.vllm_config and read_backend_plan(config) is None:
        runner._gr_native_attention = service
    return service


def _admission(config: Any) -> Any:
    from .loader import activate_current_backend

    provider = activate_current_backend()
    plan = read_backend_plan(config)
    if plan is not None:
        if getattr(provider, "backend_id", None) != plan.backend_id:
            raise ValueError("Attention admission provider differs from the engine plan")
        admission = getattr(provider, "attention_admission", None)
        if getattr(provider, "plan_api_version", None) == 1 and callable(admission):
            return admission(plan=plan)
        # Older providers only expose primitive capability queries. Never
        # instantiate their mutable device executor just to select a class.
        return AttentionExecutor(provider.get_attention_execution())
    # Unplanned native selectors have the pre-planning, primitive provider API.
    # Built-in selection can inspect options during setup without caching them.
    native = getattr(provider, "native_attention", None)
    return (
        native(config)
        if native is not None
        else AttentionExecutor(provider.get_attention_execution())
    )


def for_runner(runner: Any) -> Any:
    binding = getattr(runner, "_gr_backend_binding", None)
    surface = getattr(binding, "attention_surface", None)
    if surface is not None:
        return surface
    executor = getattr(binding, "attention_executor", None)
    if executor is not None:
        return _surface(executor)
    native = getattr(runner, "_gr_native_attention", None)
    return native if native is not None else for_config(getattr(runner, "vllm_config", None))
