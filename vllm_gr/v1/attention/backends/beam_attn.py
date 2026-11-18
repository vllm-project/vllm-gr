# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Adapted from https://github.com/vllm-project/vllm

"""Beam-Consecutive Suffix Gather Attention Backend."""

from types import ModuleType
from typing import Any, ClassVar, cast

import torch
from vllm import envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadata,
    BeamAttentionMetadataBuilder,
    _get_fp8_dtype_for_flashattn,
)

__all__ = [
    "BeamAttentionBackend",
    "BeamAttentionImpl",
    "BeamAttentionMetadata",
    "BeamAttentionMetadataBuilder",
]

logger = init_logger(__name__)


def _attention_service() -> Any:
    """Return the selected backend's Beam attention service.

    Resolution goes through the provider registry, never through a platform
    constant, so this module carries no hardware identity of its own. The backend
    answers execution-path selection, the cascade/xllm split and configuration
    admissibility.

    The service is bound once per process: this runs per attention layer per
    forward pass, and activation re-walks the provider registry.
    """
    from vllm_gr.platforms.loader import active_backend_service

    return active_backend_service("create_beam_attention_service")


def _get_npu_decode_execution() -> str:
    """Return the resolved Decode execution mode name for diagnostics.

    An unset configuration resolves per backend: ``xllm`` on Ascend when the
    xllm-ops operator library is installed, ``cascade`` everywhere else.
    Explicit ``cascade`` / ``xllm`` values always win. The answer is not cached
    in this module, so two engines cannot observe different modes.
    """
    return str(_attention_service().decode_mode())


def _get_platform_execution_module() -> ModuleType:
    """Return the module serving Decode and ordinary attention."""
    return cast(ModuleType, _attention_service().execution_module())


def _get_prefill_execution_module() -> ModuleType:
    """Return the module serving the Prefill path.

    Kept separate from Decode because a decode-only fused path must not be
    asked to serve Prefill.
    """
    return cast(ModuleType, _attention_service().prefill_module())


class BeamAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def _kv_cache_views(
        self, kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(kv_cache, tuple):
            return kv_cache
        # vLLM 0.30 allocates a logical [B, H, N, K+V] view, with
        # layout-dependent strides. The kernels consume [B, N, H, D]
        # views into that same allocation, including interleaved layers.
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        return key_cache, value_cache

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        """Refresh ordinary FIA tasks; Beam uses persistent device buffers.

        Beam decode refreshes persistent fixed-address device buffers before
        replay. A backend whose Prefill owns native attention task groups
        refreshes them through its own service.
        """
        from vllm_gr.v1.beam.context import BEAM_DATA_VAR

        info = BEAM_DATA_VAR.get()
        if isinstance(info, dict) and info.get("beam_session") is not None:
            return None
        _attention_service().update_standard_graph_params(
            update_stream, forward_context, num_tokens
        )
        return None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        # Reject an unsupported configuration before any instance state is
        # stored, so a refusal cannot leave a half-initialized implementation.
        _attention_service().validate_impl_features(
            sliding_window=sliding_window,
            alibi_slopes=alibi_slopes,
            logits_soft_cap=logits_soft_cap,
            has_sinks=sinks is not None,
        )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type
        service = _attention_service()
        self.batch_invariant_enabled = envs.VLLM_BATCH_INVARIANT and (
            service.supports_batch_invariance()
        )
        platform_execution = service.execution_module()
        self.vllm_flash_attn_version = platform_execution.get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
        )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            if not BeamAttentionBackend.supports_kv_cache_dtype(self.kv_cache_dtype):
                raise NotImplementedError(
                    "BeamAttention does not support fp8 kv-cache on this device."
                )

        self.sinks = sinks
        if self.sinks is not None:
            assert platform_execution.supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of heads in the layer"
            )
        from vllm.config import get_current_vllm_config_or_none

        from vllm_gr.config import GRConfig

        config = GRConfig.from_vllm_config(get_current_vllm_config_or_none())
        short_suffix_attention = (
            service.supports_short_suffix_attention()
            and config.beam_max_width is not None
            and config.enable_short_suffix_attention
        )
        if short_suffix_attention:
            from vllm_gr.v1.attention.backends.beam_attn_gpu import (
                validate_short_suffix_attention,
            )

            validate_short_suffix_attention(
                num_heads,
                num_kv_heads,
                head_size,
                self.vllm_flash_attn_version,
                self.sliding_window,
                self.logits_soft_cap,
                self.alibi_slopes,
                self.sinks,
            )
            logger.info_once(
                "Beam prefix uses FlashAttention %s with canonical out/LSE; "
                "short-suffix fusion is enabled independently.",
                self.vllm_flash_attn_version,
            )
        # Separate prefill from the version that defines decode prefix/LSE
        # semantics. Resolve before capture; never change the global config.
        self.prefill_flash_attn_version = self.vllm_flash_attn_version
        version_selection = service.supports_prefill_version_selection()
        if version_selection and config.prefill_flash_attn_version is not None:
            assert self.vllm_flash_attn_version is not None
            self.prefill_flash_attn_version = service.resolve_prefill_version(
                config.prefill_flash_attn_version,
                self.vllm_flash_attn_version,
                head_size=head_size,
                kv_cache_dtype=kv_cache_dtype,
                requires_alibi=self.alibi_slopes is not None,
                has_sinks=self.sinks is not None,
                batch_invariant=self.batch_invariant_enabled,
            )
            logger.info_once(
                "Beam attention: requested prefill=%s, effective prefill FA%s, decode FA%s",
                config.prefill_flash_attn_version,
                self.prefill_flash_attn_version,
                self.vllm_flash_attn_version,
            )
        elif not version_selection and config.prefill_flash_attn_version not in (None, "auto"):
            # A backend without version selection rejects an explicit request
            # instead of silently ignoring it.
            service.resolve_prefill_version(
                config.prefill_flash_attn_version,
                self.vllm_flash_attn_version,
            )
        # The fused suffix consumes model-dtype queries, not quantized Q.
        self.supports_quant_query_input = not short_suffix_attention
        self.use_cascade = True

    @staticmethod
    def cascade_attention(
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        sliding_window_size: list[int] | tuple[int, int] | None,
        scale: float,
        logits_soft_cap: float,
        vllm_flash_attn_version: int,
        s_aux: torch.Tensor | None = None,
        alibi_slopes: torch.Tensor | None = None,
        beam_pool: object = None,  # Per-layer pool, passed explicitly by forward()
    ) -> None:
        platform_execution = _get_platform_execution_module()
        platform_execution.cascade_attention(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            attn_metadata=attn_metadata,
            sliding_window_size=sliding_window_size,
            scale=scale,
            logits_soft_cap=logits_soft_cap,
            vllm_flash_attn_version=vllm_flash_attn_version,
            s_aux=s_aux,
            alibi_slopes=alibi_slopes,
            beam_pool=beam_pool,
        )

    def _prefill_cascade_kwargs(self) -> dict[str, int]:
        version = self.prefill_flash_attn_version
        if version is not None and version != self.vllm_flash_attn_version:
            # Only a backend that separates the Prefill version can produce a
            # distinct value here; the service returns the decode version when
            # it has no such selection.
            return {"prefill_flash_attn_version": version}
        return {}

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if _attention_service().requires_flash_attn_version():
            assert self.vllm_flash_attn_version is not None, "FlashAttention version not detected."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for BeamAttentionImpl"
            )
        if attn_metadata is None:
            return output.fill_(0)

        if getattr(attn_metadata, "is_prefill_graph_exec", False):
            return self.prefill_forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        platform_execution = _get_platform_execution_module()
        service = _attention_service()

        key_cache, value_cache = self._kv_cache_views(kv_cache)

        # --- Incremental decode path for beam decode ---
        # When is_beam_decode is True and beam_session is available,
        # use cache_unshared_kv + cascade_attention with pool slice
        # instead of reshape_and_cache + cascade_attention with gather.
        beam_session = getattr(attn_metadata, "beam_session", None)
        is_beam_decode = getattr(attn_metadata, "is_beam_decode", False)

        if is_beam_decode and beam_session is not None and beam_session.is_active:
            # Cache only the layer index. The pool object is session-specific:
            # multi-request GPU execution replaces each physical [B, W, ...]
            # pool with a request-major [1, B*W, ...] view. Reusing the pool
            # cached by the startup B=1 capture would bypass that view and make
            # later kernels address neighbouring request slots by logically
            # indexing past W.
            layer_idx = getattr(layer, "_beam_pool_index", None)
            if layer_idx is None:
                layer_name = getattr(layer, "layer_name", "")
                layer_idx = extract_layer_index(layer_name)
                layer._beam_pool_index = layer_idx
            pool = beam_session.pools[layer_idx]

            # Extract the beam tokens' KV from the key/value tensors (they
            # contain all scheduled tokens).  The gather runs inside the
            # captured graph: fancy indexing allocates from the graph pool
            # (address-stable across replays) and re-gathers from the
            # fixed-address prefix_indices buffer each replay.
            if attn_metadata.prefix_indices is not None:
                # Beam tokens at prefix_indices positions (always set on the
                # incremental decode path, mixed or pure batch alike).
                select_rows = service.beam_row_selector()
                if select_rows is not None:
                    beam_key = select_rows(key, attn_metadata)
                    beam_value = select_rows(value, attn_metadata)
                else:
                    beam_key = key[attn_metadata.prefix_indices]  # [W, H, D]
                    beam_value = value[attn_metadata.prefix_indices]  # [W, H, D]
            else:
                # Fallback kept for callers that do not set prefix_indices;
                # the incremental path always sets it (see beam_attn_metadata).
                beam_key = key
                beam_value = value

            # cache_unshared_kv: write current step's KV into pool.
            # Use Graph-compatible Triton kernel when decode_step_buf is
            # available on CUDA (reads decode_step from device buffer instead
            # of host Python int, enabling CUDA Graph capture/replay).
            # Non-CUDA (NPU xllm/cascade, CPU unit tests) dispatches through
            # the platform module so the xllm execution mode can swap in the
            # fused NPU operator; cascade modules forward to the original
            # beam_kernels implementation.
            decode_step_buf = getattr(beam_session.context, "decode_step_buf", None)
            if decode_step_buf is not None and decode_step_buf.is_cuda:
                from vllm_gr.v1.attention.backends.beam_graph_kernels import (
                    cache_unshared_kv_graph,
                )

                cache_unshared_kv_graph(
                    pool.unshared_key,
                    pool.unshared_value,
                    beam_key,
                    beam_value,
                    beam_session.slot,
                    decode_step_buf,
                    active_mask=getattr(attn_metadata, "beam_active_mask", None),
                )
            else:
                # Non-CUDA (NPU xllm/cascade, CPU unit tests) dispatches through
                # the platform module so the xllm execution mode can swap in the
                # fused NPU operator; cascade modules forward to the original
                # beam_kernels implementation.  The beam_session gives the xllm
                # module access to the context's fixed-address device buffers
                # (ACL graph capture/replay); cascade/gpu modules ignore it.
                platform_execution.cache_unshared_kv(
                    pool.unshared_key,
                    pool.unshared_value,
                    beam_key,
                    beam_value,
                    beam_session.slot,
                    beam_session.decode_step,
                    beam_session=beam_session,
                )

            # The suffix pool stores unquantized model activations. Its dtype
            # follows the current attention K/V, independently of a possibly
            # FP8 paged prefix cache.
            sliding_window_size = (
                list(self.sliding_window) if self.sliding_window is not None else None
            )

            # cascade_attention with pool slice (incremental path)
            platform_execution.cascade_attention(
                output=output,
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                attn_metadata=attn_metadata,
                sliding_window_size=sliding_window_size,
                scale=self.scale,
                logits_soft_cap=self.logits_soft_cap,
                vllm_flash_attn_version=self.vllm_flash_attn_version,
                s_aux=self.sinks,
                alibi_slopes=self.alibi_slopes,
                beam_pool=pool,
                **self._prefill_cascade_kwargs(),
            )

            return output

        # --- Standard path (non-beam or prefill) ---
        if self.kv_sharing_target_layer_name is None and key is not None and value is not None:
            if service.requires_layer_kv_scale():
                k_scale = layer._k_scale
                v_scale = layer._v_scale
            else:
                k_scale = None
                v_scale = None
            platform_execution.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            dtype = BeamAttentionBackend.get_fp8_dtype_for_flashattn(self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None

        platform_execution.cascade_attention(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            attn_metadata=attn_metadata,
            sliding_window_size=sliding_window_size,
            scale=self.scale,
            logits_soft_cap=self.logits_soft_cap,
            vllm_flash_attn_version=self.vllm_flash_attn_version,
            s_aux=self.sinks,
            alibi_slopes=self.alibi_slopes,
            **self._prefill_cascade_kwargs(),
        )

        return output

    @staticmethod
    def prefill_attention(
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        sliding_window_size: list[int] | tuple[int, int] | None,
        scale: float,
        logits_soft_cap: float,
        vllm_flash_attn_version: int,
        s_aux: torch.Tensor | None = None,
        alibi_slopes: torch.Tensor | None = None,
    ) -> None:
        # Prefill and Decode are resolved separately: a decode-only fused path
        # must not be asked to serve Prefill.
        _get_prefill_execution_module().prefill_attention(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            attn_metadata=attn_metadata,
            sliding_window_size=sliding_window_size,
            scale=scale,
            logits_soft_cap=logits_soft_cap,
            vllm_flash_attn_version=vllm_flash_attn_version,
            s_aux=s_aux,
            alibi_slopes=alibi_slopes,
        )

    def prefill_forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Streamlined fast-path for pure prefill execution in Full Graph mode."""
        service = _attention_service()
        platform_execution = service.execution_module()

        # Validate the captured layout before submitting even the cache write.
        # A BSND dummy has no block table and cannot fall through to paged TND.
        use_bsnd = service.uses_bsnd_prefill(attn_metadata)
        if use_bsnd:
            batch_size = attn_metadata.num_reqs
            lengths = attn_metadata.seq_lens_cpu
            tokens = query.shape[0]
            if (
                batch_size <= 0
                or tokens <= 0
                or tokens % batch_size
                or lengths is None
                or len(lengths) != batch_size
                or any(length != tokens // batch_size for length in lengths)
            ):
                raise ValueError(
                    "BSND prefill requires fixed [P]*B metadata; cannot fall back to TND"
                )
            if key is None or value is None or self.kv_sharing_target_layer_name is not None:
                raise ValueError("BSND prefill requires this layer's key/value without KV sharing")
            if key.shape[0] != tokens or value.shape[0] != tokens:
                raise ValueError("BSND prefill Q/K/V token counts must match")
            if not attn_metadata.causal or attn_metadata.npu_causal_atten_mask is None:
                raise ValueError("BSND padded prefill requires the static causal mask")
            if (
                self.sliding_window not in (None, (-1, -1))
                or self.logits_soft_cap
                or self.sinks is not None
                or self.alibi_slopes is not None
                or self.kv_cache_dtype.startswith("fp8")
            ):
                raise ValueError("BSND prefill does not support this attention configuration")

        key_cache, value_cache = self._kv_cache_views(kv_cache)

        if self.kv_sharing_target_layer_name is None and key is not None and value is not None:
            if service.requires_layer_kv_scale():
                k_scale = getattr(layer, "_k_scale", None)
                v_scale = getattr(layer, "_v_scale", None)
            else:
                k_scale = None
                v_scale = None
            platform_execution.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            dtype = BeamAttentionBackend.get_fp8_dtype_for_flashattn(self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        # Both warmup and capture use current-layer K/V; replay runs this graph.
        if use_bsnd:
            from vllm_gr.v1.attention.backends import beam_attn_npu

            beam_attn_npu.prefill_attention_bsnd(
                output=output,
                query=query,
                key=key,
                value=value,
                attn_metadata=attn_metadata,
                scale=self.scale,
            )
            return output

        sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None

        # A native full-graph pass captures attention tasks that the Beam path
        # does not own. Only a backend with native task groups consults the
        # forward context; a backend without them never needs it.
        if service.has_standard_prefill_capture():
            from vllm.config import CUDAGraphMode
            from vllm.forward_context import get_forward_context

            from vllm_gr.v1.beam.context import BEAM_DATA_VAR

            context = get_forward_context()
            beam_info = BEAM_DATA_VAR.get()
            captures_standard = (
                context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and context.capturing
                and (not isinstance(beam_info, dict) or beam_info.get("beam_session") is None)
                and service.capture_standard_prefill(
                    output,
                    query,
                    key_cache,
                    value_cache,
                    attn_metadata,
                    self.scale,
                    layer.layer_name,
                    context.model_instance.vllm_config.model_config.max_model_len,
                )
            )
            if captures_standard:
                return output

        self.prefill_attention(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            attn_metadata=attn_metadata,
            sliding_window_size=sliding_window_size,
            scale=self.scale,
            logits_soft_cap=self.logits_soft_cap,
            vllm_flash_attn_version=self.prefill_flash_attn_version,
            s_aux=self.sinks,
            alibi_slopes=self.alibi_slopes,
        )

        return output


@register_backend(AttentionBackendEnum.CUSTOM)
class BeamAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # Preserve the Ascend-compatible block size that
        # _apply_npu_default_block_size pins onto the running config (128).
        # vLLM 0.30's platform hook otherwise resolves an unspecified size back
        # to a 16-multiple geometry that the Ascend kernels and prefix-caching
        # block tables cannot represent.
        from vllm.config.vllm import get_current_vllm_config_or_none

        current = get_current_vllm_config_or_none()
        if current is not None and cls.supports_block_size(current.cache_config.block_size):
            return current.cache_config.block_size
        return super().get_preferred_block_size(default_block_size)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        # vLLM 0.30 dropped this from the base class; vllm-ascend's NPU runner
        # calls it on the selected backend to size the pooled K/V cache, so the
        # shape must match what the Ascend kernels and this module's gather
        # helpers expect: a leading "2" that splits K from V, then a
        # token-major block view. Mirror vllm-ascend's own layout switch so an
        # explicit VLLM_KV_CACHE_LAYOUT=LBHNC/HND stays consistent with it.
        from vllm import envs as vllm_envs

        if vllm_envs.VLLM_KV_CACHE_LAYOUT in ("LBHNC", "HND"):
            return (2, num_blocks, num_kv_heads, block_size, head_size)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return bool(_attention_service().supports_batch_invariance())

    @staticmethod
    def get_impl_cls() -> type["BeamAttentionImpl"]:
        return BeamAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["BeamAttentionMetadataBuilder"]:
        return BeamAttentionMetadataBuilder

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Cache writes, FlashAttention and suffix gathering consume strides.
        # Prefer token-major storage, matching the former NHD default.
        return (
            KVCacheLayout.LBNHC,
            KVCacheLayout.LBHNC,
            KVCacheLayout.BLNHC,
            KVCacheLayout.BLHNC,
            KVCacheLayout.BHLNC,
            KVCacheLayout.LHBNC,
        )

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        return _get_fp8_dtype_for_flashattn(kv_cache_dtype)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return bool(_attention_service().supports_fp8_kv_cache())
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_sink(cls) -> bool:
        return bool(_attention_service().supports_sinks())

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return bool(_attention_service().supports_compute_capability(capability))

    @classmethod
    def supports_combination(
        cls,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        use_mm_prefix,
        device_capability,
    ) -> str | None:
        service = _attention_service()
        if not _get_platform_execution_module().is_available():
            return "No FlashAttention varlen support."
        if dtype not in cls.supported_dtypes:
            return f"Unsupported dtype {dtype}."
        if not cls.supports_head_size(head_size):
            return f"Unsupported head_size={head_size}."
        if not cls.supports_kv_cache_dtype(kv_cache_dtype):
            return f"Unsupported kv_cache_dtype={kv_cache_dtype}."
        if block_size is not None and block_size % 16 != 0:
            return f"block_size={block_size} not multiple of 16."
        if not service.supports_compute_capability(device_capability):
            return "Requires GPU compute capability >= 8.0."
        if has_sink and not service.supports_sinks():
            return "No FlashAttention sink support."
        if use_mla:
            return "No MLA support."
        if use_sparse:
            return "No sparse support."
        if use_mm_prefix:
            return "No multimodal prefix support."
        return None
