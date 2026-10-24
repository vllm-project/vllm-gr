# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam-Consecutive Suffix Gather Attention Backend."""

from types import ModuleType

import torch
from vllm import envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.kv_cache_interface import is_quantized_kv_cache

from vllm_gr.utils.npu_adapter import is_cuda_available, is_npu_available
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
NPU_DEVICE = is_npu_available()
CUDA_DEVICE = is_cuda_available()

# In-process cache for the resolved vllm_gr_npu_decode_execution mode.  The
# vLLM config is process-global and set once before model load, so the first
# non-None read settles the mode for the lifetime of the process.
_NPU_DECODE_EXECUTION_MODE: str | None = None


def _get_npu_decode_execution() -> str:
    """Lazily resolve ``vllm_gr_npu_decode_execution`` (cascade | xllm).

    An unset config (None) resolves per platform: ``xllm`` on NPU when the
    xllm-ops operator library is installed, ``cascade`` everywhere else (CUDA,
    NPU without the library, or config not yet installed in this process).
    Explicit ``cascade`` / ``xllm`` values always win.
    """
    global _NPU_DECODE_EXECUTION_MODE
    if _NPU_DECODE_EXECUTION_MODE is not None:
        return _NPU_DECODE_EXECUTION_MODE
    from vllm.config import get_current_vllm_config_or_none

    from vllm_gr.config import GRConfig

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        # Config not installed yet in this process; default without caching.
        return "cascade"
    configured = GRConfig.from_vllm_config(vllm_config).npu_decode_execution
    if configured is not None:
        _NPU_DECODE_EXECUTION_MODE = configured.value
        return _NPU_DECODE_EXECUTION_MODE
    if NPU_DEVICE and not CUDA_DEVICE:
        from vllm_gr.ops.npu import beam_ops

        _NPU_DECODE_EXECUTION_MODE = "xllm" if beam_ops.is_available() else "cascade"
        if _NPU_DECODE_EXECUTION_MODE == "cascade":
            logger.info(
                "vllm_gr_npu_decode_execution is unset and the xllm-ops library "
                "is unavailable (%s); defaulting to the cascade decode path. "
                "Install xllm-ops (tools/install_xllm_ops.sh) to enable the "
                "fused xllm decode path by default.",
                beam_ops.unavailable_reason(),
            )
    else:
        _NPU_DECODE_EXECUTION_MODE = "cascade"
    return _NPU_DECODE_EXECUTION_MODE


def _get_platform_execution_module() -> ModuleType:
    npu_decode_execution = _get_npu_decode_execution()
    if NPU_DEVICE and not CUDA_DEVICE:
        if npu_decode_execution == "xllm":
            from vllm_gr.v1.attention.backends import beam_attn_npu_xllm

            return beam_attn_npu_xllm
        from vllm_gr.v1.attention.backends import beam_attn_npu

        return beam_attn_npu
    if CUDA_DEVICE and not NPU_DEVICE:
        if npu_decode_execution == "xllm":
            raise RuntimeError(
                "vllm_gr_npu_decode_execution='xllm' requires an Ascend NPU; "
                "unset the option or select 'cascade' on CUDA."
            )
        from vllm_gr.v1.attention.backends import beam_attn_gpu

        return beam_attn_gpu
    raise RuntimeError(
        "BeamAttention requires exactly one active CUDA or NPU device "
        f"(CUDA_DEVICE={CUDA_DEVICE}, NPU_DEVICE={NPU_DEVICE})."
    )


class BeamAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

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
        replay. NPU prefill updates its native attention task groups directly
        in ``NPUPrefillGraphRunner`` and does not use this shared hook.
        """
        if NPU_DEVICE and not CUDA_DEVICE:
            from vllm_gr.v1.beam.context import BEAM_DATA_VAR

            info = BEAM_DATA_VAR.get()
            if not isinstance(info, dict) or info.get("beam_session") is None:
                from vllm_gr.v1.attention.backends.npu_standard_graph import (
                    update_standard_graph_params,
                )

                update_standard_graph_params(update_stream, forward_context, num_tokens)
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
        platform_execution = _get_platform_execution_module()
        self.vllm_flash_attn_version = platform_execution.get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
        )
        self.batch_invariant_enabled = envs.VLLM_BATCH_INVARIANT if CUDA_DEVICE else False

        if is_quantized_kv_cache(self.kv_cache_dtype):
            if NPU_DEVICE:
                raise NotImplementedError("BeamAttention does not support fp8 kv-cache on NPU.")
            if not BeamAttentionBackend.supports_kv_cache_dtype(self.kv_cache_dtype):
                raise NotImplementedError(
                    "BeamAttention does not support fp8 kv-cache on this device."
                )

        self.sinks = sinks
        if self.sinks is not None:
            if NPU_DEVICE:
                raise NotImplementedError("Sinks are not yet supported on NPU.")
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
            CUDA_DEVICE
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
        if CUDA_DEVICE:
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

        # NPU provides kv_cache as a tuple (key_cache, value_cache);
        # GPU provides a stacked tensor [2, num_blocks, ...]
        if isinstance(kv_cache, tuple):
            key_cache, value_cache = kv_cache[0], kv_cache[1]
        else:
            key_cache, value_cache = kv_cache.unbind(0)

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
                select_rows = (
                    getattr(platform_execution, "select_beam_rows", None)
                    if key.device.type == "npu"
                    else None
                )
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
            )

            return output

        # --- Standard path (non-beam or prefill) ---
        if self.kv_sharing_target_layer_name is None and key is not None and value is not None:
            if NPU_DEVICE:
                k_scale = None
                v_scale = None
            else:
                k_scale = layer._k_scale
                v_scale = layer._v_scale
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
        # The prefill path is never serviced by the xllm module (decode-only
        # by design): on NPU resolve the cascade module directly so the
        # prefill branch bypasses the process-global platform-module
        # resolution entirely.
        if NPU_DEVICE and not CUDA_DEVICE:
            from vllm_gr.v1.attention.backends import beam_attn_npu

            platform_execution = beam_attn_npu
        else:
            platform_execution = _get_platform_execution_module()
        platform_execution.prefill_attention(
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
        platform_execution = _get_platform_execution_module()

        if isinstance(kv_cache, tuple):
            key_cache, value_cache = kv_cache[0], kv_cache[1]
        else:
            key_cache, value_cache = kv_cache.unbind(0)

        if self.kv_sharing_target_layer_name is None and key is not None and value is not None:
            k_scale = None if NPU_DEVICE else getattr(layer, "_k_scale", None)
            v_scale = None if NPU_DEVICE else getattr(layer, "_v_scale", None)
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

        if NPU_DEVICE and not CUDA_DEVICE:
            from vllm.config import CUDAGraphMode
            from vllm.forward_context import get_forward_context

            from vllm_gr.v1.attention.backends import beam_attn_npu
            from vllm_gr.v1.beam.context import BEAM_DATA_VAR

            context = get_forward_context()
            beam_info = BEAM_DATA_VAR.get()
            if (
                context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and context.capturing
                and not beam_attn_npu._GRAPH_CAPTURE_STATE["capturing"]
                and (not isinstance(beam_info, dict) or beam_info.get("beam_session") is None)
            ):
                from vllm_gr.v1.attention.backends.npu_standard_graph import (
                    capture_standard_fia,
                )

                capture_standard_fia(
                    output,
                    query,
                    key_cache,
                    value_cache,
                    attn_metadata,
                    self.scale,
                    layer.layer_name,
                    context.model_instance.vllm_config.model_config.max_model_len,
                )
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
            vllm_flash_attn_version=self.vllm_flash_attn_version,
            s_aux=self.sinks,
            alibi_slopes=self.alibi_slopes,
        )

        return output


@register_backend(AttentionBackendEnum.CUSTOM)
class BeamAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    supported_dtypes = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return CUDA_DEVICE

    @staticmethod
    def get_impl_cls() -> type["BeamAttentionImpl"]:
        return BeamAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["BeamAttentionMetadataBuilder"]:
        return BeamAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension: bool = False) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            return (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            return (0, 1, 3, 2, 4)
        raise ValueError(f"Unknown cache layout format {cache_layout}.")

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
            if NPU_DEVICE:
                return False  # fp8 kv-cache not yet supported on NPU
            if not CUDA_DEVICE:
                return False
            platform_execution = _get_platform_execution_module()
            return (
                platform_execution.get_flash_attn_version() == 3
                and current_platform.is_device_capability_family(90)
            )
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_sink(cls) -> bool:
        if NPU_DEVICE:
            return False  # sinks not yet supported on NPU
        if not CUDA_DEVICE:
            return False
        platform_execution = _get_platform_execution_module()
        return platform_execution.is_available() and platform_execution.supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # NPU does not use CUDA-style compute capability; accept it.
        if NPU_DEVICE:
            return True
        return capability >= DeviceCapability(8, 0)

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
        device_capability,
    ) -> str | None:
        if not NPU_DEVICE:
            if not CUDA_DEVICE:
                return "No FlashAttention varlen support."
            if not _get_platform_execution_module().is_available():
                return "No FlashAttention varlen support."
        if dtype not in cls.supported_dtypes:
            return f"Unsupported dtype {dtype}."
        if not cls.supports_head_size(head_size):
            return f"Unsupported head_size={head_size}."
        if not NPU_DEVICE and not cls.supports_kv_cache_dtype(kv_cache_dtype):
            return f"Unsupported kv_cache_dtype={kv_cache_dtype}."
        if block_size is not None and block_size % 16 != 0:
            return f"block_size={block_size} not multiple of 16."
        if not cls.supports_compute_capability(device_capability):
            return "Requires GPU compute capability >= 8.0."
        if not NPU_DEVICE and has_sink and not cls.supports_sink():
            return "No FlashAttention sink support."
        if use_mla:
            return "No MLA support."
        if use_sparse:
            return "No sparse support."
        return None
