# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam-Consecutive Suffix Gather Attention Backend."""

from types import ModuleType

import torch
import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
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

from vllm_gr.ops import cache_unshared_kv
from vllm_gr.utils.npu_adapter import is_cuda_available, is_npu_available
from vllm_gr.v1.attention.backends.beam_attn_metadata import (
    BeamAttentionMetadata,
    BeamAttentionMetadataBuilder,
    _get_fp8_dtype_for_flashattn,
)
from vllm_gr.v1.beam.context import BEAM_DATA_VAR  # noqa: F401
from vllm_gr.v1.beam.geometry import _compute_beam_bounds  # noqa: F401

__all__ = [
    "BEAM_DATA_VAR",
    "BeamAttentionBackend",
    "BeamAttentionImpl",
    "BeamAttentionMetadata",
    "BeamAttentionMetadataBuilder",
    "_compute_beam_bounds",
]

logger = init_logger(__name__)
NPU_DEVICE = is_npu_available()
CUDA_DEVICE = is_cuda_available()


def _get_platform_execution_module() -> ModuleType:
    if NPU_DEVICE and not CUDA_DEVICE:
        from vllm_gr.v1.attention.backends import beam_attn_npu

        return beam_attn_npu
    if CUDA_DEVICE and not NPU_DEVICE:
        from vllm_gr.v1.attention.backends import beam_attn_gpu

        return beam_attn_gpu
    raise RuntimeError(
        "BeamAttention requires exactly one active CUDA or NPU device "
        f"(CUDA_DEVICE={CUDA_DEVICE}, NPU_DEVICE={NPU_DEVICE})."
    )


class BeamAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

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
        self.supports_quant_query_input = True
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
        )

    @staticmethod
    def cascade_attention_incremental(
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        unshared_key: torch.Tensor,
        unshared_value: torch.Tensor,
        suffix_k_buf: torch.Tensor,
        suffix_v_buf: torch.Tensor,
        sliding_window_size: list[int] | tuple[int, int] | None,
        scale: float,
        logits_soft_cap: float,
        vllm_flash_attn_version: int,
        s_aux: torch.Tensor | None = None,
        alibi_slopes: torch.Tensor | None = None,
    ) -> None:
        """Delegate fixed-address incremental cascade execution by platform."""

        platform_execution = _get_platform_execution_module()
        platform_execution.cascade_attention_incremental(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            attn_metadata=attn_metadata,
            unshared_key=unshared_key,
            unshared_value=unshared_value,
            suffix_k_buf=suffix_k_buf,
            suffix_v_buf=suffix_v_buf,
            sliding_window_size=sliding_window_size,
            scale=scale,
            logits_soft_cap=logits_soft_cap,
            vllm_flash_attn_version=vllm_flash_attn_version,
            s_aux=s_aux,
            alibi_slopes=alibi_slopes,
        )

    def _forward_incremental(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: BeamAttentionMetadata,
        output: torch.Tensor,
        platform_execution: ModuleType,
    ) -> torch.Tensor:
        """Execute the CUDA-only Pool-backed incremental attention path."""

        if not CUDA_DEVICE or NPU_DEVICE:
            raise NotImplementedError("Incremental Beam Attention is currently CUDA-only")
        if self.kv_sharing_target_layer_name is not None:
            raise NotImplementedError("Incremental Beam Attention does not support KV sharing")
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("Incremental Beam Attention does not support quantized KV")
        if key is None or value is None:
            raise ValueError("incremental Beam Attention requires current key and value tensors")

        session = attn_metadata.beam_session
        if session is None:
            raise RuntimeError("incremental Beam Attention metadata is missing beam_session")
        if not session.is_active:
            raise RuntimeError("incremental Beam Attention requires an active beam_session")

        beam_width = attn_metadata.incremental_beam_width
        suffix_kv_len = attn_metadata.incremental_suffix_kv_len
        if beam_width != session.beam_width:
            raise ValueError("metadata beam width does not match beam_session")
        if suffix_kv_len != session.current_step + 1:
            raise ValueError("metadata suffix length does not match beam_session.current_step")
        if attn_metadata.prefix_max_seq_len != session.prefix_len:
            raise ValueError("metadata prefix length does not match beam_session")
        if attn_metadata.suffix_total_tokens != beam_width * suffix_kv_len:
            raise ValueError("metadata suffix_total_tokens does not match the active suffix")

        layer_name = getattr(layer, "layer_name", None)
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError("incremental Beam Attention requires layer.layer_name")
        pool = session.pool_for_layer(layer_name)
        active_key = pool.active_key(beam_width)
        active_value = pool.active_value(beam_width)
        if suffix_kv_len > active_key.shape[3]:
            raise ValueError("incremental suffix length exceeds the current layer Pool")

        if key.ndim != 3 or value.shape != key.shape:
            raise ValueError("current key/value must have shape [tokens, num_heads, head_dim]")
        expected_tail = active_key.shape[2], active_key.shape[4]
        if key.shape[0] < beam_width or key.shape[1:] != expected_tail:
            raise ValueError(
                f"current key/value must contain {beam_width} active rows with shape "
                f"{expected_tail}"
            )
        for name, tensor in (
            ("value", value),
            ("query", query),
            ("output", output),
            ("key_cache", key_cache),
            ("value_cache", value_cache),
            ("suffix_k_buf", session.suffix_k_buf),
            ("suffix_v_buf", session.suffix_v_buf),
        ):
            if tensor.device != active_key.device:
                raise ValueError(f"{name} must be on {active_key.device}, got {tensor.device}")
            if tensor.dtype != active_key.dtype:
                raise TypeError(f"{name} must have the same dtype as the active Pool")
        expected_buffer_tail = active_key.shape[2], active_key.shape[4]
        for name, buffer in (
            ("suffix_k_buf", session.suffix_k_buf),
            ("suffix_v_buf", session.suffix_v_buf),
        ):
            if buffer.ndim != 3 or buffer.shape[1:] != expected_buffer_tail:
                raise ValueError(f"{name} shape does not match the current layer Pool")
            if buffer.shape[0] < beam_width * suffix_kv_len:
                raise ValueError(f"{name} does not have enough active suffix capacity")
            if not buffer.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if key.device != active_key.device:
            raise ValueError(f"key must be on {active_key.device}, got {key.device}")
        if key.dtype != active_key.dtype:
            raise TypeError("key must have the same dtype as the active Pool")
        if output.shape[0] < beam_width or query.shape[0] < beam_width:
            raise ValueError("query and output must contain every active beam row")

        curr_key = key[:beam_width]
        curr_value = value[:beam_width]
        cache_unshared_kv(
            active_key,
            active_value,
            curr_key,
            curr_value,
            session.block_table_buf,
            session.decode_step_buf,
        )

        sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None
        platform_execution.cascade_attention_incremental(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            attn_metadata=attn_metadata,
            unshared_key=pool.unshared_key,
            unshared_value=pool.unshared_value,
            suffix_k_buf=session.suffix_k_buf,
            suffix_v_buf=session.suffix_v_buf,
            sliding_window_size=sliding_window_size,
            scale=self.scale,
            logits_soft_cap=self.logits_soft_cap,
            vllm_flash_attn_version=self.vllm_flash_attn_version,
            s_aux=self.sinks,
            alibi_slopes=self.alibi_slopes,
        )
        return output

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

        platform_execution = _get_platform_execution_module()

        # NPU provides kv_cache as a tuple (key_cache, value_cache);
        # GPU provides a stacked tensor [2, num_blocks, ...]
        if isinstance(kv_cache, tuple):
            key_cache, value_cache = kv_cache[0], kv_cache[1]
        else:
            key_cache, value_cache = kv_cache.unbind(0)

        if getattr(attn_metadata, "is_incremental_decode", False):
            return self._forward_incremental(
                layer,
                query,
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata,
                output,
                platform_execution,
            )

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
