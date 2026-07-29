# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam-Consecutive Suffix Gather Attention Backend."""

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
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
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
from vllm_gr.v1.attention.backends.merge_tnd_attention_outputs import (
    extract_suffix_kv_pytorch,
    merge_tnd_attention_outputs,
)
from vllm_gr.v1.beam.context import BEAM_DATA_VAR  # noqa: F401
from vllm_gr.v1.beam.geometry import _compute_beam_bounds  # noqa: F401

logger = init_logger(__name__)
# Device flag – must be checked after torch is loaded.
NPU_DEVICE = is_npu_available()
CUDA_DEVICE = is_cuda_available()

if NPU_DEVICE:
    import torch_npu  # noqa: F401
if CUDA_DEVICE:
    from vllm_gr.v1.attention.backends.beam_attn_triton import (
        extract_suffix_kv,
    )

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
else:

    def _flash_attn_varlen_unavailable(*args, **kwargs):
        raise RuntimeError("BeamAttentionBackend requires FlashAttention varlen support")

    flash_attn_varlen_func = _flash_attn_varlen_unavailable
    reshape_and_cache_flash = _flash_attn_varlen_unavailable
    merge_attn_states = _flash_attn_varlen_unavailable
    flash_attn_supports_sinks = _flash_attn_varlen_unavailable


# ---------------------------------------------------------------------------
# CORE EXECUTION REFACTOR
# ---------------------------------------------------------------------------


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
        self.vllm_flash_attn_version = (
            get_flash_attn_version(
                requires_alibi=alibi_slopes is not None,
                head_size=head_size,
            )
            if not NPU_DEVICE
            else 0
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
            assert flash_attn_supports_sinks(), "Sinks are only supported in FlashAttention 3"
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of heads in the layer"
            )
        self.supports_quant_query_input = True
        self.use_cascade = True

    @staticmethod
    def cascade_attention_gpu(
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
    ):
        if attn_metadata.std_indices is not None:
            if attn_metadata.prefix_indices is None:
                std_query = query[: attn_metadata.num_std_indices]
            else:
                std_query = query[attn_metadata.std_indices]
            std_out = flash_attn_varlen_func(
                q=std_query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=attn_metadata.std_cu_seqlens_q_device,
                max_seqlen_q=attn_metadata.std_max_q_len,
                seqused_k=attn_metadata.std_seqlens_kv_device,
                max_seqlen_k=attn_metadata.std_max_seq_len,
                softmax_scale=scale,
                causal=attn_metadata.causal,
                alibi_slopes=alibi_slopes,
                window_size=sliding_window_size,
                block_table=attn_metadata.std_block_table,
                softcap=logits_soft_cap,
                fa_version=vllm_flash_attn_version,
            )
            if attn_metadata.prefix_indices is None:
                output[: attn_metadata.num_std_indices] = std_out
            else:
                output[attn_metadata.std_indices] = std_out

        if attn_metadata.prefix_indices is not None:
            if attn_metadata.std_indices is None:
                beam_query = query[: attn_metadata.num_prefix_indices]
            else:
                beam_query = query[attn_metadata.prefix_indices]

            prefix_out, prefix_lse = flash_attn_varlen_func(
                q=beam_query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=attn_metadata.prefix_cu_seqlens_q_device,
                seqused_k=attn_metadata.prefix_seqlens_kv_device,
                max_seqlen_q=attn_metadata.prefix_max_q_len,
                max_seqlen_k=attn_metadata.prefix_max_seq_len,
                softmax_scale=scale,
                causal=False,
                alibi_slopes=alibi_slopes,
                window_size=sliding_window_size,
                block_table=attn_metadata.prefix_block_table,
                softcap=logits_soft_cap,
                return_softmax_lse=True,
                fa_version=vllm_flash_attn_version,
                scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
                num_splits=attn_metadata.max_num_splits,
                s_aux=s_aux,
            )

            contig_k, contig_v = extract_suffix_kv(
                key_cache,
                value_cache,
                attn_metadata.suffix_slot_mapping_out,
                attn_metadata.suffix_total_tokens,
                attn_metadata.block_size,
            )

            suffix_out, suffix_lse = flash_attn_varlen_func(
                q=beam_query,
                k=contig_k,
                v=contig_v,
                cu_seqlens_q=attn_metadata.suffix_cu_seqlens_q_device,
                cu_seqlens_k=attn_metadata.suffix_cu_seqlens_k_device,
                seqused_k=None,
                max_seqlen_q=attn_metadata.suffix_max_q_len,
                max_seqlen_k=attn_metadata.suffix_max_seq_len,
                softmax_scale=scale,
                causal=True,
                alibi_slopes=alibi_slopes,
                window_size=sliding_window_size,
                block_table=None,
                softcap=logits_soft_cap,
                return_softmax_lse=True,
                fa_version=vllm_flash_attn_version,
            )
            group_size = attn_metadata.num_heads // attn_metadata.num_kv_heads
            if (
                attn_metadata.suffix_max_q_len == 1
                and attn_metadata.suffix_max_seq_len == 1
                and group_size > 1
                and not vllm_flash_attn_version == 3
            ):
                suffix_lse = (
                    suffix_lse.view(attn_metadata.num_kv_heads, group_size, -1)
                    .transpose(1, 2)
                    .contiguous()
                    .view(attn_metadata.num_heads, -1)
                )

            out = torch.empty_like(beam_query)
            merge_attn_states(out, prefix_out, prefix_lse, suffix_out, suffix_lse)
            if attn_metadata.std_indices is None:
                output[: attn_metadata.num_prefix_indices] = out
            else:
                output[attn_metadata.prefix_indices] = out

    @staticmethod
    def cascade_attention_npu(
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
    ):
        # Standard attention
        if attn_metadata.std_indices is not None:
            if attn_metadata.prefix_indices is None:
                std_query = query[: attn_metadata.num_std_indices]
            else:
                std_query = query[attn_metadata.std_indices]
            std_out, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query=std_query,
                key=key_cache.view(*key_cache.shape[:-2], -1),
                value=value_cache.view(*value_cache.shape[:-2], -1),
                atten_mask=attn_metadata.npu_causal_atten_mask if attn_metadata.causal else None,
                block_table=attn_metadata.std_block_table,
                input_layout="TND",
                block_size=attn_metadata.block_size,
                actual_seq_qlen=attn_metadata.std_cu_seqlens_q_cpu,
                actual_seq_kvlen=attn_metadata.std_seqlens_kv_cpu,
                num_key_value_heads=attn_metadata.num_kv_heads,
                num_query_heads=attn_metadata.num_heads,
                softmax_scale=scale,
                sparse_mode=3 if attn_metadata.causal else 0,
            )
            if attn_metadata.prefix_indices is None:
                output[: attn_metadata.num_std_indices] = std_out
            else:
                output[attn_metadata.std_indices] = std_out
        # Prefix/Suffix attention
        if attn_metadata.prefix_indices is not None:
            if attn_metadata.std_indices is None:
                beam_query = query[: attn_metadata.num_prefix_indices]
            else:
                beam_query = query[attn_metadata.prefix_indices]
            prefix_out, prefix_lse = torch_npu.npu_fused_infer_attention_score_v2(
                query=beam_query,
                key=key_cache.view(*key_cache.shape[:-2], -1),
                value=value_cache.view(*value_cache.shape[:-2], -1),
                block_table=attn_metadata.prefix_block_table,
                input_layout="TND",
                atten_mask=None,
                block_size=attn_metadata.block_size,
                actual_seq_qlen=attn_metadata.prefix_cu_seqlens_q_cpu,
                actual_seq_kvlen=attn_metadata.prefix_seqlens_kv_cpu,
                num_key_value_heads=attn_metadata.num_kv_heads,
                num_query_heads=attn_metadata.num_heads,
                softmax_scale=scale,
                return_softmax_lse=True,
                sparse_mode=0,  # prefix attention is non-causal (full)
            )

            contig_k, contig_v = extract_suffix_kv_pytorch(
                key_cache, value_cache, attn_metadata.suffix_slot_mapping_out
            )

            suffix_out, suffix_lse = torch_npu.npu_fused_infer_attention_score_v2(
                beam_query,
                contig_k,
                contig_v,
                num_query_heads=attn_metadata.num_heads,
                num_key_value_heads=attn_metadata.num_kv_heads,
                actual_seq_qlen=attn_metadata.suffix_cu_seqlens_q_cpu,
                actual_seq_kvlen=attn_metadata.suffix_cu_seqlens_kv_cpu,
                softmax_scale=scale,
                atten_mask=attn_metadata.npu_causal_atten_mask,
                block_table=None,
                block_size=0,
                sparse_mode=3,
                inner_precise=0,
                input_layout="TND",
                return_softmax_lse=True,
            )
            out, _ = merge_tnd_attention_outputs(prefix_out, prefix_lse, suffix_out, suffix_lse)
            if attn_metadata.std_indices is None:
                output[: attn_metadata.num_prefix_indices] = out
            else:
                output[attn_metadata.prefix_indices] = out

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
    ):
        if CUDA_DEVICE and not NPU_DEVICE:
            BeamAttentionImpl.cascade_attention_gpu(
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
        elif NPU_DEVICE and not CUDA_DEVICE:
            BeamAttentionImpl.cascade_attention_npu(
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

        # NPU provides kv_cache as a tuple (key_cache, value_cache);
        # GPU provides a stacked tensor [2, num_blocks, ...]
        if isinstance(kv_cache, tuple):
            key_cache, value_cache = kv_cache[0], kv_cache[1]
        else:
            key_cache, value_cache = kv_cache.unbind(0)

        if self.kv_sharing_target_layer_name is None and key is not None and value is not None:
            if NPU_DEVICE:
                torch_npu._npu_reshape_and_cache(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                )
            elif CUDA_DEVICE:
                reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

        if self.kv_cache_dtype.startswith("fp8"):
            dtype = BeamAttentionBackend.get_fp8_dtype_for_flashattn(self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        sliding_window_size = list(self.sliding_window) if self.sliding_window is not None else None

        BeamAttentionImpl.cascade_attention(
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
            return get_flash_attn_version() == 3 and current_platform.is_device_capability_family(
                90
            )
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_sink(cls) -> bool:
        if NPU_DEVICE:
            return False  # sinks not yet supported on NPU
        return is_flash_attn_varlen_func_available() and flash_attn_supports_sinks()

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
        if not NPU_DEVICE and not is_flash_attn_varlen_func_available():
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
