# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Ascend Beam attention and physical metadata policy."""

from __future__ import annotations

from typing import Any

from ..attention import AttentionExecutor, MetadataProfile


class AscendMetadataProfile(MetadataProfile):
    def make_causal_mask(self, device: Any) -> Any:
        import torch

        return torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(device)


class AscendAttention(AttentionExecutor):
    """Own Ascend Beam attention selection and admissibility.

    Decode and Prefill are resolved separately because the fused ``xllm`` path
    serves Decode only: Prefill always needs the cascade module. Shared code
    asks for the path it is about to run instead of branching on a platform
    constant.
    """

    metadata_profile = AscendMetadataProfile(
        allocates_fixed_buffers=True,
        packs_seqlens_non_blocking=True,
        allows_identity_block_table_slice=True,
        marks_identity_prefix_rows=True,
    )

    backend_id = "ascend"

    def normalize_bootstrap_selector(
        self, selector: Any, *, config: Any, num_heads: int | None
    ) -> Any:
        # NPUModelRunner.__init__ in 0.30.0rc2 queries the backend class with
        # (0, model dtype, None) before loading any attention layer.
        if selector.head_size != 0 or selector.kv_cache_dtype is not None or num_heads is not None:
            return selector
        get_head_size = getattr(getattr(config, "model_config", None), "get_head_size", None)
        if not callable(get_head_size):
            return selector
        return selector._replace(head_size=get_head_size())

    def execution_module(self) -> Any:
        """Return the module serving Decode and ordinary attention."""
        if self.decode_mode() == "xllm":
            from vllm_gr.v1.attention.backends import beam_attn_npu_xllm

            return beam_attn_npu_xllm
        from vllm_gr.v1.attention.backends import beam_attn_npu

        return beam_attn_npu

    def prefill_module(self) -> Any:
        """Return the cascade module, which owns Ascend Prefill attention."""
        from vllm_gr.v1.attention.backends import beam_attn_npu

        return beam_attn_npu

    def __init__(self, requested: str = "auto", *, kv_layout: str | None = None) -> None:
        super().__init__(kv_layout=kv_layout)
        if requested == "auto":
            from vllm_gr.ops.npu import beam_ops

            requested = "xllm" if beam_ops.is_available() else "cascade"
        if requested not in ("xllm", "cascade"):
            raise ValueError(f"Unsupported Ascend attention mode {requested!r}")
        self._mode = requested

    def reorder_kv(
        self, parent: Any, slot: int, pools: Any, decode_step: int, **kwargs: Any
    ) -> None:
        from vllm_gr.v1.beam.reorder import (
            _fused_reorder_applies,
            _group_token_num_from_parents,
            reorder_unshared_kv,
        )

        if self._mode == "xllm" and _fused_reorder_applies(parent, pools[0].unshared_key.shape[1]):
            import torch

            from vllm_gr.ops.npu import beam_ops

            beam_ops.select_unshared_kv(
                parent.to(torch.int32).contiguous(),
                [p.unshared_key for p in pools],
                [p.unshared_value for p in pools],
                torch.tensor([slot], dtype=torch.int32, device=parent.device),
                _group_token_num_from_parents(parent).to(parent.device),
                decode_step + 1,
                pools[0].unshared_key.shape[1],
                len(pools),
            )
            return
        reorder_unshared_kv(parent, slot, pools, decode_step, **kwargs)

    def decode_mode(self) -> str:
        return self._mode

    def supports_sinks(self) -> bool:
        # Neither the cascade nor the xllm execution path implements sinks.
        return False

    def requires_layer_kv_scale(self) -> bool:
        # The Ascend cache-write operators apply no layer scale.
        return False

    def uses_bsnd_prefill(self, attn_metadata: Any) -> bool:
        return getattr(attn_metadata, "prefill_fia_layout", "TND") == "BSND"

    def beam_row_selector(self) -> Any:
        """Return the xllm dense-row reuse hook when that path is selected."""
        if self.decode_mode() != "xllm":
            return None
        from vllm_gr.v1.attention.backends.beam_attn_npu_xllm import select_beam_rows

        return select_beam_rows

    def validate_impl_features(
        self,
        *,
        sliding_window: Any,
        alibi_slopes: Any,
        logits_soft_cap: Any,
        has_sinks: bool,
    ) -> None:
        unsupported = []
        if sliding_window is not None:
            unsupported.append("sliding_window")
        if alibi_slopes is not None:
            unsupported.append("alibi_slopes")
        if logits_soft_cap not in (None, 0.0):
            unsupported.append("logits_soft_cap")
        if unsupported:
            raise NotImplementedError(
                "NPU CUSTOM Beam Attention does not support "
                + ", ".join(unsupported)
                + "; neither the cascade nor xllm execution path implements these settings."
            )
        if has_sinks:
            raise NotImplementedError("Sinks are not yet supported on NPU.")

    def resolve_prefill_version(
        self,
        requested: Any,
        decode_version: Any,
        **kwargs: Any,
    ) -> Any:
        # Prefill FlashAttention versioning is a CUDA concept; an explicit
        # request is a configuration error rather than a silently ignored one.
        del kwargs
        if requested not in (None, "auto"):
            raise ValueError("beam.prefill_flash_attn_version is CUDA-only")
        return decode_version

    def update_standard_graph_params(
        self,
        update_stream: Any,
        forward_context: Any,
        num_tokens: Any,
    ) -> None:
        from vllm_gr.v1.attention.backends.npu_standard_graph import (
            update_standard_graph_params,
        )

        update_standard_graph_params(update_stream, forward_context, num_tokens)

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
        """Capture the standard FIA task group during a native full-graph pass.

        Ascend prefill graphs capture ordinary attention tasks that the Beam
        path does not own, so shared attention must hand those to the backend
        instead of running its own Prefill kernel.
        """
        from vllm_gr.v1.attention.backends import beam_attn_npu
        from vllm_gr.v1.attention.backends.npu_standard_graph import (
            capture_standard_fia,
        )

        if beam_attn_npu.in_prefill_capture():
            return False
        capture_standard_fia(
            output,
            query,
            key_cache,
            value_cache,
            attn_metadata,
            scale,
            layer_name,
            max_model_len,
        )
        return True

    def has_standard_prefill_capture(self) -> bool:
        # Ascend captures native FIA task groups for the ordinary Prefill path.
        return True

    def validate_prefill(self, impl: Any, metadata: Any, query: Any, key: Any, value: Any) -> bool:
        if not self.uses_bsnd_prefill(metadata):
            return False
        batch_size = metadata.num_reqs
        lengths = metadata.seq_lens_cpu
        tokens = query.shape[0]
        if (
            batch_size <= 0
            or tokens <= 0
            or tokens % batch_size
            or lengths is None
            or len(lengths) != batch_size
            or any(length != tokens // batch_size for length in lengths)
        ):
            raise ValueError("BSND prefill requires fixed [P]*B metadata; cannot fall back to TND")
        if key is None or value is None or impl.kv_sharing_target_layer_name is not None:
            raise ValueError("BSND prefill requires this layer's key/value without KV sharing")
        if key.shape[0] != tokens or value.shape[0] != tokens:
            raise ValueError("BSND prefill Q/K/V token counts must match")
        if not metadata.causal or metadata.npu_causal_atten_mask is None:
            raise ValueError("BSND padded prefill requires the static causal mask")
        if (
            impl.sliding_window not in (None, (-1, -1))
            or impl.logits_soft_cap
            or impl.sinks is not None
            or impl.alibi_slopes is not None
            or impl.kv_cache_dtype.startswith("fp8")
        ):
            raise ValueError("BSND prefill does not support this attention configuration")
        return True

    def special_prefill(self, **kwargs: Any) -> None:
        self.prefill_module().prefill_attention_bsnd(**kwargs)
