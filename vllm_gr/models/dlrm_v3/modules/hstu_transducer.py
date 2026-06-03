# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Ported to vllm-GR with modifications.

"""HSTU Transducer module."""

from typing import Any, Dict, Optional, Tuple

import torch
from torch.profiler import record_function

from vllm_gr.logger import init_logger
from vllm_gr.models.dlrm_v3.common import HammerModule
from vllm_gr.models.dlrm_v3.modules.positional_encoder import HSTUPositionalEncoder
from vllm_gr.models.dlrm_v3.modules.postprocessors import (
    L2NormPostprocessor,
    OutputPostprocessor,
)
from vllm_gr.models.dlrm_v3.modules.preprocessors import InputPreprocessor
from vllm_gr.models.dlrm_v3.modules.stu import STU
from vllm_gr.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged
from vllm_gr.utils import ReliableOps

logger = init_logger(__name__)


def complete_cumsum(lengths: torch.Tensor) -> torch.Tensor:
    """Compute complete cumulative sum (starting with 0)."""
    return ReliableOps.asynchronous_complete_cumsum(lengths)


@torch.fx.wrap
def default_seq_payload(
    seq_payloads: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if seq_payloads is None:
        return {}
    else:
        return seq_payloads


class HSTUTransducer(HammerModule):
    """
    HSTU Transducer - the main sequence encoder for DLRM_v3.

    Combines preprocessing, positional encoding, STU layers, and postprocessing.
    """

    def __init__(
        self,
        stu_module: STU,
        input_preprocessor: InputPreprocessor,
        output_postprocessor: Optional[OutputPostprocessor] = None,
        input_dropout_ratio: float = 0.0,
        positional_encoder: Optional[HSTUPositionalEncoder] = None,
        return_full_embeddings: bool = False,
        listwise: bool = False,
    ) -> None:
        super().__init__()

        self._stu_module = stu_module
        self._input_preprocessor: InputPreprocessor = input_preprocessor
        self._output_postprocessor: OutputPostprocessor = (
            output_postprocessor if output_postprocessor is not None else L2NormPostprocessor()
        )

        self._positional_encoder: Optional[HSTUPositionalEncoder] = positional_encoder
        self._input_dropout_ratio: float = input_dropout_ratio
        self._return_full_embeddings: bool = return_full_embeddings
        self._listwise_training: bool = listwise and self.is_train

        # Initialize weights
        for name, m in self.named_modules():
            if "_stu_module" in name:
                continue
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, torch.nn.LayerNorm):
                if m.weight.dim() >= 2:
                    torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None and m.bias.dim() >= 2:
                    torch.nn.init.xavier_normal_(m.bias)

    def _can_use_vllm_kv_cache(
        self,
        hstu_attn_metadata: Optional[Any],
    ) -> bool:
        return bool(
            hstu_attn_metadata is not None and getattr(hstu_attn_metadata, "cache_layers", None)
        )

    def _cached_path_has_consistent_inputs(
        self,
        *,
        seq_lengths: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        hstu_attn_metadata: Optional[Any],
    ) -> bool:
        expected_tokens = int(seq_lengths.sum().item())
        actual_tokens = int(seq_embeddings.shape[0])
        if actual_tokens != expected_tokens:
            common_attn_metadata = getattr(
                hstu_attn_metadata,
                "common_attn_metadata",
                None,
            )
            slot_mapping = getattr(common_attn_metadata, "slot_mapping", None)
            num_computed_tokens = getattr(
                common_attn_metadata,
                "_num_computed_tokens_cpu",
                None,
            )
            slot_tokens = int(slot_mapping.numel()) if slot_mapping is not None else 0
            computed_tokens = (
                int(num_computed_tokens.to(dtype=torch.int64).sum().item())
                if num_computed_tokens is not None
                else 0
            )
            logger.warning(
                "Disable HSTU cached path because preprocessed embeddings do not "
                "cover the logical HSTU sequence: embeddings=%d expected=%d "
                "slot_tokens=%d computed_tokens=%d total_targets=%d",
                actual_tokens,
                expected_tokens,
                slot_tokens,
                computed_tokens,
                int(num_targets.sum().item()) if num_targets.numel() > 0 else 0,
            )
            return False
        return True

    def _should_fallback_cached_compute(
        self,
        exc: Exception,
        *,
        seq_embeddings: torch.Tensor,
    ) -> bool:
        if seq_embeddings.device.type != "npu":
            return False
        if not isinstance(exc, (RuntimeError, ValueError)):
            return False
        message = str(exc).lower()
        return (
            "expanded size of the tensor" in message
            or "must match the existing size" in message
            or "shape mismatch" in message
            or "cached compute returned mismatched embedding shape" in message
        )

    def _preprocess(
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        seq_payloads = default_seq_payload(seq_payloads)

        with record_function("hstu_input_preprocessor"):
            (
                output_max_seq_len,
                output_total_uih_len,
                output_total_targets,
                output_seq_lengths,
                output_seq_offsets,
                output_seq_timestamps,
                output_seq_embeddings,
                output_num_targets,
                output_seq_payloads,
            ) = self._input_preprocessor(
                max_uih_len=max_uih_len,
                max_targets=max_targets,
                total_uih_len=total_uih_len,
                total_targets=total_targets,
                seq_lengths=seq_lengths,
                seq_timestamps=seq_timestamps,
                seq_embeddings=seq_embeddings,
                num_targets=num_targets,
                seq_payloads=seq_payloads,
            )
        with record_function("hstu_positional_encoder"):
            if self._positional_encoder is not None:
                output_seq_embeddings = self._positional_encoder(
                    max_seq_len=output_max_seq_len,
                    seq_lengths=output_seq_lengths,
                    seq_offsets=output_seq_offsets,
                    seq_timestamps=output_seq_timestamps,
                    seq_embeddings=output_seq_embeddings,
                    num_targets=(None if self._listwise_training else output_num_targets),
                )
        output_seq_embeddings = torch.nn.functional.dropout(
            output_seq_embeddings,
            p=self._input_dropout_ratio,
            training=self.training,
        )

        return (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
            output_seq_payloads,
        )

    def _hstu_compute(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        hstu_attn_metadata: Optional[Any] = None,
    ) -> torch.Tensor:
        del seq_timestamps
        with record_function("hstu"):
            seq_embeddings = self._stu_module(
                max_seq_len=max_seq_len,
                x=seq_embeddings,
                x_lengths=seq_lengths,
                x_offsets=seq_offsets,
                num_targets=(None if self._listwise_training else num_targets),
                hstu_attn_metadata=hstu_attn_metadata,
                hstu_attention_mode="full",
            )
        return seq_embeddings

    def _hstu_compute_cached(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        hstu_attn_metadata: Optional[Any] = None,
    ) -> torch.Tensor:
        del seq_offsets
        prefix_lengths = seq_lengths - num_targets
        if prefix_lengths.numel() == 0 or int(prefix_lengths.max().item()) <= 0:
            return self._hstu_compute(
                max_seq_len=max_seq_len,
                seq_lengths=seq_lengths,
                seq_offsets=complete_cumsum(seq_lengths),
                seq_timestamps=torch.empty(0, device=seq_embeddings.device),
                seq_embeddings=seq_embeddings,
                num_targets=num_targets,
                hstu_attn_metadata=hstu_attn_metadata,
            )

        prefix_offsets = complete_cumsum(prefix_lengths)
        delta_offsets = complete_cumsum(num_targets)
        max_prefix_len = int(prefix_lengths.max().item())
        max_delta_len = int(num_targets.max().item()) if num_targets.numel() > 0 else 0

        prefix_embeddings, delta_embeddings = split_2D_jagged(
            values=seq_embeddings,
            max_seq_len=max_seq_len,
            offsets_left=prefix_offsets,
            offsets_right=delta_offsets,
            kernel=self.hammer_kernel(),
        )

        prefix_outputs = self._stu_module(
            max_seq_len=max_prefix_len,
            x=prefix_embeddings,
            x_lengths=prefix_lengths,
            x_offsets=prefix_offsets,
            num_targets=torch.zeros_like(num_targets),
            hstu_attn_metadata=hstu_attn_metadata,
            hstu_attention_mode="prefix",
        )

        if max_delta_len == 0:
            return prefix_outputs

        delta_outputs = self._stu_module.cached_forward(
            delta_x=delta_embeddings,
            num_targets=(None if self._listwise_training else num_targets),
            hstu_attn_metadata=hstu_attn_metadata,
            hstu_attention_mode="delta",
        )

        return concat_2D_jagged(
            max_seq_len=max_seq_len,
            values_left=prefix_outputs,
            values_right=delta_outputs,
            max_len_left=max_prefix_len,
            max_len_right=max_delta_len,
            offsets_left=prefix_offsets,
            offsets_right=delta_offsets,
            kernel=self.hammer_kernel(),
        )

    def _postprocess(
        self,
        max_seq_len: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        with record_function("hstu_output_postprocessor"):
            if self._return_full_embeddings:
                seq_embeddings = self._output_postprocessor(
                    seq_embeddings=seq_embeddings,
                    seq_timestamps=seq_timestamps,
                    seq_payloads=seq_payloads,
                )

            uih_offsets = complete_cumsum(seq_lengths - num_targets)
            candidates_offsets = complete_cumsum(num_targets)

            _, candidate_embeddings = split_2D_jagged(
                values=seq_embeddings,
                max_seq_len=max_seq_len,
                total_len_left=total_uih_len,
                total_len_right=total_targets,
                offsets_left=uih_offsets,
                offsets_right=candidates_offsets,
                kernel=self.hammer_kernel(),
            )

            interleave_targets: bool = self._input_preprocessor.interleave_targets()
            if interleave_targets:
                candidate_embeddings = candidate_embeddings.view(
                    -1, 2, candidate_embeddings.size(-1)
                )[:, 0, :]

            if not self._return_full_embeddings:
                _, candidate_timestamps = split_2D_jagged(
                    values=seq_timestamps.unsqueeze(-1),
                    max_seq_len=max_seq_len,
                    total_len_left=total_uih_len,
                    total_len_right=total_targets,
                    offsets_left=uih_offsets,
                    offsets_right=candidates_offsets,
                    kernel=self.hammer_kernel(),
                )
                candidate_timestamps = candidate_timestamps.squeeze(-1)
                if interleave_targets:
                    candidate_timestamps = candidate_timestamps.view(-1, 2)[:, 0]

                candidate_embeddings = self._output_postprocessor(
                    seq_embeddings=candidate_embeddings,
                    seq_timestamps=candidate_timestamps,
                    seq_payloads=seq_payloads,
                )

            return (
                seq_embeddings if self._return_full_embeddings else None,
                candidate_embeddings,
            )

    def forward(
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
        hstu_attn_metadata: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of HSTU Transducer.

        Returns:
            (candidate_embeddings, full_embeddings or None)
        """
        (
            max_seq_len,
            total_uih_len,
            total_targets,
            seq_lengths,
            seq_offsets,
            seq_timestamps,
            seq_embeddings,
            num_targets,
            seq_payloads,
        ) = self._preprocess(
            max_uih_len=max_uih_len,
            max_targets=max_targets,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
            seq_lengths=seq_lengths,
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
            seq_payloads=seq_payloads,
        )

        use_vllm_kv_cache = (
            self._can_use_vllm_kv_cache(hstu_attn_metadata) and torch.any(num_targets > 0).item()
        )
        if use_vllm_kv_cache:
            use_vllm_kv_cache = self._cached_path_has_consistent_inputs(
                seq_lengths=seq_lengths,
                seq_embeddings=seq_embeddings,
                num_targets=num_targets,
                hstu_attn_metadata=hstu_attn_metadata,
            )
        if use_vllm_kv_cache:
            try:
                encoded_embeddings = self._hstu_compute_cached(
                    max_seq_len=max_seq_len,
                    seq_lengths=seq_lengths,
                    seq_offsets=seq_offsets,
                    seq_embeddings=seq_embeddings,
                    num_targets=num_targets,
                    hstu_attn_metadata=hstu_attn_metadata,
                )
                if encoded_embeddings.shape != seq_embeddings.shape:
                    raise RuntimeError(
                        "HSTU cached compute returned mismatched embedding shape: "
                        f"expected={tuple(seq_embeddings.shape)} "
                        f"got={tuple(encoded_embeddings.shape)}"
                    )
            except Exception as exc:
                if not self._should_fallback_cached_compute(
                    exc,
                    seq_embeddings=seq_embeddings,
                ):
                    raise
                logger.warning(
                    "Fallback to full HSTU forward because cached path failed on %s with %s",
                    seq_embeddings.device,
                    exc,
                )
                encoded_embeddings = self._hstu_compute(
                    max_seq_len=max_seq_len,
                    seq_lengths=seq_lengths,
                    seq_offsets=seq_offsets,
                    seq_timestamps=seq_timestamps,
                    seq_embeddings=seq_embeddings,
                    num_targets=num_targets,
                    hstu_attn_metadata=None,
                )
        else:
            encoded_embeddings = self._hstu_compute(
                max_seq_len=max_seq_len,
                seq_lengths=seq_lengths,
                seq_offsets=seq_offsets,
                seq_timestamps=seq_timestamps,
                seq_embeddings=seq_embeddings,
                num_targets=num_targets,
                hstu_attn_metadata=hstu_attn_metadata,
            )

        encoded_embeddings, encoded_candidate_embeddings = self._postprocess(
            max_seq_len=max_seq_len,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
            seq_lengths=seq_lengths,
            seq_embeddings=encoded_embeddings,
            seq_timestamps=seq_timestamps,
            num_targets=num_targets,
            seq_payloads=seq_payloads,
        )

        return (encoded_candidate_embeddings, encoded_embeddings)
