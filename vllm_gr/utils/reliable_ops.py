# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure-PyTorch jagged-tensor operator wrappers.

These helpers used to dispatch to torch.ops.fbgemm when available; the fbgemm
dependency has been removed and all paths now execute through native PyTorch
implementations.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class ReliableOps:
    """Jagged-tensor operators implemented on top of native PyTorch."""

    @staticmethod
    def asynchronous_complete_cumsum(lengths: torch.Tensor) -> torch.Tensor:
        cumsum = torch.zeros(
            lengths.size(0) + 1,
            dtype=lengths.dtype,
            device=lengths.device,
        )
        torch.cumsum(lengths, dim=0, out=cumsum[1:])
        return cumsum

    @staticmethod
    def jagged_to_padded_dense(
        values: torch.Tensor,
        offsets: list[torch.Tensor],
        max_lengths: list[int],
        padding_value: float,
    ) -> torch.Tensor:
        device = values.device
        fallback_offsets = ReliableOps._resolve_single_offsets(
            op_name="jagged_to_padded_dense",
            offsets=offsets,
            default=torch.tensor(
                [0, values.size(0)],
                dtype=torch.long,
                device=device,
            ),
        )
        return ReliableOps._pytorch_jagged_to_padded_dense(
            values=values,
            offsets=fallback_offsets,
            max_length=max_lengths[0] if max_lengths else values.size(0),
            padding_value=padding_value,
        )

    @staticmethod
    def dense_to_jagged(
        dense: torch.Tensor,
        x_offsets: list[torch.Tensor],
    ) -> torch.Tensor:
        device = dense.device
        fallback_offsets = ReliableOps._resolve_single_offsets(
            op_name="dense_to_jagged",
            offsets=x_offsets,
            default=torch.tensor(
                [0, dense.size(1)],
                dtype=torch.long,
                device=device,
            ),
        )
        return ReliableOps._pytorch_dense_to_jagged(
            dense=dense,
            offsets=fallback_offsets,
        )

    @staticmethod
    def _resolve_single_offsets(
        op_name: str,
        offsets: list[torch.Tensor],
        default: torch.Tensor,
    ) -> torch.Tensor:
        if not offsets:
            return default

        if len(offsets) != 1:
            raise NotImplementedError(
                f"{op_name} only supports a single offsets tensor; "
                "multi-offset dispatch is not implemented in the PyTorch path."
            )

        return offsets[0]

    @staticmethod
    def _pytorch_jagged_to_padded_dense(
        values: torch.Tensor,
        offsets: torch.Tensor,
        max_length: int,
        padding_value: float,
    ) -> torch.Tensor:
        batch_size = offsets.size(0) - 1
        width = values.size(-1) if values.dim() > 1 else 1

        if values.dim() == 1:
            values = values.unsqueeze(-1)

        output = torch.full(
            (batch_size, max_length, width),
            padding_value,
            dtype=values.dtype,
            device=values.device,
        )

        for idx in range(batch_size):
            start = int(offsets[idx].item())
            end = int(offsets[idx + 1].item())
            length = min(end - start, max_length)
            if length > 0:
                output[idx, :length] = values[start : start + length]

        if width == 1:
            output = output.squeeze(-1)
        return output

    @staticmethod
    def _pytorch_dense_to_jagged(
        dense: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        results = []
        for idx in range(dense.size(0)):
            start = int(offsets[idx].item())
            end = int(offsets[idx + 1].item())
            if end > start:
                results.append(dense[idx, : end - start])

        if results:
            return torch.cat(results, dim=0)

        logger.debug("dense_to_jagged received an empty batch")
        return torch.empty(
            0,
            *dense.shape[2:],
            dtype=dense.dtype,
            device=dense.device,
        )
