# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Backend-aware operator wrappers with conservative fallbacks."""

import logging

import torch

from vllm_gr.utils.npu_adapter import get_device_adapter

logger = logging.getLogger(__name__)


class ReliableOps:
    """Run optional custom ops when available and fall back to PyTorch.

    Backend selection is delegated to DeviceAdapter so CUDA and torchrec-backed
    NPU can share the same torch.ops.fbgemm dispatch path.
    """

    @staticmethod
    def asynchronous_complete_cumsum(lengths: torch.Tensor) -> torch.Tensor:
        adapter = get_device_adapter()
        device = lengths.device

        if adapter.should_use_fbgemm(device):
            return ReliableOps._call_fbgemm_or_raise(
                op_name="asynchronous_complete_cumsum",
                device=device,
                fn=lambda: torch.ops.fbgemm.asynchronous_complete_cumsum(lengths),
            )

        cumsum = torch.zeros(
            lengths.size(0) + 1,
            dtype=lengths.dtype,
            device=device,
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
        adapter = get_device_adapter()
        device = values.device

        if adapter.should_use_fbgemm(device):
            return ReliableOps._call_fbgemm_or_raise(
                op_name="jagged_to_padded_dense",
                device=device,
                fn=lambda: torch.ops.fbgemm.jagged_to_padded_dense(
                    values=values,
                    offsets=offsets,
                    max_lengths=max_lengths,
                    padding_value=padding_value,
                ),
            )

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
        adapter = get_device_adapter()
        device = dense.device

        if adapter.should_use_fbgemm(device):
            return ReliableOps._call_fbgemm_or_raise(
                op_name="dense_to_jagged",
                device=device,
                fn=lambda: torch.ops.fbgemm.dense_to_jagged(
                    dense=dense,
                    x_offsets=x_offsets,
                )[0],
            )

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
    def _call_fbgemm_or_raise(
        op_name: str,
        device: torch.device,
        fn,
    ):
        try:
            return fn()
        except (RuntimeError, AttributeError) as exc:
            raise RuntimeError(
                "torch.ops.fbgemm."
                f"{op_name} failed on accelerated device {torch.device(device)}. "
                "CUDA/NPU paths require a working fbgemm backend and do not "
                "fall back to the PyTorch reference implementation."
            ) from exc

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
                f"{op_name} fallback only supports a single offsets tensor; "
                "multi-offset dispatch requires torch.ops.fbgemm."
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
                output[idx, :length] = values[start:start + length]

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

        logger.debug("dense_to_jagged fallback received an empty batch")
        return torch.empty(
            0,
            *dense.shape[2:],
            dtype=dense.dtype,
            device=dense.device,
        )
