# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Import-light provider entry point; discovery must not load the vendor stack."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    import torch

    from vllm_gr.constraints.backend import ConstraintBackend
    from vllm_gr.constraints.execution import ConstrainedTopKExecution
    from vllm_gr.constraints.table import ConstraintTable
    from vllm_gr.platforms.identity import PlatformIdentity


class MLUBackendProvider:
    backend_id = "mlu"
    device_type = "mlu"
    api_version = 2
    constraint_backend_api_version = 1
    gr_version_range = None
    vllm_version_range = None
    vendor_version_range = None
    vendor_distribution = "vllm-mlu"

    def matches(self, identity: PlatformIdentity) -> bool:
        return identity.device_type == self.device_type

    def check_environment(self) -> NoReturn:
        # No native Runner/KV ABI has passed the RFC's P0 gate yet. Stop before
        # core startup installs any patches for a runtime we cannot support.
        from vllm_gr.platforms.errors import UnsupportedBackendError

        raise UnsupportedBackendError(
            "vllm-gr-mlu provides constraint execution only: MLU Beam V1 requires "
            "a validated base MLU vLLM V1 Runner/Executor and attention/KV ABI (RFC #603 P0)"
        )

    def apply_runtime_patches(self, *, force: bool = False) -> NoReturn:
        self.check_environment()

    def get_model_runner_hooks(self) -> NoReturn:
        self.check_environment()

    def get_attention_execution(self) -> NoReturn:
        self.check_environment()

    def create_constraint_backend(
        self,
        *,
        artifact: ConstraintTable,
        device: torch.device,
        dtype: torch.dtype,
        max_rows: int,
        max_top_k: int,
    ) -> ConstraintBackend:
        from .constraints.backend import create_constraint_backend

        return create_constraint_backend(
            artifact=artifact, device=device, dtype=dtype, max_rows=max_rows, max_top_k=max_top_k
        )

    def create_constrained_topk_execution(
        self, *, device: str | torch.device | None = None
    ) -> ConstrainedTopKExecution:
        from .constraints.execution import MLUConstrainedTopKExecution

        return MLUConstrainedTopKExecution(device=device)


MLU_BACKEND = MLUBackendProvider()
