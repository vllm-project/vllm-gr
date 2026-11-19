# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..config import BackendConfigService

if TYPE_CHECKING:
    from vllm_gr.config import GRConfig


class CudaConfigService(BackendConfigService):
    default_worker_decision = True
    supports_constraint_vocab_slicing = True

    def startup_inputs(self, admitted: Mapping[str, Any] | None) -> dict[str, Any]:
        from vllm_gr.v1.attention.backends.beam_dense_prefill import (
            requested_dense_prefill_mode,
        )

        if admitted is None:
            selected = requested_dense_prefill_mode()
            invariant = (
                bool(int(os.environ.get("VLLM_BATCH_INVARIANT", "0"))) if selected else False
            )
        else:
            selected = admitted.get("dense_prefill_request")
            invariant = admitted.get("dense_prefill_batch_invariant", False)
        return {"dense_prefill_request": selected, "dense_prefill_batch_invariant": invariant}

    def validate_engine(self, config: Any, gr: GRConfig) -> None:
        from vllm_gr.config import BeamExecutionMode

        if (
            getattr(config, "use_v2_model_runner", False)
            and (
                gr.beam_max_width is not None
                or (gr.constraint_table and gr.constraint_table.enabled)
            )
            and gr.beam_execution_mode is not BeamExecutionMode.V1
        ):
            raise ValueError(
                "ModelRunner V2 Beam requires beam.execution_mode='v1'; "
                "select it explicitly before engine initialization, or use ModelRunner V1"
            )

    def finalize_native_config(self, config: Any, gr: GRConfig) -> None:
        from vllm_gr.config import BeamExecutionMode

        compilation = getattr(config, "compilation_config", None)
        if (
            gr.beam_execution_mode is BeamExecutionMode.V1
            and not getattr(config, "use_v2_model_runner", False)
            and compilation is not None
            and compilation.backend == "inductor"
        ):
            # Fix reduction geometry across AOT loads and preserve explicit overrides.
            compilation.inductor_compile_config.setdefault("dynamic_scale_rblock", False)
