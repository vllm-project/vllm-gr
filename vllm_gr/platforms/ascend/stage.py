# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Runner-owned Ascend GR stage and operator bindings."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_gr.v1.beam.decision_ops import DecisionOps, reference_ops, reorder_sessions

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..plan import BackendPlan


@dataclass(frozen=True)
class AscendStageExecutionService:
    """Execute, retire and pre-capture Ascend GR Beam stages."""

    implementation_id: str
    decision: DecisionOps
    worker_decision: bool

    def execute(
        self,
        worker: Any,
        scheduled: Any,
        original: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        from vllm_gr.v1.worker.npu.npu_beam_stage_runner import get_npu_stage_runner

        stage_runner = get_npu_stage_runner(worker, scheduled)
        try:
            return stage_runner.execute(scheduled, original, *args, **kwargs)
        except Exception as error:
            stage_runner.execution_error = error
            raise

    def retire(self, worker: Any, scheduled: Any) -> Any:
        from vllm_gr.v1.worker.npu.npu_beam_stage_runner import (
            execute_npu_gr_retirement,
        )

        return execute_npu_gr_retirement(worker, scheduled)

    def capture_startup_graph(
        self,
        worker: Any,
        *,
        allow_eager_fallback: bool = False,
    ) -> bool:
        del allow_eager_fallback
        from vllm_gr.v1.worker.npu.npu_model_runner_patch import (
            capture_beam_decode_graph,
        )

        return bool(capture_beam_decode_graph(worker))

    def admit_unconstrained_beam(self, worker: Any) -> None:
        """Apply the Ascend-only device admission rules for unconstrained Beam."""
        from vllm_gr.v1.engine.beam_search_v1 import validate_gr_device_config

        validate_gr_device_config(worker.vllm_config)


def _reorder(session: Any, sessions: tuple[Any, ...], indices: Any, counts: Any) -> None:
    from vllm_gr.ops.npu import beam_ops

    pools = session.context.pools
    batch, width = indices.shape
    if batch == 1 and width != pools[0].unshared_key.shape[1]:
        reorder_sessions(session, sessions, indices, counts)
        return
    steps = {item._state.decode_step for item in sessions}
    if len(steps) != 1 or tuple(counts.shape) != (batch, width):
        raise RuntimeError("Ascend batched KV reorder has inconsistent step or geometry")
    table = getattr(session.context, "xllm_block_table_buf", None)
    if table is None:
        import torch

        table = torch.tensor(
            [item.slot for item in sessions], dtype=torch.int32, device=indices.device
        )
    elif batch == 1:
        table = table[sessions[0].slot : sessions[0].slot + 1]
    else:
        table = table[:batch]
    try:
        beam_ops.select_unshared_kv(
            indices.reshape(-1).contiguous(),
            [pool.unshared_key for pool in pools],
            [pool.unshared_value for pool in pools],
            table,
            counts.reshape(-1).contiguous(),
            steps.pop() + 1,
            width,
            len(pools),
        )
    except Exception:
        logger.warning(
            "Fused Ascend KV reorder failed; using native session reorder", exc_info=True
        )
        if batch == 1:
            reorder_sessions(session, sessions, indices, counts)
        else:
            for row, item in enumerate(sessions):
                _reorder(item, (item,), indices[row : row + 1], counts[row : row + 1])
        return
    for item in sessions:
        item._state.decode_step += 1


def create_stage(*, plan: BackendPlan | None, runner: Any) -> AscendStageExecutionService:
    from vllm_gr.config import GRConfig
    from vllm_gr.ops.npu import beam_ops

    choices = plan.choices.to_dict() if plan is not None else {}
    if (
        choices.get("operator_policy", "ascend.xllm-with-reference-fallback")
        != "ascend.xllm-with-reference-fallback"
    ):
        raise RuntimeError("Ascend runner cannot bind the admitted decision operators")
    if choices.get("stage", "ascend.v1") != "ascend.v1":
        raise RuntimeError("Ascend native stage differs from the admitted backend plan")
    attention_available = beam_ops.is_available()
    available = beam_ops.decision_ops_available()
    mode = choices.get("decode_attention")
    if mode is None:
        configured = GRConfig.from_vllm_config(
            getattr(runner, "vllm_config", None)
        ).npu_decode_execution
        mode = configured.value if configured is not None else "auto"
    fused = mode == "xllm" or (mode == "auto" and attention_available)
    requested = plan.request.config.to_dict() if plan is not None else {}
    # Legacy worker selection is an explicit deployment capability. V1 chooses
    # between fixed and dynamic decisions using the submitted request instead.
    if (
        mode == "xllm"
        and requested.get("beam_execution_mode") != "v1"
        and requested.get("beam_worker_decision")
        and requested.get("beam_max_width") is not None
        and (
            not requested.get("catalog_path")
            or requested.get("constraint_backend") == "constraint_table"
        )
        and not available
    ):
        raise RuntimeError("The admitted xllm worker decision operators are unavailable")
    ops = DecisionOps(
        "ascend.xllm" if available else "torch.reference",
        beam_ops.beam_search_group if available else None,
        beam_ops.beam_search_rec_final_select if available else None,
        32,
        _reorder if attention_available else reorder_sessions,
        True,
        reference_ops(),
    )
    return AscendStageExecutionService("ascend.v1", ops, fused)
