# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="misc"
"""Test-only forward metadata/replay counters; no model-operator instrumentation."""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import Enum
from inspect import signature
from numbers import Integral
from typing import Any

import torch


def _plain(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _snapshot(value: Any) -> Any:
    """Copy small metadata without synchronizing device tensors to the host."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_snapshot(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"unavailable:{type(value).__name__}"


class NPUForwardTrace:
    """Observe the runner boundary outside compiled/captured model code."""

    def __init__(
        self,
        runner: Any,
        get_context: Callable[[], Any],
        *,
        expected_batch_size: int = 2,
        record_cold: bool = False,
    ) -> None:
        self.runner = runner
        self.events: list[dict[str, Any]] = []
        original = runner._model_forward
        call_signature = signature(original)
        self.had_override = "_model_forward" in vars(runner)
        self.original_override = vars(runner).get("_model_forward")
        self.closed = False

        def prefill_replays() -> dict[Any, int]:
            prefill = getattr(runner, "_prefill_graph_runner", None)
            return dict(getattr(prefill, "_replays_by_bucket", {}))

        @torch.inference_mode()
        def forward(*args: Any, **kwargs: Any) -> Any:
            plan = getattr(runner, "_gr_npu_stage_plan", None)
            if plan is None:
                return original(*args, **kwargs)
            execution = plan.execution
            bound = call_signature.bind(*args, **kwargs).arguments
            context = get_context()
            compilation = runner.vllm_config.compilation_config
            descriptor = getattr(context, "batch_descriptor", None)
            padded = bound["num_tokens_padded"]
            if execution.batch_size != expected_batch_size or not 0 < padded <= 512:
                raise RuntimeError("forward tracing received an unexpected or unbounded shape")
            event = dict(
                stage=execution.stage_index,
                kind=execution.kind,
                compilation_mode=_snapshot(compilation.mode),
                configured_graph_mode=_snapshot(compilation.cudagraph_mode),
                capture_sizes=list(compilation.cudagraph_capture_sizes or []),
                enforce_eager=runner.model_config.enforce_eager,
                runtime_mode=_snapshot(context.cudagraph_runtime_mode),
                skip_compiled=context.skip_compiled,
                batch_descriptor={
                    key: _snapshot(getattr(descriptor, key, None))
                    for key in ("num_tokens", "num_reqs", "uniform", "has_lora")
                },
                actual_tokens=plan.model_token_count,
                padded_tokens=int(padded),
                model_wrapper=type(runner.model).__name__,
            )
            if execution.kind == "prefill":
                if record_cold:
                    scheduled = runner._current_scheduler_output
                    event["cold_rows"] = [
                        {
                            "num_computed_tokens": int(
                                runner.requests[request_id].num_computed_tokens
                            ),
                            "prompt_tokens": len(runner.requests[request_id].prompt_token_ids),
                            "scheduled_tokens": int(scheduled.num_scheduled_tokens[request_id]),
                        }
                        for request_id in execution.request_ids
                    ]
                    event["scheduled_total"] = int(scheduled.total_num_scheduled_tokens)
                # Values are the real arguments, not reconstructed prompt IDs.
                for field in ("input_ids", "positions"):
                    tensor = bound.get(field)
                    event[field] = _snapshot(tensor)
                    event[f"{field}_shape"] = list(tensor.shape) if tensor is not None else None
                event["runner_logits_indices"] = _snapshot(getattr(runner, "logits_indices", None))
                event["input_request_ids"] = list(runner.input_batch.req_ids)
                metadata = context.attn_metadata
                # This TP=1, non-ubatched test observes the first attention layer.
                first = next(iter(metadata.values()), None) if isinstance(metadata, dict) else None
                event["attention_metadata_type"] = type(first).__name__
                event["attention"] = {
                    field: _snapshot(getattr(first, field, None))
                    for field in (
                        "attn_state",
                        "num_actual_tokens",
                        "num_prefills",
                        "num_decodes",
                        "query_start_loc",
                        "seq_lens",
                        "seq_lens_cpu",
                        "seq_lens_list",
                        "actual_seq_lengths_q",
                        "slot_mapping",
                    )
                }
            self.events.append(event)
            before = prefill_replays()
            try:
                return original(*args, **kwargs)
            finally:
                after = prefill_replays()
                # A separate GR Prefill graph can intercept _model_forward;
                # report this explicitly, not as native runtime-mode evidence.
                event["gr_prefill_replay_delta"] = {
                    str(bucket): count - before.get(bucket, 0)
                    for bucket, count in after.items()
                    if count != before.get(bucket, 0)
                }

        runner._model_forward = forward

    def finish(self) -> list[dict[str, Any]]:
        if not self.closed:
            if self.had_override:
                self.runner._model_forward = self.original_override
            else:
                del self.runner._model_forward
            self.closed = True
        events = _plain(self.events)
        self.events.clear()
        return events  # type: ignore[no-any-return]
