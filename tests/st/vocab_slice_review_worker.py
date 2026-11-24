# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Diagnostic worker: hooks are disabled for official E2E measurements."""

import os
from typing import Any, Callable

import torch
from vllm.v1.worker.gpu_worker import Worker


class ReviewWorker(Worker):  # type: ignore[misc]  # Upstream Worker is untyped.
    def load_model(self, **kwargs: Any) -> None:
        if os.environ.get("REVIEW_STRICT_BF16") == "1":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        super().load_model(**kwargs)
        self.review_active = False
        self.review_events: list[tuple[str, Any, Any, dict[str, Any]]] = []
        self.review_shadow: list[dict[str, Any]] = []
        self.review_stage: dict[str, Any] = {}
        self.review_replays = 0
        runner = self.model_runner
        self.review_projection = os.environ["REVIEW_PROJECTION"]
        if self.review_projection == "full":
            for processor in runner._gr_vocab_slice_processors:
                processor._gr_vocab_slice = None
        self.review_windows = [p._gr_vocab_slice for p in runner._gr_vocab_slice_processors]
        original_forward = runner._model_forward

        def forward(*args: Any, **kwargs: Any) -> Any:
            return self.review_timed("forward", original_forward, *args, **kwargs)

        runner._model_forward = forward
        from vllm_gr.v1.worker.gpu.beam_decode_graph import BeamDecodeGraph

        original_replay = BeamDecodeGraph.replay

        def replay(graph: Any) -> Any:
            self.review_replays += 1
            return self.review_timed("forward_graph", original_replay, graph)

        BeamDecodeGraph.replay = replay
        from vllm_gr.v1.worker.gpu.beam_stage_runner import GPUBeamStageRunner

        original_admit = GPUBeamStageRunner.admit

        def admit(stage_runner: Any, scheduler_output: Any) -> Any:
            output = original_admit(stage_runner, scheduler_output)
            m = output.entries[0].metadata
            self.review_stage = {"kind": m.kind.value, "decode_step": m.decode_step}
            return output

        GPUBeamStageRunner.admit = admit
        for processor in runner._gr_vocab_slice_processors:
            original = processor._get_logits

            def projection(
                hidden: Any,
                head: Any,
                bias: Any,
                original: Callable[..., Any] = original,
                processor: Any = processor,
            ) -> Any:
                logits = self.review_timed("projection", original, hidden, head, bias)
                if self.review_active:
                    self.review_events[-1][3]["logits_shape"] = list(logits.shape)
                    self.review_events[-1][3]["hidden_shape"] = list(hidden.shape)
                    window = processor._gr_vocab_slice
                    if window is not None and len(self.review_shadow) < 3:
                        processor._gr_vocab_slice = None
                        try:
                            full = original(hidden, head, bias)
                        finally:
                            processor._gr_vocab_slice = window
                        ref = full[:, window[0] : window[0] + window[1]]
                        delta = (logits.float() - ref.float()).abs()
                        self.review_shadow.append(
                            {
                                **self.review_stage,
                                "full_shape": list(full.shape),
                                "slice_shape": list(logits.shape),
                                "exact": torch.equal(logits, ref),
                                "max_abs_diff": delta.max().item(),
                                "mean_abs_diff": delta.mean().item(),
                                "different_values": (logits != ref).sum().item(),
                            }
                        )
                        old_precision = (
                            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                        )
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                        try:
                            strict_slice = original(hidden, head, bias)
                            processor._gr_vocab_slice = None
                            strict_full = original(hidden, head, bias)
                        finally:
                            processor._gr_vocab_slice = window
                            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                                old_precision
                            )
                        strict_ref = strict_full[:, window[0] : window[0] + window[1]]
                        self.review_shadow[-1].update(
                            strict_pair_exact=torch.equal(strict_slice, strict_ref),
                            strict_slice_vs_default_full_exact=torch.equal(strict_slice, ref),
                            strict_full_vs_default_full_exact=torch.equal(strict_full, full),
                            strict_max_abs_diff=(strict_slice.float() - strict_ref.float())
                            .abs()
                            .max()
                            .item(),
                        )
                return logits

            processor._get_logits = projection

    def review_timed(
        self, label: str, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        if not self.review_active:
            return function(*args, **kwargs)
        start, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
        start.record()
        result = function(*args, **kwargs)
        end.record()
        self.review_events.append((label, start, end, dict(self.review_stage)))
        return result

    def review_control(self, action: str) -> dict[str, Any]:
        torch.accelerator.synchronize()
        if action in ("measure", "diagnostic"):
            self.review_active = action == "diagnostic"
            self.review_events = []
            self.review_replays = 0
            torch.accelerator.reset_peak_memory_stats()
            return {"windows": self.review_windows}
        self.review_active = False
        records = [
            {"label": label, "ms": a.elapsed_time(b), **stage}
            for label, a, b, stage in self.review_events
        ]
        snapshot = torch.cuda.memory_snapshot()
        pools = [s for s in snapshot if tuple(s.get("segment_pool_id", (0, 0))) != (0, 0)]
        return {
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "windows": self.review_windows,
            "records": records,
            "shadow_projection": self.review_shadow,
            "graph_replays": self.review_replays,
            "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
            "peak_reserved_bytes": torch.accelerator.max_memory_reserved(),
            "graph_pool_reserved_bytes": sum(s["total_size"] for s in pools),
            "snapshot_has_pool_id": bool(snapshot and "segment_pool_id" in snapshot[0]),
        }
