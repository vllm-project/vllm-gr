# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Sampled CUDA events and per-row beam counts reuse existing D2H fences."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from vllm_gr.v1.metrics.telemetry import metrics_config

_COMPONENTS = (
    "forward",
    "constraint_select",
    "beam_select",
    "lineage_kv",
    "final_pack",
)


class GPUStageMetrics:
    """Two slots mirror the owner's dispatch controls; no extra synchronize()."""

    def __init__(self, owner: Any, capacity: int) -> None:
        self.config = metrics_config()
        self.enabled = self.config.enabled
        self.active_slot: int | None = None
        self.slots: list[dict] = []
        if not self.enabled:
            return
        state = owner.beam.state
        batch = len(owner.bindings)
        for _ in range(capacity):
            self.slots.append(
                {
                    "events": {
                        c: [
                            (torch.Event(enable_timing=True), torch.Event(enable_timing=True))
                            for _ in range(2 * batch + 1)
                        ]
                        for c in _COMPONENTS
                    },
                    "used": {},
                    "running": set(),
                    "sampled": False,
                    "effective_device": torch.empty(batch, dtype=torch.int32, device=state.device),
                    "effective_host": torch.empty(batch, dtype=torch.int32, pin_memory=True),
                }
            )

    def begin(self, slot: int, metadata: Any, owner: Any) -> None:
        if not self.enabled:
            return
        record = self.slots[slot]
        record["sampled"] = metadata.owner_generation % self.config.sample_every == 0
        record["used"] = {}
        record["running"] = set()
        record["width"] = 0
        record["candidate_width"] = None
        self.active_slot = slot if record["sampled"] else None
        if self.active_slot is not None:
            self.start("forward")

    def start(self, name: str) -> None:
        if self.active_slot is None:
            return
        record = self.slots[self.active_slot]
        index = record["used"].get(name, 0)
        record["events"][name][index][0].record()
        record["running"].add(name)

    def stop(self, name: str) -> None:
        if self.active_slot is None:
            return
        record = self.slots[self.active_slot]
        if name not in record["running"]:
            return
        index = record["used"].get(name, 0)
        record["events"][name][index][1].record()
        record["used"][name] = index + 1
        record["running"].remove(name)

    @contextmanager
    def measure(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def snapshot(self, owner: Any, view: Any, *, final: bool = False) -> None:
        if self.active_slot is None:
            return
        record = self.slots[self.active_slot]
        batch = len(owner.bindings)
        if final:
            # Final packing already filters candidates; its header owns the count.
            output = owner.final_output
            width = output.options.num_return_sequences
            for i in range(batch):
                start = 3 + i * output.layout.member_stride
                record["effective_device"][i : i + 1].copy_(output.device[start : start + 1])
        else:
            width = view.output_width
            record["effective_device"].copy_(owner.beam.state.num_active_beams[:batch])
        record["width"] = width

    def enqueue_copy(self, slot: int) -> None:
        if not self.enabled:
            return
        record = self.slots[slot]
        if record["sampled"] and record["width"]:
            # Caller is already on copy_stream after its producer wait_event.
            record["effective_host"].copy_(record["effective_device"], non_blocking=True)

    def note_candidate_width(self, slot: int, width: int) -> None:
        if self.enabled:
            self.slots[slot]["candidate_width"] = width

    def consume(self, slot: int) -> dict | None:
        if not self.enabled:
            return None
        record = self.slots[slot]
        if not record["sampled"]:
            return {"sampled": False, "candidate_width": record["candidate_width"]}
        timings = {
            name: sum(start.elapsed_time(end) for start, end in record["events"][name][:n]) / 1000
            for name, n in record["used"].items()
        }
        return {
            "sampled": True,
            "candidate_width": record["candidate_width"],
            "timings": timings,
            "effective_widths": (
                record["effective_host"].tolist()
                if record["width"]
                else [0] * record["effective_host"].numel()
            ),
        }
