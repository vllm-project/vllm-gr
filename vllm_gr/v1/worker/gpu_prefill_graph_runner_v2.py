# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GR Prefill policy over the native V2 capture and replay implementation.

One instance owns the FULL/PW plan. PW is captured first at the same sizes as
FULL, so its NONE-mode warmup allocates the shared output at maximum capacity.
Only successfully captured candidates become visible to serving dispatch.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
)

from vllm_gr.v1.worker.cudagraph_capture import is_recoverable_capture_error
from vllm_gr.v1.worker.prefill_graph_common import (
    get_default_buckets,
    get_max_context_len,
    is_prefill_graph_enabled,
)

logger = init_logger(__name__)


def is_prefill_phase_batch_v2(runner: Any, scheduler_output: Any) -> bool:
    """True when the scheduled batch is a pure prompt-prefill (V2 surface).

    Runs at DISPATCH time — before ``prepare_inputs`` builds this step's
    ``InputBatch`` — so it reads the persistent ``req_states`` arrays (kept
    current by the ``add_requests`` / ``update_requests`` calls at the top of
    ``execute_model``) instead of the V1 per-batch InputBatchProxy.

    Beam-decode rows (the cascade attention path), speculative-decoding and
    encoder inputs are excluded; constraint-table prefill rows ARE admitted
    (they carry no ``is_beam_decode`` flag and run the standard prefill
    path the graph models).
    """
    beam_data = getattr(scheduler_output, "beam_data", None) or {}
    for mdata in beam_data.values():
        if mdata and mdata.get("is_beam_decode", False):
            return False
    if getattr(scheduler_output, "scheduled_spec_decode_tokens", None):
        return False
    if getattr(scheduler_output, "scheduled_encoder_inputs", None):
        return False
    num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not num_scheduled:
        return False
    req_states = getattr(runner, "req_states", None)
    if req_states is None:
        return False
    req_id_to_index = getattr(req_states, "req_id_to_index", None)
    computed_np = getattr(req_states, "num_computed_tokens_np", None)
    prompt_np = getattr(getattr(req_states, "prompt_len", None), "np", None)
    if req_id_to_index is None or computed_np is None or prompt_np is None:
        return False
    for req_id in num_scheduled:
        idx = req_id_to_index.get(req_id)
        if idx is None:
            return False
        if int(computed_np[idx]) >= int(prompt_np[idx]):
            return False
    return True


@dataclass(frozen=True)
class PrefillCapturePlan:
    buckets: tuple[int, ...]
    full: bool
    piecewise: bool

    @classmethod
    def for_runner(cls, runner: Any) -> PrefillCapturePlan:
        manager = runner.cudagraph_manager
        limit = min(runner.scheduler_config.max_num_batched_tokens, get_max_context_len(runner))
        sizes = [1, 2, 16, *get_default_buckets()]
        sizes.extend(manager.compilation_config.cudagraph_capture_sizes or ())
        buckets = tuple(sorted({min(int(size), limit) for size in sizes if size > 0}))
        mode = manager.cudagraph_mode
        return cls(buckets, mode.has_full_cudagraphs(), mode.has_piecewise_cudagraphs())

    def descriptors(self) -> dict[CUDAGraphMode, list[BatchExecutionDescriptor]]:
        result = {}
        for mode, enabled, num_reqs in (
            (CUDAGraphMode.PIECEWISE, self.piecewise, None),
            (CUDAGraphMode.FULL, self.full, 1),
        ):
            if enabled and self.buckets:
                result[mode] = [
                    BatchExecutionDescriptor(mode, size, num_reqs)
                    for size in reversed(self.buckets)
                ]
        return result


class GRPrefillCudaGraphManager(ModelCudaGraphManager):
    """Instance-local plan/dispatch adapter; capture mechanics remain native."""

    def __init__(self, runner: Any, plan: PrefillCapturePlan) -> None:
        native = runner.cudagraph_manager
        self.plan = plan
        self.runner = runner
        self._prefill_scope: Any = None
        self._capture_finished = False
        self._successful: set[BatchExecutionDescriptor] = set()
        self.full_capture_failures: dict[int, str] = {}
        self.dispatch_counts = {
            mode.name: 0
            for mode in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE)
        }
        super().__init__(
            native.vllm_config, native.device, native.cudagraph_mode, native.decode_query_len
        )

    def _init_candidates(self) -> None:
        # Do not append the native mixed/decode plan: Beam owns its decode
        # graph, and the shared Prefill plan already includes native sizes.
        self._capture_descs = self.plan.descriptors()

    @contextmanager
    def prefill_scope(self, scheduler_output: Any) -> Iterator[None]:
        previous = self._prefill_scope
        self._prefill_scope = scheduler_output
        try:
            yield
        finally:
            self._prefill_scope = previous

    def capture(self, *args: Any, **kwargs: Any) -> dict:
        if self._capture_finished:
            return {}
        if self.graphs or self.hidden_states is not None:
            raise RuntimeError("GR Prefill capture must start with unbound output storage")
        plan = self._capture_descs
        captured = {}
        try:
            # The largest PW warmup (or FULL if PW is disabled) allocates
            # hidden_states for the union of both modes before any binding.
            piecewise = plan.get(CUDAGraphMode.PIECEWISE, [])
            if piecewise:
                self._capture_descs = {CUDAGraphMode.PIECEWISE: piecewise}
                captured.update(super().capture(*args, **kwargs))
                self._successful.update(piecewise)
            for desc in plan.get(CUDAGraphMode.FULL, []):
                self._capture_descs = {CUDAGraphMode.FULL: [desc]}
                try:
                    captured.update(super().capture(*args, **kwargs))
                except RuntimeError as error:
                    if not is_recoverable_capture_error(error):
                        raise
                    # CUDA capture must have unwound cleanly. A failed health
                    # check propagates instead of serving with a broken device.
                    if torch.cuda.is_current_stream_capturing():
                        raise
                    torch.accelerator.synchronize(self.device)
                    self.full_capture_failures[desc.num_tokens] = str(error)
                    logger.warning(
                        "GR Prefill FULL capture unsupported for %d tokens; "
                        "using captured PIECEWISE or NONE: %s",
                        desc.num_tokens,
                        str(error),
                    )
                else:
                    self._successful.add(desc)
            self._capture_finished = True
            return captured
        finally:
            self._capture_descs = plan
            # Native capture sets this after each sub-plan. Publish readiness
            # only when the entire adapter plan has completed successfully.
            self._graphs_captured = self._capture_finished

    def dispatch(
        self, num_reqs: int, num_tokens: int, uniform_token_count: int | None
    ) -> BatchExecutionDescriptor:
        result = BatchExecutionDescriptor(CUDAGraphMode.NONE, num_tokens, num_reqs)
        in_prefill = self._prefill_scope is not None
        if self._capture_finished and 0 < num_reqs <= self.max_num_reqs and num_tokens > 0:
            index = bisect_left(self.plan.buckets, num_tokens)
            if index < len(self.plan.buckets):
                bucket = self.plan.buckets[index]
                full = BatchExecutionDescriptor(CUDAGraphMode.FULL, bucket, 1)
                piecewise = BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, bucket, None)
                if (
                    in_prefill
                    and num_reqs == 1
                    and is_prefill_phase_batch_v2(self.runner, self._prefill_scope)
                    and full in self._successful
                    and full in self.graphs
                ):
                    result = full
                elif piecewise in self._successful:
                    result = piecewise
        if in_prefill:
            self.dispatch_counts[result.cg_mode.name] += 1
        return result


def install_prefill_graph_manager_v2(runner: Any) -> None:
    """Replace only an uncaptured, supported GR runner's native manager."""
    native = getattr(runner, "cudagraph_manager", None)
    if isinstance(native, GRPrefillCudaGraphManager):
        return
    if (
        not is_prefill_graph_enabled()
        or runner.model_config.enforce_eager
        or native is None
        or not native.cudagraph_mode
    ):
        return
    parallel = runner.parallel_config
    if (
        parallel.tensor_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.use_ubatching
        or runner.lora_config is not None
        or runner.speculator is not None
        or runner.use_aux_hidden_state_outputs
        or runner.supports_mm_inputs
    ):
        logger.info("GR Prefill plan unavailable for this configuration; keeping native V2 graphs")
        return
    if native._graphs_captured or native.graphs or native.hidden_states is not None:
        raise RuntimeError("Cannot replace a V2 graph manager after output storage is bound")
    runner.cudagraph_manager = GRPrefillCudaGraphManager(
        runner, PrefillCapturePlan.for_runner(runner)
    )
