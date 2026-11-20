# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
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

from vllm_gr.config.models import deployment_for
from vllm_gr.v1.forward_context import fresh_prefill_capture
from vllm_gr.v1.worker.cudagraph_capture import is_recoverable_capture_error
from vllm_gr.v1.worker.prefill_graph_common import (
    get_default_buckets,
    get_max_context_len,
    get_prefill_graph_capture_num_reqs,
    is_prefill_graph_enabled,
)

logger = init_logger(__name__)


def get_gpu_prefill_buckets(runner: Any) -> list[int]:
    """Use the GR token buckets, or native capture sizes when the list is unset.

    Native config has already resolved its default/list/maximum precedence.
    Do not union in extra sizes or capacity endpoints: capacity is not a
    capture budget. An empty native list must stay empty. The KV-bound knob
    changes attention coverage, not an explicitly configured native list.
    """
    from vllm_gr.config import GRConfig

    host = getattr(runner, "vllm_config", None)
    settings = GRConfig.from_vllm_config(host).deployment
    if settings.prefill_token_buckets is not None:
        return get_default_buckets(host)
    compilation = getattr(runner, "compilation_config", None)
    if compilation is None:
        compilation = getattr(getattr(runner, "vllm_config", None), "compilation_config", None)
    if compilation is None:
        compilation = getattr(
            getattr(runner, "cudagraph_manager", None), "compilation_config", None
        )
    sizes = getattr(compilation, "cudagraph_capture_sizes", None)
    return list(sizes) if sizes is not None else get_default_buckets(host)


def is_prefill_phase_batch_v2(
    runner: Any, scheduler_output: Any, *, require_fresh: bool = False
) -> bool:
    """True when the scheduled batch is a pure prompt-prefill (V2 surface).

    ``require_fresh`` additionally requires ``num_computed == 0`` for every
    request, as graphs captured with dense CUDA Prefill need.

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
        if require_fresh and int(computed_np[idx]) != 0:
            return False
    return True


@dataclass(frozen=True)
class PrefillCapturePlan:
    buckets: tuple[int, ...]
    full: bool
    piecewise: bool
    batch_sizes: tuple[int, ...] = (1,)
    max_model_len: int | None = None

    @classmethod
    def for_runner(cls, runner: Any) -> PrefillCapturePlan:
        manager = runner.cudagraph_manager
        batch_sizes = tuple(
            b
            for b in get_prefill_graph_capture_num_reqs(runner.vllm_config)
            if b <= runner.scheduler_config.max_num_seqs
        )
        context = get_max_context_len(runner)
        token_limit = runner.scheduler_config.max_num_batched_tokens
        limit = (
            min(
                token_limit,
                max(batch_sizes, default=runner.scheduler_config.max_num_seqs) * context,
            )
            if context
            else token_limit
        )
        buckets = tuple(
            sorted({size for size in get_gpu_prefill_buckets(runner) if 0 < size <= limit})
        )
        mode = manager.cudagraph_mode
        return cls(
            buckets,
            mode.has_full_cudagraphs(),
            mode.has_piecewise_cudagraphs(),
            batch_sizes,
            context or None,
        )

    def descriptors(self) -> dict[CUDAGraphMode, list[BatchExecutionDescriptor]]:
        result = {}
        if self.piecewise and self.buckets:
            result[CUDAGraphMode.PIECEWISE] = [
                BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, size, None)
                for size in reversed(self.buckets)
            ]
        if self.full and self.buckets:
            # vLLM 0.30 distributes dummy tokens evenly over request rows.
            result[CUDAGraphMode.FULL] = [
                BatchExecutionDescriptor(CUDAGraphMode.FULL, size, batch)
                for size in reversed(self.buckets)
                for batch in self.batch_sizes
                if size >= batch
                and (
                    self.max_model_len is None or (size + batch - 1) // batch <= self.max_model_len
                )
            ]
        return result


class GRPrefillCudaGraphManager(ModelCudaGraphManager):
    """Instance-local plan/dispatch adapter; capture mechanics remain native."""

    def __init__(self, runner: Any, plan: PrefillCapturePlan) -> None:
        native = runner.cudagraph_manager
        self.plan = plan
        self.runner = runner
        self._require_fresh = deployment_for(runner.vllm_config).dense_prefill != "off"
        self._prefill_scope: Any = None
        self._capture_finished = False
        self._successful: set[BatchExecutionDescriptor] = set()
        self.full_capture_failures: dict[BatchExecutionDescriptor, str] = {}
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

    def clear(self) -> None:
        """Retire this model's captures before reload or shutdown."""
        self._capture_finished = False
        self._graphs_captured = False
        self._prefill_scope = None
        self.graphs.clear()
        self.hidden_states = None
        self.aux_hidden_states.clear()
        self.intermediate_tensors = None
        self.use_aux_hidden_state_outputs = False
        self._capture_descs.clear()
        self._candidates.clear()
        self._successful.clear()
        self.full_capture_failures.clear()
        for mode in self.dispatch_counts:
            self.dispatch_counts[mode] = 0

    @contextmanager
    def prefill_scope(self, scheduler_output: Any) -> Iterator[None]:
        previous = self._prefill_scope
        self._prefill_scope = scheduler_output
        try:
            yield
        finally:
            self._prefill_scope = previous

    def capture(self, *args: Any, **kwargs: Any) -> None:
        if self._capture_finished:
            return
        if self.graphs or self.hidden_states is not None:
            raise RuntimeError("GR Prefill capture must start with unbound output storage")
        plan = self._capture_descs
        try:
            # The largest PW warmup (or FULL if PW is disabled) allocates
            # hidden_states for the union of both modes before any binding.
            piecewise = plan.get(CUDAGraphMode.PIECEWISE, [])
            if piecewise:
                self._capture_descs = {CUDAGraphMode.PIECEWISE: piecewise}
                with fresh_prefill_capture():
                    super().capture(*args, **kwargs)
                self._successful.update(piecewise)
            full = plan.get(CUDAGraphMode.FULL, [])
            if self._max_full_descs_to_capture is not None:
                # The native profiler limits the whole plan, not each of our
                # single-descriptor calls. Keep the full plan for extrapolation.
                full = full[: self._max_full_descs_to_capture]
            for desc in full:
                self._capture_descs = {CUDAGraphMode.FULL: [desc]}
                try:
                    # These graphs replay only for fresh prompts when dense
                    # Prefill is requested (see beam_dense_prefill).
                    with fresh_prefill_capture():
                        super().capture(*args, **kwargs)
                except RuntimeError as error:
                    if not is_recoverable_capture_error(error):
                        raise
                    # CUDA capture must have unwound cleanly. A failed health
                    # check propagates instead of serving with a broken device.
                    if torch.cuda.is_current_stream_capturing():
                        raise
                    torch.accelerator.synchronize(self.device)
                    self.full_capture_failures[desc] = str(error)
                    logger.warning(
                        "GR Prefill FULL capture unsupported for %d tokens, %d requests; "
                        "using captured PIECEWISE or NONE: %s",
                        desc.num_tokens,
                        desc.num_reqs,
                        str(error),
                    )
                else:
                    self._successful.add(desc)
            self._capture_finished = True
        finally:
            self._capture_descs = plan
            # Native capture sets this after each sub-plan. Publish readiness
            # only when the entire adapter plan has completed successfully.
            self._graphs_captured = self._capture_finished

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int = 0,
        max_query_len: int | None = None,
        num_ubatches: int = 1,
    ) -> BatchExecutionDescriptor:
        result = BatchExecutionDescriptor(
            CUDAGraphMode.NONE,
            num_tokens,
            num_reqs,
            num_active_loras=num_active_loras,
            num_ubatches=num_ubatches,
        )
        in_prefill = self._prefill_scope is not None
        if (
            self._capture_finished
            and 0 < num_reqs <= self.max_num_reqs
            and num_tokens > 0
            and num_active_loras == 0
            and num_ubatches == 1
        ):
            index = bisect_left(self.plan.buckets, num_tokens)
            if index < len(self.plan.buckets):
                bucket = self.plan.buckets[index]
                full = BatchExecutionDescriptor(CUDAGraphMode.FULL, bucket, num_reqs)
                piecewise = BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, bucket, None)
                if (
                    in_prefill
                    and is_prefill_phase_batch_v2(
                        self.runner,
                        self._prefill_scope,
                        require_fresh=self._require_fresh,
                    )
                    and full in self._successful
                    and full in self.graphs
                ):
                    result = full
                elif piecewise in self._successful:
                    result = piecewise
        if in_prefill:
            self.dispatch_counts[result.cg_mode.name] += 1
        return result


def uninstall_prefill_graph_manager_v2(runner: Any) -> None:
    """Drop GR Prefill bindings and restore an uncaptured native manager."""
    manager = getattr(runner, "cudagraph_manager", None)
    if not isinstance(manager, GRPrefillCudaGraphManager):
        return
    manager.clear()
    # Keep native capture usable if the next model does not enable GR Prefill.
    # Otherwise installation will build a fresh plan from the reloaded runner.
    runner.cudagraph_manager = ModelCudaGraphManager(
        manager.vllm_config, manager.device, manager.cudagraph_mode, manager.decode_query_len
    )


def install_prefill_graph_manager_v2(runner: Any) -> None:
    """Replace only an uncaptured, supported GR runner's native manager."""
    native = getattr(runner, "cudagraph_manager", None)
    if isinstance(native, GRPrefillCudaGraphManager):
        return
    if (
        not is_prefill_graph_enabled(runner)
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
