# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Platform-agnostic prefill graph components, buffers, and bucket matching."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_gr.config import (
    GRConfig,
)
from vllm_gr.config.models import (
    DEFAULT_BEAM_BATCH_SIZES,
    DEFAULT_PREFILL_TOKEN_BUCKETS,
    deployment_for,
)
from vllm_gr.v1.worker.cudagraph_capture import is_recoverable_capture_error

logger = init_logger(__name__)

# Default capture policy is deliberately narrower than engine capability.
# Static buffer capacity remains independent of the selected capture grid.
PREFILL_GRAPH_CAPTURE_NUM_REQS = DEFAULT_BEAM_BATCH_SIZES
# Small exact buckets, available as an opt-in extension of the configured
# grid (see get_default_buckets). A prefix-cache hit schedules only the
# uncached tail ((((N - 1) mod block_size) + 1) <= 16 tokens at block_size
# 16), so on the step-sized grid every hit lands in the smallest bucket (128)
# and pads 112+ rows: the replay then pays ~4 ms of extra GEMM work (erasing
# the graph's overhead saving at the median) and its M=128 linear layers no
# longer match the eager tail's M=16 bit-for-bit. Tail 16 (and tails 2..15,
# which share cuBLAS kernels with M=16 on the tested stacks) replays at
# bucket 16; tail 1 runs a separate gemv kernel and needs bucket 1 to stay
# bit-identical to eager.
#
# They are NOT part of the default grid: a small bucket only pays off
# together with a widened KV bound (gr.prefill_kv_bound), because
# 2 * bucket alone would bake a *tighter* attention extent than the step grid
# baked for the same request and would drop replays the previous grid
# admitted (128-token and 17-token cache hits). Opting in keeps the default
# grid byte-identical to the pre-extension behaviour.
# See docs/design/feature/cuda_prefill_graph.md for the capture and replay contract.
SMALL_PREFILL_BUCKETS = (1, 16)
DEFAULT_PREFILL_BUCKETS = list(DEFAULT_PREFILL_TOKEN_BUCKETS)
# Public compatibility alias. Both platforms use the configured bucket range.
NPU_DEFAULT_PREFILL_BUCKETS = list(DEFAULT_PREFILL_BUCKETS)

# Backend capability range. This is separate from the capture configuration:
# NPU can rebind request counts on one graph, while GPU captures the exact
# request counts declared by PREFILL_GRAPH_CAPTURE_NUM_REQS above.
SUPPORTED_NUM_REQS = tuple(range(1, 9))


def get_default_buckets(vllm_config: object | None = None) -> list[int]:
    """Build the configured prefill graph buckets for either platform.

    The small exact buckets are added only when the KV-bound override is set:
    they serve cache-hit tails, whose KV length is the whole context, so
    ``2 * bucket`` alone bakes a tighter attention extent than the step grid
    baked for the same request and would drop replays the previous grid
    admitted. Opting in with the bound keeps the default grid unchanged.
    """
    settings = deployment_for(vllm_config)
    buckets = list(settings.prefill_token_buckets or DEFAULT_PREFILL_BUCKETS)
    if settings.prefill_kv_bound is not None:
        buckets = sorted(set(buckets) | {b for b in SMALL_PREFILL_BUCKETS if b <= buckets[-1]})
    return buckets


def get_prefill_graph_capture_num_reqs(vllm_config: object | None = None) -> tuple[int, ...]:
    config = GRConfig.from_vllm_config(vllm_config)
    return config.deployment.beam_batch_sizes


def get_max_context_len(runner: Any) -> int:
    """Maximum supported per-request KV length from the model config.

    Used to size static graph bounds (NPU FIA workspace, GPU flash-attention
    max_seqlen_k) so replays with large cached/chunked-prefill contexts stay
    within the capture-time budgets. 0 when the config is unavailable.
    """
    model_config = getattr(runner, "model_config", None)
    if model_config is None:
        vllm_config = getattr(runner, "vllm_config", None)
        model_config = getattr(vllm_config, "model_config", None)
    max_model_len = getattr(model_config, "max_model_len", 0)
    return int(max_model_len or 0)


def _get_block_table_cols(runner: Any) -> int | None:
    """Block-table column count for the pre-allocated staging buffers.

    Matches vLLM's InputBatch width (max_model_len / block_size, rounded up)
    so the staged block table never reallocates per batch. None when the
    config is unavailable (the capture path then sizes from the metadata).
    """
    max_model_len = get_max_context_len(runner)
    if not max_model_len:
        return None
    kv_cache_config = getattr(runner, "kv_cache_config", None)
    groups = getattr(kv_cache_config, "kv_cache_groups", None) if kv_cache_config else None
    if not groups:
        return None
    try:
        block_size = groups[0].kv_cache_spec.block_size
    except Exception:  # noqa: BLE001
        return None
    if not block_size:
        return None
    return int((max_model_len + block_size - 1) // block_size)


def is_npu_platform() -> bool:
    """Return True when the active vLLM platform is Ascend NPU."""
    from vllm.platforms import current_platform

    return str(current_platform.device_type).lower() == "npu"


def is_prefill_graph_enabled(runner: Any = None) -> bool:
    return deployment_for(getattr(runner, "vllm_config", None)).prefill_graph_enabled


def is_prefill_phase_batch(runner: Any, scheduler_output: Any) -> bool:
    """True when every scheduled request is in the prompt-prefill phase.

    Shared by the GPU and NPU prefill-graph interceptions. Validates EVERY
    scheduled request via the input-batch arrays, so:
      - first chunks, cached prefill, and chunked-prefill continuations are
        admitted (any request with ``num_computed < num_prompt``);
      - ordinary decode requests are excluded (``num_computed == num_prompt``);
      - beam-decode batches are excluded (they use the cascade attention
        path), as are speculative and encoder batches, which the static
        prefill graph does not model.
    """
    beam_data = getattr(scheduler_output, "beam_data", {})
    for mdata in beam_data.values():
        if mdata and mdata.get("is_beam_decode", False):
            return False
    if getattr(scheduler_output, "scheduled_spec_decode_tokens", None):
        return False
    if getattr(scheduler_output, "scheduled_encoder_inputs", None):
        return False
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return False
    computed = getattr(input_batch, "num_computed_tokens_cpu", None)
    prompt = getattr(input_batch, "num_prompt_tokens", None)
    num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
    if computed is None or prompt is None or not num_scheduled_tokens:
        return False
    if isinstance(num_scheduled_tokens, dict):
        # num_scheduled_tokens is keyed by request ID and may cover a subset
        # of the InputBatch rows; resolve each ID through the batch's row
        # mapping instead of assuming a dense [0, num_reqs) prefix. A
        # scheduled request missing from the mapping fails the check.
        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        if req_id_to_index is None:
            return False
        for req_id in num_scheduled_tokens:
            row = req_id_to_index.get(req_id)
            if row is None or computed[row] >= prompt[row]:
                return False
        return True
    # Legacy list form: no request IDs available, so positional rows.
    num_reqs = len(num_scheduled_tokens)
    return all(computed[i] < prompt[i] for i in range(num_reqs))


def scheduled_requests_are_fresh(runner: Any, scheduler_output: Any) -> bool:
    """True when no scheduled request has computed tokens (V1 input batch).

    Graphs captured with dense CUDA Prefill attend over fresh K/V only, so they
    must not replay for a request with context (see ``beam_dense_prefill``).
    """
    input_batch = getattr(runner, "input_batch", None)
    computed = getattr(input_batch, "num_computed_tokens_cpu", None)
    num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
    if computed is None or not num_scheduled_tokens:
        return False
    if isinstance(num_scheduled_tokens, dict):
        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        if req_id_to_index is None:
            return False
        rows = [req_id_to_index.get(req_id) for req_id in num_scheduled_tokens]
    else:
        rows = list(range(len(num_scheduled_tokens)))
    return all(row is not None and int(computed[row]) == 0 for row in rows)


class PrefillBucketBuffers:
    """Pre-allocated static device memory buffers for a specific prefill bucket size K."""

    def __init__(
        self,
        bucket_size: int,
        max_num_seqs: int,
        device: torch.device,
        pad_token_id: int = 0,
        block_table_cols: int | None = None,
    ) -> None:
        self.bucket_size = bucket_size
        self.max_num_seqs = max_num_seqs
        self.device = device
        self.pad_token_id = pad_token_id

        # Static input buffers (shared across all num_reqs for this bucket)
        self.input_ids_buf = torch.full(
            (bucket_size,), pad_token_id, dtype=torch.int32, device=device
        )
        self.positions_buf = torch.zeros((bucket_size,), dtype=torch.int32, device=device)
        self.slot_mapping_buf = torch.full((bucket_size,), -1, dtype=torch.int64, device=device)

        # Static per-request staging buffers, pre-allocated once here (the
        # capture metadata is rebound to these and execute_model_prefill
        # refreshes them before every replay).
        self.seq_lens_buf = torch.zeros((SUPPORTED_NUM_REQS[-1],), dtype=torch.int32, device=device)
        self.query_start_loc_buf = torch.zeros(
            (SUPPORTED_NUM_REQS[-1] + 1,), dtype=torch.int32, device=device
        )
        # Static block-table staging buffer (prefill graph replay). Sized from
        # the model config; None when the config is unavailable (mocks) — the
        # capture path then allocates it from the metadata shape.
        self.block_table_buf: torch.Tensor | None = None
        if block_table_cols:
            self.block_table_buf = torch.zeros(
                (SUPPORTED_NUM_REQS[-1], block_table_cols), dtype=torch.int32, device=device
            )

        # Captured graphs and metadata keyed by num_reqs (1..8)
        self.captured_graphs: dict[int, Any] = {}
        self.output_bufs: dict[int, torch.Tensor] = {}
        self.static_attn_metadatas: dict[int, Any] = {}
        self.max_seq_len_bounds: dict[int, int] = {}

    def has_captured_graph(self, num_reqs: int = 1) -> bool:
        return num_reqs in self.captured_graphs

    @property
    def captured_graph(self) -> Any | None:
        return self.captured_graphs.get(1)

    @captured_graph.setter
    def captured_graph(self, value: Any | None) -> None:
        if value is not None:
            self.captured_graphs[1] = value
        elif 1 in self.captured_graphs:
            del self.captured_graphs[1]

    @property
    def output_buf(self) -> torch.Tensor | None:
        return self.output_bufs.get(1)

    @output_buf.setter
    def output_buf(self, value: torch.Tensor | None) -> None:
        if value is not None:
            self.output_bufs[1] = value
        elif 1 in self.output_bufs:
            del self.output_bufs[1]

    @property
    def static_attn_metadata(self) -> Any | None:
        return self.static_attn_metadatas.get(1)

    @static_attn_metadata.setter
    def static_attn_metadata(self, value: Any | None) -> None:
        if value is not None:
            self.static_attn_metadatas[1] = value
        elif 1 in self.static_attn_metadatas:
            del self.static_attn_metadatas[1]


class BasePrefillGraphRunner:
    """Base class for bucket matching and buffer management across hardware backends."""

    supports_multi_request_total_buckets = False

    def __init__(
        self,
        model_runner: Any,
        buckets: list[int] | None = None,
    ) -> None:
        self.model_runner = model_runner
        self.gr_settings = GRConfig.from_vllm_config(
            getattr(model_runner, "vllm_config", None)
        ).deployment
        self.device = getattr(
            model_runner, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.buckets = sorted(
            buckets or get_default_buckets(getattr(model_runner, "vllm_config", None))
        )

        # Resolve and clamp bucket sizes before planning shapes and allocating buffers.
        scheduler_config = getattr(model_runner, "scheduler_config", None)
        if scheduler_config is None:
            vllm_config = getattr(model_runner, "vllm_config", None)
            scheduler_config = getattr(vllm_config, "scheduler_config", None)
        max_num_batched_tokens = getattr(scheduler_config, "max_num_batched_tokens", None)
        max_model_len = get_max_context_len(model_runner)
        # Only GPU V1 prepares per-request positions for total-token buckets
        # larger than one model context. Other runners keep the model limit.
        supports_total_buckets = (
            self.supports_multi_request_total_buckets
            and GRConfig.from_vllm_config(getattr(model_runner, "vllm_config", None)).beam_max_width
            is not None
            and self.gr_settings.prefill_token_buckets is not None
        )
        limits = (
            (max_num_batched_tokens,)
            if supports_total_buckets
            else (max_num_batched_tokens, max_model_len)
        )
        bucket_limits = [int(limit) for limit in limits if limit]
        if bucket_limits:
            effective_max_bucket = min(bucket_limits)
            clamped = sorted({min(b, effective_max_bucket) for b in self.buckets})
            if clamped != self.buckets:
                logger.info(
                    "Prefill Graph buckets clamped to effective maximum %d: %s -> %s",
                    effective_max_bucket,
                    self.buckets,
                    clamped,
                )
                self.buckets = clamped

        max_num_seqs_cfg = getattr(
            getattr(model_runner, "vllm_config", None),
            "scheduler_config",
            None,
        )
        max_seqs = getattr(max_num_seqs_cfg, "max_num_seqs", 256) if max_num_seqs_cfg else 256

        if self.supports_multi_request_total_buckets:
            self._plan_capture_buckets(
                max_num_seqs=int(getattr(scheduler_config, "max_num_seqs", max_seqs)),
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                total_token_buckets=supports_total_buckets,
            )

        # Block-table column count from the model config (used to pre-size the
        # per-bucket staging buffers); None when the config is unavailable.
        block_table_cols = _get_block_table_cols(model_runner)

        # Eager static-buffer allocation. Only capture-unsupported failures
        # may disable this optional runner; device or model failures abort startup.
        try:
            self.buffers: dict[int, PrefillBucketBuffers] = {
                b: PrefillBucketBuffers(b, max_seqs, self.device, block_table_cols=block_table_cols)
                for b in self.buckets
            }
        except Exception as e:  # noqa: BLE001
            if not isinstance(e, RuntimeError) or not is_recoverable_capture_error(e):
                raise
            logger.warning("Prefill Graph runner disabled: static buffer allocation failed: %r", e)
            self.buckets = []
            self.buffers = {}

        # O(1) bucket-match lookup table (total_tokens is bounded by the
        # largest bucket); built once with the same policy used at runtime.
        self._bucket_lut: list[int | None] = []
        if self.buckets:
            self._bucket_lut = [None] * (self.buckets[-1] + 1)
            for t in range(1, self.buckets[-1] + 1):
                self._bucket_lut[t] = self._compute_match_bucket(t)

    def _plan_capture_buckets(
        self,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int | None,
        max_model_len: int,
        total_token_buckets: bool,
    ) -> None:
        """Let CUDA V1 filter feasible capture shapes before allocation.

        The NPU runner uses the shared bucket limits and plans its own captures.
        """

    def _compute_match_bucket(self, total_tokens: int) -> int | None:
        """Return the smallest captured bucket that can hold the request."""
        for bucket in self.buckets:
            if bucket >= total_tokens:
                return bucket

        logger.debug(
            "Prefill Graph Fallback: Total %d tokens exceeds maximum bucket size %d. Reverting to Eager mode.",
            total_tokens,
            self.buckets[-1],
        )
        return None

    def match_bucket(self, total_tokens: int) -> int | None:
        """O(1) lookup of the smallest captured bucket K >= total_tokens."""
        if total_tokens <= 0 or not self._bucket_lut:
            return None
        if total_tokens >= len(self._bucket_lut):
            logger.debug(
                "Prefill Graph Fallback: Total %d tokens exceeds maximum bucket size %d. Reverting to Eager mode.",
                total_tokens,
                self.buckets[-1],
            )
            return None
        return self._bucket_lut[total_tokens]

    def get_prefill_logit_indices(
        self, num_scheduled_tokens: list[int] | dict[Any, int]
    ) -> list[int]:
        """Calculates end-of-prompt logit indices for prefill requests."""
        token_counts = (
            list(num_scheduled_tokens.values())
            if isinstance(num_scheduled_tokens, dict)
            else list(num_scheduled_tokens)
        )
        indices = []
        curr = 0
        for cnt in token_counts:
            curr += cnt
            indices.append(curr - 1)
        return indices

    def has_captured_graph(self, bucket: int, num_reqs: int = 1) -> bool:
        """Returns True if a Graph has been captured for this bucket and num_reqs."""
        buf = self.buffers.get(bucket)
        return buf is not None and buf.has_captured_graph(num_reqs)

    def capture_graphs(self, runner: Any) -> None:
        """Subclasses implement device-specific graph capture."""
        raise NotImplementedError

    def execute_model_prefill(
        self,
        bucket: int,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
        num_scheduled_tokens: list[int] | dict[Any, int],
    ) -> torch.Tensor | None:
        """Subclasses implement device-specific graph execution."""
        raise NotImplementedError
