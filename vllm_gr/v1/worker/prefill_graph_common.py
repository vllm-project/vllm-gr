# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-agnostic prefill graph components, buffers, and bucket matching."""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

PREFILL_GRAPH_BUCKET_STEP_DEFAULT = 128
PREFILL_GRAPH_MAX_BUCKET_DEFAULT = 8192
DEFAULT_PREFILL_BUCKETS = list(
    range(
        PREFILL_GRAPH_BUCKET_STEP_DEFAULT,
        PREFILL_GRAPH_MAX_BUCKET_DEFAULT + 1,
        PREFILL_GRAPH_BUCKET_STEP_DEFAULT,
    )
)
# Public compatibility alias. Both platforms use the configured bucket range.
NPU_DEFAULT_PREFILL_BUCKETS = list(DEFAULT_PREFILL_BUCKETS)
MAX_PAD_RATIO_DEFAULT = 0.10
MAX_PAD_TOKENS_DEFAULT = 128
SUPPORTED_NUM_REQS = [1, 2, 3, 4, 5, 6, 7, 8]

ENV_ENABLE_PREFILL_GRAPH = "VLLM_ENABLE_PREFILL_CUDAGRAPH"
ENV_PREFILL_GRAPH_BUCKET_STEP = "VLLM_GR_PREFILL_GRAPH_BUCKET_STEP"
ENV_PREFILL_GRAPH_MAX_BUCKET = "VLLM_GR_PREFILL_GRAPH_MAX_BUCKET"


def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _read_env_flag(name: str, default: bool) -> bool:
    """Parse a boolean-style env flag once (truthy/falsy strings, else default)."""
    val = os.getenv(name, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


class PrefillGraphEnv:
    """Process-wide prefill-graph environment switch, read once at import.

    Every process (main / EngineCore / worker) initializes this class at
    plugin load with the environment it was spawned with, and the switch
    never changes afterwards, so per-call ``os.getenv`` reads are wasted work.
    """

    enabled: bool = _read_env_flag(ENV_ENABLE_PREFILL_GRAPH, default=True)


def get_default_buckets() -> list[int]:
    """Build the configured prefill graph buckets for either platform."""
    step = _read_positive_int_env(ENV_PREFILL_GRAPH_BUCKET_STEP, PREFILL_GRAPH_BUCKET_STEP_DEFAULT)
    max_bucket = _read_positive_int_env(
        ENV_PREFILL_GRAPH_MAX_BUCKET, PREFILL_GRAPH_MAX_BUCKET_DEFAULT
    )
    if max_bucket < step:
        raise ValueError(
            f"{ENV_PREFILL_GRAPH_MAX_BUCKET} ({max_bucket}) must be at least "
            f"{ENV_PREFILL_GRAPH_BUCKET_STEP} ({step})"
        )
    if max_bucket % step:
        raise ValueError(
            f"{ENV_PREFILL_GRAPH_MAX_BUCKET} ({max_bucket}) must be divisible by "
            f"{ENV_PREFILL_GRAPH_BUCKET_STEP} ({step})"
        )
    return list(range(step, max_bucket + 1, step))


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


def _get_gpu_dynamic_thresholds(total_tokens: int) -> tuple[float, int]:
    """GPU padding thresholds by token volume.
    Returns (max_pad_ratio, max_pad_tokens).
    A batch matches if pad_ratio <= max_pad_ratio OR pad_tokens <= max_pad_tokens.
    """
    if total_tokens <= 512:
        # Highly dispatch-bound: 30-35% padding is profitable (wins by +35% to +47%)
        return 0.35, 256
    elif total_tokens <= 768:
        # Highly dispatch-bound: 30-35% padding is profitable (wins by +35% to +47%)
        return 0.30, 200
    elif total_tokens <= 1024:
        # Moderately dispatch-bound: profitable up to 10-20% padding (~150 tokens)
        return 0.20, 150
    elif total_tokens <= 2048:
        # Transitioning to compute-bound: keep padding tight (<= 5% or <= 100 tokens)
        return 0.05, 100
    else:
        # Strictly compute-bound (> 2048 tokens): only near-exact matches (<= 2% or <= 64 tokens)
        # to avoid the -5% to -10% padding compute penalty
        return 0.02, 64


def _get_npu_dynamic_thresholds(total_tokens: int) -> tuple[float, int]:
    """NPU padding thresholds by token volume."""
    if total_tokens <= 512:
        return 0.35, 512
    elif total_tokens <= 1024:
        return 0.1, 10
    else:
        return 0.02, 64


def get_dynamic_thresholds(total_tokens: int) -> tuple[float, int]:
    """Returns (max_pad_ratio, max_pad_tokens) for the active platform.

    - Short prompts (<= 512): Highly dispatch-bound. Allow up to 35% padding or 512 tokens.
    - Medium prompts (513..1024): Moderately dispatch-bound. GPU padding is
      progressively tightened; NPU allows 10% padding or 10 tokens.
    - Transition prompts (1025..2048): Transitioning to compute-bound. Allow up to 5% padding or 100 tokens.
    - Long prompts (> 2048): Strictly compute-bound. Allow up to 2% padding or 64 tokens.
    """
    if is_npu_platform():
        return _get_npu_dynamic_thresholds(total_tokens)
    return _get_gpu_dynamic_thresholds(total_tokens)


def is_npu_platform() -> bool:
    """Return True when the active vLLM platform is Ascend NPU."""
    from vllm.platforms import current_platform

    return str(current_platform.device_type).lower() == "npu"


def is_prefill_graph_enabled() -> bool:
    """Return True if prefill CUDA/NPU graph capture is enabled (cached env)."""
    return PrefillGraphEnv.enabled


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

    def __init__(
        self,
        model_runner: Any,
        buckets: list[int] | None = None,
        max_pad_ratio: float | None = None,
        max_pad_tokens: int | None = None,
    ) -> None:
        self.model_runner = model_runner
        self.device = getattr(
            model_runner, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.buckets = sorted(buckets or get_default_buckets())

        # Requests cannot exceed either scheduler/model limit, so larger
        # graphs are unreachable and waste memory. The set() dedupes buckets
        # clamped to the same effective maximum.
        scheduler_config = getattr(model_runner, "scheduler_config", None)
        if scheduler_config is None:
            vllm_config = getattr(model_runner, "vllm_config", None)
            scheduler_config = getattr(vllm_config, "scheduler_config", None)
        max_num_batched_tokens = getattr(scheduler_config, "max_num_batched_tokens", None)
        max_model_len = get_max_context_len(model_runner)
        bucket_limits = [int(limit) for limit in (max_num_batched_tokens, max_model_len) if limit]
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

        self.max_pad_ratio = max_pad_ratio
        self.max_pad_tokens = max_pad_tokens

        self.use_dynamic_thresholds = max_pad_ratio is None and max_pad_tokens is None

        max_num_seqs_cfg = getattr(
            getattr(model_runner, "vllm_config", None),
            "scheduler_config",
            None,
        )
        max_seqs = getattr(max_num_seqs_cfg, "max_num_seqs", 256) if max_num_seqs_cfg else 256

        # Block-table column count from the model config (used to pre-size the
        # per-bucket staging buffers); None when the config is unavailable.
        block_table_cols = _get_block_table_cols(model_runner)

        # Eager static-buffer allocation: a failure here must not abort
        # worker startup. Disable the runner instead — every replay then
        # falls back to eager and the capture loops become no-ops.
        try:
            self.buffers: dict[int, PrefillBucketBuffers] = {
                b: PrefillBucketBuffers(b, max_seqs, self.device, block_table_cols=block_table_cols)
                for b in self.buckets
            }
        except Exception as e:  # noqa: BLE001
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

    def _compute_match_bucket(self, total_tokens: int) -> int | None:
        """Smallest captured bucket K >= total_tokens under the active policy."""
        # Fallback threshold evaluation
        if self.use_dynamic_thresholds and self.max_pad_ratio is None:
            max_pad_ratio, max_pad_tokens = get_dynamic_thresholds(total_tokens)
        else:
            max_pad_ratio = (
                self.max_pad_ratio if self.max_pad_ratio is not None else MAX_PAD_RATIO_DEFAULT
            )
            max_pad_tokens = (
                self.max_pad_tokens if self.max_pad_tokens is not None else MAX_PAD_TOKENS_DEFAULT
            )

        for bucket in self.buckets:
            if bucket >= total_tokens:
                pad_tokens = bucket - total_tokens
                pad_ratio = pad_tokens / float(total_tokens)
                if pad_ratio <= max_pad_ratio:
                    logger.debug(
                        "Prefill Graph Match: Total %d prompt tokens matched bucket %d (padding: %d tokens, ratio: %.2f <= max %.2f).",
                        total_tokens,
                        bucket,
                        pad_tokens,
                        pad_ratio,
                        max_pad_ratio,
                    )
                    return bucket
                if pad_tokens <= max_pad_tokens:
                    logger.debug(
                        "Prefill Graph Match: Total %d prompt tokens matched bucket %d (padding: %d tokens <= max %d).",
                        total_tokens,
                        bucket,
                        pad_tokens,
                        max_pad_tokens,
                    )
                    return bucket
                logger.debug(
                    "Prefill Graph Fallback: Total %d tokens matched bucket %d, but padding ratio %.2f exceeds max %.2f (or pad %d > %d). Reverting to Eager mode.",
                    total_tokens,
                    bucket,
                    pad_ratio,
                    max_pad_ratio,
                    pad_tokens,
                    max_pad_tokens,
                )
                return None

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
