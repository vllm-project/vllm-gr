# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-agnostic prefill graph components, buffers, and bucket matching."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

PREFILL_GRAPH_BUCKET_STEP_DEFAULT = 256
PREFILL_GRAPH_MAX_BUCKET_DEFAULT = 8192
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
# They are NOT part of the default P2D grid: a small bucket only pays off
# together with a widened KV bound (VLLM_GR_PREFILL_GRAPH_KV_BOUND) and the
# dense step-range override, because 2 * bucket alone would bake a *tighter*
# attention extent than the step grid baked for the same request and would
# drop replays the previous grid admitted (128-token and 17-token cache
# hits). See get_default_buckets for the opt-in rule.
SMALL_PREFILL_BUCKETS = (1, 16)


def _init_default_prefill_buckets() -> list[int]:
    """Generate default prefill buckets with stride 128 between 1024 and 3072."""
    return (
        [128, 256, 512, 768]
        + list(range(1024, 3073, 128))
        + [3328, 3584, 3840, 4096, 5120, 6144, 7168, 8192]
    )


DEFAULT_PREFILL_BUCKETS = _init_default_prefill_buckets()
# Kept as a public compatibility alias; both platforms now use the configured
# bucket step and maximum.
NPU_DEFAULT_PREFILL_BUCKETS = list(DEFAULT_PREFILL_BUCKETS)
# Unified monolithic P2D buckets shared by GPU and NPU (stride 128/256
# matching common recommender prompt distributions). The runner's bucket
# list feeds the standard prefill-graph capture, match_p2d_bucket, and the
# max-bucket capacity, so this is a deliberate cross-platform tradeoff:
# standard prefill graphs are captured for these sizes only (VLLM_GR_P2D_BUCKETS
# restores a wider list when needed), while tier-2 P2D stays length-agnostic
# through is_p2d_eligible's max_context override.
DEFAULT_P2D_BUCKETS = [128, 256, 512, 768, 1024]
MONOLITHIC_SUPPORTED_BUCKETS: tuple[int, ...] = (128, 256, 512, 768, 1024)
MONOLITHIC_MAX_PAD_RATIO: float = 0.10
DEFAULT_P2D_SUPPORTED_NUM_REQS = [1, 2]


def get_p2d_buckets() -> list[int]:
    """Returns the bucket list for P2D monolithic capture (default: [128, 256, 512, 768, 1024]).

    Can be overridden via VLLM_GR_P2D_BUCKETS environment variable (comma-separated ints).
    """
    env_buckets = os.getenv("VLLM_GR_P2D_BUCKETS")
    if env_buckets:
        try:
            parsed = [int(x.strip()) for x in env_buckets.split(",") if x.strip()]
            if parsed:
                return sorted(parsed)
        except ValueError as e:
            logger.warning(
                "Invalid VLLM_GR_P2D_BUCKETS=%r: %s; falling back to default.",
                env_buckets,
                e,
            )
    return list(DEFAULT_P2D_BUCKETS)


def get_p2d_supported_num_reqs() -> list[int]:
    """Returns supported batch sizes for P2D (default: [1, 2]).

    Can be overridden via VLLM_GR_P2D_BATCH_SIZES environment variable (comma-separated ints).
    """
    env_bs = os.getenv("VLLM_GR_P2D_BATCH_SIZES")
    if env_bs:
        try:
            parsed = [int(x.strip()) for x in env_bs.split(",") if x.strip()]
            if parsed:
                return sorted(parsed)
        except ValueError as e:
            logger.warning(
                "Invalid VLLM_GR_P2D_BATCH_SIZES=%r: %s; falling back to default.",
                env_bs,
                e,
            )
    return list(DEFAULT_P2D_SUPPORTED_NUM_REQS)


P2D_SUPPORTED_NUM_REQS = get_p2d_supported_num_reqs()
MAX_PAD_RATIO_DEFAULT = 0.10
MAX_PAD_TOKENS_DEFAULT = 128
SUPPORTED_NUM_REQS = [1, 2, 3, 4, 5, 6, 7, 8]
# Backward-compatible default beam-width set (monolithic capture). Prefer
# get_p2d_beam_widths() which honors VLLM_GR_P2D_BEAM_WIDTHS.
SUPPORTED_BEAM_WIDTHS = [32, 128]
# Upper bound for the tier-3 eager pool capacity (W_pool): wider requests
# fall back to the engine-driven beam loop even if --max-logprobs is larger.
DEFAULT_DECODE_STEPS = 2
P2D_SENTINEL_RANK = -999
# trace_headers key carrying the outer beam-search request's total generation
# step count (max_tokens minus reserved begin/end tokens). The frontend stamps
# it on every request; the worker refuses P2D when it is present and differs
# from the fixed 1 + DEFAULT_DECODE_STEPS tokens the P2D graph produces.
P2D_TOTAL_STEPS_HEADER = "vllm_gr_total_steps"


def _get_server_beam_max_width() -> int | None:
    """Reads --beam-max-width from CLI args (e.g. from vllm-gr serve)."""
    for idx, arg in enumerate(sys.argv):
        if arg == "--beam-max-width" and idx + 1 < len(sys.argv):
            try:
                return int(sys.argv[idx + 1])
            except ValueError:
                pass
        elif arg.startswith("--beam-max-width="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                pass
    return None


def get_p2d_pool_max_beams_cap(runner: Any = None) -> int:
    """Resolves the P2D pool max beams cap.

    Returns the server config of --beam-max-width if configured on the runner or CLI,
    otherwise defaults to 256.
    """
    if runner is not None:
        try:
            from vllm_gr.v1.worker.model_runner_common import _get_gr_config

            gr_cfg = _get_gr_config(runner)
            if getattr(gr_cfg, "beam_max_width", None) is not None:
                return int(gr_cfg.beam_max_width)
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not resolve beam_max_width from gr_config: %r", e)
        add_cfg = getattr(getattr(runner, "vllm_config", None), "additional_config", None)
        if isinstance(add_cfg, dict) and add_cfg.get("vllm_gr_beam_max_width") is not None:
            try:
                return int(add_cfg["vllm_gr_beam_max_width"])
            except (ValueError, TypeError):
                pass
    server_width = _get_server_beam_max_width()
    if server_width is not None:
        return server_width
    return 256


P2D_POOL_MAX_BEAMS_CAP = _get_server_beam_max_width() or 256


def _parse_env_int_list(name: str, default: list[int]) -> list[int]:
    """Parses a comma-separated int env list; falls back to the default."""
    env_val = os.getenv(name)
    if env_val:
        try:
            parsed = [int(x.strip()) for x in env_val.split(",") if x.strip()]
            if parsed:
                return sorted(parsed)
        except ValueError as e:
            logger.warning("Invalid %s=%r: %s; falling back to default.", name, env_val, e)
    return list(default)


def get_p2d_beam_widths() -> list[int]:
    """Beam widths captured as monolithic P2D graphs (default: [32, 128]).

    Override with VLLM_GR_P2D_BEAM_WIDTHS (comma-separated ints). Evaluated
    per call (same convention as get_p2d_supported_num_reqs).
    """
    return _parse_env_int_list("VLLM_GR_P2D_BEAM_WIDTHS", SUPPORTED_BEAM_WIDTHS)


def get_p2d_step_beam_widths_config() -> tuple[list[int], list[int], list[int]]:
    """Per-step capture sets: (P_BW, D0_BW, D1_BW).

    P_BW gates the Decode-1 graph input widths (W0), D0_BW gates the Decode-2
    graph input widths (W1). D1_BW (W2) is parsed for symmetry but shapes no
    graph — the step-2 pruning is eager and only bounded by the pool capacity.
    Override with VLLM_GR_P_BW / VLLM_GR_D0_BW / VLLM_GR_D1_BW.
    """
    p_bw = _parse_env_int_list("VLLM_GR_P_BW", SUPPORTED_BEAM_WIDTHS)
    d0_bw = _parse_env_int_list("VLLM_GR_D0_BW", SUPPORTED_BEAM_WIDTHS)
    d1_bw = _parse_env_int_list("VLLM_GR_D1_BW", SUPPORTED_BEAM_WIDTHS)
    return p_bw, d0_bw, d1_bw


def get_p2d_max_beams(
    max_logprobs: int = 0,
    runner: Any = None,
    beam_max_width: int | None = None,
) -> int:
    """Physical beam capacity of the static P2D pool (W_pool).

    Sized from the capture sets, bounded by the server config of --beam-max-width
    (or 256 if not specified), so uncaptured widths can run through the on-device eager
    tier (Tier 3) without capturing graphs for them.
    """
    p_bw, d0_bw, d1_bw = get_p2d_step_beam_widths_config()
    cap = max(max(p_bw), max(d0_bw), max(d1_bw), max(get_p2d_beam_widths()))
    pool_cap = beam_max_width if beam_max_width is not None else get_p2d_pool_max_beams_cap(runner)
    if max_logprobs and max_logprobs > 0:
        cap = max(cap, min(int(max_logprobs), pool_cap))
    else:
        cap = max(cap, pool_cap)
    return cap


class StepBeamWidths(NamedTuple):
    """Per-step beam widths for the P2D chain: W0 (after prefill), W1 (after
    decode 1), W2 (after decode 2)."""

    w0: int
    w1: int
    w2: int

    @property
    def max_w(self) -> int:
        return max(self.w0, self.w1, self.w2)

    @property
    def is_uniform(self) -> bool:
        return self.w0 == self.w1 == self.w2


class SIDSlices(NamedTuple):
    """(offset, size) vocabulary slice bounds for each hierarchical SID level."""

    sid_a: tuple[int, int]  # Stage P0 (Prefill)
    sid_b: tuple[int, int]  # Stage D0 (Decode 1)
    sid_c: tuple[int, int]  # Stage D1 (Decode 2)


def compute_sid_slices_from_tokenizer(tokenizer: Any) -> SIDSlices | None:
    """Discovers SID token ranges directly from the tokenizer with NO prior knowledge

    of codebook size or start offsets. Returns None if model does not use SID tokens.
    """
    if tokenizer is None:
        return None
    vocab: dict[str, int] = {}
    if hasattr(tokenizer, "get_vocab"):
        try:
            v = tokenizer.get_vocab()
            if isinstance(v, dict):
                vocab.update(v)
        except Exception as e:  # noqa: BLE001
            logger.debug("P2D: could not extract vocab from get_vocab: %r", e)
    if hasattr(tokenizer, "get_added_vocab"):
        try:
            v = tokenizer.get_added_vocab()
            if isinstance(v, dict):
                vocab.update(v)
        except Exception as e:  # noqa: BLE001
            logger.debug("P2D: could not extract vocab from get_added_vocab: %r", e)
    if not vocab:
        return None

    sid_ids: dict[str, list[int]] = {"a": [], "b": [], "c": []}
    pattern = re.compile(r"^<s_([abc])_\d+>$")

    for token, tid in vocab.items():
        if isinstance(token, str):
            match = pattern.match(token)
            if match:
                try:
                    sid_ids[match.group(1)].append(int(tid))
                except (ValueError, TypeError):
                    continue

    # If the model does not have hierarchical SID tokens at all 3 levels, return None
    if not all(sid_ids[lvl] for lvl in ("a", "b", "c")):
        return None

    res = {}
    for lvl in ("a", "b", "c"):
        ids = sid_ids[lvl]
        min_id, max_id = min(ids), max(ids)
        count = len(ids)
        expected_count = max_id - min_id + 1
        if count != expected_count:
            logger.warning(
                "P2D: Discovered SID_%s token IDs are not contiguous (count=%d, range=%d). Slicing disabled.",
                lvl.upper(),
                count,
                expected_count,
            )
            return None
        res[f"sid_{lvl}"] = (min_id, expected_count)

    return SIDSlices(sid_a=res["sid_a"], sid_b=res["sid_b"], sid_c=res["sid_c"])


def resolve_sid_slices_from_runner(model_runner: Any) -> SIDSlices | None:
    """Extracts the tokenizer from runner.model_config and computes SID slices.

    Safely falls back to None if model_config or tokenizer is unavailable.
    """
    try:
        model_config = getattr(model_runner, "model_config", None)
        if model_config is None:
            return None
        tokenizer_name = getattr(model_config, "tokenizer", None)
        if not tokenizer_name:
            return None
        from vllm.tokenizers import get_tokenizer

        trust_remote_code = getattr(model_config, "trust_remote_code", False)
        tokenizer = get_tokenizer(tokenizer_name, trust_remote_code=trust_remote_code)
        slices = compute_sid_slices_from_tokenizer(tokenizer)
        if slices is not None:
            logger.info(
                "P2D SID vocabulary slices discovered from tokenizer: P0=%s, D0=%s, D1=%s",
                slices.sid_a,
                slices.sid_b,
                slices.sid_c,
            )
        return slices
    except Exception as e:  # noqa: BLE001
        logger.warning("P2D: could not resolve SID slices from tokenizer: %r", e)
        return None


ENV_ENABLE_PREFILL_GRAPH = "VLLM_ENABLE_PREFILL_CUDAGRAPH"
ENV_DISABLE_P2D = "VLLM_GR_DISABLE_P2D"
ENV_PREFILL_GRAPH_BUCKET_STEP = "VLLM_GR_PREFILL_GRAPH_BUCKET_STEP"
ENV_PREFILL_GRAPH_MAX_BUCKET = "VLLM_GR_PREFILL_GRAPH_MAX_BUCKET"
ENV_PREFILL_GRAPH_KV_BOUND = "VLLM_GR_PREFILL_GRAPH_KV_BOUND"


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


class P2DEnv:
    """Process-wide P2D kill switch, read once at import.

    Universal opt-out across GPU and NPU (``VLLM_GR_DISABLE_P2D=1``): an
    unforeseen kernel/driver issue can disable P2D without redeploying code.
    P2D is enabled by default on every platform.
    """

    disabled: bool = _read_env_flag(ENV_DISABLE_P2D, default=False)


def is_p2d_enabled() -> bool:
    """True when P2D is enabled (cached universal kill switch).

    P2D is additionally gated off on NPU: the fast path is disabled on
    Ascend until the NPU implementation is validated. GPU is unaffected.
    """
    return not P2DEnv.disabled and not is_npu_platform()


def get_default_buckets() -> list[int]:
    """Returns the platform-shared default prefill bucket list.

    Both GPU and NPU use the unified P2D bucket list (standard prefill-graph
    capture, match_p2d_bucket, and max-bucket capacity). Override with
    VLLM_GR_P2D_BUCKETS when a deployment needs wider standard-prefill
    coverage.
    The legacy dense-bucket config (VLLM_GR_PREFILL_GRAPH_BUCKET_STEP /
    VLLM_GR_PREFILL_GRAPH_MAX_BUCKET) is honored for backward compatibility:
    when either variable is set, a dense range is built from the two values.

    The small exact buckets (see SMALL_PREFILL_BUCKETS) are opt-in on top of
    that dense range, together with the KV-bound override: they serve
    cache-hit tails, whose KV length is the whole context, so ``2 * bucket``
    alone would bake a tighter attention extent than the step grid baked for
    the same request and drop replays the previous grid admitted. Opting in
    with the bound keeps the default P2D grid unchanged.
    """
    if (
        os.getenv(ENV_PREFILL_GRAPH_BUCKET_STEP) is not None
        or os.getenv(ENV_PREFILL_GRAPH_MAX_BUCKET) is not None
    ):
        step = _read_positive_int_env(
            ENV_PREFILL_GRAPH_BUCKET_STEP, PREFILL_GRAPH_BUCKET_STEP_DEFAULT
        )
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
        buckets = set(range(step, max_bucket + 1, step))
        if get_kv_capture_bound() is not None:
            # Entries beyond a deliberately tiny configured maximum stay out
            # so the override keeps its meaning.
            buckets.update(b for b in SMALL_PREFILL_BUCKETS if b <= max_bucket)
        return sorted(buckets)
    return get_p2d_buckets()


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


def get_kv_capture_bound() -> int | None:
    """Capture-time KV bound override; None keeps the shipped ``2 * bucket``.

    The default bound couples the replay-eligible KV length to the bucket,
    which only holds when a batch schedules roughly its own KV length. A
    prefix-cache hit schedules just the uncached tail, so it matches the
    smallest bucket while its real KV length is the whole context and every
    such replay is rejected by the check in ``execute_model_prefill``.

    Setting this variable enables the shape-identical cache-hit replay path,
    in two coupled ways:

    * it widens the bound to at least this value for every bucket that would
      otherwise bake less -- the bucket still sizes the query extent, while
      this value raises the captured attention extent (``max_seq_len`` ->
      flash-attention ``max_seqlen_k``). Buckets whose default bound is
      already larger keep it, so miss-path graphs stay byte-identical to the
      default. ``max_seq_len`` feeds nothing else, so no other buffer changes
      size with it;
    * it adds the small exact buckets (see ``SMALL_PREFILL_BUCKETS``) to the
      grid, because on their own their ``2 * bucket`` bound would be tighter
      than the step grid's and would drop replays the default grid admitted.

    A value below the context a caller wants to replay still falls back to
    eager -- the replay gate is unchanged; this knob only raises the captured
    extent. Size it to the longest context that must replay (e.g.
    ``max_model_len``, or the workload's N).

    Read once per captured bucket, i.e. once per process at capture time, and
    once per grid build by ``get_default_buckets``.
    """
    raw = os.getenv(ENV_PREFILL_GRAPH_KV_BOUND)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{ENV_PREFILL_GRAPH_KV_BOUND} must be a positive integer, got {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"{ENV_PREFILL_GRAPH_KV_BOUND} must be a positive integer, got {raw!r}")
    return value


ENV_P2D_POOL_KV_BOUND = "VLLM_GR_P2D_KV_BOUND"


def resolve_p2d_pool_kv_bound(max_ctx: int) -> int:
    """KV extent bound for the P2D pool and its 2D/decode-step graphs.

    The bucket-less 2D and decode-step graphs bake their prefix-attention
    extent from the pool's ``max_bucket``; the tier replay gates use the same
    value, so every prompt up to this bound rides the P2D tiers. Defaults to
    the model context (minus a 2-token margin) so long prompts are not
    silently dropped to the engine path; ``VLLM_GR_P2D_KV_BOUND`` overrides
    it with a positive integer (clamped to the context) for deployments that
    want the historical tighter bounds or a custom tradeoff between captured
    extent and graph memory.
    """
    if max_ctx <= 0:
        return PREFILL_GRAPH_MAX_BUCKET_DEFAULT
    bound = max_ctx - 2
    raw = os.getenv(ENV_P2D_POOL_KV_BOUND)
    if raw is None or not raw.strip():
        return bound
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{ENV_P2D_POOL_KV_BOUND} must be a positive integer, got {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"{ENV_P2D_POOL_KV_BOUND} must be a positive integer, got {raw!r}")
    return min(value, bound)


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
    """NPU padding thresholds by token volume (buckets capped at 1024)."""
    if total_tokens <= 512:
        return 0.35, 512
    elif total_tokens <= 1024:
        return 0.1, 10
    else:
        # Unreachable with the NPU bucket cap; kept as the GPU tail.
        return 0.02, 64


def get_dynamic_thresholds(total_tokens: int) -> tuple[float, int]:
    """Returns (max_pad_ratio, max_pad_tokens) for the active platform.

    - Short prompts (<= 512): Highly dispatch-bound. Allow up to 35% padding or 512 tokens.
    - Medium prompts (513..1024): Moderately dispatch-bound. GPU allows up to
      30% padding or 300 tokens; NPU allows 10% padding or 10 tokens.
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


def _is_beam_prefill_sampling_params(sp: Any) -> bool:
    """True when SamplingParams shape marks a GR beam-search prefill.

    Every GR beam flow (online OpenAI serving and offline GRLLM) builds the
    prefill request as ``SamplingParams(logprobs=beam_width, max_tokens=1,
    detokenize=False, flat_logprobs=True)`` — the remaining steps are driven
    through beam sessions. A standard autoregressive request that merely
    asks for token logprobs (e.g. ``max_tokens=100, logprobs=5``) must never
    be promoted into P2D, so ``logprobs`` only counts as a beam width when
    both markers are present.
    """
    return (
        sp is not None
        and int(getattr(sp, "max_tokens", 0) or 0) == 1
        and bool(getattr(sp, "flat_logprobs", False))
    )


def _batch_has_beam_marker(scheduled_new_reqs: Any) -> bool:
    """True when any scheduled request carries an explicit GR beam signal.

    Used to gate the ``input_batch.num_logprobs`` fallback, which is
    populated for ordinary logprobs requests as well as beam requests: it
    may only be consulted when the batch itself is a beam batch.
    """
    for req in scheduled_new_reqs or ():
        trace_headers = getattr(req, "trace_headers", None)
        if isinstance(trace_headers, dict) and trace_headers.get("vllm_gr_step_beam_widths"):
            return True
        sp = getattr(req, "sampling_params", None)
        if sp is None:
            continue
        extra = getattr(sp, "extra_body", None)
        if isinstance(extra, dict) and (
            extra.get("beam_width") is not None
            or extra.get("step_beam_widths") is not None
            or extra.get("beam_widths") is not None
        ):
            return True
        if getattr(sp, "beam_width", None) is not None:
            return True
        if _is_beam_prefill_sampling_params(sp):
            return True
    return False


def match_p2d_beam_width(runner: Any, scheduler_output: Any) -> int | None:
    """Extract requested beam width and find the smallest supported beam width >= request."""
    if not is_p2d_enabled():
        return None
    req_bws: list[int] = []
    scheduled_new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
    if scheduled_new_reqs:
        for req in scheduled_new_reqs:
            sp = getattr(req, "sampling_params", None)
            if sp is not None:
                bw = (
                    (
                        getattr(sp, "extra_body", {}).get("beam_width")
                        if isinstance(getattr(sp, "extra_body", None), dict)
                        else None
                    )
                    or getattr(sp, "beam_width", None)
                    or (
                        getattr(sp, "logprobs", None)
                        if _is_beam_prefill_sampling_params(sp)
                        else None
                    )
                )
                if bw is not None and bw > 0:
                    req_bws.append(int(bw))

    if not req_bws:
        input_batch = getattr(runner, "input_batch", None)
        if input_batch is not None and _batch_has_beam_marker(scheduled_new_reqs):
            num_logprobs = getattr(input_batch, "num_logprobs", {})
            num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
            if isinstance(num_scheduled_tokens, dict):
                for req_id in num_scheduled_tokens:
                    if req_id in num_logprobs and num_logprobs[req_id] > 0:
                        req_bws.append(int(num_logprobs[req_id]))
            elif hasattr(input_batch, "req_ids"):
                for req_id in input_batch.req_ids:
                    if req_id in num_logprobs and num_logprobs[req_id] > 0:
                        req_bws.append(int(num_logprobs[req_id]))

    if not req_bws:
        return None

    max_req_bw = max(req_bws)
    for bw in sorted(SUPPORTED_BEAM_WIDTHS):
        if bw >= max_req_bw:
            return bw
    return None


def match_p2d_step_beam_widths(runner: Any, scheduler_output: Any) -> StepBeamWidths | None:
    """Extracts the per-step beam widths (W0, W1, W2) for the scheduled batch.

    Propagation channels, in order:
      1. ``request.trace_headers["vllm_gr_step_beam_widths"]`` — a comma-joined
         triplet encoded by the serving engine (vLLM 0.22.1 SamplingParams has
         no extra_body, so the widths travel on the request's trace headers);
      2. ``sampling_params.extra_body["step_beam_widths"]``/``["beam_widths"]``
         (future vLLM versions with extra_body);
      3. scalar ``beam_width`` and then ``logprobs``, mapped to a uniform
         triplet. Unlike ``match_p2d_beam_width`` there is NO rounding — the
         exact widths drive the tier dispatch (uncaptured widths go to the
         on-device eager tier).
    """
    extra_triplets: list[StepBeamWidths] = []
    scalar_widths: list[int] = []
    scheduled_new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
    if scheduled_new_reqs:
        for req in scheduled_new_reqs:
            trace_headers = getattr(req, "trace_headers", None)
            if isinstance(trace_headers, dict):
                encoded = trace_headers.get("vllm_gr_step_beam_widths")
                if encoded:
                    try:
                        parts = [int(x) for x in str(encoded).split(",") if x.strip()]
                    except (TypeError, ValueError) as exc:
                        logger.warning(
                            "Invalid vllm_gr_step_beam_widths=%r on request: %r; ignoring.",
                            encoded,
                            exc,
                        )
                        parts = []
                    if len(parts) == 3 and all(w > 0 for w in parts):
                        extra_triplets.append(StepBeamWidths(*parts))
                        continue
            sp = getattr(req, "sampling_params", None)
            if sp is None:
                continue
            extra = getattr(sp, "extra_body", None)
            if isinstance(extra, dict):
                triplet = extra.get("step_beam_widths") or extra.get("beam_widths")
                if triplet is not None:
                    try:
                        w0, w1, w2 = (int(x) for x in triplet)
                    except (TypeError, ValueError) as exc:
                        logger.warning(
                            "Invalid step_beam_widths=%r on request: %r; ignoring.",
                            triplet,
                            exc,
                        )
                    else:
                        if w0 > 0 and w1 > 0 and w2 > 0:
                            extra_triplets.append(StepBeamWidths(w0, w1, w2))
                        continue
            bw = (
                (extra.get("beam_width") if isinstance(extra, dict) else None)
                or getattr(sp, "beam_width", None)
                or (getattr(sp, "logprobs", None) if _is_beam_prefill_sampling_params(sp) else None)
            )
            if bw is not None and bw > 0:
                scalar_widths.append(int(bw))

    if extra_triplets:
        # All scheduled requests must share the same step_beam_widths triplet.
        # Batches with heterogeneous triplets or mixed triplet/scalar requests
        # cannot be executed under a single P2D CUDA graph topology; fall back
        # to the engine path.
        if scalar_widths or any(t != extra_triplets[0] for t in extra_triplets):
            logger.warning(
                "P2D: heterogeneous step_beam_widths in scheduled batch (triplets=%r, scalars=%r); "
                "falling back to engine path.",
                extra_triplets,
                scalar_widths,
            )
            return None
        return extra_triplets[0]

    if not scalar_widths:
        input_batch = getattr(runner, "input_batch", None)
        if input_batch is not None and _batch_has_beam_marker(scheduled_new_reqs):
            num_logprobs = getattr(input_batch, "num_logprobs", {})
            num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
            if isinstance(num_scheduled_tokens, dict):
                for req_id in num_scheduled_tokens:
                    if req_id in num_logprobs and num_logprobs[req_id] > 0:
                        scalar_widths.append(int(num_logprobs[req_id]))
            elif hasattr(input_batch, "req_ids"):
                for req_id in input_batch.req_ids:
                    if req_id in num_logprobs and num_logprobs[req_id] > 0:
                        scalar_widths.append(int(num_logprobs[req_id]))

    if not scalar_widths:
        return None
    w = max(scalar_widths)
    return StepBeamWidths(w, w, w)


def _has_catalog(runner: Any, scheduler_output: Any) -> bool:
    """Check whether an item catalog is configured on the runner, config, or scheduler."""
    if getattr(runner, "catalog", None) is not None or getattr(runner, "catalog_path", None):
        return True
    if getattr(scheduler_output, "catalog", None) is not None or getattr(
        scheduler_output, "catalog_path", None
    ):
        return True
    models = getattr(runner, "models", None)
    if getattr(models, "catalog", None) is not None:
        return True
    vllm_config = getattr(runner, "vllm_config", None)
    if vllm_config is not None:
        additional_config = getattr(vllm_config, "additional_config", None)
        if isinstance(additional_config, dict) and additional_config.get("vllm_gr_catalog_path"):
            return True
        try:
            from vllm_gr.config import GRConfig

            if GRConfig.from_vllm_config(vllm_config).catalog_path is not None:
                return True
        except (AttributeError, ValueError, TypeError) as e:
            logger.debug("Could not inspect GRConfig for catalog_path: %s", e)
    return False


def resolve_constraint_backend(runner: Any) -> Any | None:
    """Return the ready CUDA constraint backend bound to the runner, else None.

    Capability-gated instead of isinstance: the CUDA backend is the only
    runtime exposing ``select_out_unchecked`` (fixed-address out-variant the
    captured graphs require) plus ``_invoke``; NPU and CPU backends lack it
    and stay on the engine-driven path.
    """
    backend = getattr(runner, "constraint_backend", None)
    if backend is None:
        backend = getattr(getattr(runner, "model_runner", None), "constraint_backend", None)
    if backend is None:
        return None
    if (
        getattr(backend, "is_ready", False)
        and callable(getattr(backend, "select_out_unchecked", None))
        and hasattr(backend, "_invoke")
    ):
        return backend
    return None


def is_constraint_backend_ready(runner: Any) -> bool:
    """True when the runner or worker holds an initialized CUDAConstraintBackend."""
    return resolve_constraint_backend(runner) is not None


def _requested_total_steps(scheduler_output: Any) -> int | None:
    """Read the outer request's total step count from its trace_headers.

    Returns ``None`` when the batch carries no stamped step count (legacy or
    non-GR callers), so callers can keep their existing eligibility behavior.
    If scheduled requests specify conflicting step counts, returns -1 to indicate
    ineligibility for P2D.
    """
    scheduled_new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
    if not scheduled_new_reqs:
        return None
    step_counts: set[int] = set()
    for req in scheduled_new_reqs:
        headers = getattr(req, "trace_headers", None)
        if isinstance(headers, Mapping):
            raw = headers.get(P2D_TOTAL_STEPS_HEADER)
            if raw is not None:
                try:
                    step_counts.add(int(raw))
                except (TypeError, ValueError):
                    return None
    if len(step_counts) > 1:
        logger.warning(
            "P2D: heterogeneous requested step counts in scheduled batch %r; "
            "falling back to engine path.",
            step_counts,
        )
        return -1
    return step_counts.pop() if step_counts else None


def _model_uses_beam_attention(runner: Any) -> bool:
    """True when the model's attention layers use the beam attention impl.

    The P2D passes replay the model forward with ``P2DMetadata``; only
    ``BeamAttentionImpl`` dispatches that metadata to the P2D attention
    kernels.  Standard backends (e.g. FLASH_ATTN) raise on the unknown
    metadata, so a non-beam impl disqualifies the batch and the engine falls
    back to the per-step loop.  Runners without an inspectable model
    (unit-test fakes) are allowed through: the gate only downgrades a batch
    whose impl it can positively identify as non-beam.
    """
    model = getattr(runner, "model", None)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        return True
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionImpl

    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        # Most architectures expose the vllm Attention module (and its backend
        # impl) as ``self_attn`` itself; models with a per-architecture
        # attention wrapper (e.g. qwen3) nest it one level deeper at
        # ``self_attn.attn``.
        impl = getattr(self_attn, "impl", None)
        if impl is None:
            impl = getattr(getattr(self_attn, "attn", None), "impl", None)
        if impl is not None and not isinstance(impl, BeamAttentionImpl):
            return False
    return True


def is_p2d_eligible(
    runner: Any,
    scheduler_output: Any,
    num_reqs: int,
    total_tokens: int,
    bucket: int | None = None,
    beam_width: int | None = None,
    step_bws: StepBeamWidths | None = None,
) -> bool:
    """True when scheduled batch qualifies for the P2D fast path.

    When ``step_bws`` is provided, the per-step triplet drives the check (the
    max width must fit the static pool capacity; exact widths are matched to
    capture sets by the tier dispatch). Otherwise the legacy scalar
    ``beam_width`` path applies (rounded up to a supported width).
    """
    if not is_p2d_enabled():
        logger.debug("P2D: batch is not eligible because P2D is disabled (kill switch).")
        return False
    if not is_prefill_graph_enabled():
        return False
    if getattr(scheduler_output, "gr_stage_metadata", None):
        logger.debug(
            "P2D: batch is not eligible because gr_stage_metadata is present "
            "(handled by GPUBeamStageRunner)."
        )
        return False
    if not _model_uses_beam_attention(runner):
        logger.debug(
            "P2D: batch is not eligible because the model attention layers do "
            "not use the beam attention backend (BeamAttentionImpl)."
        )
        return False
    # The P2D graph generates exactly 1 + DEFAULT_DECODE_STEPS tokens per
    # beam. Requests with a different step budget must stay on the normal
    # per-step loop: fewer steps would over-generate, more steps would be
    # silently truncated by the frontend sentinel fast path.
    total_steps = _requested_total_steps(scheduler_output)
    if total_steps is not None and total_steps != 1 + DEFAULT_DECODE_STEPS:
        logger.debug(
            "P2D: batch is not eligible because the request needs %d steps; "
            "P2D generates exactly %d.",
            total_steps,
            1 + DEFAULT_DECODE_STEPS,
        )
        return False
    from vllm_gr.ops.sliced_logits import is_vocab_slice_enabled

    if is_vocab_slice_enabled():
        sid_slices = (
            getattr(runner, "sid_slices", None)
            or getattr(getattr(runner, "model_runner", runner), "sid_slices", None)
            or getattr(
                getattr(runner, "_prefill_graph_runner", None)
                or getattr(runner, "prefill_graph_runner", None),
                "sid_slices",
                None,
            )
        )
        has_sid = sid_slices is not None
        if not (has_sid or is_constraint_backend_ready(runner)):
            logger.debug(
                "P2D: batch is not eligible because sliced-logits is active without SID slices."
            )
            return False
        if sid_slices is not None:
            from vllm_gr.ops.sliced_logits import get_vocab_slice_config

            win_off, win_sz, _ = get_vocab_slice_config()
            if win_off is not None and win_sz is not None:
                sid_start = sid_slices.sid_a[0]
                sid_end = sid_slices.sid_c[0] + sid_slices.sid_c[1]
                if not (win_off <= sid_start and win_off + win_sz >= sid_end):
                    logger.debug(
                        "P2D: batch is not eligible because active vocab slice window [%d, %d) "
                        "does not contain SID range [%d, %d).",
                        win_off,
                        win_off + win_sz,
                        sid_start,
                        sid_end,
                    )
                    return False
    # P2D CUDA Graph execution generates all decode tokens in a single fast path.
    # Constraint-table catalogs are enforced worker-side by the ready CUDA
    # backend (the constrained kernels are baked into the captured graphs);
    # a catalog without a ready worker backend (e.g. legacy Trie) stays gated
    # out and falls back to the engine-driven loop. The captured P2D pool must
    # additionally be bound to that exact backend — a backend injected after
    # startup capture can never replay unconstrained graphs for catalog
    # batches.
    backend = resolve_constraint_backend(runner)
    if _has_catalog(runner, scheduler_output):
        if backend is None:
            logger.debug(
                "P2D: batch is not eligible because catalog is configured without a "
                "ready worker constraint backend."
            )
            return False
        p_runner = getattr(runner, "_prefill_graph_runner", None) or getattr(
            runner, "prefill_graph_runner", None
        )
        pool = getattr(p_runner, "p2d_pool", None)
        if pool is None or getattr(pool, "constraint_backend", None) is not backend:
            logger.debug(
                "P2D: batch is not eligible because the P2D pool is not bound to "
                "the ready constraint backend (graphs captured unconstrained)."
            )
            return False
    if num_reqs not in get_p2d_supported_num_reqs():
        return False
    effective_max_tokens = bucket
    if effective_max_tokens is None:
        p_runner = getattr(runner, "_prefill_graph_runner", None) or getattr(
            runner, "prefill_graph_runner", None
        )
        if p_runner is not None and hasattr(p_runner, "max_bucket"):
            effective_max_tokens = p_runner.max_bucket
        elif p_runner is not None and getattr(p_runner, "buckets", None):
            effective_max_tokens = max(p_runner.buckets)
        else:
            effective_max_tokens = PREFILL_GRAPH_MAX_BUCKET_DEFAULT
        # Load-bearing override: tier-2 (Eager-P + 2D decode graph) is
        # bucket-less, so prompts beyond the monolithic buckets stay eligible
        # up to the model's max context (the GPU pool caps at 8192, the NPU
        # FIA workspace is sized by max_kvlen).
        max_context = get_max_context_len(runner)
        if max_context > 0:
            effective_max_tokens = max(effective_max_tokens, max_context)
    if total_tokens <= 0 or total_tokens > effective_max_tokens:
        return False
    if step_bws is not None:
        # Constraint-backend capacity gate: each constrained launch presents
        # at most max(num_reqs, max_w) rows (step 0 batches all requests) and
        # top_k <= max_w, both bounded by the backend's runtime capacities.
        if backend is not None:
            info = getattr(backend, "info", None)
            max_top_k = getattr(info, "max_top_k", None)
            max_rows = getattr(info, "max_rows", None)
            if isinstance(max_top_k, int) and step_bws.max_w > max_top_k:
                logger.debug(
                    "P2D: batch is not eligible because max step width %d exceeds the "
                    "constraint backend max_top_k %d.",
                    step_bws.max_w,
                    max_top_k,
                )
                return False
            if isinstance(max_rows, int) and max(num_reqs, step_bws.max_w) > max_rows:
                logger.debug(
                    "P2D: batch is not eligible because constrained rows %d exceed the "
                    "constraint backend max_rows %d.",
                    max(num_reqs, step_bws.max_w),
                    max_rows,
                )
                return False
        # Pool-capacity gate: the widest step must fit the static buffers;
        # wider requests fall back to the engine-driven beam loop. The pool
        # capacity derives from the server's --max-logprobs (bounded by
        # P2D_POOL_MAX_BEAMS_CAP), so the gate protects the hard cap.
        model_config = getattr(runner, "model_config", None)
        max_logprobs = int(getattr(model_config, "max_logprobs", 0) or 0)
        pool_cap = get_p2d_max_beams(max_logprobs, runner=runner)
        if step_bws.max_w > pool_cap:
            logger.debug(
                "P2D: batch is not eligible because max step width %d exceeds the "
                "pool capacity %d.",
                step_bws.max_w,
                pool_cap,
            )
            return False
    else:
        if beam_width is None:
            beam_width = match_p2d_beam_width(runner, scheduler_output)
        if beam_width is None or not any(bw >= beam_width for bw in SUPPORTED_BEAM_WIDTHS):
            return False
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return False
    computed = getattr(input_batch, "num_computed_tokens_cpu", None)
    prompt_tokens = getattr(input_batch, "num_prompt_tokens", None)
    if computed is None:
        return False
    num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
    if isinstance(num_scheduled_tokens, dict):
        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        if req_id_to_index is None:
            return False
        for req_id, sched_cnt in num_scheduled_tokens.items():
            row = req_id_to_index.get(req_id)
            if row is None:
                return False
            # Verify prompt will be fully computed after this step
            if prompt_tokens is not None:
                if computed[row] + sched_cnt != prompt_tokens[row]:
                    return False
            elif computed[row] > 0:
                return False
    else:
        for i in range(num_reqs):
            sched_cnt = num_scheduled_tokens[i] if num_scheduled_tokens is not None else 0
            if prompt_tokens is not None:
                if computed[i] + sched_cnt != prompt_tokens[i]:
                    return False
            elif computed[i] > 0:
                return False
    if getattr(scheduler_output, "scheduled_spec_decode_tokens", None):
        return False
    return not bool(getattr(scheduler_output, "scheduled_encoder_inputs", None))


class P2DMetadata:
    """Carries P2D graph execution state for the attention layers."""

    def __init__(
        self,
        p2d_pool: P2DBufferPool,
        p2d_pass: int = 0,
        bucket: int = 0,
        num_reqs: int = 1,
        beam_width: int = 128,
        num_decode_steps: int = DEFAULT_DECODE_STEPS,
        total_prompt_tokens: int = 0,
        cu_seqlens_q_0: torch.Tensor | None = None,
        cu_seqlens_q_prefix: torch.Tensor | None = None,
        cu_seqlens_k_prompt: torch.Tensor | None = None,
        cu_seqlens_q_suffix: torch.Tensor | None = None,
        cu_seqlens_k_suffix_1: torch.Tensor | None = None,
        cu_seqlens_k_suffix_2: torch.Tensor | None = None,
        # NPU-specific fields for the paged-cache decode graph (prefix
        # attention over the KV cache via FIA task groups).  GPU leaves
        # them at their defaults.
        block_table: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        seqused_k: torch.Tensor | None = None,
        num_heads: int = 0,
        num_kv_heads: int = 0,
        block_size: int = 16,
        npu_causal_atten_mask: torch.Tensor | None = None,
        prefix_kvlen_list: list[int] | None = None,
        # NPU monolithic pass-0 host lists (FIA-v2 convention): the
        # per-request chunk qlens (cumsum without the leading zero) and the
        # full-prompt kvlens (computed + scheduled, so cached prefixes are
        # attended through the paged cache).
        pass0_qlen_list: list[int] | None = None,
        pass0_kvlen_list: list[int] | None = None,
    ) -> None:
        self.is_p2d_graph_exec: bool = True
        self.p2d_pool = p2d_pool
        self.p2d_pass = p2d_pass
        self.bucket = bucket
        self.num_reqs = num_reqs
        self.beam_width = beam_width
        self.num_decode_steps = num_decode_steps
        self.total_prompt_tokens = total_prompt_tokens
        self.cu_seqlens_q_0 = cu_seqlens_q_0
        self.cu_seqlens_q_prefix = cu_seqlens_q_prefix
        self.cu_seqlens_k_prompt = cu_seqlens_k_prompt
        self.cu_seqlens_q_suffix = cu_seqlens_q_suffix
        self.cu_seqlens_k_suffix_1 = cu_seqlens_k_suffix_1
        self.cu_seqlens_k_suffix_2 = cu_seqlens_k_suffix_2
        self.block_table = block_table
        self.slot_mapping = slot_mapping
        self.seqused_k = seqused_k
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.npu_causal_atten_mask = npu_causal_atten_mask
        self.prefix_kvlen_list = prefix_kvlen_list
        self.pass0_qlen_list = pass0_qlen_list
        self.pass0_kvlen_list = pass0_kvlen_list


def stage_p2d_block_table(
    staged_buf: torch.Tensor,
    live_block_table: torch.Tensor,
    num_reqs: int,
    beam_width: int | None = None,
) -> torch.Tensor:
    """Stages the live block table into the preallocated buffer.

    If beam_width is provided (for NPU FIA), rows are repeated per beam:
        shape [num_reqs * beam_width, num_cols].
    If beam_width is None (for GPU FlashAttention varlen), rows match num_reqs:
        shape [num_reqs, num_cols].
    """
    num_cols = min(live_block_table.shape[1], staged_buf.shape[1])
    if beam_width is not None and beam_width > 1:
        total_rows = num_reqs * beam_width
        out = staged_buf[:total_rows, :num_cols]
        out.copy_(
            live_block_table[:num_reqs, :num_cols].repeat_interleave(beam_width, dim=0),
            non_blocking=True,
        )
    else:
        out = staged_buf[:num_reqs, :num_cols]
        out.copy_(live_block_table[:num_reqs, :num_cols], non_blocking=True)
    return out


class P2DBufferPool:
    """Dedicated static buffer pool for Prefill + 2 Decodes (P2D) execution.

    Allocated once per worker at startup. Sized to the maximum batch size,
    maximum bucket length, and maximum beam width across all layers.
    """

    def __init__(
        self,
        max_batch: int = 2,
        max_bucket: int = 8192,
        max_beams: int = 128,
        num_decode_steps: int = DEFAULT_DECODE_STEPS,
        num_layers: int = 28,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        vocab_size: int = 176256,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
        alloc_prompt_kv: bool = False,
        max_blocks: int = 2048,
        sid_slices: SIDSlices | None = None,
        slot_mapping_dtype: torch.dtype = torch.int64,
        constraint_backend: Any = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda")
        self.max_batch = max_batch
        self.max_bucket = max_bucket
        self.max_beams = max_beams
        self.num_decode_steps = num_decode_steps
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.device = device
        self.max_blocks = max_blocks
        self.sid_slices = sid_slices

        # Prompt KV: [num_layers, 2 (K/V), max_batch * max_bucket, H_kv, D]
        # NPU decodes read the prefix from the paged KV cache (block-table
        # FIA), so the contiguous prompt pool is skipped there
        # (alloc_prompt_kv=False) — a ~1.75 GB saving per worker.
        # Decodes read the prefix directly from the paged KV cache (block_table),
        # so this contiguous prompt pool is skipped by default (alloc_prompt_kv=False)
        # — saving ~1.88 GB per GPU worker.
        self.prompt_kv: torch.Tensor | None = None
        if alloc_prompt_kv:
            self.prompt_kv = torch.zeros(
                (num_layers, 2, max_batch * max_bucket, num_kv_heads, head_dim),
                dtype=dtype,
                device=device,
            )

        # Block table buffer for paged KV cache prefix attention
        self.block_table = torch.zeros((max_batch, max_blocks), dtype=torch.int32, device=device)
        # Static slot mapping buffer for monolithic Pass 0. NPU's ATB
        # ReshapeAndCache op requires int32 slot mappings, so the dtype is
        # platform-selectable (GPU keeps int64).
        self.slot_mapping_0 = torch.zeros(
            (max_batch * max_bucket,), dtype=slot_mapping_dtype, device=device
        )
        # Exact KV sequence lengths for FlashAttention block_table decode passes
        self.seqused_k = torch.zeros((max_batch,), dtype=torch.int32, device=device)

        # Suffix KV: [num_layers, 2 (K/V), max_batch * max_beams, num_decode_steps, H_kv, D]
        self.suffix_kv = torch.zeros(
            (num_layers, 2, max_batch * max_beams, num_decode_steps, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        # Pass 0 inputs
        self.input_ids_0 = torch.zeros((max_batch * max_bucket,), dtype=torch.int32, device=device)
        self.positions_0 = torch.zeros((max_batch * max_bucket,), dtype=torch.int64, device=device)
        self.cu_seqlens_q_0 = torch.zeros((max_batch + 1,), dtype=torch.int32, device=device)
        self.logits_indices_0 = torch.zeros((max_batch,), dtype=torch.int64, device=device)

        # Pass 1 & 2 inputs (for N * W beams)
        self.input_ids_1 = torch.zeros((max_batch * max_beams,), dtype=torch.int32, device=device)
        self.positions_1 = torch.zeros((max_batch * max_beams,), dtype=torch.int64, device=device)
        self.input_ids_2 = torch.zeros((max_batch * max_beams,), dtype=torch.int32, device=device)
        self.positions_2 = torch.zeros((max_batch * max_beams,), dtype=torch.int64, device=device)

        # Prefix attention seqlens across decode passes
        self.cu_seqlens_q_prefix = torch.zeros((max_batch + 1,), dtype=torch.int32, device=device)
        self.cu_seqlens_k_prompt = torch.zeros((max_batch + 1,), dtype=torch.int32, device=device)

        # Suffix attention seqlens (1 query per beam)
        self.cu_seqlens_q_suffix = torch.arange(
            max_batch * max_beams + 1, dtype=torch.int32, device=device
        )
        # Step 0: 1 token per beam
        self.cu_seqlens_k_suffix_1 = torch.arange(
            max_batch * max_beams + 1, dtype=torch.int32, device=device
        )
        # Step 1: 2 tokens per beam
        self.cu_seqlens_k_suffix_2 = torch.arange(
            0, 2 * max_batch * max_beams + 1, 2, dtype=torch.int32, device=device
        )

        # Logits buffers
        self.logits_0 = torch.zeros((max_batch, vocab_size), dtype=dtype, device=device)
        self.logits_1 = torch.zeros((max_batch * max_beams, vocab_size), dtype=dtype, device=device)
        self.logits_2 = torch.zeros((max_batch * max_beams, vocab_size), dtype=dtype, device=device)

        # Intermediate beam states
        self.tokens_0 = torch.zeros((max_batch, max_beams), dtype=torch.int64, device=device)
        self.cum_lps_0 = torch.zeros((max_batch, max_beams), dtype=torch.float32, device=device)
        self.tokens_1 = torch.zeros((max_batch, max_beams), dtype=torch.int64, device=device)
        self.parents_1 = torch.zeros((max_batch, max_beams), dtype=torch.int64, device=device)
        self.cum_lps_1 = torch.zeros((max_batch, max_beams), dtype=torch.float32, device=device)
        self.tokens_2 = torch.zeros((max_batch, max_beams), dtype=torch.int64, device=device)
        self.parents_2 = torch.zeros((max_batch, max_beams), dtype=torch.int64, device=device)
        self.cum_lps_2 = torch.zeros((max_batch, max_beams), dtype=torch.float32, device=device)

        # Final outputs: shape [max_batch, max_beams, num_decode_steps + 1]
        self.final_tokens = torch.zeros(
            (max_batch, max_beams, num_decode_steps + 1), dtype=torch.int64, device=device
        )
        self.final_cum_lps = torch.zeros((max_batch, max_beams), dtype=torch.float32, device=device)

        # Per-beam local step scores [max_batch, max_beams, num_decode_steps+1]:
        # the true per-token logprobs (log_softmax of each step's candidates)
        # for every surviving beam. The serving engine exposes them instead of
        # dividing the final cumulative score evenly across the tokens.
        self.step_scores = torch.zeros(
            (max_batch, max_beams, num_decode_steps + 1), dtype=torch.float32, device=device
        )

        # Constraint-table execution state. Flat candidate slabs keep every
        # Triton output contiguous: per-call views are taken as
        # ``flat[offset : offset + rows * top_k].view(rows, top_k)``, which the
        # constrained Top-K validation accepts (1D slices are contiguous while
        # 2D column slices of a wider buffer are not). The single-element
        # temperatures tensor broadcasts to any row count inside the kernel.
        self.constraint_backend = constraint_backend
        total_cand_slots = max_batch * max_beams * max_beams
        self.cand_tokens_flat = torch.zeros(total_cand_slots, dtype=torch.int32, device=device)
        self.cand_logprobs_flat = torch.zeros(total_cand_slots, dtype=torch.float32, device=device)
        self.cand_mask_flat = torch.zeros(total_cand_slots, dtype=torch.bool, device=device)
        self.cand_valid_counts = torch.zeros(
            (max_batch * max_beams,), dtype=torch.int32, device=device
        )
        self.temperatures = torch.ones((1,), dtype=torch.float32, device=device)
        self.prefixes_0 = torch.zeros((max_batch, 3), dtype=torch.int32, device=device)
        self.prefixes_1 = torch.zeros((max_batch * max_beams, 3), dtype=torch.int32, device=device)
        self.prefixes_2 = torch.zeros((max_batch * max_beams, 3), dtype=torch.int32, device=device)
        self.valid_beam_counts = torch.zeros((max_batch,), dtype=torch.int32, device=device)


def _pool_constraint_backend(pool: P2DBufferPool) -> Any | None:
    """Ready constraint backend bound to the pool, else None."""
    backend = getattr(pool, "constraint_backend", None)
    if backend is None or not getattr(backend, "is_ready", False):
        return None
    if not callable(getattr(backend, "select_out_unchecked", None)):
        return None
    return backend


def _run_constrained_topk(
    pool: P2DBufferPool,
    backend: Any,
    *,
    logits: torch.Tensor,
    prefixes: torch.Tensor,
    step: int,
    top_k: int,
    rows: int,
    cand_offset: int,
    valid_offset: int,
) -> torch.Tensor:
    """Launch one constrained Top-K into static pool slabs (device-only).

    Writes contiguous ``[rows, top_k]`` views of the flat candidate slabs:
    global token IDs into ``cand_tokens_flat`` and validity-masked logprobs
    (invalid slots are ``-inf``) into ``cand_logprobs_flat``. Returns the
    token view. No allocations and no host syncs.
    """
    from vllm_gr.constraints.backend import ConstraintMetadata

    out_tokens = pool.cand_tokens_flat[cand_offset : cand_offset + rows * top_k].view(rows, top_k)
    out_logprobs = pool.cand_logprobs_flat[cand_offset : cand_offset + rows * top_k].view(
        rows, top_k
    )
    out_mask = pool.cand_mask_flat[cand_offset : cand_offset + rows * top_k].view(rows, top_k)
    out_counts = pool.cand_valid_counts[valid_offset : valid_offset + rows]
    backend.select_out_unchecked(
        logits,
        ConstraintMetadata(
            prefix_token_ids=prefixes,
            temperatures=pool.temperatures,
            current_step=step,
            top_k=top_k,
        ),
        out_token_ids=out_tokens,
        out_logprobs=out_logprobs,
        out_valid_mask=out_mask,
        out_valid_counts=out_counts,
    )
    out_logprobs.masked_fill_(~out_mask, float("-inf"))
    return out_tokens


def step0_select_topk(
    logits_0: torch.Tensor,
    num_reqs: int,
    beam_width: int,
    pool: P2DBufferPool,
    inv_temperature: torch.Tensor | None = None,
) -> None:
    """Device-side Step 0 Top-W branch selection into static pool buffers.

    Performs log_softmax and topk(k=beam_width) purely on device (GPU or NPU)
    without a CPU roundtrip. Populates pool.tokens_0, pool.cum_lps_0, and
    pool.input_ids_1.
    """
    total_beam_tokens = num_reqs * beam_width
    backend = _pool_constraint_backend(pool)
    if backend is not None:
        # Constrained step 0: one batched launch over all requests (rows =
        # num_reqs <= max_rows). The kernel writes global catalog token IDs
        # and validity-masked logprobs directly; invalid slots receive -inf
        # cumulative logprobs, so later pruning never promotes them.
        cand_tokens = _run_constrained_topk(
            pool,
            backend,
            logits=logits_0[:num_reqs],
            prefixes=pool.prefixes_0[:num_reqs],
            step=0,
            top_k=beam_width,
            rows=num_reqs,
            cand_offset=0,
            valid_offset=0,
        )
        pool.tokens_0[:num_reqs, :beam_width].copy_(cand_tokens, non_blocking=True)
        pool.cum_lps_0[:num_reqs, :beam_width].copy_(
            pool.cand_logprobs_flat[: num_reqs * beam_width].view(num_reqs, beam_width),
            non_blocking=True,
        )
        pool.input_ids_1[:total_beam_tokens].copy_(
            pool.tokens_0[:num_reqs, :beam_width].reshape(-1), non_blocking=True
        )
        return

    sid_slices = getattr(pool, "sid_slices", None)
    sid_a = getattr(sid_slices, "sid_a", None) if sid_slices is not None else None
    use_slice_a = False
    from vllm_gr.ops.sliced_logits import get_vocab_slice_config, is_vocab_slice_enabled

    is_sliced = is_vocab_slice_enabled()
    win_off = 0
    if is_sliced:
        win_off_val, _, _ = get_vocab_slice_config()
        win_off = win_off_val or 0

    offset_a = win_off if is_sliced else 0
    if sid_a is not None and sid_a[1] >= beam_width:
        if is_sliced:
            local_start = sid_a[0] - win_off
            if 0 <= local_start and local_start + sid_a[1] <= logits_0.shape[-1]:
                logits = logits_0[:num_reqs, local_start : local_start + sid_a[1]].float()
                use_slice_a = True
                offset_a = sid_a[0]
        elif logits_0.shape[-1] >= sid_a[0] + sid_a[1]:
            logits = logits_0[:num_reqs, sid_a[0] : sid_a[0] + sid_a[1]].float()
            use_slice_a = True
            offset_a = sid_a[0]

    if not use_slice_a:
        logits = logits_0[:num_reqs].float()

    if inv_temperature is not None:
        logits = logits * inv_temperature[:num_reqs].unsqueeze(-1)

    lps_0 = torch.log_softmax(logits, dim=-1)
    val_0, idx_0 = torch.topk(lps_0, k=beam_width, dim=-1)
    if offset_a:
        idx_0 = idx_0 + offset_a

    pool.tokens_0[:num_reqs, :beam_width].copy_(idx_0, non_blocking=True)
    pool.cum_lps_0[:num_reqs, :beam_width].copy_(val_0, non_blocking=True)
    pool.input_ids_1[:total_beam_tokens].copy_(
        pool.tokens_0[:num_reqs, :beam_width].reshape(-1), non_blocking=True
    )


# Backward-compatible alias (the function moved from the GPU runner).
gpu_step0_select_topk = step0_select_topk


def _slice_logits_for_topk(
    logits: torch.Tensor,
    sid_level: tuple[int, int] | None,
    k: int,
) -> tuple[torch.Tensor, int]:
    """Returns (float logits ready for topk, global-offset add-back).

    Slices the SID level when configured and large enough for ``k``; when the
    slice is smaller than ``k`` the full vocabulary is used instead (exactness
    preserved — never silently capping the candidate count).
    """
    from vllm_gr.ops.sliced_logits import get_vocab_slice_config, is_vocab_slice_enabled

    is_sliced = is_vocab_slice_enabled()
    fallback_offset = 0
    win_off = 0
    if is_sliced:
        win_off_val, _, _ = get_vocab_slice_config()
        win_off = win_off_val or 0
        fallback_offset = win_off

    if sid_level is not None:
        offset, size = sid_level
        if size >= k:
            if is_sliced:
                local_start = offset - win_off
                if 0 <= local_start and local_start + size <= logits.shape[-1]:
                    return logits[:, local_start : local_start + size].float(), offset
            elif logits.shape[-1] >= offset + size:
                return logits[:, offset : offset + size].float(), offset
        else:
            logger.warning(
                "P2D: SID slice size %d < candidate count %d; using the full "
                "vocabulary for this step.",
                size,
                k,
            )
    return logits.float(), fallback_offset


def step1_prune_and_kv_gather(
    p2d_pool: P2DBufferPool,
    num_reqs: int,
    w0: int,
    w1: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """Prunes W0 x K_cand candidates to the top-W1 beams and reorders the
    step-0 suffix KV along the beam dimension from [B, w0] to [B, w1].

    ``K_cand = w1`` (exactness: the global top-w1 is contained in the union of
    per-parent top-w1 continuations). Populates pool.tokens_1, parents_1,
    cum_lps_1, reorders suffix_kv slot 0, and fills input_ids_2.
    """
    sid_slices = getattr(p2d_pool, "sid_slices", None)
    sid_b = getattr(sid_slices, "sid_b", None) if sid_slices is not None else None
    total_in = num_reqs * w0
    backend = _pool_constraint_backend(p2d_pool)
    if backend is not None:
        # Constrained step 1: vectorized prefix fill (col 0 = step-0 root;
        # cols 1/2 stay zero) followed by per-request launches so each call's
        # rows (w0) stays within the backend max_rows capacity. Candidate
        # logprobs are already masked to -inf for invalid slots.
        p2d_pool.prefixes_1[:total_in, 0].copy_(
            p2d_pool.tokens_0[:num_reqs, :w0].reshape(-1), non_blocking=True
        )
        # Explicitly keep the unread columns zero (step 1 reads only col 0).
        p2d_pool.prefixes_1[:total_in, 1:].zero_()
        for r in range(num_reqs):
            _run_constrained_topk(
                p2d_pool,
                backend,
                logits=p2d_pool.logits_1[r * w0 : (r + 1) * w0],
                prefixes=p2d_pool.prefixes_1[r * w0 : (r + 1) * w0],
                step=1,
                top_k=w1,
                rows=w0,
                cand_offset=r * w0 * w1,
                valid_offset=r * w0,
            )
        val_1 = p2d_pool.cand_logprobs_flat[: num_reqs * w0 * w1].view(num_reqs, w0, w1)
        idx_1 = p2d_pool.cand_tokens_flat[: num_reqs * w0 * w1].view(num_reqs, w0, w1)
    else:
        logits_d1, offset_b = _slice_logits_for_topk(p2d_pool.logits_1[:total_in], sid_b, w1)

        lps_1 = torch.log_softmax(logits_d1, dim=-1)
        val_1, idx_1 = torch.topk(lps_1, k=w1, dim=-1)
        if offset_b:
            idx_1 = idx_1 + offset_b

    scores_1 = (
        p2d_pool.cum_lps_0[:num_reqs, :w0].unsqueeze(-1) + val_1.view(num_reqs, w0, w1)
    ).view(num_reqs, w0 * w1)
    best_val_1, best_idx_1 = torch.topk(scores_1, k=w1, dim=-1)
    best_p_1 = best_idx_1 // w1
    best_tok_1 = torch.gather(idx_1.view(num_reqs, w0 * w1), dim=-1, index=best_idx_1)
    p2d_pool.tokens_1[:num_reqs, :w1].copy_(best_tok_1)
    p2d_pool.parents_1[:num_reqs, :w1].copy_(best_p_1)
    p2d_pool.cum_lps_1[:num_reqs, :w1].copy_(best_val_1)

    # Suffix KV reorder: step-0 KV gathered according to best_p_1, from
    # [B, w0] source rows to [B, w1] target rows (explicit bounds — the two
    # widths may differ).
    s_kv = p2d_pool.suffix_kv[:, :, :total_in, 0:1].view(
        num_layers, 2, num_reqs, w0, 1, num_kv_heads, head_dim
    )
    p_idx = best_p_1.view(1, 1, num_reqs, w1, 1, 1, 1).expand(
        num_layers, 2, num_reqs, w1, 1, num_kv_heads, head_dim
    )
    gathered = torch.gather(s_kv, dim=3, index=p_idx)
    target = p2d_pool.suffix_kv[:, :, : num_reqs * w1, 0:1].view(
        num_layers, 2, num_reqs, w1, 1, num_kv_heads, head_dim
    )
    target.copy_(gathered)

    p2d_pool.input_ids_2[: num_reqs * w1].copy_(p2d_pool.tokens_1[:num_reqs, :w1].reshape(-1))


def step2_prune_and_backtrack(
    p2d_pool: P2DBufferPool,
    num_reqs: int,
    w0: int,
    w1: int,
    w2: int,
) -> None:
    """Final Top-W2 selection with 3-token history backtracking.

    ``K_cand = w2``. Writes final_tokens[:B, :w2, :] (step 2 token, step 1
    token, step 0 root) and final_cum_lps[:B, :w2].
    """
    sid_slices = getattr(p2d_pool, "sid_slices", None)
    sid_c = getattr(sid_slices, "sid_c", None) if sid_slices is not None else None
    total_in = num_reqs * w1
    backend = _pool_constraint_backend(p2d_pool)
    if backend is not None:
        # Constrained step 2: fully vectorized prefix construction — col 0 is
        # the step-0 root gathered through parents_1, col 1 the step-1 token;
        # col 2 stays zero and is never read. Per-request launches keep rows
        # (w1) within the backend max_rows capacity.
        t0 = torch.gather(
            p2d_pool.tokens_0[:num_reqs, :w0], dim=-1, index=p2d_pool.parents_1[:num_reqs, :w1]
        )
        p2d_pool.prefixes_2[:total_in, 0].copy_(t0.reshape(-1), non_blocking=True)
        p2d_pool.prefixes_2[:total_in, 1].copy_(
            p2d_pool.tokens_1[:num_reqs, :w1].reshape(-1), non_blocking=True
        )
        # Explicitly keep the unread column zero (step 2 reads cols 0/1 only).
        p2d_pool.prefixes_2[:total_in, 2].zero_()
        for r in range(num_reqs):
            _run_constrained_topk(
                p2d_pool,
                backend,
                logits=p2d_pool.logits_2[r * w1 : (r + 1) * w1],
                prefixes=p2d_pool.prefixes_2[r * w1 : (r + 1) * w1],
                step=2,
                top_k=w2,
                rows=w1,
                cand_offset=r * w1 * w2,
                valid_offset=r * w1,
            )
        val_2 = p2d_pool.cand_logprobs_flat[: num_reqs * w1 * w2].view(num_reqs, w1, w2)
        idx_2 = p2d_pool.cand_tokens_flat[: num_reqs * w1 * w2].view(num_reqs, w1, w2)
    else:
        logits_d2, offset_c = _slice_logits_for_topk(p2d_pool.logits_2[:total_in], sid_c, w2)

        lps_2 = torch.log_softmax(logits_d2, dim=-1)
        val_2, idx_2 = torch.topk(lps_2, k=w2, dim=-1)
        if offset_c:
            idx_2 = idx_2 + offset_c

    scores_2 = (
        p2d_pool.cum_lps_1[:num_reqs, :w1].unsqueeze(-1) + val_2.view(num_reqs, w1, w2)
    ).view(num_reqs, w1 * w2)
    best_val_2, best_idx_2 = torch.topk(scores_2, k=w2, dim=-1)
    best_p_2 = best_idx_2 // w2
    best_tok_2 = torch.gather(idx_2.view(num_reqs, w1 * w2), dim=-1, index=best_idx_2)
    p2d_pool.tokens_2[:num_reqs, :w2].copy_(best_tok_2)
    p2d_pool.parents_2[:num_reqs, :w2].copy_(best_p_2)
    p2d_pool.cum_lps_2[:num_reqs, :w2].copy_(best_val_2)

    p2d_pool.final_tokens[:num_reqs, :w2, 2].copy_(best_tok_2)
    p2d_pool.final_tokens[:num_reqs, :w2, 1].copy_(
        torch.gather(p2d_pool.tokens_1[:num_reqs, :w1], dim=-1, index=best_p_2)
    )
    p1_for_p2 = torch.gather(p2d_pool.parents_1[:num_reqs, :w1], dim=-1, index=best_p_2)
    p2d_pool.final_tokens[:num_reqs, :w2, 0].copy_(
        torch.gather(p2d_pool.tokens_0[:num_reqs, :w0], dim=-1, index=p1_for_p2)
    )
    p2d_pool.final_cum_lps[:num_reqs, :w2].copy_(best_val_2)

    # Reconstruct the true per-token local scores for every surviving beam:
    # the cumulative chain telescopes into the three step logprobs. Dead
    # beams stay -inf (guarded so -inf - -inf can never produce NaN).
    parent_cum_1 = torch.gather(p2d_pool.cum_lps_1[:num_reqs, :w1], dim=-1, index=best_p_2)
    root_cum_0 = torch.gather(p2d_pool.cum_lps_0[:num_reqs, :w0], dim=-1, index=p1_for_p2)
    valid_2 = torch.isfinite(best_val_2)
    p2d_pool.step_scores[:num_reqs, :w2, 2].copy_(
        torch.where(valid_2, best_val_2 - parent_cum_1, float("-inf"))
    )
    p2d_pool.step_scores[:num_reqs, :w2, 1].copy_(
        torch.where(valid_2, parent_cum_1 - root_cum_0, float("-inf"))
    )
    p2d_pool.step_scores[:num_reqs, :w2, 0].copy_(torch.where(valid_2, root_cum_0, float("-inf")))

    # Per-request count of finite (valid) beams: constrained catalogs may
    # yield fewer than w2 legal triples; the bookkeeping path truncates to
    # this count so dead beams never reach the serving engine.
    p2d_pool.valid_beam_counts[:num_reqs].copy_(
        torch.isfinite(best_val_2).sum(dim=-1).to(torch.int32), non_blocking=True
    )


def check_p2d_vocab_size(p2d_pool: Any, logits: torch.Tensor, context_name: str) -> None:
    """Verifies that logits width matches p2d_pool.vocab_size.

    Fails fast with an informative error message if there is a mismatch
    between model.compute_logits output dimension and P2D buffer allocation.
    """
    expected_vocab = getattr(p2d_pool, "vocab_size", None)
    if expected_vocab is None:
        logits_buf = getattr(p2d_pool, "logits_0", None)
        if logits_buf is not None and hasattr(logits_buf, "shape") and len(logits_buf.shape) > 0:
            expected_vocab = logits_buf.shape[-1]
    if expected_vocab is not None and expected_vocab > 0:
        assert logits.shape[-1] == expected_vocab, (
            f"P2D {context_name} error: compute_logits returned width {logits.shape[-1]} "
            f"but p2d_pool.vocab_size is {expected_vocab}. "
            f"Verify vocabulary slicing and TP configuration."
        )


def run_decode_pass_eager(
    runner: Any,
    p2d_pool: P2DBufferPool,
    attn_meta: P2DMetadata,
    pass_idx: int,
    total_beam_tokens: int,
    forward_ctx_fn: Any,
) -> None:
    """Eager single decode pass: model forward + LM head into the pool logits.

    Shared by the on-device eager tier (Tier 3) and as the capture body of the
    decoupled decode-step graphs (Tier 2).
    """
    if pass_idx == 1:
        input_ids = p2d_pool.input_ids_1[:total_beam_tokens]
        positions = p2d_pool.positions_1[:total_beam_tokens]
        logits_buf = p2d_pool.logits_1
    else:
        input_ids = p2d_pool.input_ids_2[:total_beam_tokens]
        positions = p2d_pool.positions_2[:total_beam_tokens]
        logits_buf = p2d_pool.logits_2
    with forward_ctx_fn(runner, attn_meta, total_beam_tokens):
        hidden = runner.model(input_ids=input_ids, positions=positions)
    logits = runner.model.compute_logits(hidden)
    check_p2d_vocab_size(p2d_pool, logits, f"decode pass {pass_idx}")
    logits_buf[:total_beam_tokens].copy_(logits)


def run_p2d_stepwise_tail(
    runner: Any,
    p2d_pool: P2DBufferPool,
    hidden: torch.Tensor,
    sample_indices: list[int],
    attn_meta_1: P2DMetadata,
    attn_meta_2: P2DMetadata,
    num_reqs: int,
    w0: int,
    w1: int,
    w2: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    forward_ctx_fn: Any,
    decode1_fn: Any,
    decode2_fn: Any,
) -> None:
    """Shared P2D tail: step-0 topk, decode 1, prune+KV gather, decode 2,
    backtrack. ``decode1_fn(total_beam_tokens)`` / ``decode2_fn(total_beam_tokens)``
    execute the single decode passes (graph replays for Tier 2, eager forwards
    for Tier 3) — no host round-trips between the stages.
    """
    # Index with the pre-staged pool device tensor: both a Python list and a
    # fresh ``torch.tensor(list, device=...)`` force a stream sync inside the
    # copy (pageable H2D), serializing step 0 and both decode passes behind
    # the full GPU prefill.  Callers fill ``logits_indices_0`` during staging.
    sample_hidden = hidden[p2d_pool.logits_indices_0[:num_reqs]]
    logits_0 = runner.model.compute_logits(sample_hidden)
    check_p2d_vocab_size(p2d_pool, logits_0, "step-0")
    p2d_pool.logits_0[:num_reqs].copy_(logits_0)
    step0_select_topk(
        logits_0=p2d_pool.logits_0,
        num_reqs=num_reqs,
        beam_width=w0,
        pool=p2d_pool,
    )
    decode1_fn(num_reqs * w0)
    step1_prune_and_kv_gather(p2d_pool, num_reqs, w0, w1, num_layers, num_kv_heads, head_dim)
    decode2_fn(num_reqs * w1)
    step2_prune_and_backtrack(p2d_pool, num_reqs, w0, w1, w2)


def run_decode_passes(
    runner: Any,
    p2d_pool: P2DBufferPool,
    attn_meta_1: P2DMetadata,
    attn_meta_2: P2DMetadata,
    total_beam_tokens: int,
    num_reqs: int,
    beam_width: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    forward_ctx_fn: Any,
) -> None:
    """Executes Pass 1 (Decode 1) and Pass 2 (Decode 2) on the active device.

    Shared by the GPU and NPU runners. ``forward_ctx_fn(runner, attn_meta,
    num_tokens)`` returns the platform's forward-context manager for the
    decode passes (GPU: ``set_forward_context``; NPU:
    ``set_ascend_forward_context`` with the per-layer metadata dict).
    The attention dispatch is handled by the platform execution module
    through ``BeamAttentionImpl.p2d_attention``; the beam math is delegated to
    ``step1_prune_and_kv_gather`` / ``step2_prune_and_backtrack`` (uniform
    widths here — identical ops to the per-step tier, bitwise-compatible).
    """

    def _run_decode_pass(pass_idx: int, attn_meta: P2DMetadata) -> None:
        run_decode_pass_eager(
            runner,
            p2d_pool,
            attn_meta,
            pass_idx,
            total_beam_tokens,
            forward_ctx_fn,
        )

    _run_decode_pass(1, attn_meta_1)
    step1_prune_and_kv_gather(
        p2d_pool, num_reqs, beam_width, beam_width, num_layers, num_kv_heads, head_dim
    )
    _run_decode_pass(2, attn_meta_2)
    step2_prune_and_backtrack(p2d_pool, num_reqs, beam_width, beam_width, beam_width)


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

        # Captured graphs and metadata keyed by (num_reqs, beam_width) or num_reqs
        self.captured_graphs: dict[Any, Any] = {}
        self.output_bufs: dict[Any, torch.Tensor] = {}
        self.static_attn_metadatas: dict[Any, Any] = {}
        self.max_seq_len_bounds: dict[int, int] = {}

    def has_captured_graph(self, num_reqs: int = 1, beam_width: int | None = None) -> bool:
        if beam_width is not None:
            return (num_reqs, beam_width) in self.captured_graphs
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

        # Clamp every bucket to the scheduler's max_num_batched_tokens:
        # batches can never exceed it, so larger graphs are wasted memory
        # (and unschedulable token counts). The set() dedupes buckets clamped
        # to the same max.
        scheduler_config = getattr(model_runner, "scheduler_config", None)
        if scheduler_config is None:
            vllm_config = getattr(model_runner, "vllm_config", None)
            scheduler_config = getattr(vllm_config, "scheduler_config", None)
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

        if buckets is not None:
            self._max_bucket = (
                max(self.buckets) if self.buckets else PREFILL_GRAPH_MAX_BUCKET_DEFAULT
            )
        else:
            scheduler_max = max_num_batched_tokens or PREFILL_GRAPH_MAX_BUCKET_DEFAULT
            self._max_bucket = min(PREFILL_GRAPH_MAX_BUCKET_DEFAULT, scheduler_max)

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

    def match_p2d_bucket(self, total_tokens: int) -> int | None:
        """Return the smallest captured bucket >= total_tokens for P2D execution."""
        if not is_p2d_enabled() or total_tokens <= 0:
            return None
        for b in self.buckets:
            if b >= total_tokens:
                return b
        return None

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

    @property
    def max_bucket(self) -> int:
        """Maximum configured prefill bucket size."""
        if hasattr(self, "_max_bucket") and self._max_bucket:
            return self._max_bucket
        return max(self.buckets) if self.buckets else PREFILL_GRAPH_MAX_BUCKET_DEFAULT

    def has_captured_graph(
        self, bucket: int, num_reqs: int = 1, beam_width: int | None = None
    ) -> bool:
        """Returns True if a Graph has been captured for this bucket and num_reqs."""
        buf = self.buffers.get(bucket)
        return buf is not None and buf.has_captured_graph(num_reqs, beam_width)

    def has_2d_graph(self, num_reqs: int = 1, beam_width: int = 128) -> bool:
        """Returns True if a 2D (Decodes 1 & 2) Graph has been captured for this configuration."""
        if not is_p2d_enabled():
            return False
        return False

    def has_monolithic_graph(self, bucket: int, num_reqs: int = 1, beam_width: int = 128) -> bool:
        """Returns True if a monolithic P2D graph has been captured for this configuration."""
        if not is_p2d_enabled():
            return False
        return False

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

    def execute_p2d(
        self,
        bucket: int,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        num_scheduled_tokens: list[int] | dict[Any, int],
        beam_width: int = 128,
        block_table: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Subclasses implement device-specific P2D graph execution."""
        if not is_p2d_enabled():
            return None
        raise NotImplementedError

    def execute_eager_p2d(
        self,
        runner: Any,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        num_scheduled_tokens: list[int] | dict[Any, int],
        beam_width: int = 128,
        block_table: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        computed_tokens: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Subclasses implement device-specific Eager Prefill + 2D Graph execution."""
        if not is_p2d_enabled():
            return None
        raise NotImplementedError
