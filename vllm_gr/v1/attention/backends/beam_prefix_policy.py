# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-local prefix plans for the framework segmentation path.

The cost model is measured once per device at engine start-up (see
``beam_prefix_calibrate``) and cached in a YAML file; nothing is benchmarked on
a serving request and no GPU product name is dispatched on.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

# Length buckets are aligned to 1024 tokens (see ``prefix_bucket``).
_LENGTH_BUCKET_STEP = 1024
# Segment counts the framework path may produce. The range stays small because
# every extra segment costs one query copy and one merge pass.
MAX_SEGMENTS = 8
# A plan is only installed when it beats native by this factor.
MIN_SPEEDUP = 1.05
# ... and only when the unsplit serial chain is this many times the reachable
# floor. Below it the shape is compute/byte bound and splitting cannot pay for
# its extra query copies and merges (measured: 10240/B1/W256 predicted a 1.3x
# win but actually lost 12%).
MIN_HEADROOM = 1.5
# ... and only when the unsplit chain is long enough in absolute terms to pay
# for two extra kernel launches per call. Below this the prefix call is launch
# bound, so the relative win is unreal: 1024/B1/W128 predicted 1.5x and measured
# 5% slower.
MIN_CHAIN_US = 150.0


@dataclass(frozen=True)
class PrefixPlan:
    fa_version: int
    num_splits: int = 0
    segments: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.fa_version) is not int
            or self.fa_version not in (2, 3, 4)
            or type(self.num_splits) is not int
            or self.num_splits not in (0, 1, 2, 4, 8)
            or type(self.segments) is not int
            or not 1 <= self.segments <= MAX_SEGMENTS
        ):
            raise ValueError("Invalid prefix plan")
        if self.fa_version == 2 and self.num_splits > 1:
            raise ValueError("FA2 varlen has no native split support")
        if self.segments > 1 and self.num_splits != 1:
            raise ValueError("Framework segmentation must disable native splitting")


def prefix_bucket(length: int, block_size: int, model_bound: int) -> int:
    """Smallest 1024-aligned bound >= length, clamped to model capacity.

    The ladder step (1024) is a multiple of ``block_size * 8``, so every bucket
    is evenly divisible by the chunk size of any supported segment count
    (<= 8). That keeps the last KV segment from being empty, keeps the number of
    captured graphs bounded, and makes the bucket usable as the graph signature
    for the whole (bucket - 1024, bucket] length range.
    """
    aligned_bound = (model_bound + block_size - 1) // block_size * block_size
    step = max(block_size, _LENGTH_BUCKET_STEP)
    if length <= step:
        bucket = (max(length, 1) + block_size - 1) // block_size * block_size
    else:
        bucket = (length + step - 1) // step * step
    return min(aligned_bound, bucket)


@lru_cache(maxsize=16)
def device_fingerprint(device_index: int | None) -> dict[str, Any]:
    import torch
    import vllm

    prop = torch.cuda.get_device_properties(device_index)
    return {
        "name": prop.name,
        "capability": [prop.major, prop.minor],
        "sm_count": prop.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": vllm.__version__,
    }


@lru_cache(maxsize=16)
def available_prefix_versions(device_index: int | None) -> tuple[int, ...]:
    """Check optional adapter imports too, before a plan enters capture.

    An optional backend counts as available only if it imports cleanly: a broken
    CuTe install (for example an ``AttributeError`` for a missing symbol in
    ``cutlass.cute.core`` on some SM90 images) must not abort selection or the
    offline tuner. ImportError is the normal "not built" signal; any other
    exception is logged once and treated the same way.
    """
    import torch
    from vllm.logger import init_logger
    from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported

    logger = init_logger(__name__)
    versions = []
    with torch.accelerator.device_index(device_index):
        for version in (2, 3, 4):
            try:
                if not is_fa_version_supported(version):
                    continue
                if version == 4:
                    from vllm.vllm_flash_attn.cute import interface  # noqa: F401
            except ImportError:
                continue
            except Exception as error:  # noqa: BLE001 - optional dependency probe
                logger.warning(
                    "vllm_gr: FlashAttention %d is unavailable (%s: %s)",
                    version,
                    type(error).__name__,
                    error,
                )
                continue
            versions.append(version)
    return tuple(versions)


# The request path cannot rely on a vLLM config context (it can be absent while
# a request is being executed), so the operator's explicit version choice is
# captured once at initialization and reused here.
_explicit_fa_version: int | None = None
_explicit_fa_probed = False


def remember_explicit_fa_version(version: Any) -> None:
    """Persist an explicit FlashAttention version chosen by the operator."""
    global _explicit_fa_version, _explicit_fa_probed
    _explicit_fa_version = int(version) if version is not None else None
    _explicit_fa_probed = True


def reset_explicit_fa_version() -> None:
    """Test helper: forget both the remembered value and the probe flag."""
    global _explicit_fa_version, _explicit_fa_probed
    _explicit_fa_version = None
    _explicit_fa_probed = False


def explicit_fa_version() -> int | None:
    """Explicit FA version, falling back to a best-effort one-time config read."""
    global _explicit_fa_probed
    if not _explicit_fa_probed:
        _explicit_fa_probed = True
        try:
            from vllm.config import get_current_vllm_config_or_none

            config = get_current_vllm_config_or_none()
        except Exception:  # noqa: BLE001 - config context is optional here
            config = None
        attention = getattr(config, "attention_config", None)
        if attention is not None:
            remember_explicit_fa_version(getattr(attention, "flash_attn_version", None))
    return _explicit_fa_version


def remember_worker_flash_attn_version(worker: Any) -> None:
    """Capture the explicit FA version from the worker at initialization.

    ``worker.vllm_config`` is authoritative and always present in the worker; the
    context-based fallback only covers callers that pass a bare model runner.
    """
    config = getattr(worker, "vllm_config", None)
    if config is None:
        try:
            from vllm.config import get_current_vllm_config_or_none

            config = get_current_vllm_config_or_none()
        except Exception:  # noqa: BLE001
            config = None
    if config is None:
        return
    attention = getattr(config, "attention_config", None)
    if attention is not None:
        remember_explicit_fa_version(getattr(attention, "flash_attn_version", None))


# ---------------------------------------------------------------------------
# Shape, calibration and default-on planning
#
# The default is ON for CUDA/FA2. Selection order is: measured per-device
# calibration -> native. FA3/FA4 keep their own split-K heuristic
# (num_splits=0): it is measured against the kernel it dispatches and its
# per-split overhead is a single fused combine, so this module never overrides it.
# ---------------------------------------------------------------------------

SEGMENT_CANDIDATES = (1, 2, 4, 5, 8)
ENV_PREFIX_POLICY = "VLLM_GR_PREFIX_POLICY"
ENV_CALIBRATION = "VLLM_GR_PREFIX_CALIBRATION"
ENV_CALIBRATION_PATH = "VLLM_GR_PREFIX_CALIBRATION_PATH"


@dataclass(frozen=True, slots=True)
class PrefixShape:
    """One prefix attention geometry, as seen by the runtime."""

    batch: int
    width: int
    length: int
    heads: int
    kv_heads: int
    dim: int
    dtype: str
    block_size: int

    @property
    def query_blocks(self) -> int:
        """FlashAttention tiles queries in 128-row blocks for head_dim <= 128."""
        return (self.width + 127) // 128

    @property
    def ctas(self) -> int:
        """CTAs launched by one unsplit prefix call."""
        return self.batch * self.heads * self.query_blocks


@dataclass(frozen=True, slots=True)
class PrefixCalibration:
    """Device-local coefficients of the prefix cost model.

    ``chain`` shrinks with the segment count, ``comp`` and ``mem`` do not, so
    the last two cap the achievable speed-up; ``m`` prices the per-segment
    launch/merge overhead of the framework path.
    """

    u_ns_per_token: float
    c_comp_us: float
    c_mem_us: float
    m_us: float

    def terms_us(
        self, shape: PrefixShape, segments: int, sm_count: int
    ) -> tuple[float, float, float]:
        """Return the three irreducible terms: (chain, compute floor, byte floor)."""
        segments = max(1, int(segments))
        chunk = -(-shape.length // segments)
        waves = -(-(shape.ctas * segments) // max(1, sm_count))
        chain = waves * chunk * self.u_ns_per_token / 1000.0
        comp = self.c_comp_us * shape.batch * shape.width * shape.length
        mem = self.c_mem_us * shape.batch * shape.length
        return chain, comp, mem

    def estimate_us(self, shape: PrefixShape, segments: int, sm_count: int) -> float:
        chain, comp, mem = self.terms_us(shape, segments, sm_count)
        return max(chain, comp, mem) + self.m_us * max(1, int(segments))

    def headroom(self, shape: PrefixShape, sm_count: int) -> float:
        """How far the unsplit chain sits above the reachable floor.

        Splitting only removes the chain term, so a headroom below the install
        threshold means the shape is already compute/byte bound and segmenting
        would only add query copies and merges.
        """
        chain, comp, mem = self.terms_us(shape, 1, sm_count)
        floor = max(comp, mem)
        return chain / floor if floor > 0 else float("inf")

    def best(self, shape: PrefixShape, sm_count: int) -> tuple[int, float, float]:
        native = self.estimate_us(shape, 1, sm_count)
        best_segments, best = 1, native
        pages = -(-shape.length // max(1, shape.block_size))
        for segments in SEGMENT_CANDIDATES[1:]:
            # Only exact splits: a remainder would leave the last segment empty
            # and charge its launch/merge cost for nothing.
            if pages % segments:
                continue
            estimate = self.estimate_us(shape, segments, sm_count)
            if estimate < best:
                best_segments, best = segments, estimate
        return best_segments, best, native

    def as_mapping(self) -> dict[str, float]:
        return {
            "u_ns_per_token": round(self.u_ns_per_token, 4),
            "c_comp_us": self.c_comp_us,
            "c_mem_us": self.c_mem_us,
            "m_us": round(self.m_us, 4),
        }


_CALIBRATION_HEADER = """\
# vllm-gr Beam prefix-segmentation calibration (auto-generated, schema 1).
#
# Measured once per (device, model geometry) when the engine starts and reused by
# later starts. Delete this file to force a re-measurement; start the engine with
# VLLM_GR_PREFIX_CALIBRATION=off to skip calibration and the default-on path.
#
# Cost model for one prefix attention call (length = graph length bucket):
#   chain(S) = ceil(B * heads * ceil(W/128) * S / sm_count) * (length/S) * u
#   comp     = c_comp * B * W * length   (attention FLOPs, independent of S)
#   mem      = c_mem  * B * length       (KV bytes read,  independent of S)
#   T(S)     = max(chain, comp, mem) + m * S
# and the installed plan minimises T over {1, 2, 4, 5, 8}. Splitting can never
# beat max(comp, mem); those two floors cap the achievable speed-up, which is
# why compute-bound wide-beam shapes stay native.
#
# Units: u = ns per KV token per CTA; c_comp/c_mem = us per the product above;
# m = us per segment (kernel launches + LSE merge).
"""


def calibration_path() -> Path:
    override = os.environ.get(ENV_CALIBRATION_PATH)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "vllm_gr" / "prefix_calibration.yaml"


def _plain(value: Any) -> Any:
    """Plain JSON types only: torch versions are str subclasses."""
    return json.loads(json.dumps(value, default=str))


def _device_key(device: dict[str, Any]) -> str:
    """Readable key, so the file stays inspectable without a hash lookup."""
    return f"{device.get('name')}-{device.get('sm_count')}-{device.get('cuda')}"


# Populated only during worker startup. Requests neither insert entries nor
# consult the filesystem; immutable coefficients live for the worker process.
_runtime_calibrations: dict[tuple[str, str], PrefixCalibration] = {}


def _calibration_key(device: dict[str, Any], geometry: dict[str, Any]) -> tuple[str, str]:
    return (
        json.dumps(device, sort_keys=True, default=str),
        json.dumps(geometry, sort_keys=True, default=str),
    )


def remember_calibration(
    device: dict[str, Any],
    geometry: dict[str, Any],
    calibration: PrefixCalibration | None,
) -> None:
    """Publish a startup result for this device and geometry in the worker."""
    key = _calibration_key(device, geometry)
    if calibration is None:
        _runtime_calibrations.pop(key, None)
    else:
        _runtime_calibrations[key] = calibration


def get_calibration(device: dict[str, Any], geometry: dict[str, Any]) -> PrefixCalibration | None:
    """Look up startup coefficients in memory; a miss keeps native attention."""
    return _runtime_calibrations.get(_calibration_key(device, geometry))


def load_calibration(device: dict[str, Any], geometry: dict[str, Any]) -> PrefixCalibration | None:
    """Read persisted calibration during startup, never on a serving request."""
    import yaml

    try:
        data = yaml.safe_load(calibration_path().read_text(encoding="utf-8"))
        entry = data["entries"][_device_key(device)]
    except Exception:  # noqa: BLE001 - a missing or unreadable file is not an error
        return None
    if not isinstance(entry, dict) or entry.get("device") != _plain(device):
        return None
    if entry.get("geometry") != _plain(geometry):
        return None
    try:
        return PrefixCalibration(
            u_ns_per_token=float(entry["u_ns_per_token"]),
            c_comp_us=float(entry["c_comp_us"]),
            c_mem_us=float(entry["c_mem_us"]),
            m_us=float(entry["m_us"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def store_calibration(
    device: dict[str, Any],
    geometry: dict[str, Any],
    calibration: PrefixCalibration,
) -> None:
    """Write this device's calibration.

    Every worker measures its own GPU and writes the same bytes, so no lock is
    needed; a torn file simply fails to parse next start-up and is re-measured.
    """
    import yaml

    path = calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    try:
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and isinstance(existing.get("entries"), dict):
            entries = dict(existing["entries"])
    except Exception:  # noqa: BLE001 - start from an empty file
        entries = {}
    entries[_device_key(device)] = {
        "device": _plain(device),
        "geometry": _plain(geometry),
        **calibration.as_mapping(),
    }
    body = _CALIBRATION_HEADER + yaml.safe_dump(
        {"schema": 1, "entries": entries}, sort_keys=True, default_flow_style=False
    )
    path.write_text(body, encoding="utf-8")


def policy_mode() -> str:
    """``auto`` (default) or ``off``; anything else keeps the safe ``auto``."""
    raw = os.environ.get(ENV_PREFIX_POLICY)
    if raw is not None and raw.strip().lower() == "off":
        return "off"
    return "auto"


def resolve_prefix_plan(
    shape: PrefixShape,
    *,
    default_version: int,
    fingerprint: dict[str, Any],
    supported_versions: tuple[int, ...],
    sm_count: int,
    explicit_version: int | None = None,
    batch_invariant: bool = False,
) -> PrefixPlan | None:
    """Decide the prefix plan; ``None`` keeps the backend's native behaviour.

    ``None`` is the safe answer: it preserves FA3/FA4 auto splitting, keeps any
    pre-computed scheduler metadata, and costs no extra CUDA graph.
    """
    if batch_invariant or policy_mode() == "off":
        return None
    if explicit_version is None:
        # The request path may run without a vLLM config context; use the value
        # captured at initialization so an explicit FA2/FA3 choice always wins.
        explicit_version = explicit_fa_version()
    if explicit_version is not None and explicit_version >= 3:
        return None
    if default_version >= 3 and explicit_version is None:
        # FA3/FA4: their split-K heuristic already runs on every call and pays a
        # single fused combine, so overriding it is a regression risk.
        return None
    if 2 not in supported_versions:
        return None
    calibration = get_calibration(
        fingerprint,
        {
            "heads": shape.heads,
            "kv_heads": shape.kv_heads,
            "dim": shape.dim,
            "dtype": shape.dtype,
            "block_size": shape.block_size,
        },
    )
    if calibration is None:
        return None
    segments, predicted, native = calibration.best(shape, sm_count)
    if segments < 2 or predicted <= 0 or native / predicted < MIN_SPEEDUP:
        return None
    if calibration.headroom(shape, sm_count) < MIN_HEADROOM:
        # Already compute/byte bound at S=1: the model's own win would not
        # survive the extra query copies and merges.
        return None
    if calibration.terms_us(shape, 1, sm_count)[0] < MIN_CHAIN_US:
        # Short prefix: the call is launch bound, so a relative win does not pay
        # for the extra launches and merges.
        return None
    return PrefixPlan(2, 1, segments)


@lru_cache(maxsize=16)
def multi_processor_count(device_index: int | None) -> int:
    """SM count of the current device, cached for the hot attention path."""
    import torch

    return torch.cuda.get_device_properties(device_index).multi_processor_count
