# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in host-timing probes for Beam V1/V2 latency attribution.

Set ``VLLM_GR_STAGE_PROFILE=/path/prefix`` to enable. Each process appends one
JSON object per event to ``<prefix>.<pid>.jsonl`` and flushes immediately, so a
killed engine still leaves usable data. With the variable unset the probes cost
a couple of function calls and write nothing.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from typing import Any

_PREFIX = os.environ.get("VLLM_GR_STAGE_PROFILE")
_ENABLED = bool(_PREFIX)
_LOCK = threading.RLock()
_STREAM: Any = None
_COUNTS: dict[str, int] = {}
_RECORDS = 0
_FLUSH_EVERY = 500


def enabled() -> bool:
    """Return whether profiling output was requested for this process."""
    return _ENABLED


def now() -> int:
    """Span timestamp; cheap enough to call unconditionally."""
    return time.perf_counter_ns()


def _stream() -> Any:
    global _STREAM
    if _STREAM is None:
        _STREAM = open(f"{_PREFIX}.{os.getpid()}.jsonl", "a", encoding="utf-8")
        # LIFO atexit order: flush counters before the stream is closed.
        atexit.register(_STREAM.close)
        atexit.register(_flush_counts)
    return _STREAM


def record(kind: str, **fields: Any) -> None:
    """Append one event. No-op unless profiling is enabled."""
    global _RECORDS
    if not _ENABLED:
        return
    fields["kind"] = kind
    fields["pid"] = os.getpid()
    fields["ts_ns"] = time.perf_counter_ns()
    line = json.dumps(fields, separators=(",", ":"))
    with _LOCK:
        handle = _stream()
        handle.write(line + "\n")
        handle.flush()
        _RECORDS += 1
        if _RECORDS % _FLUSH_EVERY == 0:
            _flush_counts_locked()


def durations(kind: str, *, start_ns: int, **fields: Any) -> None:
    """Record elapsed microseconds for a span that started at ``start_ns``."""
    if not _ENABLED:
        return
    fields["us"] = (time.perf_counter_ns() - start_ns) / 1000.0
    record(kind, **fields)


def count(name: str, amount: int = 1) -> None:
    """Increment a per-process counter, flushed periodically and at exit."""
    if not _ENABLED:
        return
    with _LOCK:
        _COUNTS[name] = _COUNTS.get(name, 0) + amount


def _flush_counts_locked() -> None:
    if not _COUNTS:
        return
    fields = dict(_COUNTS)
    _COUNTS.clear()
    fields["kind"] = "counts"
    fields["pid"] = os.getpid()
    fields["ts_ns"] = time.perf_counter_ns()
    handle = _stream()
    handle.write(json.dumps(fields, separators=(",", ":")) + "\n")
    handle.flush()


def _flush_counts() -> None:
    with _LOCK:
        _flush_counts_locked()
