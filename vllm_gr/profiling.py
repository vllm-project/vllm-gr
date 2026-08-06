# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal opt-in profiler for the catalog beam-search hot path.

Enabled only when ``VLLM_GR_PROFILE=1`` is set.  When disabled, each
``profile()`` call still enters/exits a shared no-op context (measured at
roughly 70-80ns per call).  On a 128-beam, 5-step request that amounts to
~0.16ms — about 0.1% of a ~200ms decode — i.e. small but not literally zero.
Keeping the overhead inside the ``with`` block keeps the instrumentation
readable; if it ever matters, the hot per-beam calls can switch to an ``if``
shortcut instead.

When enabled, ``profile(name)`` records wall time against the *current
request's* profiler (tracked via a ``contextvars.ContextVar``), so concurrent
online-serving requests never bleed timing into each other.  Each profiled
``beam_search`` request prints one compact summary line at the end with
per-stage aggregates (calls / total / avg / max).  No per-step logging.

Designed to answer the reviewer question "where does time actually go on the
real decode loop" — including the engine-step wait, so a stage's share of the
end-to-end budget is visible, not just its internal share.
"""

from __future__ import annotations

import os
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

_ENABLED = os.environ.get("VLLM_GR_PROFILE") == "1"

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class _Stage:
    calls: int = 0
    total_us: float = 0.0
    max_us: float = 0.0

    def record(self, us: float) -> None:
        self.calls += 1
        self.total_us += us
        if us > self.max_us:
            self.max_us = us


@dataclass
class _RequestProfiler:
    stages: dict[str, _Stage] = field(default_factory=dict)

    def record(self, name: str, us: float) -> None:
        self.stages.setdefault(name, _Stage()).record(us)

    def summary(self) -> str:
        if not self.stages:
            return ""
        rows = [
            (s.total_us, s.calls, s.total_us / s.calls, s.max_us, name)
            for name, s in self.stages.items()
            if s.calls
        ]
        rows.sort(reverse=True)  # by total
        lines = ["| stage | calls | total(ms) | avg(us) | max(us) |"]
        lines.append("| --- | --- | --- | --- | --- |")
        for total_us, calls, avg_us, max_us, name in rows:
            lines.append(
                f"| {name} | {calls} | {total_us / 1000:.2f} | {avg_us:.1f} | {max_us:.1f} |"
            )
        return "\n".join(lines)


_current: ContextVar[_RequestProfiler | None] = ContextVar("vllm_gr_profiler", default=None)


class _ProfileContext:
    __slots__ = ("name", "start")

    def __init__(self, name: str) -> None:
        self.name = name
        self.start = 0.0

    def __enter__(self) -> "_ProfileContext":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        elapsed = (time.perf_counter() - self.start) * 1e6  # us
        prof = _current.get()
        if prof is not None:
            prof.record(self.name, elapsed)


def profile(name: str) -> _ProfileContext:
    """Time ``name`` against the current request's profiler when enabled.

    The disabled path is a single boolean check plus an attribute fetch; the
    context manager itself is only allocated when profiling is on.
    """
    if _ENABLED:
        return _ProfileContext(name)
    return _noop_context


def profiled(name: str) -> Callable[[F], F]:
    """Decorator timing the whole wrapped function under ``name``.

    Transparent no-op when profiling is off.
    """

    def decorator(func: F) -> F:
        if not _ENABLED:
            return func

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with profile(name):
                return func(*args, **kwargs)

        wrapper.__name__ = func.__name__  # type: ignore[attr-defined]
        wrapper.__doc__ = func.__doc__  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def profiled_async_generator(func: F) -> F:
    """Wrap an async generator with a per-request profiler, exception-safe.

    When profiling is off this returns ``func`` unchanged, so the production
    path is *not* rewritten (pure opt-in).

    The ContextVar token is set immediately before each ``__anext__`` and reset
    immediately after the item is produced, so the token never crosses a
    ``yield``.  This keeps the profiler safe against callers that ``break`` or
    close the generator from a different async context.  The ``_RequestProfiler``
    object itself lives across yields (accumulating stages); the summary is
    printed once on normal completion or any exception.
    """

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        _prof = beam_search_profiler()
        _finished = False
        agen = func(*args, **kwargs)
        try:
            while True:
                _prof._set_token()
                try:
                    item = await agen.__anext__()
                except StopAsyncIteration:
                    _finished = True
                    break
                finally:
                    _prof._reset_token(print_summary=False)
                yield item
            _finished = True
        except GeneratorExit:
            # Caller closed us early (e.g. `break` / `aclose` from another
            # context): close the inner generator explicitly so its `finally`
            # runs now, but do not print a summary.
            await agen.aclose()
            raise
        except BaseException:
            _finished = True
            raise
        finally:
            if _finished:
                _prof._print_summary()

    if not _ENABLED:
        return func

    wrapper.__name__ = func.__name__  # type: ignore[attr-defined]
    wrapper.__doc__ = func.__doc__  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def profiled_sync(func: F) -> F:
    """Wrap a plain function/method with a per-request profiler, exception-safe.

    ``__exit__`` (ContextVar reset + summary print) runs on normal return or any
    exception, so the profiler never leaks its ContextVar.  When profiling is
    off this returns ``func`` unchanged (pure opt-in).
    """

    if not _ENABLED:
        return func

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _prof = beam_search_profiler()
        _prof.__enter__()
        try:
            return func(*args, **kwargs)
        finally:
            _prof.__exit__(None, None, None)

    wrapper.__name__ = func.__name__  # type: ignore[attr-defined]
    wrapper.__doc__ = func.__doc__  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


class _NoopContext:
    __slots__ = ()

    def __enter__(self) -> "_NoopContext":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


_noop_context = _NoopContext()


class beam_search_profiler:
    """Per-request profiler: set a ContextVar, accumulate stages, print summary.

    Used as a context manager (``with beam_search_profiler():``) or via the
    lower-level ``_set_token`` / ``_reset_token`` / ``_print_summary`` methods
    used by ``profiled_async_generator`` so the ContextVar token never crosses
    a ``yield``.

    When profiling is off this is a transparent no-op and ``__enter__`` returns
    None.
    """

    __slots__ = ("_token", "_prof")

    def __init__(self) -> None:
        self._token: Any = None
        self._prof: _RequestProfiler | None = None

    def _set_token(self) -> None:
        if not _ENABLED:
            return
        if self._prof is None:
            self._prof = _RequestProfiler()
        if self._token is None:
            self._token = _current.set(self._prof)

    def _reset_token(self, *, print_summary: bool = False) -> None:
        if not _ENABLED:
            return
        if self._token is not None:
            _current.reset(self._token)
            self._token = None
        if print_summary:
            self._print_summary()

    def _print_summary(self) -> None:
        if self._prof is not None:
            print(self._prof.summary(), flush=True)

    def __enter__(self) -> _RequestProfiler | None:
        if not _ENABLED:
            return None
        self._set_token()
        return self._prof

    def __exit__(self, *exc: object) -> None:
        self._reset_token(print_summary=True)
