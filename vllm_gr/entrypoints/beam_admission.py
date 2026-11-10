# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend admission control for online beam-search requests.

An online beam session keeps the engine's execution owner until it retires, so
the number of sessions a server runs -- and the queue waiting behind them --
decides whether overload is answered with fast rejections or with unbounded
latency.  This module bounds both in the serving process, before a request
reaches the engine.

The limiter is process-local on purpose: every API server process owns its own
engine client and its own admission budget, and the engine keeps its own
capacity check behind it.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any

from vllm.exceptions import GracefulHTTPError
from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_INFLIGHT_ENV = "VLLM_GR_BEAM_MAX_INFLIGHT"
MAX_WAITING_ENV = "VLLM_GR_BEAM_MAX_WAITING"
ADMISSION_TIMEOUT_ENV = "VLLM_GR_BEAM_ADMISSION_TIMEOUT_S"
OUTPUT_TOP_K_ENV = "VLLM_GR_BEAM_OUTPUT_TOP_K"

_REJECT_LOG_EVERY = 100


class BeamAdmissionRejected(GracefulHTTPError):
    """A beam request refused before submission, reported as HTTP 429.

    ``GracefulHTTPError`` is vLLM's contract for an expected refusal -- the
    serving layer answers it with the status it carries instead of turning it
    into an internal server error -- so backpressure stays distinguishable from
    a broken engine.  ``Retry-After`` is advisory: callers should still use
    jittered backoff rather than a fixed delay.
    """

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message, HTTPStatus.TOO_MANY_REQUESTS)
        self.retry_after_s = retry_after_s


@dataclass
class _Waiter:
    """One queued request holding a future the releaser hands a slot to."""

    future: asyncio.Future[None]
    granted: bool = False


@dataclass
class BeamAdmissionLimiter:
    """Bounded in-flight sessions plus a bounded, timed waiting queue.

    ``max_inflight`` is the number of sessions allowed to run at once.
    ``max_waiting`` is how many may queue behind them; ``wait_timeout_s`` is how
    long a queued request may wait for a slot.  A timeout or a full queue is
    rejected immediately (HTTP 429) rather than parked, so admitted requests
    keep a bounded latency instead of inheriting the whole backlog's.
    """

    max_inflight: int | None = None
    max_waiting: int = 0
    wait_timeout_s: float = 0.0

    _inflight: int = field(default=0, init=False)
    _waiters: list[_Waiter] = field(default_factory=list, init=False)
    _admitted: int = field(default=0, init=False)
    _rejected: int = field(default=0, init=False)
    _timed_out: int = field(default=0, init=False)
    _released: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_inflight is not None and self.max_inflight <= 0:
            raise ValueError("max_inflight must be positive when set")
        if self.max_waiting < 0:
            raise ValueError("max_waiting must be non-negative")
        if self.wait_timeout_s < 0:
            raise ValueError("wait_timeout_s must be non-negative")

    @property
    def enabled(self) -> bool:
        return self.max_inflight is not None

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def waiting(self) -> int:
        return len(self._waiters)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_inflight": self.max_inflight,
            "max_waiting": self.max_waiting,
            "wait_timeout_s": self.wait_timeout_s,
            "inflight": self._inflight,
            "waiting": len(self._waiters),
            "admitted": self._admitted,
            "released": self._released,
            "rejected": self._rejected,
            "timed_out": self._timed_out,
        }

    async def acquire(self) -> None:
        """Take one execution slot, or raise ``BeamAdmissionRejected``."""
        if not self.enabled:
            return
        assert self.max_inflight is not None
        if self._inflight < self.max_inflight:
            self._inflight += 1
            self._admitted += 1
            return
        if (
            self.max_waiting <= 0
            or self.wait_timeout_s <= 0.0
            or len(self._waiters) >= self.max_waiting
        ):
            if self.max_waiting <= 0 or self.wait_timeout_s <= 0.0:
                detail = f"in-flight limit {self.max_inflight} reached; waiting disabled"
            else:
                detail = (
                    f"in-flight limit {self.max_inflight} and waiting limit "
                    f"{self.max_waiting} both reached"
                )
            raise self._reject(detail)

        waiter = _Waiter(asyncio.get_running_loop().create_future())
        self._waiters.append(waiter)
        try:
            # Shield keeps a handed-over slot observable after a timeout: the
            # future is not cancelled underneath us, so ``granted`` stays true
            # and the slot can be returned instead of leaked.
            await asyncio.wait_for(asyncio.shield(waiter.future), self.wait_timeout_s)
        except TimeoutError:
            if waiter.granted:
                self.release()
            else:
                self._discard(waiter)
                self._timed_out += 1
                raise self._rejection(
                    f"waited {self.wait_timeout_s:.3f}s for an execution slot"
                ) from None
        except asyncio.CancelledError:
            if waiter.granted:
                self.release()
            else:
                self._discard(waiter)
            raise
        self._admitted += 1

    def release(self) -> None:
        """Give a slot back, handing it to the oldest waiter when one exists."""
        if not self.enabled or self._inflight <= 0:
            return
        self._released += 1
        while self._waiters:
            waiter = self._waiters.pop(0)
            if waiter.future.done():
                continue
            # Ownership moves to the waiter; the in-flight count is unchanged.
            waiter.granted = True
            waiter.future.set_result(None)
            return
        self._inflight -= 1

    def _discard(self, waiter: _Waiter) -> None:
        try:
            self._waiters.remove(waiter)
        except ValueError:
            pass

    def _reject(self, detail: str) -> BeamAdmissionRejected:
        self._rejected += 1
        if self._rejected == 1 or self._rejected % _REJECT_LOG_EVERY == 0:
            logger.warning(
                "vllm-gr: beam admission rejected %d request(s) so far (%s; %s)",
                self._rejected,
                detail,
                self.stats(),
            )
        return self._rejection(detail)

    def _rejection(self, detail: str) -> BeamAdmissionRejected:
        return BeamAdmissionRejected(
            "vllm-gr: beam request rejected by frontend admission control "
            f"({detail}; inflight={self._inflight}, waiting={len(self._waiters)})",
            retry_after_s=self.wait_timeout_s if self.wait_timeout_s > 0 else None,
        )


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("vllm-gr: ignoring non-integer %s=%r", name, raw)
        return None
    return value if value > 0 else None


def _env_float(name: str) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        logger.warning("vllm-gr: ignoring non-numeric %s=%r", name, raw)
        return 0.0
    return max(value, 0.0)


_limiter: BeamAdmissionLimiter | None = None
_limiter_env: tuple[int | None, int, float] | None = None


def beam_admission_limiter() -> BeamAdmissionLimiter:
    """Return this process's limiter, rebuilt if the environment changed."""
    global _limiter, _limiter_env
    max_inflight = _env_int(MAX_INFLIGHT_ENV)
    max_waiting = _env_int(MAX_WAITING_ENV) or 0
    wait_timeout_s = _env_float(ADMISSION_TIMEOUT_ENV)
    env = (max_inflight, max_waiting, wait_timeout_s)
    if _limiter is None or env != _limiter_env:
        _limiter = BeamAdmissionLimiter(
            max_inflight=max_inflight,
            max_waiting=max_waiting,
            wait_timeout_s=wait_timeout_s,
        )
        _limiter_env = env
        if _limiter.enabled:
            logger.info("vllm-gr: beam admission control %s", _limiter.stats())
    return _limiter


def reset_beam_admission_limiter() -> None:
    """Forget the cached limiter (tests and environment changes)."""
    global _limiter, _limiter_env
    _limiter = None
    _limiter_env = None


def beam_output_top_k() -> int | None:
    """Cap the number of beam results converted into public outputs."""
    return _env_int(OUTPUT_TOP_K_ENV)


_CLOSE_TASKS: set[asyncio.Task[None]] = set()


def close_async_generator(stream: Any) -> None:
    """Close ``stream`` from a task that outlives the caller's cancellation.

    Closing the outer stream is what turns a cancelled request into a session
    abort; driving it from its own task keeps that work alive when the caller's
    task is already cancelled, and the retained task reference keeps asyncio
    from collecting it mid-flight.
    """
    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return

    async def _close() -> None:
        try:
            await aclose()
        except Exception:
            logger.exception("vllm-gr: closing an aborted beam stream failed")

    try:
        task = asyncio.get_running_loop().create_task(_close())
    except RuntimeError:
        return
    _CLOSE_TASKS.add(task)
    task.add_done_callback(_CLOSE_TASKS.discard)


async def admitted_stream(inner: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Yield from ``inner`` while holding one admission slot.

    The slot is held for exactly the length of the request: a rejected request
    never reaches the engine, and a cancelled one closes the inner stream so
    its session is aborted instead of running on.
    """
    limiter = beam_admission_limiter()
    await limiter.acquire()
    exhausted = False
    try:
        async for output in inner:
            yield output
        exhausted = True
    finally:
        limiter.release()
        if not exhausted:
            close_async_generator(inner)
