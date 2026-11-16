# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise timing instrumentation failure paths without a model runtime."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import pytest

from tests.st import test_onerec_v1_scheduler_st as scheduler

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


class EngineTimingTests(unittest.TestCase):
    def test_sync_measurement_keeps_shutdown_on_request_failure(self) -> None:
        events = []

        class Engine:
            def __init__(self) -> None:
                events.append("startup")

            def __enter__(self) -> object:
                return self

            def __exit__(self, exc_type: object, *args: object) -> bool:
                events.append(exc_type)
                return False

        with patch.object(scheduler, "_MEASUREMENTS", {"phases": {}}):
            with (
                self.assertRaisesRegex(RuntimeError, "request failed"),
                scheduler._measured_sync_engine(Engine),
            ):
                raise RuntimeError("request failed")
            self.assertEqual(events, ["startup", RuntimeError])
            self.assertEqual(
                set(scheduler._MEASUREMENTS["phases"]),
                {"engine_startup_seconds", "engine_shutdown_seconds"},
            )

    def test_startup_failure_is_recorded_and_propagated(self) -> None:
        def fail_startup() -> None:
            raise RuntimeError("startup failed")

        with patch.object(scheduler, "_MEASUREMENTS", {"phases": {}}):
            with (
                self.assertRaisesRegex(RuntimeError, "startup failed"),
                scheduler._measured_sync_engine(fail_startup),
            ):
                self.fail("failed engine must not be yielded")
            self.assertEqual(set(scheduler._MEASUREMENTS["phases"]), {"engine_startup_seconds"})

    def test_async_measurement_keeps_shutdown_on_request_failure(self) -> None:
        events = []

        class Engine:
            async def __aenter__(self) -> object:
                events.append("startup")
                return self

            async def __aexit__(self, exc_type: object, *args: object) -> bool:
                events.append(exc_type)
                return False

        async def run() -> None:
            async with scheduler._measured_engine(lambda args: Engine(), None):
                raise RuntimeError("request failed")

        with patch.object(scheduler, "_MEASUREMENTS", {"phases": {}}):
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                asyncio.run(run())
            self.assertEqual(events, ["startup", RuntimeError])
            self.assertEqual(
                set(scheduler._MEASUREMENTS["phases"]),
                {"engine_startup_seconds", "engine_shutdown_seconds"},
            )
