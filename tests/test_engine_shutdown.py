# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_gr.utils.engine_lifecycle import shutdown_sync_engine


@pytest.mark.parametrize(  # type: ignore[misc]
    "configured, expected", [(0, None), (1, 1), (10, 10), (30, 30)]
)
def test_sync_core_shutdown_receives_configured_drain_budget(
    configured: int, expected: int | None
) -> None:
    core = SimpleNamespace(shutdown=Mock())
    engine = SimpleNamespace(
        engine_core=core, vllm_config=SimpleNamespace(shutdown_timeout=configured)
    )

    shutdown_sync_engine(engine)

    core.shutdown.assert_called_once_with(timeout=expected)


def test_sync_core_without_config_keeps_default_cleanup_window() -> None:
    core = SimpleNamespace(shutdown=Mock())

    shutdown_sync_engine(SimpleNamespace(engine_core=core))

    core.shutdown.assert_called_once_with(timeout=None)


def test_engine_owned_shutdown_keeps_its_protocol() -> None:
    engine = SimpleNamespace(
        shutdown=Mock(),
        engine_core=SimpleNamespace(shutdown=Mock()),
        vllm_config=SimpleNamespace(shutdown_timeout=10),
    )

    shutdown_sync_engine(engine)

    engine.shutdown.assert_called_once_with()
    engine.engine_core.shutdown.assert_not_called()


def test_shutdown_error_is_not_retried_without_the_budget() -> None:
    shutdown = Mock(side_effect=RuntimeError("shutdown failed"))
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(shutdown=shutdown),
        vllm_config=SimpleNamespace(shutdown_timeout=10),
    )

    with pytest.raises(RuntimeError, match="shutdown failed"):
        shutdown_sync_engine(engine)

    shutdown.assert_called_once_with(timeout=10)


pytestmark = pytest.mark.cpu_test
