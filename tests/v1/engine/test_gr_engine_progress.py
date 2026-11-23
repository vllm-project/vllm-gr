# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""V1 idle backoff must not delay pending execution or lose native outputs."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm_gr.v1.engine.engine_core_patch import make_patched_process_engine_step


def _native_step(engine: Any) -> bool:
    raise AssertionError("the V1 step adapter should execute step_fn directly")


@pytest.mark.parametrize(  # type: ignore[misc]
    ("model_executed", "progress", "in_flight", "has_requests", "should_sleep"),
    [
        (False, False, False, True, True),
        (True, False, False, True, False),
        (False, True, False, True, False),
        (False, False, True, True, False),
        (False, False, False, False, False),
    ],
)
def test_progress_backoff_preserves_native_outputs(
    monkeypatch: pytest.MonkeyPatch,
    model_executed: bool,
    progress: bool,
    in_flight: bool,
    has_requests: bool,
    should_sleep: bool,
) -> None:
    from vllm_gr.v1.engine import engine_core_patch

    events: list[Any] = []
    sleep = Mock()
    monkeypatch.setattr(engine_core_patch.time, "sleep", sleep)
    scheduler = SimpleNamespace(_gr_progress=0, has_requests=lambda: has_requests)
    output = object()

    def step() -> tuple[dict[int, object], bool]:
        scheduler._gr_progress += int(progress)
        return {2: output}, model_executed

    engine = SimpleNamespace(
        scheduler=scheduler,
        step_fn=step,
        batch_queue=[object()] if in_flight else [],
        output_queue=SimpleNamespace(put_nowait=lambda item: events.append(("output", item))),
        post_step=lambda executed: events.append(("post", executed)),
    )
    wrapper = make_patched_process_engine_step(_native_step)
    assert make_patched_process_engine_step(wrapper) is wrapper
    assert wrapper(engine) is model_executed
    assert events == [("output", (2, output)), ("post", model_executed)]
    if should_sleep:
        sleep.assert_called_once_with(0.001)
    else:
        sleep.assert_not_called()


def test_step_failure_propagates_without_fabricating_output() -> None:
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(_gr_progress=0),
        step_fn=Mock(side_effect=RuntimeError("worker failed")),
        output_queue=SimpleNamespace(put_nowait=Mock()),
        post_step=Mock(),
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        make_patched_process_engine_step(_native_step)(engine)
    engine.output_queue.put_nowait.assert_not_called()
    engine.post_step.assert_not_called()


pytestmark = pytest.mark.cpu_test
