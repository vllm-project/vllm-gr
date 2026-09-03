# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from vllm_gr.v1.engine.engine_core_patch import make_patched_abort_requests


def test_abort_releases_only_persistent_gr_worker_state() -> None:
    original = mock.Mock(return_value="done")
    abort = make_patched_abort_requests(original)
    executor = mock.Mock()
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(
            requests={
                "gr-0": SimpleNamespace(gr_persistent_runner_state=True),
                "plain-0": SimpleNamespace(gr_persistent_runner_state=False),
            }
        ),
        model_executor=executor,
    )

    assert abort(engine, ["gr-0", "plain-0", "missing"]) == "done"
    original.assert_called_once_with(engine, ["gr-0", "plain-0", "missing"])
    executor.collective_rpc.assert_called_once_with(
        "release_gr_model_runner_states",
        args=(["gr-0"],),
        non_block=True,
    )


def test_plain_abort_does_not_dispatch_worker_rpc() -> None:
    original = mock.Mock()
    abort = make_patched_abort_requests(original)
    executor = mock.Mock()
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(
            requests={"plain-0": SimpleNamespace(gr_persistent_runner_state=False)}
        ),
        model_executor=executor,
    )

    abort(engine, ["plain-0"])

    executor.collective_rpc.assert_not_called()
