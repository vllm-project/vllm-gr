# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-free regressions for accelerator cleanup after engine shutdown."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_gr.utils import npu_adapter


@pytest.mark.parametrize("initialized", [False, True])  # type: ignore[misc]
def test_release_memory_preserves_npu_initialization_state(
    monkeypatch: pytest.MonkeyPatch, initialized: bool
) -> None:
    state = {"initialized": initialized}

    def synchronize() -> None:
        # Model torch-npu's real lazy-init side effect of synchronize().
        state["initialized"] = True

    npu = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: state["initialized"],
        synchronize=Mock(side_effect=synchronize),
        empty_cache=Mock(),
        ipc_collect=Mock(),
    )
    cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=Mock(),
        empty_cache=Mock(),
        ipc_collect=Mock(),
    )
    monkeypatch.setattr(npu_adapter, "torch", SimpleNamespace(npu=npu, cuda=cuda))

    npu_adapter.release_accelerator_memory()

    assert state["initialized"] is initialized
    if initialized:
        npu.synchronize.assert_called_once_with()
        npu.empty_cache.assert_called_once_with()
    else:
        npu.synchronize.assert_not_called()
        npu.empty_cache.assert_not_called()
    npu.ipc_collect.assert_not_called()
    cuda.synchronize.assert_called_once_with()
    cuda.empty_cache.assert_called_once_with()
    cuda.ipc_collect.assert_called_once_with()
