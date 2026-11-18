# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Contract tests for the CUDA stage hooks ABC.

The base class guarantees completeness: a subclass that leaves a required
operation abstract cannot be instantiated. These tests pin that guarantee so a future
change back to a structural Protocol cannot silently drop it.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from tools.pre_commit import mypy as mypy_hook
from vllm_gr.v1.worker.cuda.gpu_stage_hooks import (
    CudaStageHooks,
    validate_stage_hooks_type,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_every_operation_is_abstract() -> None:
    """Each declared operation must be abstract, or omission stops failing."""
    assert {
        "persistent_graph_install",
        "supports_transposed_head",
        "specialize_constraint_steps",
        "required_constraint_rows",
        "forward_hidden",
        "prefill_logits",
        "finish_native_output",
        "prefill_stage_fence",
        "consume_native_state",
        "decode_block_ids",
        "initialize_inputs",
        "prepare_new_adapter",
        "decode_graph",
        "decode_logits",
        "consume_retirement_state",
    } <= CudaStageHooks.__abstractmethods__


def test_incomplete_implementation_cannot_be_instantiated() -> None:
    """A missing operation fails at construction, on every device, without a checker."""

    # The test mypy group skips production imports, so it sees this base as Any.
    class Incomplete(CudaStageHooks):  # type: ignore[misc]
        persistent_graph_install = False
        supports_transposed_head = False
        specialize_constraint_steps = False

        @staticmethod
        def required_constraint_rows(batch_size: int, beam_width: int) -> int:
            return beam_width

        def forward_hidden(self, *args: Any, **kwargs: Any) -> Any: ...
        def prefill_logits(self, *args: Any, **kwargs: Any) -> Any: ...
        def finish_native_output(self) -> None: ...
        def prefill_stage_fence(self) -> None: ...
        def consume_native_state(self, scheduler_output: Any) -> None: ...
        # decode_block_ids deliberately omitted
        def initialize_inputs(self, owner: Any, rows: int) -> None: ...
        def prepare_new_adapter(self, view: Any) -> None: ...
        def decode_graph(self, batch_size: int, beam_width: int) -> Any: ...
        def decode_logits(self, owner: Any) -> Any: ...
        def consume_retirement_state(self, scheduler_output: Any, bindings: Any) -> None: ...

    with pytest.raises(TypeError, match="decode_block_ids"):
        Incomplete(SimpleNamespace())


def test_validate_rejects_a_different_hooks_class() -> None:
    """One Worker keeps one concrete hooks type for its whole lifetime."""

    # Keep the real runtime ABC despite the test group's skipped-import typing.
    class First(CudaStageHooks):  # type: ignore[misc]
        persistent_graph_install = False
        supports_transposed_head = False
        specialize_constraint_steps = False

        @staticmethod
        def required_constraint_rows(batch_size: int, beam_width: int) -> int:
            return beam_width

        def forward_hidden(self, *args: Any, **kwargs: Any) -> Any: ...
        def prefill_logits(self, *args: Any, **kwargs: Any) -> Any: ...
        def finish_native_output(self) -> None: ...
        def prefill_stage_fence(self) -> None: ...
        def consume_native_state(self, scheduler_output: Any) -> None: ...
        def decode_block_ids(self, session_id: str) -> list[list[int]]:
            return []

        def initialize_inputs(self, owner: Any, rows: int) -> None: ...
        def prepare_new_adapter(self, view: Any) -> None: ...
        def decode_graph(self, batch_size: int, beam_width: int) -> Any: ...
        def decode_logits(self, owner: Any) -> Any: ...
        def consume_retirement_state(self, scheduler_output: Any, bindings: Any) -> None: ...

    class Second(First): ...

    validate_stage_hooks_type(First(SimpleNamespace()), First)
    with pytest.raises(ValueError, match="cannot change"):
        validate_stage_hooks_type(First(SimpleNamespace()), Second)


def test_implementations_subclass_the_contract() -> None:
    """Both CUDA runners must inherit the ABC, not merely match it structurally."""
    from vllm_gr.v1.worker.cuda.v1.gpu_beam_stage_runner_v1 import CudaStageHooksV1
    from vllm_gr.v1.worker.cuda.v2.gpu_beam_stage_runner_v2 import CudaStageHooksV2

    for implementation in (CudaStageHooksV1, CudaStageHooksV2):
        assert issubclass(implementation, CudaStageHooks)
        assert not implementation.__abstractmethods__


@pytest.mark.parametrize(  # type: ignore[misc]
    "changed_file",
    sorted(
        {
            *mypy_hook.CUDA_STAGE_FILES,
            "tools/pre_commit/mypy.py",
            # Pin moved owners independently of the routing list itself.
            "vllm_gr/v1/worker/model_runner_contracts.py",
            "vllm_gr/v1/worker/gr_gc.py",
            "vllm_gr/v1/worker/gr_context.py",
            "vllm_gr/v1/worker/cuda/v2/gpu_model_runner_v2_patch.py",
            "vllm_gr/v1/worker/cuda/v2/gpu_model_runner_v2_resources.py",
        }
    ),
)
def test_mypy_checks_complete_boundary(changed_file: str) -> None:
    """Changing any boundary member checks its peers and negative cases too."""
    groups = mypy_hook.group_files([changed_file, "tests/test_gr_startup.py"])
    assert groups.pop(mypy_hook.CUDA_STAGE_GROUP) == mypy_hook.CUDA_STAGE_FILES
    assert groups.pop("tests") == ["tests/test_gr_startup.py"]
    assert not any(groups.values())


@pytest.mark.parametrize(  # type: ignore[misc]
    "changed_file",
    sorted(mypy_hook.NPU_WORKER_FILES),
)
def test_npu_worker_mypy_checks_all_callers(changed_file: str) -> None:
    """Changing either NPU helper or caller checks the whole NPU worker boundary."""
    groups = mypy_hook.group_files([changed_file, "tests/test_gr_startup.py"])
    assert groups.pop(mypy_hook.NPU_WORKER_GROUP) == mypy_hook.NPU_WORKER_FILES
    assert groups.pop("tests") == ["tests/test_gr_startup.py"]
    assert not any(groups.values())


def test_mypy_boundary_avoids_cache_and_requires_negative_errors(monkeypatch: Any) -> None:
    """Pin checker flags without starting a checker subprocess in pytest."""
    commands: list[list[str]] = []

    def run(args: list[str], *, check: bool) -> Any:
        assert check is False
        commands.append(args)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(mypy_hook.subprocess, "run", run)
    assert mypy_hook.mypy(mypy_hook.CUDA_STAGE_FILES, "3.12", "skip", cuda_stage=True) == 7
    command = commands[0]
    assert "--no-incremental" in command
    assert command[command.index("--cache-dir") + 1] == mypy_hook.os.devnull
    assert "--warn-unused-ignores" in command
    assert command[-len(mypy_hook.CUDA_STAGE_FILES) :] == mypy_hook.CUDA_STAGE_FILES
