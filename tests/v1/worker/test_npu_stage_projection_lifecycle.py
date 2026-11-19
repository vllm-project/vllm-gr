# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Exercise stage projection ownership through the real worker lifecycle."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from vllm_gr.config import (
    BeamExecutionMode,
    ConstraintTableBackendEnum,
    ConstraintTableConfig,
    GRConfig,
)
from vllm_gr.config.parser import gr_config_to_mapping
from vllm_gr.ops.npu_stage_logits import make_npu_stage_reload_weights
from vllm_gr.v1.worker.model_runner_patch import (
    make_load_model_with_constraint_backend,
    make_shutdown_with_constraint_backend,
)


class _Backend:
    def __init__(self) -> None:
        self.info = SimpleNamespace(
            backend=ConstraintTableBackendEnum.NPU,
            device_memory_bytes=32,
            table_metadata=SimpleNamespace(vocab_size=384),
        )
        self.stage_vocab: SimpleNamespace | None = SimpleNamespace(
            vocab_size=128,
            stage_token_ids=(
                torch.tensor([17, 100]),
                torch.tensor([3]),
                torch.tensor([1, 89, 321]),
            ),
        )
        self.close_calls = 0

    def install_runner_integration(
        self,
        runner: Any,
        *,
        parallel_config: Any,
        vocab_size: int,
    ) -> int:
        """Delegate to the real Ascend integration so the test tracks production."""
        from vllm_gr.constraints.npu_backend import NPUConstraintBackend

        return NPUConstraintBackend.install_runner_integration(
            self,
            runner,
            parallel_config=parallel_config,
            vocab_size=vocab_size,
        )

    def close(self) -> None:
        self.close_calls += 1


def _replace_model(runner: Any, value: float) -> None:
    model = torch.nn.Module()
    head: Any = torch.nn.Linear(8, 384)
    with torch.no_grad():
        head.weight.fill_(value)
        head.bias.fill_(value / 2)
    head.quant_method = UnquantizedEmbeddingMethod()
    head.tp_size = 1
    # vLLM 0.30 CustomOps read the current config at construction time (same
    # pattern as tests/test_sliced_logits.py::_native_processor).
    with set_current_vllm_config(VllmConfig()):
        processor = LogitsProcessor(384, scale=0.7, soft_cap=13.0)
    processor._gather_logits = lambda logits: logits
    model.add_module("lm_head", head)
    model.add_module("logits_processor", processor)
    runner.model = model
    runner.model_memory_usage = 100


@pytest.fixture
def lifecycle(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    # Earlier NPU integration tests register AscendLogitsProcessor as the
    # out-of-tree LogitsProcessor, and vllm-ascend 0.30 dispatches it by
    # lm_head.parallel_mode — an attribute this fixture's CPU fakes do not
    # model. Construct the native processor for the worker-lifecycle logic.
    from vllm.model_executor.custom_op import op_registry_oot

    monkeypatch.delitem(op_registry_oot, "LogitsProcessor", raising=False)
    config = GRConfig(
        beam_execution_mode=BeamExecutionMode.V1,
        beam_max_width=4,
        beam_max_decode_steps=3,
        constraint_table=ConstraintTableConfig(
            backend=ConstraintTableBackendEnum.NPU,
            enabled=True,
            max_top_k=4,
            path="/models/test-constraint-table.safetensors",
            artifact_digest="sha256:" + "1" * 64,
            tokenizer_digest="sha256:" + "2" * 64,
        ),
    )
    plan = SimpleNamespace(execution=SimpleNamespace(stage_index=0, num_execution_rows=2))
    runner = SimpleNamespace(
        constraint_backend=None,
        vllm_config=SimpleNamespace(
            additional_config={"vllm_gr": gr_config_to_mapping(config)},
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
            lora_config=None,
            speculative_config=None,
        ),
        lora_config=None,
        model_config=SimpleNamespace(dtype=torch.float32, get_vocab_size=lambda: 384),
        model_memory_usage=0,
        device=torch.device("cpu"),
        _gr_npu_stage_plan=plan,
        _gr_npu_stage_runner=SimpleNamespace(pending_plan=plan),
    )
    created: list[_Backend] = []

    def load_backend(**kwargs: Any) -> _Backend:
        assert kwargs["device"] == torch.device("cpu")
        assert kwargs["vocab_size"] == 384
        assert runner.constraint_backend is None
        backend = _Backend()
        if not kwargs.get("enable_npu_stage_vocab", False):
            backend.stage_vocab = None
        created.append(backend)
        return backend

    monkeypatch.setenv("VLLM_GR_NPU_STAGE_LM_HEAD", "1")
    monkeypatch.setattr("vllm_gr.constraints.loader.load_constraint_backend", load_backend)
    yield SimpleNamespace(runner=runner, backends=created)
    make_shutdown_with_constraint_backend()(runner)


@pytest.mark.parametrize("shutdown_fails", [False, True])
def test_reload_and_shutdown_restore_hooks_and_close_each_backend_once(
    lifecycle: Any, shutdown_fails: bool
) -> None:
    runner = lifecycle.runner
    processors: list[Any] = []
    originals: list[Any] = []
    projections: list[Any] = []

    def load_weights(self: Any) -> str:
        assert self.constraint_backend is None
        if projections:
            assert projections[-1].closed
            assert processors[-1]._get_logits == originals[-1]
            assert lifecycle.backends[-1].close_calls == 1
        _replace_model(self, float(len(processors) + 1))
        processors.append(self.model.logits_processor)
        originals.append(self.model.logits_processor._get_logits)
        return "loaded"

    load = make_load_model_with_constraint_backend(load_weights)
    hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 20
    previous = None
    for _ in range(2):
        assert load(runner) == "loaded"
        owner = runner._gr_npu_stage_projection
        projections.append(owner)
        assert runner.model_memory_usage == 132 + owner.device_memory_bytes
        head = runner.model.lm_head
        actual = runner.model.logits_processor.forward(head, hidden, head.bias)
        full = torch.tanh(F.linear(hidden, head.weight, head.bias) / 13.0) * 13.0 * 0.7
        torch.testing.assert_close(actual[:, :2], full[:, [17, 100]])
        if previous is not None:
            assert not torch.allclose(actual, previous)
        previous = actual

    shutdown_calls: list[int] = []

    def native_shutdown(self: Any) -> None:
        shutdown_calls.append(1)
        assert self.constraint_backend is None
        assert self._gr_npu_stage_projection is None
        assert projections[-1].closed
        assert processors[-1]._get_logits == originals[-1]
        assert lifecycle.backends[-1].close_calls == 1
        self.model = None
        if shutdown_fails:
            raise RuntimeError("native shutdown failed")

    shutdown = make_shutdown_with_constraint_backend(native_shutdown)
    for _ in range(2):
        if shutdown_fails:
            with pytest.raises(RuntimeError, match="native shutdown failed"):
                shutdown(runner)
        else:
            shutdown(runner)
    assert len(shutdown_calls) == 2
    assert [backend.close_calls for backend in lifecycle.backends] == [1, 1]


@pytest.mark.parametrize("failure", ["weights", "install"])
def test_failed_reload_restores_old_hook_and_does_not_publish_partial_owner(
    lifecycle: Any, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = lifecycle.runner
    fail_next = False

    def load_weights(self: Any) -> None:
        assert self.constraint_backend is None
        if fail_next and failure == "weights":
            raise RuntimeError("weights failed")
        _replace_model(self, 1.0 if not fail_next else 2.0)

    load = make_load_model_with_constraint_backend(load_weights)
    load(runner)
    old = runner._gr_npu_stage_projection
    old_processor = runner.model.logits_processor
    fail_next = True
    if failure == "install":

        def fail_install(*args: Any, **kwargs: Any) -> None:
            assert old.closed
            assert lifecycle.backends[0].close_calls == 1
            raise RuntimeError("install failed")

        monkeypatch.setattr(
            "vllm_gr.ops.npu_stage_logits.install_npu_stage_projection", fail_install
        )
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        load(runner)
    assert old.closed
    assert "_get_logits" not in vars(old_processor)
    assert runner.constraint_backend is None
    assert runner._gr_npu_stage_projection is None
    assert len(lifecycle.backends) == (1 if failure == "weights" else 2)
    assert all(backend.close_calls == 1 for backend in lifecycle.backends)
    make_shutdown_with_constraint_backend()(runner)
    assert all(backend.close_calls == 1 for backend in lifecycle.backends)


@pytest.mark.parametrize(
    "case", ["pipeline_parallel", "tensor_parallel", "lora", "speculative", "disabled", "unset"]
)
def test_unsupported_worker_never_requests_compact_vocab_or_installs_projection(
    lifecycle: Any, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    runner = lifecycle.runner
    if case == "pipeline_parallel":
        runner.vllm_config.parallel_config.pipeline_parallel_size = 2
    elif case == "tensor_parallel":
        runner.vllm_config.parallel_config.tensor_parallel_size = 2
    elif case == "lora":
        runner.lora_config = SimpleNamespace()
        runner.vllm_config.lora_config = runner.lora_config
    elif case == "speculative":
        runner.vllm_config.speculative_config = SimpleNamespace()
    elif case == "disabled":
        monkeypatch.setenv("VLLM_GR_NPU_STAGE_LM_HEAD", "0")
    else:
        mapping = runner.vllm_config.additional_config["vllm_gr"]
        mapping["beam"].pop("execution_mode")

    make_load_model_with_constraint_backend(lambda self: _replace_model(self, 1.0))(runner)
    assert runner.constraint_backend.get() is lifecycle.backends[-1]
    assert runner.constraint_backend.stage_vocab is None
    assert getattr(runner, "_gr_npu_stage_projection", None) is None
    assert "_get_logits" not in vars(runner.model.logits_processor)
    hidden = torch.ones(2, 8)
    head = runner.model.lm_head
    actual = runner.model.logits_processor.forward(head, hidden, head.bias)
    assert actual.shape == (2, 384)


def test_failed_native_reload_requires_complete_load_to_resume(lifecycle: Any) -> None:
    runner = lifecycle.runner
    load = make_load_model_with_constraint_backend(lambda self: _replace_model(self, 1.0))
    load(runner)
    old = runner._gr_npu_stage_projection
    old_processor = runner.model.logits_processor
    native_calls = []

    def native_reload(self: Any, weights_path: str | None = None) -> None:
        native_calls.append(weights_path)
        with torch.no_grad():
            self.model.lm_head.weight.add_(1)
        raise RuntimeError("native partial reload failed")

    reload_weights = make_npu_stage_reload_weights(native_reload)
    with pytest.raises(RuntimeError, match="native partial reload failed"):
        reload_weights(runner, weights_path="replacement")
    assert old.snapshot()["update_failed"]
    hidden = torch.ones(2, 8)
    head = runner.model.lm_head
    with pytest.raises(RuntimeError):
        old_processor.forward(head, hidden, head.bias)
    # A second update may not disguise the possibly partial source model.
    with pytest.raises(RuntimeError):
        reload_weights(runner, weights_path="retry")
    assert native_calls == ["replacement"]
    load(runner)
    assert old.closed
    assert "_get_logits" not in vars(old_processor)
    assert runner._gr_npu_stage_projection is not old
    assert runner._gr_npu_stage_projection.snapshot()["weights_valid"]
    head = runner.model.lm_head
    actual = runner.model.logits_processor.forward(head, hidden, head.bias)
    full = torch.tanh(F.linear(hidden, head.weight, head.bias) / 13.0) * 13.0 * 0.7
    torch.testing.assert_close(actual[:, :2], full[:, [17, 100]])
    make_shutdown_with_constraint_backend()(runner)
    assert all(backend.close_calls == 1 for backend in lifecycle.backends)


pytestmark = pytest.mark.cpu_test
