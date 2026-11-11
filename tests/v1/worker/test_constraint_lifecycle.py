# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.config import ConstraintTableConfig, GRConfig
from vllm_gr.config.parser import gr_config_to_mapping
from vllm_gr.v1.worker.model_runner_patch import (
    _runner_vocab_size,
    make_load_model_with_constraint_backend,
    make_shutdown_with_constraint_backend,
)

_ARTIFACT_DIGEST = "sha256:" + "1" * 64
_TOKENIZER_DIGEST = "sha256:" + "2" * 64


class _Backend:
    def __init__(self, *, device_memory_bytes: int = 32) -> None:
        self.info = SimpleNamespace(device_memory_bytes=device_memory_bytes)
        self.close_calls = 0
        self.vocab_slice: tuple[int, int] | None = None

    def close(self) -> None:
        self.close_calls += 1


def _vllm_config(
    *,
    enabled: bool = True,
    beam_max_width: int | None = 4,
    max_top_k: int = 4,
) -> SimpleNamespace:
    config = GRConfig(
        beam_max_width=beam_max_width,
        beam_max_decode_steps=3,
        constraint_table=ConstraintTableConfig(
            max_top_k=max_top_k,
            enabled=enabled,
            path="/models/constraint-table.safetensors" if enabled else None,
            artifact_digest=_ARTIFACT_DIGEST if enabled else None,
            tokenizer_digest=_TOKENIZER_DIGEST if enabled else None,
        ),
    )
    return SimpleNamespace(
        additional_config={"vllm_gr": gr_config_to_mapping(config)},
    )


def _runner(
    *,
    enabled: bool = True,
    beam_max_width: int | None = 4,
    max_top_k: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=torch.nn.Module(),
        constraint_backend=None,
        vllm_config=_vllm_config(
            enabled=enabled,
            beam_max_width=beam_max_width,
            max_top_k=max_top_k,
        ),
        model_config=SimpleNamespace(
            dtype=torch.float16,
            get_vocab_size=lambda: 32000,
        ),
        device=torch.device("cpu"),
        model_memory_usage=0,
    )


def test_runner_vocab_size_prefers_upstream_api() -> None:
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            vocab_size=8,
            get_vocab_size=lambda: 32000,
        )
    )
    assert _runner_vocab_size(runner) == 32000


@pytest.mark.parametrize("max_batch", [1, 4])  # type: ignore[misc]
def test_weights_then_readiness_then_publication_and_memory(
    monkeypatch: pytest.MonkeyPatch,
    max_batch: int,
) -> None:
    runner = _runner()
    backend = _Backend(device_memory_bytes=37)
    events: list[str] = []

    def original(self: SimpleNamespace) -> str:
        assert self.constraint_backend is None
        events.append("weights")
        self.model_memory_usage = 100
        return "loaded"

    def load_backend(**kwargs: Any) -> _Backend:
        events.append("readiness")
        assert runner.constraint_backend is None
        assert kwargs["max_rows"] == 4 * max_batch
        assert kwargs["vocab_size"] == 32000
        assert kwargs["dtype"] is torch.float16
        return backend

    monkeypatch.setattr(
        "vllm_gr.constraints.loader.load_constraint_backend",
        load_backend,
    )

    def prepare_backend(owner: Any, candidate: Any) -> None:
        assert owner is runner
        assert candidate is backend
        assert runner.constraint_backend is None
        events.append("prepare")

    wrapped = (
        make_load_model_with_constraint_backend(original)
        if max_batch == 1
        else make_load_model_with_constraint_backend(
            original, max_batch=max_batch, prepare_backend=prepare_backend
        )
    )
    assert wrapped(runner) == "loaded"
    assert events == ["weights", "readiness"] + (["prepare"] if max_batch > 1 else [])
    assert runner.constraint_backend.get() is backend
    assert runner.model_memory_usage == 137


@pytest.mark.l1  # type: ignore[misc]
def test_preparation_failure_closes_unpublished_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    backend = _Backend()
    monkeypatch.setattr("vllm_gr.constraints.loader.load_constraint_backend", lambda **kw: backend)

    def fail_preparation(owner: Any, candidate: Any) -> None:
        assert owner.constraint_backend is None
        assert candidate is backend
        raise RuntimeError("kernel warmup failed")

    with pytest.raises(RuntimeError, match="kernel warmup failed"):
        make_load_model_with_constraint_backend(
            lambda self: None, prepare_backend=fail_preparation
        )(runner)
    assert runner.constraint_backend is None
    assert backend.close_calls == 1


def test_loader_failure_never_publishes_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()

    def original(self: SimpleNamespace) -> None:
        self.model_memory_usage = 100

    def fail_readiness(**kwargs: Any) -> None:
        del kwargs
        assert runner.constraint_backend is None
        raise RuntimeError("readiness failed")

    monkeypatch.setattr(
        "vllm_gr.constraints.loader.load_constraint_backend",
        fail_readiness,
    )

    with pytest.raises(RuntimeError, match="readiness failed"):
        make_load_model_with_constraint_backend(original)(runner)
    assert runner.constraint_backend is None


def test_model_load_failure_closes_old_backend() -> None:
    runner = _runner()
    old_backend = _Backend()
    runner.constraint_backend = old_backend

    def fail_weights(self: SimpleNamespace) -> None:
        assert self.constraint_backend is None
        raise RuntimeError("weights failed")

    with pytest.raises(RuntimeError, match="weights failed"):
        make_load_model_with_constraint_backend(fail_weights)(runner)
    assert runner.constraint_backend is None
    assert old_backend.close_calls == 1


def test_complete_reload_and_shutdown_close_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    created: list[_Backend] = []

    def original_load(self: SimpleNamespace) -> None:
        self.model_memory_usage = 10

    def load_backend(**kwargs: Any) -> _Backend:
        del kwargs
        backend = _Backend()
        created.append(backend)
        return backend

    monkeypatch.setattr(
        "vllm_gr.constraints.loader.load_constraint_backend",
        load_backend,
    )
    load = make_load_model_with_constraint_backend(original_load)
    load(runner)
    load(runner)

    assert len(created) == 2
    assert created[0].close_calls == 1
    assert runner.constraint_backend.get() is created[1]

    native_shutdown_calls: list[int] = []

    def native_shutdown(self: SimpleNamespace) -> None:
        del self
        native_shutdown_calls.append(1)

    shutdown = make_shutdown_with_constraint_backend(native_shutdown)
    shutdown(runner)
    shutdown(runner)
    assert created[1].close_calls == 1
    assert runner.constraint_backend is None
    assert len(native_shutdown_calls) == 2


def test_shutdown_is_installed_when_runner_has_no_native_method() -> None:
    runner = _runner()
    backend = _Backend()
    runner.constraint_backend = backend

    shutdown = make_shutdown_with_constraint_backend()
    shutdown(runner)
    shutdown(runner)

    assert backend.close_calls == 1
    assert runner.constraint_backend is None


@pytest.mark.parametrize("shutdown_fails", [False, True])  # type: ignore[misc]
def test_shutdown_releases_model_bytecode_hooks_only(shutdown_fails: bool) -> None:
    from torch._dynamo.convert_frame import _bytecode_hooks, register_bytecode_hook
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    class CompiledLayer(
        torch.nn.Linear,  # type: ignore[misc]
        TorchCompileWithNoGuardsWrapper,  # type: ignore[misc]
    ):
        pass

    runner = _runner()
    runner.model = torch.nn.Sequential(CompiledLayer(2, 2))
    layer_ref = weakref.ref(runner.model[0])
    weight_ref = weakref.ref(runner.model[0].weight)
    owned = register_bytecode_hook(runner.model[0].bytecode_hook)
    runner.model[0]._bytecode_hook_handle = owned
    other_model = CompiledLayer(2, 2)
    foreign = register_bytecode_hook(other_model.bytecode_hook)
    other_model._bytecode_hook_handle = foreign
    unrelated = register_bytecode_hook(lambda old, new: None)

    def native_shutdown(self: SimpleNamespace) -> None:
        self.model = None
        if shutdown_fails:
            raise RuntimeError("native shutdown failed")

    try:
        shutdown = make_shutdown_with_constraint_backend(native_shutdown)
        for _ in range(2):
            if shutdown_fails:
                with pytest.raises(RuntimeError, match="native shutdown failed"):
                    shutdown(runner)
            else:
                shutdown(runner)
        gc.collect()
        assert layer_ref() is None
        assert weight_ref() is None
        assert owned.id not in _bytecode_hooks
        assert _bytecode_hooks[foreign.id].__self__ is other_model
        assert unrelated.id in _bytecode_hooks
    finally:
        for handle in (owned, foreign, unrelated):
            handle.remove()


def test_disabled_configuration_never_calls_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(enabled=False)

    def forbidden(**kwargs: Any) -> None:
        raise AssertionError(f"disabled configuration called loader: {kwargs}")

    monkeypatch.setattr(
        "vllm_gr.constraints.loader.load_constraint_backend",
        forbidden,
    )

    def original(self: SimpleNamespace) -> str:
        self.model_memory_usage = 9
        return "loaded"

    assert make_load_model_with_constraint_backend(original)(runner) == "loaded"
    assert runner.constraint_backend is None
    assert runner.model_memory_usage == 9


@pytest.mark.parametrize(  # type: ignore[misc]
    ("beam_max_width", "max_top_k", "message"),
    [
        (None, 4, "beam_max_width"),
        (5, 4, "max_top_k"),
    ],
)
def test_canonical_capacity_validation_happens_after_weights(
    beam_max_width: int | None,
    max_top_k: int,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(
        beam_max_width=beam_max_width,
        max_top_k=max_top_k,
    )
    events: list[str] = []

    def original(self: SimpleNamespace) -> None:
        del self
        events.append("weights")

    def forbidden(**kwargs: Any) -> None:
        raise AssertionError(f"invalid capacity called loader: {kwargs}")

    monkeypatch.setattr(
        "vllm_gr.constraints.loader.load_constraint_backend",
        forbidden,
    )

    with pytest.raises(ValueError, match=message):
        make_load_model_with_constraint_backend(original)(runner)
    assert events == ["weights"]
    assert runner.constraint_backend is None


def test_canonical_cuda_tp_falls_back_to_full_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.platforms import current_platform

    if current_platform.device_type == "npu":
        pytest.skip("CUDA vocabulary projection slicing is not used on Ascend NPU")

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    from vllm_gr.config import ConstraintTableBackendEnum
    from vllm_gr.ops.sliced_logits import get_vocab_slice_config, set_vocab_slice_config
    from vllm_gr.ops.sliced_logits_patch import apply_sliced_logits_patch

    apply_sliced_logits_patch()
    runner = _runner()
    runner.vllm_config.additional_config["vllm_gr"]["constraint_table"]["backend"] = "cuda"
    runner.vllm_config.parallel_config = SimpleNamespace(tensor_parallel_size=2)
    backend = _Backend()
    backend.info.backend = ConstraintTableBackendEnum.CUDA
    backend.vocab_slice = (3, 5)
    monkeypatch.setattr(
        "vllm_gr.constraints.loader.load_constraint_backend", lambda **kwargs: backend
    )
    gathered: list[torch.Tensor] = []

    def gather(logits: torch.Tensor) -> torch.Tensor:
        gathered.append(logits)
        return logits

    def weights(runner: Any) -> str:
        with set_current_vllm_config(VllmConfig()):
            runner.model = torch.nn.Sequential(LogitsProcessor(32000))
        runner.model[0]._gather_logits = gather
        return "loaded"

    head = torch.nn.Linear(8, 256, bias=False)
    head.tp_size = 2
    head.quant_method = SimpleNamespace(
        apply=lambda layer, hidden, bias=None: torch.nn.functional.linear(
            hidden, layer.weight, bias
        )
    )
    hidden = torch.randn(2, 8)
    old = get_vocab_slice_config()
    set_vocab_slice_config(99, 5)
    try:
        assert make_load_model_with_constraint_backend(weights)(runner) == "loaded"
        assert runner.constraint_backend.get() is backend
        assert backend.vocab_slice is None
        assert runner.model[0]._gr_vocab_slice is None
        actual = runner.model[0]._get_logits(hidden, head, None)
        torch.testing.assert_close(actual, head(hidden))
        assert len(gathered) == 1 and gathered[0].shape == (2, 256)
        assert not hasattr(head, "_sliced_weight")
        assert get_vocab_slice_config()[:2] == (99, 5)
    finally:
        make_shutdown_with_constraint_backend()(runner)
        set_vocab_slice_config(*old)


def test_two_runners_keep_independent_projection_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.platforms import current_platform

    if current_platform.device_type == "npu":
        pytest.skip("CUDA vocabulary projection slicing is not used on Ascend NPU")

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    from vllm_gr.config import ConstraintTableBackendEnum
    from vllm_gr.constraints import loader
    from vllm_gr.ops.sliced_logits import get_vocab_slice_config, set_vocab_slice_config
    from vllm_gr.ops.sliced_logits_patch import apply_sliced_logits_patch

    apply_sliced_logits_patch()
    backends = []
    for window in ((3, 5), (40, 5), (0, 32000)):
        backend = _Backend()
        backend.info.backend = ConstraintTableBackendEnum.CUDA
        backend.vocab_slice = window
        backends.append(backend)
    queue = iter(backends)
    monkeypatch.setattr(loader, "load_constraint_backend", lambda **kwargs: next(queue))

    def weights(runner: Any) -> None:
        runner.model_memory_usage = 0
        with set_current_vllm_config(VllmConfig()):
            runner.model = torch.nn.Sequential(LogitsProcessor(32000))
        runner.model[0]._gather_logits = lambda logits: logits

    load = make_load_model_with_constraint_backend(weights)
    close = make_shutdown_with_constraint_backend()
    a, b, full = _runner(), _runner(), _runner()
    old = get_vocab_slice_config()
    set_vocab_slice_config(99, 5)
    try:
        load(a)
        load(b)
        load(full)
        head = torch.nn.Linear(8, 256, bias=False)
        head.tp_size = 1
        head.quant_method = type(
            "UnquantizedEmbeddingMethod",
            (),
            {
                "process_weights_after_loading": lambda self, layer: None,
                "apply": lambda self, layer, hidden, bias=None: torch.nn.functional.linear(
                    hidden, layer.weight, bias
                ),
            },
        )()
        hidden = torch.randn(2, 8)
        expected = head(hidden)
        for runner, offset in ((a, 3), (b, 40)):
            actual = runner.model[0]._get_logits(hidden, head, None)
            torch.testing.assert_close(actual, expected[:, offset : offset + 5])
        close(b)
        torch.testing.assert_close(a.model[0]._get_logits(hidden, head, None), expected[:, 3:8])
        torch.testing.assert_close(full.model[0]._get_logits(hidden, head, None), expected)
        assert full.model[0]._gr_vocab_slice is None
        assert get_vocab_slice_config()[:2] == (99, 5)
        close(a)
        assert backends[0].close_calls == backends[1].close_calls == 1
    finally:
        close(a)
        close(b)
        close(full)
        set_vocab_slice_config(*old)


def test_dispatcher_import_without_cuda_or_triton() -> None:
    import subprocess
    import sys

    code = """
import importlib.abc
import sys
import torch
import vllm.logger
class BlockCUDA(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'triton' or fullname.startswith('triton.') or fullname.startswith('vllm_gr.ops.cuda') or fullname == 'vllm_gr.constraints.cuda_backend':
            raise ImportError('blocked CUDA/Triton dependency: ' + fullname)
sys.meta_path.insert(0, BlockCUDA())
import vllm_gr.plugin
from vllm_gr.v1.worker import model_runner_patch
assert 'vllm_gr.constraints.cuda_backend' not in sys.modules
assert 'vllm_gr.ops.cuda.rec_constrained_topk' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


@pytest.mark.parametrize("v2", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("operation", ["reload", "shutdown"])  # type: ignore[misc]
def test_runtime_cleanup_is_opt_in_even_when_native_operation_fails(
    v2: bool, operation: str
) -> None:
    from unittest.mock import Mock

    from vllm_gr.v1.worker.gpu_model_runner_v2_resources import close_v2_runtime

    runner = _runner()
    backend = runner.constraint_backend = _Backend()
    graph = runner._beam_decode_graph = SimpleNamespace(uninstall=Mock(), clear=Mock())
    runtime = {
        "_gr_input_buffers": object(),
        "_gr_gpu_stage_runner": object(),
        "_gr_decode_capture_failures": {2: "refused"},
        "_gr_runtime_decode_capture_failures": {(1, 4): "refused"},
        "_gr_async_decode_graph_capture_failed": True,
        "_gr_async_decode_graph_capture_error": "refused",
    }
    for name, value in runtime.items():
        setattr(runner, name, value)

    def native(self: Any) -> None:
        assert self.constraint_backend is None
        assert backend.close_calls == 1
        assert (self._beam_decode_graph is None) is v2
        raise RuntimeError("native failure")

    options = {"close_runtime": close_v2_runtime} if v2 else {}
    wrapped = (
        make_load_model_with_constraint_backend(native, **options)
        if operation == "reload"
        else make_shutdown_with_constraint_backend(native, **options)
    )
    with pytest.raises(RuntimeError, match="native failure"):
        wrapped(runner)
    if v2:
        graph.uninstall.assert_called_once()
        graph.clear.assert_called_once()
        assert all(not hasattr(runner, name) for name in runtime)
    else:
        graph.uninstall.assert_not_called()
        graph.clear.assert_not_called()
        assert all(getattr(runner, name) is value for name, value in runtime.items())


pytestmark = pytest.mark.cpu_test
