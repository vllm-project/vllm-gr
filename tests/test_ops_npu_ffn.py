# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-independent contracts for the opt-in FFN adapter.

Real device execution and graph replay are covered by test_npu_fused_ffn.py.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm_gr.ops.npu import ffn


@pytest.mark.parametrize("enabled", [None, "0"])  # type: ignore[misc]
def test_disabled_ffn_keeps_native_worker_and_does_not_load_extension(
    monkeypatch: pytest.MonkeyPatch, enabled: str | None
) -> None:
    if enabled is None:
        monkeypatch.delenv(ffn.ENV_ENABLE, raising=False)
    else:
        monkeypatch.setenv(ffn.ENV_ENABLE, enabled)
    load = Mock(side_effect=AssertionError("disabled FFN tried to load an extension"))
    monkeypatch.setattr(ffn, "_load_backend", load)

    class Worker:
        def load_model(self) -> str:
            return "native"

    native = Worker.load_model
    assert not ffn.install_npu_fused_ffn_worker_hook(Worker)
    assert Worker.load_model is native
    runner = SimpleNamespace()
    ffn.install_npu_fused_ffn(runner)
    assert runner._gr_npu_fused_ffn_bindings == ()
    assert runner._gr_npu_fused_ffn_backend is None
    load.assert_not_called()


@pytest.mark.parametrize("load_fails", [False, True])  # type: ignore[misc]
def test_worker_load_hook_preserves_order_errors_and_idempotence(
    monkeypatch: pytest.MonkeyPatch, load_fails: bool
) -> None:
    monkeypatch.setenv(ffn.ENV_ENABLE, "1")
    previous_runner = object()
    loaded_runner = object()
    result = object()
    events: list[tuple[str, Any]] = []

    class Worker:
        def __init__(self) -> None:
            self.model_runner = previous_runner

        def load_model(self, marker: object, *, enabled: bool) -> object:
            assert marker is result and enabled is True
            events.append(("native_load", self.model_runner))
            if load_fails:
                raise RuntimeError("native model load failed")
            self.model_runner = loaded_runner
            return result

    monkeypatch.setattr(ffn, "_restore_bindings", lambda runner: events.append(("restore", runner)))
    monkeypatch.setattr(
        ffn, "install_npu_fused_ffn", lambda runner: events.append(("bind", runner))
    )
    native = Worker.load_model
    assert ffn.install_npu_fused_ffn_worker_hook(Worker)
    wrapped = Worker.load_model
    assert wrapped.__wrapped__ is native  # type: ignore[attr-defined]
    assert not ffn.install_npu_fused_ffn_worker_hook(Worker)
    assert Worker.load_model is wrapped
    worker = Worker()
    if load_fails:
        with pytest.raises(RuntimeError, match="native model load failed"):
            worker.load_model(result, enabled=True)
        assert worker.model_runner is previous_runner
        assert events == [("restore", previous_runner), ("native_load", previous_runner)]
    else:
        assert worker.load_model(result, enabled=True) is result
        assert worker.model_runner is loaded_runner
        assert events == [
            ("restore", previous_runner),
            ("native_load", previous_runner),
            ("bind", loaded_runner),
        ]


@pytest.mark.parametrize("rows", [128, 256, 512, 1024, 4096, 16384])  # type: ignore[misc]
@pytest.mark.parametrize("capturing", [False, True, None])  # type: ignore[misc]
def test_runtime_route_does_not_fuse_excluded_capture_shapes(
    rows: int, capturing: bool | None
) -> None:
    query = Mock(return_value=capturing) if capturing is not None else None
    binding = SimpleNamespace(is_capturing=query)
    expected = rows in (512, 4096, 16384) or (rows == 128 and capturing is False)
    assert ffn._route_supported(torch.empty((rows, 1)), binding) is expected
    if rows == 128 and query is not None:
        query.assert_called_once_with()
    elif query is not None:
        query.assert_not_called()


@pytest.mark.parametrize("rows", [128, 1024])  # type: ignore[misc]
def test_unsupported_input_runs_complete_original_mlp(
    monkeypatch: pytest.MonkeyPatch, rows: int
) -> None:
    import vllm.forward_context as forward_context

    # CPU tensors exercise the real runtime rejection and original callback;
    # they must never reach the NPU-only extension.
    if "torch_npu" not in sys.modules:
        monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    native = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4))
    x = torch.linspace(-1, 1, rows * 4).reshape(rows, 4)
    expected = native(x)
    binding = SimpleNamespace(
        layer=SimpleNamespace(gate_up_proj=SimpleNamespace(weight=native[0].weight)),
        original_forward=native,
    )
    layer = SimpleNamespace(_gr_npu_fused_ffn_binding=binding)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"mlp": layer}),
    )
    kernel = Mock(side_effect=AssertionError("unsupported input reached the fused kernel"))
    monkeypatch.setattr(ffn, "_KERNEL", kernel)
    torch.testing.assert_close(ffn._ffn_impl(x, "mlp"), expected, atol=0, rtol=0)
    kernel.assert_not_called()


def test_eager_configuration_never_prepares_optional_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ffn.ENV_ENABLE, "1")
    prepare = Mock(side_effect=AssertionError("eager configuration prepared an operator"))
    monkeypatch.setattr(ffn, "_prepare_operator", prepare)
    original_config: dict[str, Any] = {"caller": object()}
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=True), additional_config=original_config
        )
    )
    ffn.install_npu_fused_ffn(runner)
    assert runner.vllm_config.additional_config is original_config
    assert runner._gr_npu_fused_ffn_bindings == ()
    prepare.assert_not_called()


def test_restore_is_instance_local_and_preserves_later_owners() -> None:
    caller_value = object()
    previous = {"caller": caller_value}
    config = SimpleNamespace(additional_config=previous)
    context: dict[str, Any] = {"attention": object()}
    first = SimpleNamespace()
    second = SimpleNamespace()
    bindings = []
    for name, layer in (("first", first), ("second", second)):
        fused = object()
        layer.forward = fused
        binding = ffn._Binding(layer, name, context, lambda x: x, ffn._MISSING, fused, None, None)
        setattr(layer, ffn._BINDING_ATTR, binding)
        context[name] = layer
        bindings.append(binding)
    runner = SimpleNamespace(_gr_npu_fused_ffn_bindings=tuple(bindings))
    ffn._publish_compile_marker(runner, config, previous, bindings)
    assert config.additional_config is not previous
    assert config.additional_config["caller"] is caller_value
    assert ffn._COMPILE_MARKER_KEY not in previous

    # Other users may add config or replace a layer's forward after binding.
    later_forward = object()
    second.forward = later_forward
    config.additional_config["later"] = caller_value
    ffn._restore_bindings(runner)
    assert not hasattr(first, "forward")
    assert second.forward is later_forward
    assert set(context) == {"attention"}
    assert config.additional_config == {"caller": caller_value, "later": caller_value}
    assert all(not hasattr(layer, ffn._BINDING_ATTR) for layer in (first, second))
    assert runner._gr_npu_fused_ffn_bindings == ()
    assert runner._gr_npu_fused_ffn_backend is None


def test_missing_backend_is_rejected_before_extension_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    missing = tmp_path / "missing-libgr_ffn_opapi.so"
    monkeypatch.setenv("GR_FFN_OPAPI_LIBRARY", str(missing))
    with pytest.raises(ImportError, match="existing absolute path"):
        ffn._load_backend()


def test_backend_preflight_resolves_before_any_kernel_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    library = tmp_path / "libgr_ffn_opapi.so"
    library.touch()
    monkeypatch.setenv("GR_FFN_OPAPI_LIBRARY", str(library))
    kernel = Mock(side_effect=AssertionError("preflight executed the fused kernel"))
    preflight = Mock(return_value=None)
    extension = ModuleType("gr_ffn_ops")
    extension.fused_gate_up_swiglu = kernel  # type: ignore[attr-defined]
    extension.preflight = preflight  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gr_ffn_ops", extension)

    assert ffn._load_backend() is kernel

    preflight.assert_called_once_with()
    kernel.assert_not_called()


@pytest.mark.parametrize(  # type: ignore[misc]
    "failure",
    ["old_extension", "uncallable_preflight", "dependency", "operator_symbol", "workspace_symbol"],
)
def test_backend_preflight_failure_keeps_complete_native_installation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, failure: str
) -> None:
    from vllm.model_executor.models import qwen2

    class NativeMLP(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = torch.nn.Linear(4, 8, bias=False)
            self.down_proj = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            return self.down_proj(torch.nn.functional.silu(gate) * up)

    monkeypatch.setenv(ffn.ENV_ENABLE, "1")
    library = tmp_path / "libgr_ffn_opapi.so"
    library.touch()
    monkeypatch.setenv("GR_FFN_OPAPI_LIBRARY", str(library))
    monkeypatch.setattr(ffn, "_OPERATOR_LIBRARY", None)
    monkeypatch.setattr(ffn, "_KERNEL", None)
    # Only device/config eligibility is simulated. The actual preparation,
    # backend loading, installation and rollback paths remain under test.
    monkeypatch.setattr(ffn, "_graph_configuration_reason", lambda config: None)
    monkeypatch.setattr(ffn, "_layer_reason", lambda *args: None)
    monkeypatch.setattr(ffn, "_hardware_reason", lambda *args: None)
    monkeypatch.setattr(qwen2, "Qwen2MLP", NativeMLP)
    monkeypatch.setattr(
        ffn,
        "_current_prefetch_method",
        lambda: SimpleNamespace(mlp=SimpleNamespace(enable=False, is_active_this_forward=False)),
    )
    npu = ModuleType("torch_npu")
    npu.npu = SimpleNamespace(is_current_stream_capturing=lambda: False)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    kernel = Mock(side_effect=AssertionError("failed backend executed the fused kernel"))
    extension = ModuleType("gr_ffn_ops")
    extension.fused_gate_up_swiglu = kernel  # type: ignore[attr-defined]
    preflight = None
    if failure == "uncallable_preflight":
        extension.preflight = None  # type: ignore[attr-defined]
    elif failure != "old_extension":
        message = {
            "dependency": "Cannot load Gr FFN OPAPI library: missing dependency or incompatible ABI",
            "operator_symbol": "Missing Gr FFN API symbol: aclnnGrGateUpSwiGlu",
            "workspace_symbol": "Missing Gr FFN API symbol: aclnnGrGateUpSwiGluGetWorkspaceSize",
        }[failure]
        preflight = Mock(side_effect=RuntimeError(message))
        extension.preflight = preflight  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gr_ffn_ops", extension)
    registration = Mock(side_effect=AssertionError("failed preflight registered a custom operator"))
    monkeypatch.setattr(torch.library, "Library", registration)

    model = torch.nn.Sequential(NativeMLP(), NativeMLP())
    original_forwards = tuple(layer.forward for layer in model)
    x = torch.linspace(-1, 1, 16).reshape(4, 4)
    expected = model(x)
    original_config: dict[str, Any] = {"caller": object()}
    original_context: dict[str, Any] = {"attention": object()}
    runner = SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(
            additional_config=original_config,
            compilation_config=SimpleNamespace(static_forward_context=original_context),
        ),
    )

    ffn.install_npu_fused_ffn(runner)

    assert runner._gr_npu_fused_ffn_bindings == ()
    assert runner._gr_npu_fused_ffn_backend is None
    assert runner.vllm_config.additional_config is original_config
    assert ffn._COMPILE_MARKER_KEY not in original_config
    assert not hasattr(runner, ffn._COMPILE_OWNER_ATTR)
    assert set(original_context) == {"attention"}
    for layer, original in zip(model, original_forwards, strict=True):
        assert layer.forward == original
        assert "forward" not in vars(layer)
        assert not hasattr(layer, ffn._BINDING_ATTR)
    torch.testing.assert_close(model(x), expected, atol=0, rtol=0)
    assert ffn._KERNEL is None and ffn._OPERATOR_LIBRARY is None
    if preflight is not None:
        preflight.assert_called_once_with()
    registration.assert_not_called()
    kernel.assert_not_called()
