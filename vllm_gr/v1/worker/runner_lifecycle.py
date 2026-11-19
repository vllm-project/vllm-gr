# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Bind the selected native runner's already-installed methods to its engine plan."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial, wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_gr.platforms.plan import BackendPlan


@dataclass
class NativeRunnerAdapter:
    plan: BackendPlan
    implementation_id: str
    execute_model: Callable[..., Any]
    prepare_inputs: Callable[..., Any]
    sample_tokens: Callable[..., Any] | None

    def close(self) -> None:
        # Drop bound callables and their strong references to the native runner.
        self.execute_model = self.prepare_inputs = self.sample_tokens = _closed


def _closed(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("Runner adapter has been closed")


def _route_method(original: Any, phase: str) -> Any:
    @wraps(original)
    def routed(runner: Any, *args: Any, **kwargs: Any) -> Any:
        binding = getattr(runner, "_gr_backend_binding", None)
        if binding is None:
            # Native warmup before model-load binding and unplanned runners.
            return original(runner, *args, **kwargs)
        return getattr(binding.runner_adapter, phase)(*args, **kwargs)

    setattr(routed, "_gr_runner_implementation", original)
    return routed


def install_runner_lifecycle(runner_class: Any, implementation_id: str, *, prepare: str) -> None:
    """Keep native signatures and the execute/sample completion boundaries separate."""
    marker = runner_class.__dict__.get("_gr_runner_implementation_id")
    if marker == implementation_id:
        if not has_runner_lifecycle(runner_class, implementation_id):
            raise RuntimeError(f"Native runner hooks for {implementation_id!r} were replaced")
        return
    if marker is not None:
        raise RuntimeError(f"Native runner already belongs to {marker!r}")
    methods = {"execute_model": "execute_model", "prepare_inputs": prepare}
    if callable(getattr(runner_class, "sample_tokens", None)):
        methods["sample_tokens"] = "sample_tokens"
    wrappers = {}
    for phase, name in methods.items():
        original = getattr(runner_class, name)
        # A subclass can inherit a routed entry point from a native base.
        original = getattr(original, "_gr_runner_implementation", original)
        if not callable(original):
            raise RuntimeError(f"Native runner does not implement {name}()")
        wrappers[name] = _route_method(original, phase)
    for name, wrapper in wrappers.items():
        setattr(runner_class, name, wrapper)
    runner_class._gr_runner_methods = methods
    runner_class._gr_runner_implementation_id = implementation_id


def has_runner_lifecycle(runner_class: Any, implementation_id: str) -> bool:
    if getattr(runner_class, "_gr_runner_implementation_id", None) != implementation_id:
        return False
    return all(
        callable(getattr(getattr(runner_class, name, None), "_gr_runner_implementation", None))
        for name in runner_class._gr_runner_methods.values()
    )


def bind_native_runner(
    plan: BackendPlan, runner: Any, *, implementation_id: str | None = None
) -> NativeRunnerAdapter:
    expected = (
        implementation_id if implementation_id is not None else plan.choices.to_dict()["runner"]
    )
    if getattr(type(runner), "_gr_runner_implementation_id", None) != expected:
        raise RuntimeError("Worker native runner differs from the admitted backend plan")
    bound = {}
    for phase, name in type(runner)._gr_runner_methods.items():
        implementation = getattr(getattr(type(runner), name), "_gr_runner_implementation", None)
        if implementation is None:
            raise RuntimeError(f"Worker is missing the installed {phase} entry point")
        bound[phase] = partial(implementation, runner)
    return NativeRunnerAdapter(
        plan, expected, bound["execute_model"], bound["prepare_inputs"], bound.get("sample_tokens")
    )
