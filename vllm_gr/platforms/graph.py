# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Per-model graph admission, native dispatch and generation cleanup."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from inspect import isabstract
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from vllm_gr.v1.worker.prefill_graph_common import BasePrefillGraphRunner

    from .plan import BackendPlan

T = TypeVar("T")


def native_graph_counters(runner: Any) -> dict[str, Any]:
    """Counters shared by both native worker hooks, including unbound legacy runners."""
    return {
        "captured_widths": sorted(int(width) for width in getattr(runner, "_beam_graph_keys", ())),
        "full_dispatches": int(getattr(runner, "_beam_graph_full_dispatch_count", 0)),
        "gr_prefill_dispatches": dict(getattr(runner, "_gr_prefill_graph_dispatches", {})),
    }


class GraphExecution:
    """The native graph owns addresses. The runner's service owns their generation."""

    implementation_id = "native"

    def __init__(self, runner: Any, plan: BackendPlan | None) -> None:
        self.runner = runner
        self.plan = plan
        choices = plan.choices.to_dict() if plan else {}
        eager = bool(getattr(getattr(runner, "model_config", None), "enforce_eager", False))
        if plan is None:
            from vllm_gr.config import GRConfig

            config = getattr(runner, "_vllm_gr_config", None)
            if config is None:
                config = GRConfig.from_vllm_config(getattr(runner, "vllm_config", None))
            self.decode_enabled = not eager and bool(getattr(config, "beam_graph_enabled", False))
            from vllm_gr.config.models import deployment_for

            self.prefill_enabled = (
                not eager
                and deployment_for(getattr(runner, "vllm_config", None)).prefill_graph_enabled
            )
        else:
            mode = choices.get("graph", "eager")
            if mode not in ("eager", "capture-with-eager-fallback"):
                raise ValueError(f"Unsupported admitted GR graph policy: {mode}")
            self.decode_enabled = mode != "eager"
            self.prefill_enabled = bool(
                choices.get(
                    "prefill_graph",
                    not eager
                    and plan.request.config.to_dict()
                    .get("deployment", {})
                    .get("prefill_graph_enabled", True),
                )
            )
        self.closed = False
        self._captures: Counter[str] = Counter()
        self._replays: Counter[str] = Counter()
        self._fallbacks: dict[str, Counter[str]] = {}

    def _check(self, phase: str) -> None:
        if self.closed:
            raise RuntimeError("Graph belongs to a closed model generation")
        if phase not in ("decode", "prefill"):
            raise ValueError(f"Unknown GR graph phase: {phase}")
        if not (self.decode_enabled if phase == "decode" else self.prefill_enabled):
            raise RuntimeError(f"Graph execution is disabled for {phase} in this generation")

    def capture(self, phase: str, native: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        self._check(phase)
        result = native(*args, **kwargs)
        self._captures[phase] += 1
        return result

    def captured(self, phase: str) -> None:
        self._check(phase)
        self._captures[phase] += 1

    def replay(self, phase: str, native: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        self._check(phase)
        result = native(*args, **kwargs)
        self._replays[phase] += 1
        return result

    def replayed(self, phase: str) -> None:
        self._check(phase)
        self._replays[phase] += 1

    def fallback(self, phase: str, reason: str) -> None:
        if self.closed:
            raise RuntimeError("Graph belongs to a closed model generation")
        if phase not in self._fallbacks:
            self._fallbacks[phase] = Counter()
        self._fallbacks[phase][reason] += 1

    def stats(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation_id,
            "decode_enabled": self.decode_enabled,
            "prefill_enabled": self.prefill_enabled,
            "captures": dict(self._captures),
            "replays": dict(self._replays),
            "fallbacks": {phase: dict(reasons) for phase, reasons in self._fallbacks.items()},
        }

    @staticmethod
    def prefill_runner_class() -> Any:
        raise NotImplementedError("Provider has no native Prefill graph runner")

    def check_idle(self) -> None:
        if self.closed:
            return
        runner = self.runner
        if getattr(runner, "_beam_sessions", None):
            raise RuntimeError("Graph cleanup requires released Beam sessions")
        context = getattr(runner, "_beam_context", None)
        if getattr(context, "_gr_execution_owner", None) is not None:
            raise RuntimeError("Graph cleanup requires a released Beam context")

    def _release(self) -> None:
        pass

    def close(self) -> None:
        if not self.closed:
            self.check_idle()
            self._release()
            self.closed = True
            self.runner = None


def existing_graph(runner: Any) -> Any:
    graph = getattr(getattr(runner, "_gr_backend_binding", None), "graph", None)
    return graph if graph is not None else getattr(runner, "_gr_native_graph", None)


def shared_prefill_runner_class(graph: Any) -> type[BasePrefillGraphRunner]:
    """Resolve the selected runner without allocating its device buffers."""
    factory = getattr(graph, "prefill_runner_class", None)
    if callable(factory):
        try:
            runner_class = factory()
        except NotImplementedError as exc:
            raise RuntimeError(
                "Shared Prefill graph requires a concrete prefill_runner_class()"
            ) from exc
        if (
            isinstance(runner_class, type)
            and not isabstract(runner_class)
            and all(
                callable(getattr(runner_class, method, None))
                for method in ("capture_graphs", "execute_model_prefill")
            )
        ):
            return runner_class
    raise RuntimeError(
        "Shared Prefill graph requires prefill_runner_class() returning a concrete runner class"
    )


def graph_for(runner: Any, native_factory: Any = None) -> Any:
    """Resolve once for an unplanned native runner, use its admitted service otherwise."""
    graph = existing_graph(runner)
    if graph is None:
        from .loader import activate_current_backend
        from .plan import read_backend_plan

        plan = read_backend_plan(getattr(runner, "vllm_config", None))
        implementation = plan.required_services.get("graph") if plan else None
        if plan is not None and plan.requirements is not None and not implementation:
            raise RuntimeError("This runner plan did not admit a graph service")
        if plan is not None and implementation:
            # Native model initialization may ask for graphs before the full
            # bundle is published. Use the same admitted provider and adopt
            # this exact owner at readiness, including third-party services.
            factory = getattr(activate_current_backend(), "create_graph_executor", None)
            if not callable(factory):
                raise RuntimeError("Backend did not provide its admitted graph executor")
            graph = factory(plan=plan, runner=runner)
            if getattr(graph, "implementation_id", None) != implementation:
                graph.close()
                raise RuntimeError("Worker graph implementation differs from the admitted plan")
        else:
            factory = native_factory or getattr(
                activate_current_backend(), "native_graph_factory", None
            )
            if factory is None:
                raise RuntimeError("Runner has no installed GR graph service")
            graph = factory(runner, plan)
        runner._gr_native_graph_fingerprint = plan.fingerprint if implementation and plan else None
        runner._gr_native_graph = graph
    return graph


def bind_graph(plan: BackendPlan, runner: Any, factory: type[GraphExecution]) -> GraphExecution:
    graph = getattr(runner, "_gr_native_graph", None)
    if isinstance(graph, factory) and graph.plan == plan and not graph.closed:
        return graph
    if graph is not None:
        graph.check_idle()
        graph.close()
        del runner._gr_native_graph
    return factory(runner, plan)
