# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""OpenAI serving Constraint Tables injection-once semantics.

``_ensure_constraint_tables_injected`` must await the collective RPC broadcast
before marking the tables injected: a fire-and-forget coroutine would set the
flag while the workers never received the tables, and the Worker raw-logits
hook would then fail at runtime on the first constrained step.

The module under test imports real vLLM symbols at import time, so the vLLM
module graph is faked here to keep this test runnable without a vLLM install.
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _install_fake_vllm_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the vLLM modules serving_models.py imports at module scope."""
    engine = ModuleType("vllm.engine")
    protocol = ModuleType("vllm.engine.protocol")
    setattr(protocol, "EngineClient", type("EngineClient", (), {}))

    entrypoints = ModuleType("vllm.entrypoints")
    openai = ModuleType("vllm.entrypoints.openai")
    models = ModuleType("vllm.entrypoints.openai.models")
    serving = ModuleType("vllm.entrypoints.openai.models.serving")
    setattr(serving, "BaseModelPath", type("BaseModelPath", (), {}))
    setattr(serving, "LoRAModulePath", type("LoRAModulePath", (), {}))
    setattr(serving, "OpenAIServingModels", type("OpenAIServingModels", (), {}))

    for name, module in {
        "vllm.engine": engine,
        "vllm.engine.protocol": protocol,
        "vllm.entrypoints": entrypoints,
        "vllm.entrypoints.openai": openai,
        "vllm.entrypoints.openai.models": models,
        "vllm.entrypoints.openai.models.serving": serving,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _make_engine() -> SimpleNamespace:
    return SimpleNamespace(
        engine_client=SimpleNamespace(),
        _constraint_resource=object(),
        _constraint_tables_injected=False,
        _constraint_injection_lock=asyncio.Lock(),
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "model_config",
    [
        SimpleNamespace(vocab_size=176384),
        SimpleNamespace(get_vocab_size=lambda: 176384),
    ],
)
def test_model_vocab_size_supports_field_and_method(
    monkeypatch: pytest.MonkeyPatch,
    model_config: SimpleNamespace,
) -> None:
    _install_fake_vllm_modules(monkeypatch)

    from vllm_gr.entrypoints.openai import serving_models

    assert serving_models._model_vocab_size(SimpleNamespace(model_config=model_config)) == 176384


@pytest.mark.parametrize(  # type: ignore[misc]
    "model_config",
    [
        None,
        SimpleNamespace(vocab_size=None),
        SimpleNamespace(get_vocab_size=lambda: 0),
        SimpleNamespace(get_vocab_size=lambda: True),
    ],
)
def test_model_vocab_size_rejects_missing_or_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    model_config: SimpleNamespace | None,
) -> None:
    _install_fake_vllm_modules(monkeypatch)

    from vllm_gr.entrypoints.openai import serving_models

    with pytest.raises(ValueError, match="requires a positive"):
        serving_models._model_vocab_size(SimpleNamespace(model_config=model_config))


def test_ensure_constraint_tables_injected_awaits_before_marking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vllm_modules(monkeypatch)

    from vllm_gr.entrypoints.openai import serving_models

    engine = _make_engine()
    events: list[str] = []

    async def fake_inject(engine_client: Any, resource: Any) -> list[dict[str, object]]:
        # While the broadcast is in flight the flag must still be False; only
        # the awaited call may mark injection complete.
        assert engine_client is engine.engine_client
        assert engine._constraint_tables_injected is False
        events.append("inject")
        return [{"digest": "d", "cached": True}]

    monkeypatch.setattr(serving_models, "inject_constraint_resources_async", fake_inject)

    asyncio.run(serving_models._ensure_constraint_tables_injected(engine))

    assert engine._constraint_tables_injected is True
    assert events == ["inject"]

    # Idempotent: a second call must not re-broadcast.
    asyncio.run(serving_models._ensure_constraint_tables_injected(engine))
    assert events == ["inject"]


def test_ensure_constraint_tables_injected_noop_without_constraint_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trie / no-catalog instances must not AttributeError on the global method.

    ``_ensure_constraint_tables_injected`` is mounted on every
    ``OpenAIServingModels`` instance, but only constraint-table instances
    initialize the constraint fields.  Calling it on a bare instance (no
    ``_constraint_resource``, no ``_constraint_injection_lock``) must be a
    no-op — otherwise an ordinary (non-constrained) beam_search crashes here.
    """
    _install_fake_vllm_modules(monkeypatch)

    from vllm_gr.entrypoints.openai import serving_models

    engine = SimpleNamespace()  # no constraint fields, no engine_client
    events: list[str] = []

    async def fake_inject(engine_client: Any, resource: Any) -> list[dict[str, object]]:
        events.append("inject")
        return [{"digest": "d", "cached": True}]

    monkeypatch.setattr(serving_models, "inject_constraint_resources_async", fake_inject)

    asyncio.run(serving_models._ensure_constraint_tables_injected(engine))
    assert events == []


def test_ensure_constraint_tables_injected_is_serialized_by_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vllm_modules(monkeypatch)

    from vllm_gr.entrypoints.openai import serving_models

    inflight = 0
    max_inflight = 0
    calls = 0

    async def fake_inject(engine_client: Any, resource: Any) -> list[dict[str, object]]:
        nonlocal inflight, max_inflight, calls
        calls += 1
        inflight += 1
        max_inflight = max(max_inflight, inflight)
        await asyncio.sleep(0)
        inflight -= 1
        return [{"digest": "d", "cached": True}]

    monkeypatch.setattr(serving_models, "inject_constraint_resources_async", fake_inject)

    async def main() -> Any:
        engine = _make_engine()
        await asyncio.gather(
            serving_models._ensure_constraint_tables_injected(engine),
            serving_models._ensure_constraint_tables_injected(engine),
            serving_models._ensure_constraint_tables_injected(engine),
        )
        return engine

    engine = asyncio.run(main())
    assert engine._constraint_tables_injected is True
    assert calls == 1  # the lock serializes; only one broadcast happens
    assert max_inflight == 1
