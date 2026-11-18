# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Backend configuration policy for tests that exercise shared configuration.

Shared configuration asks the selected backend for its policy instead of
comparing a device name, so a test supplies that policy rather than patching a
platform predicate.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest import mock

#: A policy that behaves like the native CUDA backend.
_CUDA_DEFAULTS: dict[str, Any] = {
    "backend_id": "cuda",
    "promotes_constraint_tables": False,
    "supports_constraint_vocab_slicing": True,
    "requires_persistent_v2_protocol": True,
}

#: A policy that behaves like the Ascend backend.
_ASCEND_DEFAULTS: dict[str, Any] = {
    "backend_id": "ascend",
    "promotes_constraint_tables": True,
    "supports_constraint_vocab_slicing": False,
    "requires_persistent_v2_protocol": False,
}

#: A policy for a host with no accelerator behaviour at all.
_NEUTRAL_DEFAULTS: dict[str, Any] = {
    "backend_id": "other",
    "promotes_constraint_tables": False,
    "supports_constraint_vocab_slicing": False,
    "requires_persistent_v2_protocol": False,
}


def config_service(
    *,
    backend_id: str = "cuda",
    promotes_constraint_tables: bool | None = None,
    supports_constraint_vocab_slicing: bool | None = None,
    requires_persistent_v2_protocol: bool | None = None,
    apply_config_defaults: Any = None,
    validate_unconstrained_runner: Any = None,
    install_engine_core_config_sanitizer: Any = None,
    supports_worker_decision_by_default: bool = True,
) -> Any:
    """Return a configuration policy with overridable, backend-like defaults."""
    defaults = {"ascend": _ASCEND_DEFAULTS, "cuda": _CUDA_DEFAULTS}.get(
        backend_id, _NEUTRAL_DEFAULTS
    )
    overrides = {
        "promotes_constraint_tables": promotes_constraint_tables,
        "supports_constraint_vocab_slicing": supports_constraint_vocab_slicing,
        "requires_persistent_v2_protocol": requires_persistent_v2_protocol,
    }
    answers = {
        name: value if value is not None else defaults[name] for name, value in overrides.items()
    }
    return SimpleNamespace(
        backend_id=backend_id,
        supports_worker_decision_by_default=lambda: supports_worker_decision_by_default,
        apply_config_defaults=apply_config_defaults or (lambda config: None),
        validate_unconstrained_runner=validate_unconstrained_runner or (lambda config: None),
        install_engine_core_config_sanitizer=(
            install_engine_core_config_sanitizer or (lambda: None)
        ),
        **{name: (lambda answer=value: answer) for name, value in answers.items()},
        default_constraint_table_backend=lambda: "npu" if backend_id == "ascend" else "cuda",
    )


def use_config_service(monkeypatch: Any, service: Any) -> Any:
    """Point shared configuration at ``service`` for the duration of a test."""
    import vllm_gr.platforms.loader as loader

    provider = SimpleNamespace(create_config_service=lambda: service)
    monkeypatch.setattr(loader, "activate_current_backend", lambda: provider)
    return service


@contextmanager
def config_policy(**kwargs: Any) -> Any:
    """Inject a backend configuration policy into shared configuration code.

    Replaces patching a platform predicate: shared code asks the backend, so a
    test supplies the backend's answers.
    """
    import vllm_gr.engine.arg_utils_gr as arg_utils_gr

    service = config_service(**kwargs)
    with mock.patch.object(arg_utils_gr, "_backend_config_service", lambda: service):
        yield service


__all__ = ["config_policy", "config_service", "use_config_service"]
