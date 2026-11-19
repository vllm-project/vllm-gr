# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Extension guarantees for Constraint Table implementation identity (issue #714)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from tests.constraints.test_table import make_table
from vllm_gr.config import ConstraintTableBackendEnum, ConstraintTableConfig
from vllm_gr.constraints import loader, registry
from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.backend import (
    ConstraintBackendInfo,
    ConstraintMetadata,
    ConstraintOutput,
    build_constraint_output,
)
from vllm_gr.constraints.registry import (
    ConstraintImplementation,
    ConstraintImplementationError,
    ConstraintTableRequest,
    register_implementation,
    registered_implementation_ids,
    resolve_constraint_implementation,
)
from vllm_gr.constraints.validity import valid_mask_from_counts


class _EchoBackend:
    """A minimal Backend that satisfies the contract without touching a device."""

    def __init__(self, implementation_id: str) -> None:
        self._info = ConstraintBackendInfo(
            table_metadata=make_table().metadata,
            backend=ConstraintTableBackendEnum.CPU,
            max_top_k=4,
            max_rows=4,
            host_memory_bytes=0,
            device_memory_bytes=0,
        )
        self.implementation_id = implementation_id
        self.closed = False

    @property
    def info(self) -> ConstraintBackendInfo:
        return self._info

    def select(self, logits: torch.Tensor, metadata: ConstraintMetadata) -> ConstraintOutput:
        rows = logits.shape[0]
        counts = torch.full((rows,), metadata.top_k, dtype=torch.int64)
        return build_constraint_output(
            torch.zeros((rows, metadata.top_k), dtype=torch.int32),
            torch.zeros((rows, metadata.top_k), dtype=torch.float32),
            valid_mask_from_counts(counts, metadata.top_k),
            logits=logits,
            top_k=metadata.top_k,
        )

    def install_runner_integration(
        self,
        runner: Any,
        *,
        parallel_config: Any,
        vocab_size: int,
    ) -> int:
        del runner, parallel_config, vocab_size
        return 0

    def prepare_runner(self, model_config: Any) -> None:
        del model_config

    def close(self) -> None:
        self.closed = True


def _implementation(implementation_id: str) -> ConstraintImplementation:
    """An implementation that records the request it was handed."""
    seen: list[Any] = []

    def admit(config: Any, device: torch.device, dtype: torch.dtype) -> Any:
        seen.append(("admit", config, device, dtype))
        return None

    def build(request: ConstraintTableRequest, admitted: Any) -> Any:
        del admitted
        seen.append(("build", request))
        backend = _EchoBackend(implementation_id)
        backend.seen = seen  # type: ignore[attr-defined]
        return backend

    implementation = ConstraintImplementation(
        implementation_id=implementation_id,
        legacy_enum=None,
        admit=admit,
        build=build,
    )
    return implementation


@pytest.fixture  # type: ignore[misc]
def external_implementation() -> Any:
    """Register an implementation no core enum mentions, then remove it."""
    implementation = _implementation("acme-tpu")
    register_implementation(implementation)
    try:
        yield implementation
    finally:
        registry._REGISTRY.pop("acme-tpu", None)


def _artifact(tmp_path: Path) -> tuple[str, str]:
    table = make_table()
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(table, path)
    return str(path), digest


def _config(
    path: str,
    digest: str,
    *,
    implementation_id: str | None = None,
    backend: ConstraintTableBackendEnum = ConstraintTableBackendEnum.CPU,
) -> ConstraintTableConfig:
    return ConstraintTableConfig(
        enabled=True,
        backend=backend,
        implementation_id=implementation_id,
        path=path,
        artifact_digest=digest,
        tokenizer_digest="sha256:" + "b" * 64,
        max_top_k=4,
    )


def test_an_external_id_runs_without_a_core_enum_member(
    tmp_path: Path, external_implementation: Any
) -> None:
    """The #714 extension guarantee: a new ID needs no enum or branch edit."""
    path, digest = _artifact(tmp_path)

    backend = loader.load_constraint_backend(
        _config(path, digest, implementation_id="acme-tpu"),
        device=torch.device("cpu"),
        vocab_size=make_table().metadata.vocab_size,
        dtype=torch.float32,
        max_rows=2,
    )

    assert backend is not None
    assert backend.implementation_id == "acme-tpu"
    assert "acme-tpu" in registered_implementation_ids()
    assert "acme-tpu" not in {member.value for member in ConstraintTableBackendEnum}


def test_admission_runs_before_the_artifact_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An implementation that cannot serve a config fails without artifact I/O."""
    path, digest = _artifact(tmp_path)

    def refuse(config: Any, device: torch.device, dtype: torch.dtype) -> Any:
        del config, device, dtype
        raise RuntimeError("this implementation cannot serve the configuration")

    register_implementation(
        ConstraintImplementation(
            implementation_id="acme-tpu",
            legacy_enum=None,
            admit=refuse,
            build=lambda request, admitted: pytest.fail("build must not run"),
        )
    )
    monkeypatch.setattr(
        loader,
        "load_constraint_table_artifact",
        lambda *args, **kwargs: pytest.fail("Artifact I/O must not run"),
    )
    try:
        with pytest.raises(RuntimeError, match="cannot serve"):
            loader.load_constraint_backend(
                _config(path, digest, implementation_id="acme-tpu"),
                device=torch.device("cpu"),
                vocab_size=make_table().metadata.vocab_size,
                dtype=torch.float32,
                max_rows=2,
            )
    finally:
        registry._REGISTRY.pop("acme-tpu", None)


def test_an_unknown_id_fails_with_the_registered_set() -> None:
    with pytest.raises(ConstraintImplementationError, match="registered implementations"):
        resolve_constraint_implementation(
            ConstraintTableConfig(enabled=False, max_top_k=4, implementation_id="not-registered")
        )


def test_an_omitted_id_keeps_the_legacy_enum_behavior() -> None:
    """A payload written before implementation_id keeps its implementation."""
    for backend in ConstraintTableBackendEnum:
        implementation = resolve_constraint_implementation(
            ConstraintTableConfig(enabled=False, max_top_k=4, backend=backend)
        )
        assert implementation.legacy_enum is backend
        assert implementation.implementation_id == backend.value


def test_a_duplicate_legacy_enum_is_rejected() -> None:
    with pytest.raises(ConstraintImplementationError, match="already maps to"):
        register_implementation(
            ConstraintImplementation(
                implementation_id="acme-tpu",
                legacy_enum=ConstraintTableBackendEnum.CPU,
                admit=lambda config, device, dtype: None,
                build=lambda request, admitted: pytest.fail("not used"),
            )
        )


def test_a_duplicate_id_is_rejected() -> None:
    with pytest.raises(ConstraintImplementationError, match="already registered"):
        register_implementation(_implementation("cuda"))


def test_an_empty_id_is_rejected() -> None:
    with pytest.raises(ConstraintImplementationError, match="must not be empty"):
        register_implementation(_implementation("   "))


def test_the_loader_consults_the_registry_not_a_local_list(
    tmp_path: Path, external_implementation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing a registered implementation changes the loader's choice."""
    path, digest = _artifact(tmp_path)
    replacement = _implementation("acme-tpu")
    monkeypatch.setitem(registry._REGISTRY, "acme-tpu", replacement)

    backend = loader.load_constraint_backend(
        _config(path, digest, implementation_id="acme-tpu"),
        device=torch.device("cpu"),
        vocab_size=make_table().metadata.vocab_size,
        dtype=torch.float32,
        max_rows=2,
    )

    assert backend is not None
    assert backend.seen[0][0] == "admit"  # type: ignore[attr-defined]
    assert backend.seen[1][0] == "build"  # type: ignore[attr-defined]


def test_the_registered_id_is_carried_into_the_config(tmp_path: Path) -> None:
    """The canonical mapping round-trips implementation_id without the enum."""
    from vllm_gr.config.parser import gr_config_from_mapping

    raw = {
        "schema_version": 1,
        "constraint_table": {
            "enabled": False,
            "max_top_k": 4,
            "implementation_id": "acme-tpu",
        },
    }
    config = gr_config_from_mapping(raw)
    assert config.constraint_table is not None
    assert config.constraint_table.implementation_id == "acme-tpu"


def test_the_enum_only_payload_still_parses() -> None:
    """Historical payloads name only backend; that must keep working."""
    from vllm_gr.config.parser import gr_config_from_mapping

    raw = {"schema_version": 1, "constraint_table": {"enabled": False, "max_top_k": 4, "backend": "npu"}}
    config = gr_config_from_mapping(raw)
    assert config.constraint_table is not None
    assert config.constraint_table.backend is ConstraintTableBackendEnum.NPU
    assert config.constraint_table.implementation_id is None


def test_an_invalid_implementation_id_is_rejected_at_the_config_boundary() -> None:
    with pytest.raises(ValueError, match="implementation_id"):
        ConstraintTableConfig(enabled=False, max_top_k=4, implementation_id="  ")
    with pytest.raises(ValueError, match="implementation_id"):
        ConstraintTableConfig(enabled=False, max_top_k=4, implementation_id=7)  # type: ignore[arg-type]


def test_both_ids_resolve_to_the_explicit_one() -> None:
    """An explicit implementation_id wins over a disagreeing legacy enum."""
    implementation = resolve_constraint_implementation(
        ConstraintTableConfig(
            enabled=False,
            max_top_k=4,
            backend=ConstraintTableBackendEnum.CPU,
            implementation_id="cuda",
        )
    )
    assert implementation.implementation_id == "cuda"


def test_builtin_ids_are_registered_at_import() -> None:
    """The loader module registers exactly the built-in placements."""
    assert set(registered_implementation_ids()) >= {"cpu", "cuda", "npu"}


pytestmark = pytest.mark.cpu_test
