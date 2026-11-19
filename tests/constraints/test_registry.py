# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Extension guarantees for Constraint Table implementation identity (issue #714)."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from tests.constraints.test_table import make_table
from vllm_gr.config import ConstraintTableBackendEnum, ConstraintTableConfig
from vllm_gr.constraints import loader, registry
from vllm_gr.constraints.artifact import serialize_constraint_table_artifact
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
            backend=None,
            max_top_k=4,
            max_rows=4,
            host_memory_bytes=0,
            device_memory_bytes=0,
        )
        self.implementation_id = implementation_id
        self.seen: list[Any] = []
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
        assert not self.closed
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
        backend.seen = seen
        return backend

    implementation = ConstraintImplementation(
        implementation_id=implementation_id,
        legacy_enum=None,
        admit=admit,
        build=build,
        supports_hot_reload=True,
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
    data = serialize_constraint_table_artifact(table)
    path.write_bytes(data)
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
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
    output = backend.select(
        torch.zeros((1, make_table().metadata.vocab_size)),
        ConstraintMetadata(torch.zeros((1, 3), dtype=torch.int32), torch.ones(1), 0, 2),
    )
    assert output.valid_mask.tolist() == [[True, True]]
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

    assert isinstance(backend, _EchoBackend)
    assert backend.seen[0][0] == "admit"
    assert backend.seen[1][0] == "build"


def test_a_factory_cannot_substitute_another_implementation(
    tmp_path: Path,
    external_implementation: ConstraintImplementation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest = _artifact(tmp_path)
    wrong = _EchoBackend("other")
    monkeypatch.setitem(
        registry._REGISTRY, "acme-tpu", replace(external_implementation, build=lambda *_: wrong)
    )
    with pytest.raises(ValueError, match="factory 'acme-tpu' returned implementation ID 'other'"):
        loader.load_constraint_backend(
            _config(path, digest, implementation_id="acme-tpu"),
            device=torch.device("cpu"),
            vocab_size=make_table().metadata.vocab_size,
            dtype=torch.float32,
            max_rows=2,
        )
    assert wrong.closed


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

    raw = {
        "schema_version": 1,
        "constraint_table": {"enabled": False, "max_top_k": 4, "backend": "npu"},
    }
    config = gr_config_from_mapping(raw)
    assert config.constraint_table is not None
    assert config.constraint_table.backend is ConstraintTableBackendEnum.NPU
    assert config.constraint_table.implementation_id is None


def test_an_invalid_implementation_id_is_rejected_at_the_config_boundary() -> None:
    with pytest.raises(ValueError, match="implementation_id"):
        ConstraintTableConfig(enabled=False, max_top_k=4, implementation_id="  ")
    invalid: Any = 7
    with pytest.raises(ValueError, match="implementation_id"):
        ConstraintTableConfig(enabled=False, max_top_k=4, implementation_id=invalid)


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


@pytest.mark.l1
def test_provider_load_options_reach_initial_and_replacement_factories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm_gr.config import GRConfig, write_canonical_gr_config
    from vllm_gr.v1.worker.hot_update import reload_trie
    from vllm_gr.v1.worker.model_runner_patch import (
        make_load_model_with_constraint_backend,
    )

    path, digest = _artifact(tmp_path)
    table = _config(path, digest, implementation_id="vendor-options")
    native = SimpleNamespace(additional_config={})
    write_canonical_gr_config(native, GRConfig(beam_max_width=4, constraint_table=table))
    runner = SimpleNamespace(
        model=torch.nn.Module(),
        vllm_config=native,
        device=torch.device("cpu"),
        model_config=SimpleNamespace(
            get_vocab_size=lambda: make_table().metadata.vocab_size, dtype=torch.float32
        ),
    )

    def options(owner: Any, gr_config: GRConfig) -> dict[str, Any]:
        assert owner is runner and gr_config.constraint_table is not None
        assert gr_config.constraint_table.path is not None
        return {"vendor_index": Path(gr_config.constraint_table.path).name}

    implementation = replace(_implementation("vendor-options"), runner_load_options=options)
    monkeypatch.setitem(registry._REGISTRY, "vendor-options", implementation)
    make_load_model_with_constraint_backend(lambda owner: None)(runner)
    tables = runner.constraint_backend
    first = tables.get()
    assert first.seen[-1][1].options == {"vendor_index": "table.safetensors"}
    replacement = tmp_path / "replacement.safetensors"
    replacement.write_bytes(Path(path).read_bytes())
    assert reload_trie(SimpleNamespace(model_runner=runner), str(replacement), "v2")["ready"]
    second = tables.get("v2")
    assert second is not first
    assert second.seen[-1][1].options == {"vendor_index": replacement.name}
    tables.close()


@pytest.mark.parametrize(
    "choices",
    [
        {"constraint": "none"},
        {"constraint": "canonical"},
        {"constraint": "canonical", "constraint_implementation": "different"},
    ],
)
def test_plan_cannot_drop_or_replace_an_explicit_constraint(
    choices: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm_gr.config import GRConfig, write_canonical_gr_config
    from vllm_gr.platforms import loader as provider_loader
    from vllm_gr.platforms.cuda import CUDABackendProvider
    from vllm_gr.platforms.plan import BackendPlan, PlanData, read_backend_plan
    from vllm_gr.platforms.planning import admit_backend_plan

    native = SimpleNamespace(additional_config={})
    table = _config(*_artifact(tmp_path), implementation_id="vendor")
    write_canonical_gr_config(native, GRConfig(beam_max_width=4, constraint_table=table))
    monkeypatch.setattr(provider_loader, "activate_current_backend", CUDABackendProvider)
    monkeypatch.setattr(
        CUDABackendProvider,
        "resolve",
        lambda provider, request: BackendPlan(
            provider.backend_id, provider.api_version, request, PlanData(choices)
        ),
    )
    with pytest.raises(ValueError, match="constraint|canonical"):
        admit_backend_plan(native)
    assert read_backend_plan(native) is None


@pytest.mark.parametrize("explicit_services", [False, True])
def test_canonical_plan_binds_the_registered_factory_and_preserves_runner_ownership(
    tmp_path: Path,
    external_implementation: Any,
    monkeypatch: pytest.MonkeyPatch,
    explicit_services: bool,
) -> None:
    from vllm_gr.config import BeamExecutionMode, GRConfig, write_canonical_gr_config
    from vllm_gr.platforms import loader as provider_loader
    from vllm_gr.platforms.binding import (
        bind_backend_services,
        close_backend_binding,
        runner_constraint_backend,
        validate_worker_readiness,
    )
    from vllm_gr.platforms.constraint_plan import (
        canonical_constraint_id,
        constraint_plan_choices,
    )
    from vllm_gr.platforms.interface import ModelRunnerAdapter, StageExecution
    from vllm_gr.platforms.plan import BackendPlan, PlanData, ServiceRequirements
    from vllm_gr.platforms.planning import admit_backend_plan
    from vllm_gr.v1.beam.decision_ops import reference_ops
    from vllm_gr.v1.engine.hot_update import reload_trie as reload_engine
    from vllm_gr.v1.worker.hot_update import reload_trie, versioned_constraint_table

    class ExternalProvider:
        api_version = 2
        backend_id = "external"
        plan_api_version = 1
        adapter_revision = "fixture-v1"

        def resolve(self, request: Any) -> BackendPlan:
            if explicit_services:
                return BackendPlan(
                    self.backend_id,
                    self.api_version,
                    request,
                    PlanData({}),
                    requirements=ServiceRequirements(constraint="acme-tpu"),
                )
            choices = constraint_plan_choices(request.config.to_dict(), legacy_id="cpu")
            return BackendPlan(self.backend_id, self.api_version, request, PlanData(choices))

        def validate_worker(self, **kwargs: Any) -> None:
            pass

        def create_model_runner_adapter(self, **kwargs: Any) -> Any:
            stage = Mock(
                spec=StageExecution,
                implementation_id="fixture.stage",
                worker_decision=False,
                decision=reference_ops(),
            )
            return Mock(spec=ModelRunnerAdapter, implementation_id="fixture.runner", stage=stage)

        def create_attention_executor(self, **kwargs: Any) -> Any:
            pytest.fail("Attention was not admitted")

        def create_constrained_topk_execution(self, **kwargs: Any) -> Any:
            pytest.fail("Canonical artifacts use their registered factory")

    provider = ExternalProvider()
    monkeypatch.setattr(provider_loader, "activate_current_backend", lambda: provider)
    path, digest = _artifact(tmp_path)
    table = _config(path, digest, implementation_id="acme-tpu")
    native = SimpleNamespace(
        additional_config={},
        scheduler_config=SimpleNamespace(async_scheduling=True),
        model_config=SimpleNamespace(model="fixture"),
    )
    write_canonical_gr_config(
        native,
        GRConfig(
            beam_execution_mode=BeamExecutionMode.V1, beam_max_width=4, constraint_table=table
        ),
    )
    plan = admit_backend_plan(native)
    assert plan is not None
    assert canonical_constraint_id(table, plan) == "acme-tpu"
    assert plan.constraint_implementation == "acme-tpu"
    with pytest.raises(ValueError, match="differs from the admitted request"):
        canonical_constraint_id(replace(table, implementation_id="other"), plan)

    def load(_: str) -> Any:
        return loader.load_constraint_backend(
            table,
            plan=plan,
            device=torch.device("cpu"),
            vocab_size=make_table().metadata.vocab_size,
            dtype=torch.float32,
            max_rows=2,
        )

    first, second = load(path), load(path)
    assert first is not second
    assert first.seen[-1][1].plan is plan
    assert first.seen[-1][1].device == torch.device("cpu")
    runner = SimpleNamespace(
        vllm_config=native,
        device="cpu",
        model_config=SimpleNamespace(
            get_vocab_size=lambda: make_table().metadata.vocab_size, dtype=torch.float32
        ),
    )
    tables = versioned_constraint_table(runner, first)
    runner.constraint_backend = tables
    binding = bind_backend_services(runner)
    assert binding is not None
    assert binding.canonical_constraint is tables
    assert binding.report()["services"]["constraint"] == "acme-tpu"
    validate_worker_readiness(plan, [binding.report()], expected_workers=1)
    bad = binding.report()
    bad["services"]["constraint"] = "other"
    with pytest.raises(RuntimeError, match="canonical constraint implementation mismatch"):
        validate_worker_readiness(plan, [bad], expected_workers=1)
    assert runner_constraint_backend(runner) is tables
    worker = SimpleNamespace(model_runner=runner)

    def rpc(method: str, *, args: tuple[str, str]) -> list[Any]:
        assert method == "reload_trie"
        return [reload_trie(worker, *args)]

    core = SimpleNamespace(vllm_config=native, model_executor=SimpleNamespace(collective_rpc=rpc))
    assert reload_engine(core, path, "revision-2") == {"ok": True, "version": "revision-2"}
    assert tables.get() is not first
    assert tables.get().seen[-1][1].plan is binding.plan
    assert binding.canonical_constraint.get() is tables.get("revision-2")
    close_backend_binding(runner)
    assert not first.closed
    tables.close()
    assert first.closed and not second.closed
    second.close()


def test_reloading_another_implementation_keeps_the_active_version() -> None:
    from vllm_gr.constraints.versioned_table import VersionedConstraintTable

    original, wrong = _EchoBackend("acme-tpu"), _EchoBackend("other")
    tables = VersionedConstraintTable(original, loader=lambda _: wrong)
    with pytest.raises(ValueError, match="cannot change the constraint implementation"):
        tables.reload("new-table", "revision-2")
    assert tables.get() is original
    assert wrong.closed and not original.closed
    tables.close()
    assert original.closed


def test_persistent_device_stage_admission_accepts_an_external_backend_by_capability() -> None:
    from vllm_gr.constraints.backend import validate_device_stage_backend

    class ExternalStage(_EchoBackend):
        is_ready = True

        def select_out_unchecked(self, *args: Any, **kwargs: Any) -> None:
            pytest.fail("admission must not execute a stage")

        def validate_stage_device(self, device: torch.device) -> None:
            if device != torch.device("cpu"):
                raise ValueError("wrong worker device")

    external = ExternalStage("acme-tpu")
    size = external.info.table_metadata.vocab_size
    validate_device_stage_backend(external, torch.device("cpu"), size)
    with pytest.raises(ValueError, match="workspace selection"):
        validate_device_stage_backend(_EchoBackend("ordinary"), torch.device("cpu"), size)
    with pytest.raises(ValueError, match="wrong worker device"):
        validate_device_stage_backend(external, torch.device("meta"), size)
    with pytest.raises(ValueError, match="model/device binding differs"):
        validate_device_stage_backend(external, torch.device("cpu"), size + 1)


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
