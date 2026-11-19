# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise a single installed provider across two independently loaded runners."""

from pathlib import Path

import pytest

from tests.platforms.test_external_provider_fixture import (
    _PROVIDER_MODULE,
    _install_provider,
    _run,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]

_SERVICES = """
from types import SimpleNamespace
from vllm_gr.constraints.backend import ConstraintBackendInfo
from vllm_gr.constraints.cpu_backend import CPUConstraintBackend
from vllm_gr.constraints.registry import ConstraintImplementation, register_implementation
from vllm_gr.platforms.plan import BackendPlan, PlanData
from vllm_gr.v1.beam.decision_ops import reference_ops

class Table(CPUConstraintBackend):
    implementation_id = "fixture.table"

def build_table(request, _):
    info = ConstraintBackendInfo(request.table.metadata, None, request.max_top_k,
                                 request.max_rows, request.table.total_bytes, 0)
    table = Table(request.table, info)
    table.plan = request.plan
    return table

register_implementation(ConstraintImplementation(
    Table.implementation_id, None, lambda config, device, dtype: None, build_table))

class Attention:
    implementation_id = "fixture.attention"
    def __init__(self, plan):
        self.plan = plan
        self.close_count = 0
    def execution_module(self):
        return self
    def close(self):
        self.close_count += 1

class Stage:
    implementation_id = "fixture.stage"
    worker_decision = False
    def __init__(self, plan):
        self.plan = plan
        self.decision = reference_ops()
    def execute(self, worker, scheduled, original, *args, **kwargs):
        return original(worker, scheduled)
    def retire(self, worker, scheduled):
        return scheduled
    def capture_startup_graph(self, worker, **kwargs):
        return False
    def admit_unconstrained_beam(self, worker):
        return None

class Provider(HostOnlyProvider):
    plan_api_version = 1
    adapter_revision = "fixture.services.v1"
    def resolve(self, request):
        return BackendPlan(self.backend_id, self.api_version, request, PlanData({
            "stage": "fixture.stage", "operator_policy": "fixture.decision",
            "requires_attention": True, "constraint": "canonical",
            "constraint_implementation": Table.implementation_id}))
    def validate_worker(self, *, plan, native_runner):
        assert native_runner.device.type == "cpu"
    def create_attention_executor(self, *, plan, device):
        assert device.type == "cpu"
        return Attention(plan)
    def create_model_runner_adapter(self, *, plan, native_runner):
        return SimpleNamespace(implementation_id="fixture.runner", stage=Stage(plan),
                               prepare_inputs=lambda data: data, execute_model=lambda data: data)

BACKEND = Provider()
"""


def test_two_runners_share_no_bound_execution_services(tmp_path: Path) -> None:
    site = _install_provider(
        tmp_path,
        distribution="gr-execution-fixture",
        module="gr_execution_fixture",
        backend_id="host_only",
        source=_PROVIDER_MODULE.replace("BACKEND = HostOnlyProvider()", _SERVICES),
    )
    result = _run(
        """
import hashlib, logging
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from tests.constraints.test_table import make_table
from vllm_gr.config import ConstraintConfig, ConstraintMode, ConstraintTableSpec, GRConfig
from vllm_gr.config.parser import write_canonical_gr_config
from vllm_gr.constraints.artifact import serialize_constraint_table_artifact
from vllm_gr.constraints.backend import ConstraintMetadata
from vllm_gr.constraints.loader import load_constraint_backend
from vllm_gr.platforms.attention import for_config
from vllm_gr.platforms.binding import bind_after_model_load, worker_backend_readiness
from vllm_gr.platforms.plan import read_backend_plan
from vllm_gr.platforms.planning import admit_backend_plan
from vllm_gr.v1.worker.runner_lifecycle import get_stage_execution
logger = ModuleType("vllm.logger")
logger.init_logger = logging.getLogger
sys.modules["vllm.logger"] = logger
from vllm_gr.v1.worker.model_runner_patch import close_constraint_backend

def load(runner, table_config):
    runner.layer_execution = for_config(runner.vllm_config)
    runner.constraint_backend = load_constraint_backend(
        table_config, plan=read_backend_plan(runner.vllm_config), device=runner.device,
        vocab_size=6, dtype=torch.float32, max_rows=1)

with TemporaryDirectory() as directory:
    data = serialize_constraint_table_artifact(make_table())
    filename = Path(directory) / "table.safetensors"
    filename.write_bytes(data)
    table = ConstraintTableSpec(enabled=True, implementation_id="fixture.table",
                                  path=str(filename.resolve()), max_top_k=2,
                                  tokenizer_digest=make_table().metadata.tokenizer_digest,
                                  artifact_digest="sha256:" + hashlib.sha256(data).hexdigest())
    runners = []
    for width in (1, 2):
        runner = SimpleNamespace(device=torch.device("cpu"),
                                 vllm_config=SimpleNamespace(additional_config={}))
        write_canonical_gr_config(runner.vllm_config, GRConfig(
            beam_max_width=width, constraint=ConstraintConfig(ConstraintMode.TABLE, table)))
        plan = admit_backend_plan(runner.vllm_config)
        bind_after_model_load(load)(runner, table)
        report = worker_backend_readiness(SimpleNamespace(model_runner=runner))
        assert report["fingerprint"] == plan.fingerprint
        assert report["services"] == dict(runner="fixture.runner", stage="fixture.stage",
                                          operator="torch.reference", attention="fixture.attention",
                                          constraint="fixture.table"), report
        binding = runner._gr_backend_binding
        assert runner.layer_execution is binding.attention_executor
        assert runner.constraint_backend is binding.canonical_constraint
        assert get_stage_execution(runner) is binding.runner_adapter.stage
        assert plan == runner.layer_execution.plan == runner.constraint_backend.plan == get_stage_execution(runner).plan
        runners.append(runner)
    first, second = runners
    assert first.layer_execution.plan.fingerprint != second.layer_execution.plan.fingerprint
    assert first.layer_execution is not second.layer_execution
    assert first.constraint_backend is not second.constraint_backend
    assert get_stage_execution(first) is not get_stage_execution(second)
    old = first._gr_backend_binding
    def load_layers(runner):
        runner.layer_execution = for_config(runner.vllm_config)
    bind_after_model_load(load_layers)(first)
    new = first._gr_backend_binding
    assert new is not old and new.plan == old.plan
    assert old.attention_executor.close_count == 1
    assert first.layer_execution is new.attention_executor
    assert new.attention_executor.close_count == 0
    assert worker_backend_readiness(SimpleNamespace(model_runner=first))["ready"]
    assert first._gr_backend_binding is new
    close_constraint_backend(first)
    close_constraint_backend(first)
    assert first.layer_execution.close_count == 1
    assert old.attention_executor.close_count == 1
    assert second.layer_execution.close_count == 0
    meta = ConstraintMetadata(torch.zeros((1, 3), dtype=torch.int32), torch.ones(1), 0, 2)
    result = second.constraint_backend.select(torch.arange(6, dtype=torch.float32)[None], meta)
    assert result.token_ids.tolist() == [[3, 1]]
    close_constraint_backend(second)
    assert second.layer_execution.close_count == 1
""",
        site_packages=site,
        project_root=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stdout + result.stderr
