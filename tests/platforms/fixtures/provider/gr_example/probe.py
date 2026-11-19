# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Scenarios invoked in fresh worker processes from the installed example wheel."""

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gr_example import ExampleAttention, ExampleGraph, ExamplePlatform, ExampleRunner

from vllm_gr.config import gr_config_from_mapping
from vllm_gr.platforms import loader
from vllm_gr.platforms.attention import for_config, for_runner
from vllm_gr.platforms.binding import (
    bind_backend_services,
    validate_worker_readiness,
    worker_backend_readiness,
)
from vllm_gr.platforms.errors import (
    AmbiguousBackendError,
    InvalidBackendProviderError,
    UnsupportedBackendError,
)
from vllm_gr.platforms.identity import platform_identity
from vllm_gr.platforms.plan import read_backend_plan
from vllm_gr.platforms.planning import admit_backend_plan
from vllm_gr.v1.worker.runner_lifecycle import get_stage_execution

ORIGIN = "vllm-gr-conformance-provider==1.0.0"


@contextmanager
def rejected(error_type, *details):
    try:
        yield
    except error_type as error:
        assert all(detail in str(error) for detail in details), str(error)
    else:
        raise AssertionError(f"Expected {error_type.__name__} with {details}")


def check_imports():
    for module in (
        "torch_npu",
        "vllm_ascend",
        "vllm_gr.ops.npu",
        "vllm_gr.platforms.cuda.attention",
        "vllm_gr.platforms.ascend.attention",
        "vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch",
        "vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_patch",
        "vllm_gr.v1.worker.npu.npu_model_runner_patch",
    ):
        assert module not in sys.modules, module


def engine():
    configs = []
    for tile, mode in ((16, "shared"), (32, "provider")):
        config = SimpleNamespace(
            additional_config={"vllm_gr_backend_options": {"tile": tile, "prefill_graph": mode}}
        )
        gr_config_from_mapping(
            {
                "attention-config": {"backend": "CUSTOM"},
                "gr": {
                    "beam_width": [4] * 3,
                    "candidate_width": [4] * 3,
                    "worker_decision": True,
                    "constraint_mode": "none",
                },
            }
        ).write_to_vllm_config(config)
        plan = admit_backend_plan(config)
        assert ["vllm-gr-conformance-provider", "1.0.0"] in (
            plan.request.compatibility.to_dict()["provider_distributions"]
        )
        configs.append(config.additional_config)
    Path("configs.json").write_text(json.dumps(configs), encoding="utf-8")


def make_native_layer():
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionImpl

    return BeamAttentionImpl(1, 8, 1.0, 1, None, None, "auto")


def check_suffix_write(runner, tile, native):
    import torch

    key = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8)
    value = key + tile
    empty = torch.full((2, 2, 1, 2, 8), -1.0)
    pool = SimpleNamespace(unshared_key=empty.clone(), unshared_value=empty.clone())
    rows = torch.tensor([1, 0])
    session = SimpleNamespace(slot=1, decode_step=0, pools=[pool], is_active=True)
    metadata = SimpleNamespace(
        beam_session=session, is_beam_decode=True, prefix_indices=rows, suffix_max_q_len=1
    )
    if native:
        prefix = torch.zeros_like(key)
        cache = (prefix.clone(), prefix.clone())
        output = runner.layer.forward(
            SimpleNamespace(_beam_pool_index=0),
            key,
            key,
            value,
            cache,
            metadata,
            output=torch.empty_like(value),
        )
        assert torch.equal(output, value)
        assert all(torch.equal(part, prefix) for part in cache)
    else:
        for_runner(runner).write_suffix_kv(pool, key[rows], value[rows], session, metadata)
    for actual, source in ((pool.unshared_key, key), (pool.unshared_value, value)):
        expected = empty.clone()
        expected[1, :, :, 0] = source[rows]
        assert torch.equal(actual, expected)


def check_execution(runner, tile, native):
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"example-request": 1},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        finished_req_ids=set(),
        gr_stage_metadata={"example-request": SimpleNamespace(owner_generation=1)},
    )

    def execute(owner, local, **kwargs):
        if native:
            assert local is not scheduled
        return owner.execute_model(owner.prepare_inputs(sum(local.num_scheduled_tokens.values())))

    stage = get_stage_execution(runner)
    if native:
        from vllm_gr.v1.engine.beam_search_v1 import GRRetireMetadata
        from vllm_gr.v1.engine.gr_async_scheduler import consume_gr_retirement
        from vllm_gr.v1.worker.gr_execution import (
            execute_gr_retirement,
            execute_gr_stage,
        )

        assert execute_gr_stage(runner, scheduled, execute) == (tile, 1)
        metadata = GRRetireMetadata("example-request", 1, 3)
        retire = SimpleNamespace(
            gr_retire_metadata=metadata,
            gr_stage_metadata={},
            total_num_scheduled_tokens=0,
            num_scheduled_tokens={},
            finished_req_ids={metadata.session_id},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        )
        session = SimpleNamespace(
            retirement=metadata, worker_resources_allocated=True, terminal_result_pending=True
        )
        scheduler = SimpleNamespace(_gr_v1_sessions={metadata.session_id: session})
        proof = execute_gr_retirement(runner, retire)
        assert consume_gr_retirement(scheduler, retire, proof) is session
        assert not session.worker_resources_allocated and not stage.active
    else:
        assert stage.execute(runner, scheduled, execute) == (tile, 1)
        assert stage.active == {"example-request": 1}
    service = for_runner(runner)
    assert service.execution_module() is runner._gr_backend_binding.attention_executor
    if native:
        import torch

        assert runner.layer._execution.execution_module() is service.execution_module()
        query = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
        value = query + tile
        cache = (torch.zeros_like(query), torch.zeros_like(query))
        result = runner.layer.forward(
            SimpleNamespace(_k_scale=1, _v_scale=1),
            query,
            query,
            value,
            cache,
            SimpleNamespace(slot_mapping=[0]),
            output=torch.empty_like(query),
        )
        assert torch.equal(result, value) and torch.equal(cache[0], query)
    else:
        assert runner.layer.execution_module() is service.execution_module()
        key_cache, value_cache, output = [0], [0], [0]
        primitives = service.execution_module()
        primitives.reshape_and_cache([1], [tile], key_cache, value_cache, [0])
        primitives.cascade_attention(output=output, value_cache=value_cache)
        assert output == [tile] and key_cache == [1]
    check_suffix_write(runner, tile, native)


def worker(configs, native):
    runners = []
    reports = []
    for additional in configs:
        runner = ExampleRunner(
            SimpleNamespace(additional_config=additional),
            make_layer=make_native_layer if native else for_config,
        )
        plan = read_backend_plan(runner.vllm_config)
        assert runner.load_model() == "loaded"
        binding = bind_backend_services(runner)
        report = worker_backend_readiness(SimpleNamespace(model_runner=runner))
        reports.append(report)
        validate_worker_readiness(plan, [report], expected_workers=1)
        assert report["services"]["runner"] == "example.runner"
        assert report["services"]["attention"] == "example.attention"
        assert report["services"]["stage"] == "example.stage"
        assert report["services"]["operator"] == "torch.reference"
        assert report["services"]["graph"] == "example.graph"
        assert binding.attention_executor.device == runner.device
        check_execution(runner, plan.choices.to_dict()["tile"], native)
        runners.append((runner, binding))
    (first, a), (second, b) = runners
    assert a.plan.fingerprint != b.plan.fingerprint
    assert a.graph is not b.graph
    from vllm_gr.v1.worker.prefill_graph_runner import BeamPrefillGraphRunner

    prefill = BeamPrefillGraphRunner(first, buckets=[16])
    prefill.capture_graphs(first)
    assert prefill.model_runner is first and prefill.captured_buckets == (16,)
    assert prefill.execute_model_prefill(16, [42], None, None, [1]) == (42,)
    assert a.graph.stats()["captures"]["prefill"] == 1
    with rejected(RuntimeError, "Provider-managed Prefill"):
        BeamPrefillGraphRunner(second, buckets=[32])
    assert b.graph.capture("prefill", tuple, [32]) == (32,)
    assert b.graph.stats()["captures"]["prefill"] == 1
    assert a.graph.capture("decode", lambda: 13) == 13
    assert a.graph.replay("decode", lambda: 21) == 21
    assert a.graph.stats()["replays"] == {"prefill": 1, "decode": 1}
    assert b.graph.stats()["replays"] == {}
    first.shutdown()
    first.shutdown()
    assert a.attention_executor.close_count == 1
    with rejected(RuntimeError, "closed"):
        a.graph.replay("decode", lambda: 21)
    assert b.attention_executor.close_count == 0
    assert second.execute_model(7) == (32, 7)
    second.shutdown()
    assert b.attention_executor.close_count == 1
    return reports


def incomplete_worker(configs, provider):
    config = SimpleNamespace(additional_config=configs[0])
    owner = SimpleNamespace(model_runner=ExampleRunner(config))
    with patch.object(ExampleRunner, "_gr_stage_factory", None):
        with rejected(RuntimeError, "required Beam stage"):
            worker_backend_readiness(owner)
    with patch.object(provider, "create_attention_executor", return_value=object()):
        with rejected(RuntimeError, "CUSTOM attention", "get_flash_attn_version"):
            worker_backend_readiness(owner)
    with patch.object(
        provider,
        "create_graph_executor",
        side_effect=lambda *, plan, runner: ExampleGraph(runner, plan),
    ):
        with rejected(RuntimeError, "Shared Prefill", "prefill_runner_class"):
            worker_backend_readiness(owner)
    incomplete = ExampleAttention("example:0")
    with (
        patch.object(incomplete, "cache_unshared_kv", None),
        patch.object(provider, "create_attention_executor", return_value=incomplete),
        rejected(RuntimeError, "Beam Decode attention", "cache_unshared_kv"),
    ):
        worker_backend_readiness(owner)
    assert incomplete.close_count == 1
    assert getattr(owner.model_runner, "_gr_backend_binding", None) is None


def run(scenario, identity):
    if scenario in ("ambiguous", "explicit"):
        os.environ["GR_CONFORMANCE_PLATFORM"] = "ambiguous"
    elif scenario in ("unsupported", "version"):
        os.environ["GR_CONFORMANCE_PLATFORM"] = scenario
    if scenario == "explicit":
        os.environ["VLLM_GR_BACKEND"] = "alternate"
    if identity == "fixture":
        # Only the native discovery boundary is simulated. The registry, identity
        # bridge, compatibility checks, admission, binding and cleanup are real.
        loader.current_platform_identity = lambda: platform_identity(ExamplePlatform())
    else:
        assert loader.current_platform_identity().has_lineage("gr_example.ExamplePlatform")

    if scenario == "bootstrap":
        from vllm.plugins import load_general_plugins

        from vllm_gr.patch import get_last_beam_patch_report
        from vllm_gr.platforms.registry import get_active_backend

        load_general_plugins()
        selected = get_active_backend()
        assert selected.backend_id == "example"
        assert selected.get_model_runner_hooks().is_patched(ExampleRunner)
        report = get_last_beam_patch_report()
        assert report is not None and report.results
        assert all(result.status.value != "failed" for result in report.results)
        engine()
    elif scenario == "unsupported":
        with rejected(UnsupportedBackendError, "No vllm-gr backend", "OOT"):
            loader.activate_current_backend()
    elif scenario == "ambiguous":
        with rejected(AmbiguousBackendError, "entrypoint:example", "entrypoint:alternate", ORIGIN):
            loader.activate_current_backend()
    elif scenario == "version":
        with rejected(InvalidBackendProviderError, "vllm-gr version", "<0", "incompatible"):
            loader.activate_current_backend()
    else:
        from vllm_gr.platforms.patches import CORE_PATCH_IDS, SelectedPatchClaims

        provider = loader.activate_current_backend()
        assert provider.check_environment()["backend"] == (
            "alternate" if scenario == "explicit" else "example"
        )
        claims = SelectedPatchClaims.from_provider(provider, CORE_PATCH_IDS)
        if scenario == "patch":
            os.environ["GR_CONFORMANCE_REPLACED"] = "1"
            with rejected(RuntimeError, "example.runner.lifecycle", ORIGIN):
                claims.install(force=True)
        else:
            claims.install(force=False)
            claims.install(force=True)
            assert provider.get_model_runner_hooks().is_patched(ExampleRunner)
            if scenario == "engine":
                engine()
            elif scenario.startswith("worker-") or scenario in (
                "agreement",
                "mismatch",
                "incomplete",
            ):
                configs = json.loads(Path("configs.json").read_text(encoding="utf-8"))
                if scenario.startswith("worker-"):
                    Path(f"{scenario}.json").write_text(
                        json.dumps(worker(configs, identity == "native")), encoding="utf-8"
                    )
                elif scenario == "incomplete":
                    incomplete_worker(configs, provider)
                elif scenario == "agreement":
                    ranks = [
                        json.loads(Path(f"worker-{rank}.json").read_text(encoding="utf-8"))
                        for rank in range(2)
                    ]
                    for index, additional in enumerate(configs):
                        plan = read_backend_plan(SimpleNamespace(additional_config=additional))
                        validate_worker_readiness(
                            plan, [rank[index] for rank in ranks], expected_workers=2
                        )
                else:
                    config = SimpleNamespace(additional_config=configs[0])
                    config.additional_config["vllm_gr_backend_options"]["tile"] = 32
                    runner = ExampleRunner(config)
                    with rejected(RuntimeError, "plan mismatch", "engine=", "worker="):
                        bind_backend_services(runner)
                    assert getattr(runner, "_gr_backend_binding", None) is None
    check_imports()
    print(f"{scenario} passed (pid {os.getpid()})", flush=True)
