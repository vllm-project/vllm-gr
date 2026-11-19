# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""A separately packaged host-only provider, including its native platform identity."""

import os
from dataclasses import dataclass

from vllm_gr.platforms.binding import bind_after_model_load, close_backend_binding
from vllm_gr.platforms.graph import GraphExecution
from vllm_gr.platforms.plan import BackendPlan, PlanData, ServiceRequirements
from vllm_gr.v1.beam.decision_ops import reference_ops, reorder_sessions
from vllm_gr.v1.worker.runner_lifecycle import (
    bind_native_runner,
    has_runner_lifecycle,
    install_runner_lifecycle,
)


class ExamplePlatform:
    _enum = "OOT"
    device_type = "example"

    @property
    def device_name(self):
        return os.environ.get("GR_CONFORMANCE_PLATFORM", "example")


def native_platform():
    return "gr_example.native.HostPlatform"


class ExampleRunner:
    def __init__(self, config, make_layer=None):
        self.device = "example:0"
        self.vllm_config = config
        self.make_layer = make_layer

    def load_model(self):
        if self.make_layer is not None:
            self.layer = self.make_layer()
        return "loaded"

    def prepare_inputs(self, data):
        return data

    def execute_model(self, data):
        return self._gr_backend_binding.plan.choices.to_dict()["tile"], data

    def shutdown(self):
        close_backend_binding(self)


class RunnerHooks:
    def discover_worker_runners(self):
        return (("ExampleRunner", ExampleRunner),)

    def install(self):
        install_runner_lifecycle(
            ExampleRunner, "example.runner", prepare="prepare_inputs", stage_factory=ExampleStage
        )
        ExampleRunner.load_model = bind_after_model_load(ExampleRunner.load_model)

    def is_patched(self, runner_class):
        return has_runner_lifecycle(runner_class, "example.runner")


class ExampleDecision:
    implementation_id = "torch.reference"
    reorder = staticmethod(reorder_sessions)

    def select(self, *args, **kwargs):
        return reference_ops().select(*args, **kwargs)


class ExampleStage:
    implementation_id = "example.stage"
    worker_decision = True

    def __init__(self, *, plan, runner):
        self.decision = ExampleDecision()
        self.active = {}
        self.last_release = None

    def execute(self, runner, scheduled, original, *args, **kwargs):
        for name, stage in scheduled.gr_stage_metadata.items():
            if self.active.setdefault(name, stage.owner_generation) != stage.owner_generation:
                raise RuntimeError("Previous stage owner has outstanding resources")
        return original(runner, scheduled, *args, **kwargs)

    def retire(self, runner, scheduled):
        from vllm.v1.outputs import ModelRunnerOutput

        from vllm_gr.v1.engine.beam_search_v1 import GRRetireResult

        metadata = scheduled.gr_retire_metadata
        if self.last_release is not None and self.last_release[0] == metadata:
            return self.last_release[1]
        if (
            scheduled.num_scheduled_tokens
            or scheduled.total_num_scheduled_tokens
            or scheduled.gr_stage_metadata
            or metadata.session_id not in scheduled.finished_req_ids
            or self.active.get(metadata.session_id) != metadata.owner_generation
        ):
            raise RuntimeError("Retirement does not match the stage owner")
        del self.active[metadata.session_id]
        output = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
        output.gr_retire_result = GRRetireResult(
            metadata.session_id, metadata.owner_generation, metadata.retire_id, True
        )
        self.last_release = (metadata, output)
        return output

    def capture_startup_graph(self, runner, *, allow_eager_fallback):
        return False

    def close(self):
        self.active.clear()
        self.last_release = None


@dataclass
class ExampleAttention:
    device: str
    close_count: int = 0
    implementation_id = "example.attention"

    def close(self):
        self.close_count += 1

    def is_available(self):
        return True

    def get_flash_attn_version(self, **kwargs):
        return None

    def supports_sinks(self):
        return False

    def reshape_and_cache(self, key, value, key_cache, value_cache, slot_mapping, *args):
        for row, slot in enumerate(slot_mapping):
            key_cache[slot] = key[row]
            value_cache[slot] = value[row]

    def cache_unshared_kv(self, keys, values, key, value, slot, step, *, beam_session):
        keys[slot, : len(key), :, step].copy_(key)
        values[slot, : len(value), :, step].copy_(value)

    def cascade_attention(
        self, *, output, value_cache, beam_pool=None, attn_metadata=None, **kwargs
    ):
        if beam_pool is not None:
            session = attn_metadata.beam_session
            if session.decode_step != 0:
                raise ValueError("example attention expects one cached suffix token per beam")
            output[attn_metadata.prefix_indices] = beam_pool.unshared_value[session.slot, :, :, 0]
            return
        # One-token self-attention has weight one on its only cached value.
        if len(value_cache) != 1 or len(output) != 1:
            raise ValueError("example attention only supports the one-token conformance case")
        output[:] = value_cache[:]

    prefill_attention = cascade_attention


class ExampleProvider:
    api_version = 2
    plan_api_version = 1
    adapter_revision = "example-v1"
    device_type = "example"
    gr_version_range = ">=0.dev0"
    vllm_version_range = vendor_version_range = vendor_distribution = None
    patch_ids = ("example.runner.lifecycle",)
    runtime_patch_checks = {
        "example.runner.lifecycle": lambda: RunnerHooks().is_patched(ExampleRunner)
    }

    def __init__(self, backend_id):
        self.backend_id = backend_id

    def matches(self, identity):
        if not identity.has_lineage("gr_example.ExamplePlatform"):
            return False
        if self.backend_id == "alternate":
            return identity.device_name == "ambiguous"
        if self.backend_id == "incompatible":
            return identity.device_name == "version"
        return identity.device_name in {"example", "ambiguous"}

    def check_environment(self):
        return {"backend": self.backend_id}

    def apply_runtime_patches(self, *, force=False):
        self.get_model_runner_hooks().install()
        if os.environ.get("GR_CONFORMANCE_REPLACED"):
            ExampleRunner.execute_model = ExampleRunner.execute_model._gr_runner_implementation
        return 0

    def get_model_runner_hooks(self):
        return RunnerHooks()

    def get_attention_execution(self):
        return ExampleAttention("example:0")

    def resolve(self, request):
        tile = request.options.to_dict()["tile"]
        if type(tile) is not int or tile not in (16, 32):
            raise ValueError("example.tile must be 16 or 32")
        return BackendPlan(
            self.backend_id,
            self.api_version,
            request,
            PlanData(
                {
                    "runner": "example.runner",
                    "tile": tile,
                    "graph": "capture-with-eager-fallback",
                    "prefill_graph": request.options.to_dict().get("prefill_graph", "provider"),
                }
            ),
            requirements=ServiceRequirements(
                stage="example.stage", attention="example.attention", graph="example.graph"
            ),
        )

    def validate_worker(self, *, plan, native_runner):
        if not self.get_model_runner_hooks().is_patched(type(native_runner)):
            raise RuntimeError("example.runner lifecycle is missing")

    def create_model_runner_adapter(self, *, plan, native_runner):
        return bind_native_runner(plan, native_runner)

    def create_attention_executor(self, *, plan, device):
        return ExampleAttention(device)

    def create_constrained_topk_execution(self, *, plan=None, device=None):
        return None

    def create_graph_executor(self, *, plan, runner):
        graph = ExampleSharedGraph if plan.prefill_graph_mode == "shared" else ExampleGraph
        return graph(runner, plan)


class ExampleGraph(GraphExecution):
    implementation_id = "example.graph"


class ExampleSharedGraph(ExampleGraph):
    @staticmethod
    def prefill_runner_class():
        from .prefill import ExamplePrefillRunner

        return ExamplePrefillRunner


PROVIDER = ExampleProvider("example")
ALTERNATE = ExampleProvider("alternate")
INCOMPATIBLE = ExampleProvider("incompatible")
INCOMPATIBLE.gr_version_range = "<0"
