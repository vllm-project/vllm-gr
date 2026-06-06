# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


"""Shared fake runtime harness for platform smoke tests."""

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"
from __future__ import annotations

import importlib
import logging
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _purge_modules(*prefixes: str) -> dict[str, types.ModuleType]:
    saved: dict[str, types.ModuleType] = {}
    for name, module in list(sys.modules.items()):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            saved[name] = module
            sys.modules.pop(name, None)
    return saved


def _restore_modules(saved: dict[str, types.ModuleType]) -> None:
    for name, module in saved.items():
        sys.modules[name] = module


def _build_fake_hf_config() -> None:
    return SimpleNamespace(
        hstu_uih_feature_names=["item_id"],
        user_embedding_feature_names=["item_id"],
        hstu_candidate_feature_names=["item_candidate_id"],
        item_embedding_feature_names=["item_candidate_id"],
        uih_post_id_feature_name="item_id",
        candidate_primary_id_feature_name="item_candidate_id",
        merge_uih_candidate_feature_mapping=[
            ["item_id", "item_candidate_id"],
        ],
        task_names=["is_click"],
        task_types=["binary_classification"],
    )


@dataclass
class SmokeHarness:
    state: dict[str, object]
    current_platform: SimpleNamespace

    def set_platform(self, device_type: str) -> None:
        self.current_platform.device_type = device_type

    def create_llm(self, **kwargs) -> None:
        module = importlib.import_module("vllm_gr.entrypoints.gr")
        kwargs.setdefault("model_type", "hstu")
        return module.GRLLM(model="test-model", **kwargs)

    def prompt(self) -> dict[str, object]:
        return {
            "user_id": "u1",
            "user_history": [{"item_id": 1}, {"item_id": 2}],
            "candidates": [
                {"item_candidate_id": 11},
                {"item_candidate_id": 12},
            ],
        }


@contextmanager
def smoke_harness(initial_device_type: str = "cuda") -> None:
    prefixes = ("vllm", "vllm_ascend", "vllm_gr")
    saved = _purge_modules(*prefixes)

    state = {
        "active_platform": None,
        "active_engines": 0,
        "trace": [],
        "patch_calls": 0,
    }

    fake_vllm = types.ModuleType("vllm")
    fake_engine_pkg = types.ModuleType("vllm.engine")
    fake_engine_arg_utils = types.ModuleType("vllm.engine.arg_utils")
    fake_platforms = types.ModuleType("vllm.platforms")
    fake_config = types.ModuleType("vllm.config")
    fake_usage_pkg = types.ModuleType("vllm.usage")
    fake_usage_lib = types.ModuleType("vllm.usage.usage_lib")
    fake_v1 = types.ModuleType("vllm.v1")
    fake_v1_engine_pkg = types.ModuleType("vllm.v1.engine")
    fake_llm_engine_module = types.ModuleType("vllm.v1.engine.llm_engine")
    fake_executor_module = types.ModuleType("vllm.v1.executor")
    fake_request_module = types.ModuleType("vllm.v1.request")
    fake_core_pkg = types.ModuleType("vllm.v1.core")
    fake_sched_pkg = types.ModuleType("vllm.v1.core.sched")
    fake_sched_output_module = types.ModuleType("vllm.v1.core.sched.output")
    fake_multimodal = types.ModuleType("vllm.multimodal")
    fake_metrics_pkg = types.ModuleType("vllm.v1.metrics")
    fake_metrics_loggers = types.ModuleType("vllm.v1.metrics.loggers")
    fake_entrypoints_pkg = types.ModuleType("vllm.entrypoints")
    fake_entrypoints_llm = types.ModuleType("vllm.entrypoints.llm")
    fake_utils_pkg = types.ModuleType("vllm.utils")
    fake_counter_module = types.ModuleType("vllm.utils.counter")
    fake_pooling_params = types.ModuleType("vllm.pooling_params")
    fake_outputs = types.ModuleType("vllm.outputs")
    fake_logger = types.ModuleType("vllm.logger")
    fake_gpu_worker_module = types.ModuleType("vllm.v1.worker.gpu_worker")
    fake_gpu_runner_module = types.ModuleType("vllm.v1.worker.gpu_model_runner")

    fake_ascend_pkg = types.ModuleType("vllm_ascend")
    fake_ascend_worker_pkg = types.ModuleType("vllm_ascend.worker")
    fake_npu_worker_module = types.ModuleType("vllm_ascend.worker.worker")
    fake_npu_runner_module = types.ModuleType("vllm_ascend.worker.model_runner_v1")

    current_platform = SimpleNamespace(device_type=initial_device_type)

    @dataclass
    class FakeEngineArgs:
        model: str = ""
        trust_remote_code: bool = False
        max_model_len: int | None = None
        skip_tokenizer_init: bool = False
        runner: str = ""
        enforce_eager: bool = False
        enable_prefix_caching: bool = True
        worker_cls: str = ""
        dtype: str = "auto"
        tensor_parallel_size: int = 1
        hf_overrides: dict | None = None
        load_format: str | None = None
        disable_log_stats: bool = False
        gpu_memory_utilization: float = 0.9

        def __post_init__(self) -> None:
            return None

        def create_engine_config(self, usage_context=None, headless: bool = False) -> None:
            state["trace"].append(("engine_config", current_platform.device_type))
            return SimpleNamespace(
                worker_cls=self.worker_cls,
                platform_device_type=current_platform.device_type,
                model=current_platform.device_type,
                model_config=SimpleNamespace(hf_config=_build_fake_hf_config()),
                scheduler_config=SimpleNamespace(stream_interval=1),
            )

    @dataclass
    class FakeAsyncEngineArgs(FakeEngineArgs):
        enable_log_requests: bool = False

    class UsageContext:
        ENGINE_CONTEXT = "engine"
        LLM_CLASS = "llm"

    class MultiModalRegistry:
        pass

    class StatLoggerFactory:
        pass

    class LLM:
        pass

    class Counter:
        def __init__(self, start: int = 0) -> None:
            self.value = start

        def __next__(self) -> int:
            current = self.value
            self.value += 1
            return current

    @dataclass
    class PoolingParams:
        task: str

    @dataclass
    class PoolingOutput:
        data: object

    class PoolingRequestOutput:
        def __init__(
            self,
            request_id,
            outputs,
            prompt_token_ids,
            num_cached_tokens,
            finished,
        ) -> None:
            self.request_id = request_id
            self.outputs = outputs
            self.prompt_token_ids = prompt_token_ids
            self.num_cached_tokens = num_cached_tokens
            self.finished = finished

    class BaseWorker:
        def __init__(self, vllm_config, device_type: str) -> None:
            self.vllm_config = vllm_config
            self.device = SimpleNamespace(type=device_type)
            self.model_runner = None
            self.closed = False

        def init_device(self) -> None:
            state["trace"].append((type(self).__name__, self.device.type))

        def shutdown(self) -> None:
            self.closed = True
            state["trace"].append((f"{type(self).__name__}.shutdown", self.device.type))

    class GPUWorker(BaseWorker):
        def __init__(self, vllm_config) -> None:
            super().__init__(vllm_config, "cuda")

    class NPUWorker(BaseWorker):
        def __init__(self, vllm_config) -> None:
            super().__init__(vllm_config, "npu")

    class GPUModelRunner:
        def __init__(self, vllm_config, device) -> None:
            self.vllm_config = vllm_config
            self.device = device

    class NPUModelRunner:
        def __init__(self, vllm_config, device) -> None:
            self.vllm_config = vllm_config
            self.device = device

    class FakeExecutor:
        def __init__(self, vllm_config) -> None:
            device_type = vllm_config.platform_device_type
            active_platform = state["active_platform"]
            if active_platform is not None and active_platform != device_type:
                raise RuntimeError("Residual platform state detected before executor init")

            state["active_platform"] = device_type
            state["active_engines"] += 1
            state["trace"].append(("executor", device_type))

            module_name, class_name = vllm_config.worker_cls.rsplit(".", 1)
            worker_cls = getattr(importlib.import_module(module_name), class_name)
            self.worker = worker_cls(vllm_config)
            self.worker.init_device()

        @classmethod
        def get_class(cls, _vllm_config) -> None:
            return cls

        def run_requests(self, requests) -> None:
            return [self.worker.model_runner.run(request) for request in requests]

        def shutdown(self) -> None:
            if getattr(self, "_shutdown_complete", False):
                return
            self._shutdown_complete = True
            if getattr(self, "worker", None) is not None:
                self.worker.shutdown()
            state["active_engines"] -= 1
            if state["active_engines"] <= 0:
                state["active_engines"] = 0
                state["active_platform"] = None
            state["trace"].append(("executor.shutdown", current_platform.device_type))

        close = shutdown

    class LLMEngine:
        def __init__(
            self,
            vllm_config,
            executor_class,
            log_stats,
            aggregate_engine_logging=False,
            usage_context=None,
            stat_loggers=None,
            mm_registry=None,
            use_cached_outputs=False,
            multiprocess_mode=False,
        ) -> None:
            self.vllm_config = vllm_config
            self.log_stats = log_stats
            self.tokenizer = object()
            self.executor = executor_class(vllm_config)
            self.model_executor = self.executor
            self._pending_requests = []

        def add_request(self, request_id, prompt, params) -> None:
            self._pending_requests.append(
                {
                    "request_id": request_id,
                    "prompt": prompt,
                    "params": params,
                }
            )

        def has_unfinished_requests(self) -> None:
            return bool(self._pending_requests)

        def step(self) -> None:
            requests = list(self._pending_requests)
            self._pending_requests.clear()
            return self.executor.run_requests(requests)

        def shutdown(self) -> None:
            if getattr(self, "_shutdown_complete", False):
                return None
            self._shutdown_complete = True

            executor = getattr(self, "executor", None)
            if executor is not None:
                shutdown = getattr(executor, "shutdown", None)
                if callable(shutdown):
                    shutdown()

            return None

    class EngineCoreRequest:
        pass

    class Request:
        def __init__(self, *args, **kwargs) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class NewRequestData:
        pass

    class FakeGRConfig:
        def __init__(self, max_seq_len=512, max_candidates=1000) -> None:
            self.max_seq_len = max_seq_len
            self.max_candidates = max_candidates

    class FakeGRInputProcessor:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeGROutputProcessor:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.num_pending_requests = 0

    class HSTUForRecommendation:
        def __init__(self, backend: str) -> None:
            self.backend = backend

        def recommend(self, prompt: dict[str, object]) -> list[int]:
            state["trace"].append(("hstu", self.backend))
            candidates = prompt["candidates"]
            return [candidate["item_candidate_id"] for candidate in candidates]

    class BaseGRRunner:
        runner_name = "base"

        def __init__(self, vllm_config, device, gr_config=None) -> None:
            self.vllm_config = vllm_config
            self.device = device
            self.model = HSTUForRecommendation(device.type)
            state["trace"].append((self.runner_name, device.type))

        def run(self, request: dict[str, object]) -> None:
            from vllm_gr.outputs import GRRequestOutput, RecommendedItem

            item_ids = self.model.recommend(request["prompt"])
            recommendations = [
                RecommendedItem(item_id=item_id, score=1.0 / (idx + 1))
                for idx, item_id in enumerate(item_ids)
            ]
            return GRRequestOutput(
                request_id=request["request_id"],
                user_id=request["prompt"]["user_id"],
                recommendations=recommendations,
                finished=True,
            )

        def shutdown(self) -> None:
            state["trace"].append((f"{self.runner_name}.shutdown", self.device.type))

        close = shutdown

    class GRModelRunner(BaseGRRunner):
        runner_name = "gr_model_runner"

    class GRNPUModelRunner(BaseGRRunner):
        runner_name = "gr_npu_model_runner"

    def init_logger(name) -> None:
        return logging.getLogger(name)

    fake_engine_arg_utils.EngineArgs = FakeEngineArgs
    fake_engine_arg_utils.AsyncEngineArgs = FakeAsyncEngineArgs
    fake_platforms.current_platform = current_platform
    fake_config.VllmConfig = SimpleNamespace
    fake_usage_lib.UsageContext = UsageContext
    fake_multimodal.MULTIMODAL_REGISTRY = object()
    fake_multimodal.MultiModalRegistry = MultiModalRegistry
    fake_metrics_loggers.StatLoggerFactory = StatLoggerFactory
    fake_entrypoints_llm.LLM = LLM
    fake_counter_module.Counter = Counter
    fake_pooling_params.PoolingParams = PoolingParams
    fake_outputs.PoolingOutput = PoolingOutput
    fake_outputs.PoolingRequestOutput = PoolingRequestOutput
    fake_logger.init_logger = init_logger
    fake_llm_engine_module.LLMEngine = LLMEngine
    fake_executor_module.Executor = FakeExecutor
    fake_executor_module.get_class = FakeExecutor.get_class
    fake_v1_engine_pkg.EngineCoreRequest = EngineCoreRequest
    fake_request_module.Request = Request
    fake_sched_output_module.NewRequestData = NewRequestData
    fake_gpu_worker_module.Worker = GPUWorker
    fake_gpu_runner_module.GPUModelRunner = GPUModelRunner
    fake_npu_worker_module.NPUWorker = NPUWorker
    fake_npu_runner_module.NPUModelRunner = NPUModelRunner

    fake_gr_config = types.ModuleType("vllm_gr.config.gr_config")
    fake_gr_input = types.ModuleType("vllm_gr.engine.gr_input_processor")
    fake_gr_output = types.ModuleType("vllm_gr.engine.gr_output_processor")
    fake_patch = types.ModuleType("vllm_gr.patch")
    fake_hf_config = types.ModuleType("vllm_gr.config.hstu_vllm_registration")
    fake_registry = types.ModuleType("vllm_gr.model_executor.models.registry")
    fake_hstu_module = types.ModuleType("vllm_gr.model_executor.models.hstu_vllm")
    fake_gr_model_runner = types.ModuleType("vllm_gr.worker.gr_model_runner")
    fake_gr_npu_model_runner = types.ModuleType("vllm_gr.worker.gr_npu_model_runner")
    fake_torchrec = types.ModuleType("torchrec")
    fake_torchrec_sparse = types.ModuleType("torchrec.sparse")
    fake_torchrec_jagged = types.ModuleType("torchrec.sparse.jagged_tensor")
    fake_v1_input_processor = types.ModuleType("vllm.v1.engine.input_processor")
    fake_sampling_params = types.ModuleType("vllm.sampling_params")
    fake_lora = types.ModuleType("vllm.lora")
    fake_lora_request = types.ModuleType("vllm.lora.request")
    fake_sequence = types.ModuleType("vllm.sequence")
    fake_v1_outputs = types.ModuleType("vllm.v1.outputs")
    fake_inputs = types.ModuleType("vllm.inputs")

    def re_apply_patches() -> None:
        state["patch_calls"] += 1
        state["trace"].append(("patch", current_platform.device_type))
        return state["patch_calls"]

    fake_patch.re_apply_patches = re_apply_patches
    fake_hf_config.ensure_hstu_config_registered = lambda: True
    fake_registry.register_gr_models_to_vllm = lambda: True
    fake_hstu_module.HSTUForRecommendation = HSTUForRecommendation
    fake_gr_config.GRConfig = FakeGRConfig
    fake_gr_input.GRInputProcessor = FakeGRInputProcessor
    fake_gr_output.GROutputProcessor = FakeGROutputProcessor
    fake_gr_output.GRRequestState = object
    fake_gr_model_runner.GRModelRunner = GRModelRunner
    fake_gr_npu_model_runner.GRNPUModelRunner = GRNPUModelRunner

    # Mock torchrec.KeyedJaggedTensor for adapter imports
    fake_torchrec_jagged.KeyedJaggedTensor = type("KeyedJaggedTensor", (), {})

    class FakeInputProcessor:
        pass

    fake_v1_input_processor.InputProcessor = FakeInputProcessor

    class FakeSamplingParams:
        pass

    fake_sampling_params.SamplingParams = FakeSamplingParams

    class FakeLoRARequest:
        pass

    fake_lora_request.LoRARequest = FakeLoRARequest

    class FakeIntermediateTensors:
        pass

    fake_sequence.IntermediateTensors = FakeIntermediateTensors

    class FakeModelRunnerOutput:
        pass

    fake_v1_outputs.ModelRunnerOutput = FakeModelRunnerOutput

    class FakePromptType:
        pass

    fake_inputs.PromptType = FakePromptType

    sys.modules.update(
        {
            "vllm": fake_vllm,
            "vllm.engine": fake_engine_pkg,
            "vllm.engine.arg_utils": fake_engine_arg_utils,
            "vllm.platforms": fake_platforms,
            "vllm.config": fake_config,
            "vllm.usage": fake_usage_pkg,
            "vllm.usage.usage_lib": fake_usage_lib,
            "vllm.v1": fake_v1,
            "vllm.v1.engine": fake_v1_engine_pkg,
            "vllm.v1.engine.llm_engine": fake_llm_engine_module,
            "vllm.v1.executor": fake_executor_module,
            "vllm.v1.request": fake_request_module,
            "vllm.v1.core": fake_core_pkg,
            "vllm.v1.core.sched": fake_sched_pkg,
            "vllm.v1.core.sched.output": fake_sched_output_module,
            "vllm.multimodal": fake_multimodal,
            "vllm.v1.metrics": fake_metrics_pkg,
            "vllm.v1.metrics.loggers": fake_metrics_loggers,
            "vllm.entrypoints": fake_entrypoints_pkg,
            "vllm.entrypoints.llm": fake_entrypoints_llm,
            "vllm.utils": fake_utils_pkg,
            "vllm.utils.counter": fake_counter_module,
            "vllm.pooling_params": fake_pooling_params,
            "vllm.outputs": fake_outputs,
            "vllm.logger": fake_logger,
            "vllm.v1.worker.gpu_worker": fake_gpu_worker_module,
            "vllm.v1.worker.gpu_model_runner": fake_gpu_runner_module,
            "vllm_ascend": fake_ascend_pkg,
            "vllm_ascend.worker": fake_ascend_worker_pkg,
            "vllm_ascend.worker.worker": fake_npu_worker_module,
            "vllm_ascend.worker.model_runner_v1": fake_npu_runner_module,
            "vllm_gr.config.gr_config": fake_gr_config,
            "vllm_gr.engine.gr_input_processor": fake_gr_input,
            "vllm_gr.engine.gr_output_processor": fake_gr_output,
            "vllm_gr.patch": fake_patch,
            "vllm_gr.config.hstu_vllm_registration": fake_hf_config,
            "vllm_gr.model_executor.models.registry": fake_registry,
            "vllm_gr.model_executor.models.hstu_vllm": fake_hstu_module,
            "vllm_gr.worker.gr_model_runner": fake_gr_model_runner,
            "vllm_gr.worker.gr_npu_model_runner": fake_gr_npu_model_runner,
            "torchrec": fake_torchrec,
            "torchrec.sparse": fake_torchrec_sparse,
            "torchrec.sparse.jagged_tensor": fake_torchrec_jagged,
            "vllm.v1.engine.input_processor": fake_v1_input_processor,
            "vllm.sampling_params": fake_sampling_params,
            "vllm.lora": fake_lora,
            "vllm.lora.request": fake_lora_request,
            "vllm.sequence": fake_sequence,
            "vllm.v1.outputs": fake_v1_outputs,
            "vllm.inputs": fake_inputs,
        }
    )

    try:
        yield SmokeHarness(state=state, current_platform=current_platform)
    finally:
        _purge_modules(*prefixes)
        _restore_modules(saved)
