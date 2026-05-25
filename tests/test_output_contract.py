# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for raw-score output semantics."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.machinery
import importlib.util
from pathlib import Path
from collections import namedtuple
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch


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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_stub_package(name: str) -> None:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module


def _load_jagged_entrypoint_module():
    _install_stub_package("vllm_gr.models")
    _install_stub_package("vllm_gr.models.dlrm_v3")
    _install_stub_package("vllm_gr.ops")
    _install_stub_package("vllm_gr.ops.pytorch")

    _load_module(
        "vllm_gr.models.dlrm_v3.common",
        ROOT / "vllm_gr" / "models" / "dlrm_v3" / "common.py",
    )
    _load_module(
        "vllm_gr.ops.pytorch.pt_jagged_tensors",
        ROOT / "vllm_gr" / "ops" / "pytorch" / "pt_jagged_tensors.py",
    )
    return _load_module(
        "vllm_gr.ops.jagged_tensors",
        ROOT / "vllm_gr" / "ops" / "jagged_tensors.py",
    )


def _load_pt_position_module():
    _install_stub_package("vllm_gr.models")
    _install_stub_package("vllm_gr.models.dlrm_v3")
    _install_stub_package("vllm_gr.ops")
    _install_stub_package("vllm_gr.ops.pytorch")

    _load_module(
        "vllm_gr.models.dlrm_v3.common",
        ROOT / "vllm_gr" / "models" / "dlrm_v3" / "common.py",
    )
    return _load_module(
        "vllm_gr.ops.pytorch.pt_position",
        ROOT / "vllm_gr" / "ops" / "pytorch" / "pt_position.py",
    )


@contextmanager
def _fake_vllm_outputs():
    saved = _purge_modules("vllm", "vllm.outputs", "vllm_gr.outputs")

    fake_vllm = types.ModuleType("vllm")
    fake_outputs = types.ModuleType("vllm.outputs")

    class PoolingOutput:
        def __init__(self, data):
            self.data = data

    class PoolingRequestOutput:
        def __init__(
            self,
            request_id,
            outputs,
            prompt_token_ids,
            num_cached_tokens,
            finished,
        ):
            self.request_id = request_id
            self.outputs = outputs
            self.prompt_token_ids = prompt_token_ids
            self.num_cached_tokens = num_cached_tokens
            self.finished = finished

    fake_outputs.PoolingOutput = PoolingOutput
    fake_outputs.PoolingRequestOutput = PoolingRequestOutput

    sys.modules.update({
        "vllm": fake_vllm,
        "vllm.outputs": fake_outputs,
    })

    try:
        yield
    finally:
        _purge_modules("vllm", "vllm.outputs", "vllm_gr.outputs")
        _restore_modules(saved)


@contextmanager
def _fake_dlrm_hstu_module():
    module_name = "vllm_gr.models.dlrm_v3.modules.dlrm_hstu"
    saved = _purge_modules(module_name)
    fake_module = types.ModuleType(module_name)
    fake_module.SequenceEmbedding = namedtuple(
        "SequenceEmbedding",
        ["lengths", "embedding"],
    )
    sys.modules[module_name] = fake_module
    try:
        yield fake_module.SequenceEmbedding
    finally:
        _purge_modules(module_name)
        _restore_modules(saved)


@contextmanager
def _fake_torchrec_embedding_config_module(*, with_bf16: bool):
    saved = _purge_modules(
        "torchrec",
        "torchrec.modules",
        "torchrec.modules.embedding_configs",
    )
    fake_torchrec = types.ModuleType("torchrec")
    fake_torchrec_modules = types.ModuleType("torchrec.modules")
    fake_embedding_configs = types.ModuleType("torchrec.modules.embedding_configs")

    class FakeDataType:
        FP16 = object()
        FP32 = object()
        if with_bf16:
            BF16 = object()

    class FakeEmbeddingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_embedding_configs.DataType = FakeDataType
    fake_embedding_configs.EmbeddingConfig = FakeEmbeddingConfig
    fake_torchrec.__spec__ = importlib.machinery.ModuleSpec("torchrec", loader=None)
    fake_torchrec_modules.__spec__ = importlib.machinery.ModuleSpec(
        "torchrec.modules", loader=None
    )
    fake_embedding_configs.__spec__ = importlib.machinery.ModuleSpec(
        "torchrec.modules.embedding_configs", loader=None
    )

    sys.modules.update(
        {
            "torchrec": fake_torchrec,
            "torchrec.modules": fake_torchrec_modules,
            "torchrec.modules.embedding_configs": fake_embedding_configs,
        }
    )

    try:
        yield
    finally:
        _purge_modules(
            "torchrec",
            "torchrec.modules",
            "torchrec.modules.embedding_configs",
        )
        _restore_modules(saved)


@contextmanager
def _fake_torchrec_jagged_tensor_module():
    saved = _purge_modules(
        "torchrec",
        "torchrec.sparse",
        "torchrec.sparse.jagged_tensor",
    )
    fake_torchrec = types.ModuleType("torchrec")
    fake_torchrec_sparse = types.ModuleType("torchrec.sparse")
    fake_jagged_tensor = types.ModuleType("torchrec.sparse.jagged_tensor")

    class FakeKeyedJaggedTensor:
        @classmethod
        def from_lengths_sync(cls, **kwargs):
            return types.SimpleNamespace(**kwargs)

    fake_jagged_tensor.KeyedJaggedTensor = FakeKeyedJaggedTensor
    fake_torchrec.__spec__ = importlib.machinery.ModuleSpec("torchrec", loader=None)
    fake_torchrec_sparse.__spec__ = importlib.machinery.ModuleSpec(
        "torchrec.sparse", loader=None
    )
    fake_jagged_tensor.__spec__ = importlib.machinery.ModuleSpec(
        "torchrec.sparse.jagged_tensor", loader=None
    )
    sys.modules.update(
        {
            "torchrec": fake_torchrec,
            "torchrec.sparse": fake_torchrec_sparse,
            "torchrec.sparse.jagged_tensor": fake_jagged_tensor,
        }
    )

    try:
        yield FakeKeyedJaggedTensor
    finally:
        _purge_modules(
            "torchrec",
            "torchrec.sparse",
            "torchrec.sparse.jagged_tensor",
        )
        _restore_modules(saved)


@contextmanager
def _fake_hstu_config_runtime():
    saved = _purge_modules(
        "transformers",
        "vllm",
        "vllm.transformers_utils",
        "vllm.transformers_utils.model_arch_config_convertor",
        "vllm_gr.config.hstu_config",
        "vllm_gr.config.hstu_vllm_registration",
        "hstu_hf_config_contract_test_module",
    )

    fake_transformers = types.ModuleType("transformers")

    class _PretrainedConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _AutoConfig:
        @classmethod
        def register(cls, *_args, **_kwargs):
            return None

    fake_transformers.PretrainedConfig = _PretrainedConfig
    fake_transformers.AutoConfig = _AutoConfig

    fake_vllm = types.ModuleType("vllm")
    fake_vllm_transformers_utils = types.ModuleType("vllm.transformers_utils")
    fake_convertor = types.ModuleType(
        "vllm.transformers_utils.model_arch_config_convertor"
    )
    fake_convertor.MODEL_ARCH_CONFIG_CONVERTORS = {}

    class _ModelArchConfigConvertorBase:
        pass

    fake_convertor.ModelArchConfigConvertorBase = _ModelArchConfigConvertorBase

    sys.modules.update(
        {
            "transformers": fake_transformers,
            "vllm": fake_vllm,
            "vllm.transformers_utils": fake_vllm_transformers_utils,
            "vllm.transformers_utils.model_arch_config_convertor": fake_convertor,
        }
    )

    try:
        yield
    finally:
        _purge_modules(
            "transformers",
            "vllm",
            "vllm.transformers_utils",
            "vllm.transformers_utils.model_arch_config_convertor",
            "vllm_gr.config.hstu_config",
            "vllm_gr.config.hstu_vllm_registration",
            "hstu_hf_config_contract_test_module",
        )
        _restore_modules(saved)


@contextmanager
def _fake_gr_input_processor_runtime():
    saved = _purge_modules(
        "vllm",
        "vllm.config",
        "vllm.inputs",
        "vllm.lora",
        "vllm.lora.request",
        "vllm.multimodal",
        "vllm.pooling_params",
        "vllm.sampling_params",
        "vllm.tokenizers",
        "vllm.v1",
        "vllm.v1.engine",
        "vllm.v1.engine.input_processor",
        "vllm_gr.engine",
        "gr_input_processor_contract_test_module",
    )

    fake_vllm = types.ModuleType("vllm")
    fake_config = types.ModuleType("vllm.config")
    fake_inputs = types.ModuleType("vllm.inputs")
    fake_lora_pkg = types.ModuleType("vllm.lora")
    fake_lora_request = types.ModuleType("vllm.lora.request")
    fake_multimodal = types.ModuleType("vllm.multimodal")
    fake_pooling_params = types.ModuleType("vllm.pooling_params")
    fake_sampling_params = types.ModuleType("vllm.sampling_params")
    fake_tokenizers = types.ModuleType("vllm.tokenizers")
    fake_v1 = types.ModuleType("vllm.v1")
    fake_v1_engine = types.ModuleType("vllm.v1.engine")
    fake_input_processor = types.ModuleType("vllm.v1.engine.input_processor")
    fake_gr_engine = types.ModuleType("vllm_gr.engine")

    class _InputProcessor:
        def __init__(self, *_args, **_kwargs):
            return None

        def process_inputs(self, *_args, **_kwargs):
            raise AssertionError("super().process_inputs() should not be called")

    class _PoolingParams:
        def __init__(self, task: str = "embed"):
            self.task = task

    class _SamplingParams:
        pass

    fake_config.VllmConfig = object
    fake_inputs.PromptType = object
    fake_lora_request.LoRARequest = object
    fake_multimodal.MULTIMODAL_REGISTRY = object()
    fake_multimodal.MultiModalRegistry = object
    fake_pooling_params.PoolingParams = _PoolingParams
    fake_sampling_params.SamplingParams = _SamplingParams
    fake_tokenizers.TokenizerLike = object
    fake_v1_engine.EngineCoreRequest = object
    fake_input_processor.InputProcessor = _InputProcessor
    fake_gr_engine.GRFeaturesPayload = object
    fake_gr_engine.GREngineCoreRequest = object

    sys.modules.update(
        {
            "vllm": fake_vllm,
            "vllm.config": fake_config,
            "vllm.inputs": fake_inputs,
            "vllm.lora": fake_lora_pkg,
            "vllm.lora.request": fake_lora_request,
            "vllm.multimodal": fake_multimodal,
            "vllm.pooling_params": fake_pooling_params,
            "vllm.sampling_params": fake_sampling_params,
            "vllm.tokenizers": fake_tokenizers,
            "vllm.v1": fake_v1,
            "vllm.v1.engine": fake_v1_engine,
            "vllm.v1.engine.input_processor": fake_input_processor,
            "vllm_gr.engine": fake_gr_engine,
        }
    )

    try:
        yield
    finally:
        _purge_modules(
            "vllm",
            "vllm.config",
            "vllm.inputs",
            "vllm.lora",
            "vllm.lora.request",
            "vllm.multimodal",
            "vllm.pooling_params",
            "vllm.sampling_params",
            "vllm.tokenizers",
            "vllm.v1",
            "vllm.v1.engine",
            "vllm.v1.engine.input_processor",
            "vllm_gr.engine",
            "gr_input_processor_contract_test_module",
        )
        _restore_modules(saved)


@contextmanager
def _fake_arg_utils_runtime():
    saved = _purge_modules(
        "vllm",
        "vllm.engine",
        "vllm.engine.arg_utils",
        "vllm.platforms",
        "arg_utils_contract_test_module",
    )

    fake_vllm = types.ModuleType("vllm")
    fake_engine_pkg = types.ModuleType("vllm.engine")
    fake_engine_arg_utils = types.ModuleType("vllm.engine.arg_utils")
    fake_platforms = types.ModuleType("vllm.platforms")

    from dataclasses import dataclass

    @dataclass
    class _EngineArgs:
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

        def __post_init__(self):
            return None

    @dataclass
    class _AsyncEngineArgs(_EngineArgs):
        enable_log_requests: bool = False

    fake_engine_arg_utils.EngineArgs = _EngineArgs
    fake_engine_arg_utils.AsyncEngineArgs = _AsyncEngineArgs
    fake_platforms.current_platform = types.SimpleNamespace(device_type="cuda")

    sys.modules.update(
        {
            "vllm": fake_vllm,
            "vllm.engine": fake_engine_pkg,
            "vllm.engine.arg_utils": fake_engine_arg_utils,
            "vllm.platforms": fake_platforms,
        }
    )

    try:
        yield
    finally:
        _purge_modules(
            "vllm",
            "vllm.engine",
            "vllm.engine.arg_utils",
            "vllm.platforms",
            "arg_utils_contract_test_module",
        )
        _restore_modules(saved)


@contextmanager
def _fake_gr_engine_runtime():
    saved = _purge_modules(
        "vllm",
        "vllm.config",
        "vllm.usage",
        "vllm.usage.usage_lib",
        "vllm.v1",
        "vllm.v1.engine",
        "vllm.v1.engine.llm_engine",
        "vllm.v1.executor",
        "vllm.multimodal",
        "vllm.v1.metrics",
        "vllm.v1.metrics.loggers",
        "vllm_gr.config.gr_config",
        "vllm_gr.engine.arg_utils",
        "vllm_gr.engine.gr_input_processor",
        "vllm_gr.engine.gr_output_processor",
        "vllm_gr.engine.gr_engine",
    )

    fake_vllm = types.ModuleType("vllm")
    fake_vllm_config = types.ModuleType("vllm.config")
    fake_vllm_config.VllmConfig = object

    fake_usage_pkg = types.ModuleType("vllm.usage")
    fake_usage = types.ModuleType("vllm.usage.usage_lib")

    class _UsageContext:
        ENGINE_CONTEXT = object()
        LLM_CLASS = object()

    fake_usage.UsageContext = _UsageContext

    fake_v1_pkg = types.ModuleType("vllm.v1")
    fake_v1_engine_pkg = types.ModuleType("vllm.v1.engine")
    fake_llm_engine = types.ModuleType("vllm.v1.engine.llm_engine")
    fake_executor = types.ModuleType("vllm.v1.executor")
    fake_metrics_pkg = types.ModuleType("vllm.v1.metrics")
    fake_metrics = types.ModuleType("vllm.v1.metrics.loggers")

    class _LLMEngine:
        pass

    class _Executor:
        @staticmethod
        def get_class(_config):
            return object

    fake_llm_engine.LLMEngine = _LLMEngine
    fake_executor.Executor = _Executor
    fake_metrics.StatLoggerFactory = object

    fake_multimodal = types.ModuleType("vllm.multimodal")
    fake_multimodal.MULTIMODAL_REGISTRY = object()

    class _MultiModalRegistry:
        pass

    fake_multimodal.MultiModalRegistry = _MultiModalRegistry

    fake_gr_config = types.ModuleType("vllm_gr.config.gr_config")

    class _GRConfig:
        def __init__(self, max_seq_len=512, max_candidates=1000):
            self.max_seq_len = max_seq_len
            self.max_candidates = max_candidates

    fake_gr_config.GRConfig = _GRConfig

    fake_arg_utils = types.ModuleType("vllm_gr.engine.arg_utils")
    fake_arg_utils.GREngineArgs = object

    fake_input_processor = types.ModuleType("vllm_gr.engine.gr_input_processor")
    fake_input_processor.GRInputProcessor = object

    fake_output_processor = types.ModuleType("vllm_gr.engine.gr_output_processor")
    fake_output_processor.GROutputProcessor = object

    sys.modules.update(
        {
            "vllm": fake_vllm,
            "vllm.config": fake_vllm_config,
            "vllm.usage": fake_usage_pkg,
            "vllm.usage.usage_lib": fake_usage,
            "vllm.v1": fake_v1_pkg,
            "vllm.v1.engine": fake_v1_engine_pkg,
            "vllm.v1.engine.llm_engine": fake_llm_engine,
            "vllm.v1.executor": fake_executor,
            "vllm.v1.metrics": fake_metrics_pkg,
            "vllm.v1.metrics.loggers": fake_metrics,
            "vllm.multimodal": fake_multimodal,
            "vllm_gr.config.gr_config": fake_gr_config,
            "vllm_gr.engine.arg_utils": fake_arg_utils,
            "vllm_gr.engine.gr_input_processor": fake_input_processor,
            "vllm_gr.engine.gr_output_processor": fake_output_processor,
        }
    )

    try:
        yield
    finally:
        _purge_modules(
            "vllm",
            "vllm.config",
            "vllm.usage",
            "vllm.usage.usage_lib",
            "vllm.v1",
            "vllm.v1.engine",
            "vllm.v1.engine.llm_engine",
            "vllm.v1.executor",
            "vllm.multimodal",
            "vllm.v1.metrics",
            "vllm.v1.metrics.loggers",
            "vllm_gr.config.gr_config",
            "vllm_gr.engine.arg_utils",
            "vllm_gr.engine.gr_input_processor",
            "vllm_gr.engine.gr_output_processor",
            "vllm_gr.engine.gr_engine",
        )
        _restore_modules(saved)


class TestOutputContract(unittest.TestCase):

    def test_top_recommendation_uses_max_score_without_reordering(self):
        with _fake_vllm_outputs():
            from vllm_gr.outputs import GRRequestOutput, RecommendedItem

            output = GRRequestOutput(
                request_id="req-1",
                user_id="u1",
                recommendations=[
                    RecommendedItem(item_id=101, score=0.1),
                    RecommendedItem(item_id=102, score=0.9),
                    RecommendedItem(item_id=103, score=0.4),
                ],
                finished=True,
            )

            self.assertEqual(output.item_ids, [101, 102, 103])
            self.assertEqual(output.scores, [0.1, 0.9, 0.4])
            self.assertEqual(output.top_recommendation.item_id, 102)
            self.assertEqual(output.top_recommendation.score, 0.9)

    def test_to_dict_preserves_optional_task_scores(self):
        with _fake_vllm_outputs():
            from vllm_gr.outputs import GRRequestOutput, RecommendedItem

            output = GRRequestOutput(
                request_id="req-2",
                user_id="u2",
                recommendations=[
                    RecommendedItem(
                        item_id=201,
                        score=0.3,
                        task_scores={"is_click": 0.3, "is_like": 0.8},
                    ),
                ],
                finished=True,
            )

            data = output.to_dict()

            self.assertEqual(data["recommendations"][0]["item_id"], 201)
            self.assertEqual(
                data["recommendations"][0]["task_scores"],
                {"is_click": 0.3, "is_like": 0.8},
            )

    def test_sparse_transfer_preserves_embedding_dtype_without_target_dtype(self):
        with _fake_dlrm_hstu_module() as SequenceEmbedding:
            from vllm_gr.utils.hstu_runtime import move_hstu_sparse_outputs_to_device

            seq_embeddings = {
                "item_id": SequenceEmbedding(
                    lengths=torch.tensor([2], dtype=torch.int64),
                    embedding=torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0]],
                        dtype=torch.float16,
                    ),
                ),
            }
            payload_features = {
                "payload": torch.tensor([1.0, 2.0], dtype=torch.float16),
            }
            uih_seq_lengths = torch.tensor([2], dtype=torch.int64)
            num_candidates = torch.tensor([1], dtype=torch.int64)

            moved_seq_embeddings, moved_payload_features, _, _ = (
                move_hstu_sparse_outputs_to_device(
                    seq_embeddings=seq_embeddings,
                    payload_features=payload_features,
                    uih_seq_lengths=uih_seq_lengths,
                    num_candidates=num_candidates,
                    device=torch.device("cpu"),
                    source_device=torch.device("cuda"),
                )
            )

            self.assertEqual(
                moved_seq_embeddings["item_id"].embedding.dtype,
                torch.float16,
            )
            self.assertEqual(
                moved_payload_features["payload"].dtype,
                torch.float16,
            )

    def test_sparse_transfer_casts_embeddings_to_dense_forward_dtype(self):
        with _fake_dlrm_hstu_module() as SequenceEmbedding:
            from vllm_gr.utils.hstu_runtime import move_hstu_sparse_outputs_to_device

            seq_embeddings = {
                "item_id": SequenceEmbedding(
                    lengths=torch.tensor([2], dtype=torch.int64),
                    embedding=torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0]],
                        dtype=torch.float16,
                    ),
                ),
            }
            payload_features = {
                "payload": torch.tensor([1.0, 2.0], dtype=torch.float16),
            }
            uih_seq_lengths = torch.tensor([2], dtype=torch.int64)
            num_candidates = torch.tensor([1], dtype=torch.int64)

            moved_seq_embeddings, moved_payload_features, _, _ = (
                move_hstu_sparse_outputs_to_device(
                    seq_embeddings=seq_embeddings,
                    payload_features=payload_features,
                    uih_seq_lengths=uih_seq_lengths,
                    num_candidates=num_candidates,
                    device=torch.device("cpu"),
                    source_device=torch.device("cuda"),
                    target_dtype=torch.bfloat16,
                )
            )

            self.assertEqual(
                moved_seq_embeddings["item_id"].embedding.dtype,
                torch.bfloat16,
            )
            self.assertEqual(
                moved_payload_features["payload"].dtype,
                torch.float16,
            )

    def test_runtime_devices_default_dense_to_active_accelerator(self):
        from vllm_gr.utils import hstu_runtime

        with mock.patch.object(
            hstu_runtime,
            "is_npu_available",
            return_value=False,
        ), mock.patch.object(
            hstu_runtime,
            "is_cuda_available",
            return_value=True,
        ), mock.patch.object(
            torch.cuda,
            "current_device",
            return_value=3,
        ):
            devices = hstu_runtime.resolve_hstu_runtime_devices()

        self.assertEqual(devices.dense_device, torch.device("cuda", 3))
        self.assertEqual(devices.embedding_device, torch.device("cpu"))

    def test_sparse_inputs_move_to_embedding_device_before_preprocess(self):
        class FakeTensor:
            def __init__(self, value: torch.Tensor, device: torch.device):
                self._value = value
                self.device = device

            def to(self, device):
                return self._value.to(device)

        class FakeFeatures:
            def __init__(self, keys, values, lengths):
                self._keys = keys
                self._values = values
                self._lengths = lengths

            def keys(self):
                return self._keys

            def values(self):
                return self._values

            def lengths(self):
                return self._lengths

            def weights_or_none(self):
                return None

        with _fake_torchrec_jagged_tensor_module():
            from vllm_gr.utils.hstu_runtime import move_hstu_sparse_inputs_to_device

            fake_values = FakeTensor(
                torch.tensor([1, 2, 3], dtype=torch.int64),
                torch.device("cuda", 0),
            )
            fake_lengths = FakeTensor(
                torch.tensor([2, 1], dtype=torch.int64),
                torch.device("cuda", 0),
            )
            features = FakeFeatures(
                keys=["item_id"],
                values=fake_values,
                lengths=fake_lengths,
            )

            moved_uih, moved_candidates = move_hstu_sparse_inputs_to_device(
                uih_features=features,
                candidates_features=features,
                device=torch.device("cpu"),
            )

        self.assertEqual(moved_uih.values.device, torch.device("cpu"))
        self.assertEqual(moved_uih.lengths.device, torch.device("cpu"))
        self.assertEqual(moved_candidates.values.device, torch.device("cpu"))
        self.assertEqual(moved_candidates.lengths.device, torch.device("cpu"))

    def test_initialize_runtime_fails_fast_when_registration_fails(self):
        plugin_path = ROOT / "vllm_gr" / "plugin.py"
        saved = _purge_modules(
            "vllm_gr.plugin",
            "vllm_gr.patch",
            "vllm_gr.config.hstu_vllm_registration",
            "vllm_gr.model_executor.models.registry",
        )

        fake_patch = types.ModuleType("vllm_gr.patch")
        fake_patch.re_apply_patches = lambda: 0

        fake_config = types.ModuleType("vllm_gr.config.hstu_vllm_registration")
        fake_config.ensure_hstu_config_registered = lambda: False

        fake_registry = types.ModuleType("vllm_gr.model_executor.models.registry")
        fake_registry.register_gr_model_architectures = lambda: True

        sys.modules.update(
            {
                "vllm_gr.patch": fake_patch,
                "vllm_gr.config.hstu_vllm_registration": fake_config,
                "vllm_gr.model_executor.models.registry": fake_registry,
            }
        )

        try:
            module = _load_module("vllm_gr.plugin", plugin_path)
            with self.assertRaisesRegex(RuntimeError, "config registration"):
                module.initialize_runtime()
        finally:
            _purge_modules(
                "vllm_gr.plugin",
                "vllm_gr.patch",
                "vllm_gr.config.hstu_vllm_registration",
                "vllm_gr.model_executor.models.registry",
            )
            _restore_modules(saved)

    def test_candidate_primary_id_resolution_requires_explicit_contract(self):
        adapter_path = ROOT / "vllm_gr" / "adapters" / "hstu" / "adapter.py"

        with _fake_gr_engine_runtime():
            module = _load_module("adapter_contract_test_module", adapter_path)

        # The resolver now lives on HSTUModelAdapter._resolve_candidate_primary_id_key_from_config
        resolver = module.HSTUModelAdapter._resolve_candidate_primary_id_key_from_config

        explicit_config = types.SimpleNamespace(
            candidate_primary_id_feature_name="explicit_item_id",
            merge_uih_candidate_feature_mapping=[],
            uih_post_id_feature_name="item_id",
        )
        self.assertEqual(
            resolver(
                explicit_config,
                ["explicit_item_id", "category_id"],
            ),
            "explicit_item_id",
        )

        mapped_config = types.SimpleNamespace(
            candidate_primary_id_feature_name="",
            merge_uih_candidate_feature_mapping=[
                ["item_id", "item_candidate_id"],
            ],
            uih_post_id_feature_name="item_id",
        )
        self.assertEqual(
            resolver(
                mapped_config,
                ["item_candidate_id", "category_id"],
            ),
            "item_candidate_id",
        )

        missing_contract_config = types.SimpleNamespace(
            candidate_primary_id_feature_name="",
            merge_uih_candidate_feature_mapping=[],
            uih_post_id_feature_name="item_id",
        )
        with self.assertRaisesRegex(ValueError, "Unable to resolve candidate primary id"):
            resolver(
                missing_contract_config,
                ["item_candidate_id", "category_id"],
            )

    def test_gr_engine_shutdown_delegates_to_engine_core(self):
        engine_path = ROOT / "vllm_gr" / "engine" / "gr_engine.py"

        with _fake_gr_engine_runtime():
            module = _load_module("gr_engine_shutdown_test_module", engine_path)

        engine = module.GRLLMEngine.__new__(module.GRLLMEngine)
        fake_engine_core = mock.Mock()
        engine.engine_core = fake_engine_core

        engine.shutdown()

        fake_engine_core.shutdown.assert_called_once_with()
        self.assertIsNone(engine.engine_core)

    def test_hstu_config_allows_unresolved_candidate_primary_id_until_runtime(self):
        config_path = ROOT / "vllm_gr" / "config" / "hstu_hf_config.py"

        with _fake_hstu_config_runtime():
            module = _load_module(
                "hstu_hf_config_contract_test_module",
                config_path,
            )

        HSTUConfig = module.HSTUConfig
        shared_kwargs = {
            "user_embedding_feature_names": ["item_id"],
            "item_embedding_feature_names": ["item_candidate_id"],
            "hstu_uih_feature_names": ["item_id"],
            "hstu_candidate_feature_names": ["item_candidate_id"],
            "embedding_tables": [
                {
                    "name": "item_embedding",
                    "embedding_dim": 16,
                    "num_embeddings": 32,
                    "feature_names": ["item_id", "item_candidate_id"],
                },
            ],
        }

        config = HSTUConfig(
            merge_uih_candidate_feature_mapping=[["item_id", "item_candidate_id"]],
            **shared_kwargs,
        )
        self.assertEqual(
            config.candidate_primary_id_feature_name,
            "item_candidate_id",
        )

        unresolved = HSTUConfig(**shared_kwargs)
        self.assertEqual(unresolved.candidate_primary_id_feature_name, "")

    def test_hstu_config_rejects_candidate_primary_id_outside_item_embeddings(self):
        config_path = ROOT / "vllm_gr" / "config" / "hstu_hf_config.py"

        with _fake_hstu_config_runtime():
            module = _load_module(
                "hstu_hf_config_item_id_contract_test_module",
                config_path,
            )

        HSTUConfig = module.HSTUConfig
        shared_kwargs = {
            "user_embedding_feature_names": ["item_id"],
            "item_embedding_feature_names": ["item_candidate_id"],
            "hstu_uih_feature_names": ["item_id"],
            "hstu_candidate_feature_names": ["item_candidate_id", "item_query_time"],
            "embedding_tables": [
                {
                    "name": "item_embedding",
                    "embedding_dim": 16,
                    "num_embeddings": 32,
                    "feature_names": ["item_id", "item_candidate_id"],
                },
            ],
        }

        with self.assertRaisesRegex(ValueError, "item_embedding_feature_names"):
            HSTUConfig(
                candidate_primary_id_feature_name="item_query_time",
                **shared_kwargs,
            )

        with self.assertRaisesRegex(ValueError, "item_embedding_feature_names"):
            HSTUConfig(
                merge_uih_candidate_feature_mapping=[["item_id", "item_query_time"]],
                **shared_kwargs,
            )

    def test_gr_input_processor_rejects_non_gr_prompt(self):
        input_processor_path = ROOT / "vllm_gr" / "engine" / "gr_input_processor.py"

        with _fake_gr_input_processor_runtime():
            module = _load_module(
                "gr_input_processor_contract_test_module",
                input_processor_path,
            )

        processor = module.GRInputProcessor(
            vllm_config=None,
            tokenizer=None,
        )

        with self.assertRaisesRegex(TypeError, "only supports GRPrompt"):
            processor.process_inputs(
                request_id="req-1",
                prompt="not-a-gr-prompt",
                params=object(),
            )

    def test_arg_utils_rejects_local_model_dir_without_weights(self):
        arg_utils_path = ROOT / "vllm_gr" / "engine" / "arg_utils.py"

        with _fake_arg_utils_runtime():
            module = _load_module(
                "arg_utils_contract_test_module",
                arg_utils_path,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "No model weight files found"):
                module.GREngineArgs(
                    model=tmpdir,
                    model_family="hstu",
                )

    def test_arg_utils_allows_explicit_dummy_loader_for_weightless_local_model_dir(self):
        arg_utils_path = ROOT / "vllm_gr" / "engine" / "arg_utils.py"

        with _fake_arg_utils_runtime():
            module = _load_module(
                "arg_utils_contract_test_module",
                arg_utils_path,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            args = module.GREngineArgs(
                model=tmpdir,
                model_family="hstu",
                allow_missing_weights=True,
            )

        self.assertEqual(args.load_format, "dummy")

    def test_arg_utils_fails_fast_on_invalid_local_config(self):
        arg_utils_path = ROOT / "vllm_gr" / "engine" / "arg_utils.py"

        with _fake_arg_utils_runtime():
            module = _load_module(
                "arg_utils_contract_test_module",
                arg_utils_path,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("{ invalid json", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Failed to read local model config"):
                module.GREngineArgs(model=tmpdir)

    def test_reliable_ops_fails_fast_when_accelerated_fbgemm_call_breaks(self):
        from vllm_gr.utils import reliable_ops as module

        fake_adapter = types.SimpleNamespace(
            should_use_fbgemm=lambda _device: True,
        )
        lengths = torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu")

        with mock.patch.object(module, "get_device_adapter", return_value=fake_adapter):
            with mock.patch.object(
                module.torch.ops.fbgemm,
                "asynchronous_complete_cumsum",
                side_effect=RuntimeError("backend failure"),
                create=True,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "failed on accelerated device",
                ):
                    module.ReliableOps.asynchronous_complete_cumsum(lengths)

    def test_reliable_ops_keeps_cpu_fallback_when_no_fbgemm_backend(self):
        from vllm_gr.utils import reliable_ops as module

        fake_adapter = types.SimpleNamespace(
            should_use_fbgemm=lambda _device: False,
        )
        lengths = torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu")

        with mock.patch.object(module, "get_device_adapter", return_value=fake_adapter):
            result = module.ReliableOps.asynchronous_complete_cumsum(lengths)

        torch.testing.assert_close(
            result,
            torch.tensor([0, 1, 3, 6], dtype=torch.int64),
        )

    def test_pt_position_uses_fbgemm_jagged_add_on_accelerated_backend(self):
        module = _load_pt_position_module()
        fake_adapter = types.SimpleNamespace(
            should_use_fbgemm=lambda _device: True,
        )
        expected = torch.tensor([[9.0], [8.0]], dtype=torch.float32)

        with mock.patch.object(module, "get_device_adapter", return_value=fake_adapter):
            with mock.patch.object(
                module.torch.ops.fbgemm,
                "jagged_dense_elementwise_add_jagged_output",
                return_value=(expected,),
                create=True,
            ) as mocked_op:
                result = module.pytorch_add_timestamp_positional_embeddings(
                    seq_embeddings=torch.zeros(2, 1, dtype=torch.float32),
                    seq_offsets=torch.tensor([0, 2], dtype=torch.int64),
                    pos_embeddings=torch.zeros(4, 1, dtype=torch.float32),
                    ts_embeddings=torch.zeros(3, 1, dtype=torch.float32),
                    timestamps=torch.tensor([1, 2], dtype=torch.int64),
                    max_seq_len=2,
                    max_contextual_seq_len=0,
                    seq_lengths=torch.tensor([2], dtype=torch.int64),
                    num_targets=None,
                    interleave_targets=False,
                    time_bucket_fn="sqrt",
                )

        self.assertTrue(mocked_op.called)
        self.assertIs(result, expected)

    def test_pt_position_keeps_cpu_fallback_without_fbgemm_backend(self):
        module = _load_pt_position_module()
        fake_adapter = types.SimpleNamespace(
            should_use_fbgemm=lambda _device: False,
        )

        with mock.patch.object(module, "get_device_adapter", return_value=fake_adapter):
            with mock.patch.object(
                module.torch.ops.fbgemm,
                "jagged_dense_elementwise_add_jagged_output",
                side_effect=AssertionError("accelerated op should not run"),
                create=True,
            ):
                result = module.pytorch_add_timestamp_positional_embeddings(
                    seq_embeddings=torch.tensor(
                        [[1.0], [2.0]],
                        dtype=torch.float32,
                    ),
                    seq_offsets=torch.tensor([0, 2], dtype=torch.int64),
                    pos_embeddings=torch.tensor(
                        [[0.0], [10.0], [20.0], [30.0]],
                        dtype=torch.float32,
                    ),
                    ts_embeddings=torch.tensor(
                        [[100.0], [200.0], [300.0]],
                        dtype=torch.float32,
                    ),
                    timestamps=torch.tensor([10, 70], dtype=torch.int64),
                    max_seq_len=2,
                    max_contextual_seq_len=0,
                    seq_lengths=torch.tensor([2], dtype=torch.int64),
                    num_targets=None,
                    interleave_targets=False,
                    time_bucket_fn="sqrt",
                )

        torch.testing.assert_close(
            result,
            torch.tensor([[121.0], [112.0]], dtype=torch.float32),
        )

    def test_pooling_split_mismatch_fails_fast(self):
        from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

        with self.assertRaisesRegex(ValueError, "pooling split mismatch"):
            HSTUForRecommendation._split_pooling_predictions(
                predictions=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
                num_candidates=torch.tensor([1, 1], dtype=torch.int64),
            )

    def test_hstu_l2_helpers_preserve_contextual_insertion_semantics(self):
        module_path = (
            ROOT / "vllm_gr" / "ops" / "pytorch" / "pt_jagged_tensors.py"
        )
        module = _load_module(
            "pt_jagged_tensors_contract_test_module",
            module_path,
        )

        contextual_seq_len = 2
        max_prefix_len = 2
        max_l2_len = 4
        max_seq_len = max_prefix_len + max_l2_len

        prefix_offsets = torch.tensor([0, 2, 3], dtype=torch.int64)
        l2_offsets = torch.tensor([0, 4, 7], dtype=torch.int64)

        x = torch.tensor(
            [
                [10.0],
                [11.0],
                [20.0],
                [21.0],
                [30.0],
                [31.0],
                [110.0],
                [111.0],
                [120.0],
                [130.0],
            ],
            dtype=torch.float32,
        )

        prefix_x, l2_x = module.pytorch_hstu_split_l2_embeddings(
            max_seq_len=max_seq_len,
            x=x,
            prefix_offsets=prefix_offsets,
            l2_offsets=l2_offsets,
            contextual_seq_len=contextual_seq_len,
        )

        expected_prefix_x = torch.tensor(
            [[20.0], [21.0], [120.0]],
            dtype=torch.float32,
        )
        expected_l2_x = torch.tensor(
            [[10.0], [11.0], [30.0], [31.0], [110.0], [111.0], [130.0]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(prefix_x, expected_prefix_x)
        torch.testing.assert_close(l2_x, expected_l2_x)

        reconstructed_x = module.pytorch_hstu_concat_l2_embeddings(
            max_prefix_len=max_prefix_len,
            prefix_x=prefix_x,
            prefix_offsets=prefix_offsets,
            max_l2_len=max_l2_len,
            l2_x=l2_x,
            l2_offsets=l2_offsets,
            contextual_seq_len=contextual_seq_len,
        )
        torch.testing.assert_close(reconstructed_x, x)

    def test_plain_jagged_split_matches_golden_left_right_semantics(self):
        module_path = (
            ROOT / "vllm_gr" / "ops" / "pytorch" / "pt_jagged_tensors.py"
        )
        module = _load_module(
            "pt_jagged_tensors_plain_split_contract_test_module",
            module_path,
        )

        offsets_left = torch.tensor([0, 2, 3], dtype=torch.int64)
        offsets_right = torch.tensor([0, 1, 3], dtype=torch.int64)
        values = torch.tensor(
            [
                [10.0],
                [11.0],
                [20.0],
                [110.0],
                [120.0],
                [121.0],
            ],
            dtype=torch.float32,
        )

        left, right = module.pytorch_split_2D_jagged(
            max_seq_len=4,
            values=values,
            offsets_left=offsets_left,
            offsets_right=offsets_right,
        )

        torch.testing.assert_close(
            left,
            torch.tensor([[10.0], [11.0], [110.0]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            right,
            torch.tensor([[20.0], [120.0], [121.0]], dtype=torch.float32),
        )

    def test_plain_jagged_concat_reconstructs_split_values(self):
        module_path = (
            ROOT / "vllm_gr" / "ops" / "pytorch" / "pt_jagged_tensors.py"
        )
        module = _load_module(
            "pt_jagged_tensors_plain_concat_contract_test_module",
            module_path,
        )

        offsets_left = torch.tensor([0, 2, 3], dtype=torch.int64)
        offsets_right = torch.tensor([0, 1, 3], dtype=torch.int64)
        left = torch.tensor([[10.0], [11.0], [110.0]], dtype=torch.float32)
        right = torch.tensor([[20.0], [120.0], [121.0]], dtype=torch.float32)

        reconstructed = module.pytorch_concat_2D_jagged(
            values_left=left,
            values_right=right,
            offsets_left=offsets_left,
            offsets_right=offsets_right,
        )

        torch.testing.assert_close(
            reconstructed,
            torch.tensor(
                [
                    [10.0],
                    [11.0],
                    [20.0],
                    [110.0],
                    [120.0],
                    [121.0],
                ],
                dtype=torch.float32,
            ),
        )

    def test_jagged_entrypoint_rejects_mismatched_feature_widths(self):
        module = _load_jagged_entrypoint_module()

        with self.assertRaisesRegex(AssertionError, "shape\\[1\\].*equal"):
            module.concat_2D_jagged(
                max_seq_len=4,
                values_left=torch.zeros(2, 3),
                values_right=torch.zeros(2, 4),
                offsets_left=torch.tensor([0, 2], dtype=torch.int64),
                offsets_right=torch.tensor([0, 2], dtype=torch.int64),
            )

    def test_jagged_entrypoint_requires_split_length_contract(self):
        module = _load_jagged_entrypoint_module()

        with self.assertRaisesRegex(
            AssertionError,
            "offsets_left and offsets_right cannot be None",
        ):
            module.split_2D_jagged(
                max_seq_len=4,
                values=torch.zeros(4, 1),
            )

    def test_embedding_bf16_requires_runtime_support(self):
        model_path = ROOT / "vllm_gr" / "model_executor" / "models" / "hstu_vllm.py"
        saved = _purge_modules("hstu_vllm_bf16_contract_test_module")

        with _fake_torchrec_embedding_config_module(with_bf16=False):
            module = _load_module("hstu_vllm_bf16_contract_test_module", model_path)
            model = object.__new__(module.HSTUForRecommendation)
            model.config = types.SimpleNamespace(
                embedding_table_data_type="bf16",
                embedding_tables=[
                    {
                        "name": "item_id",
                        "embedding_dim": 8,
                        "num_embeddings": 16,
                        "feature_names": ["item_id"],
                    },
                ],
            )

            with self.assertRaisesRegex(ValueError, "bf16.*not supported"):
                model._build_embedding_configs()

        _purge_modules("hstu_vllm_bf16_contract_test_module")
        _restore_modules(saved)


if __name__ == "__main__":
    unittest.main()
