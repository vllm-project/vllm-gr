# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for GRAsyncLLM adapter wiring."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import types
import unittest

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


@contextmanager
def _fake_async_engine_runtime():
    """Set up fakes for all modules imported by async_gr_engine.py."""
    saved = _purge_modules(
        "vllm",
        "vllm.config",
        "vllm.lora",
        "vllm.lora.request",
        "vllm.pooling_params",
        "vllm.usage",
        "vllm.usage.usage_lib",
        "vllm.v1",
        "vllm.v1.engine",
        "vllm.v1.engine.async_llm",
        "vllm.v1.metrics",
        "vllm.v1.metrics.loggers",
        "vllm_gr.config.gr_config",
        "vllm_gr.engine.gr_input_processor",
        "vllm_gr.engine.gr_output_processor",
        "vllm_gr.inputs",
        "vllm_gr.inputs.data",
        "vllm_gr.outputs",
        "vllm_gr.adapters",
        "vllm_gr.adapters.base",
        "vllm_gr.adapters.capabilities",
        "vllm_gr.adapters.hstu",
        "vllm_gr.adapters.hstu.adapter",
        "vllm_gr.adapters.hstu.feature_extractor",
        "vllm_gr.adapters.hstu.kjt_builder",
    )

    # vllm core
    fake_vllm = types.ModuleType("vllm")
    fake_vllm_config = types.ModuleType("vllm.config")
    fake_vllm_config.VllmConfig = object

    fake_usage_pkg = types.ModuleType("vllm.usage")
    fake_usage = types.ModuleType("vllm.usage.usage_lib")

    class _UsageContext:
        ENGINE_CONTEXT = object()

    fake_usage.UsageContext = _UsageContext

    # v1 async engine
    _install_stub_package("vllm.v1")
    _install_stub_package("vllm.v1.engine")

    class _AsyncLLM:
        pass

    fake_async_llm = types.ModuleType("vllm.v1.engine.async_llm")
    fake_async_llm.AsyncLLM = _AsyncLLM

    # Other vllm deps
    fake_lora_pkg = types.ModuleType("vllm.lora")
    fake_lora = types.ModuleType("vllm.lora.request")
    fake_lora.LoRARequest = type("LoRARequest", (), {})

    fake_pooling = types.ModuleType("vllm.pooling_params")

    class _PoolingParams:
        def __init__(self, task=""):
            self.task = task

    fake_pooling.PoolingParams = _PoolingParams

    fake_metrics_pkg = types.ModuleType("vllm.v1.metrics")
    fake_metrics = types.ModuleType("vllm.v1.metrics.loggers")
    fake_metrics.StatLoggerFactory = object

    # vllm_gr deps
    fake_gr_config = types.ModuleType("vllm_gr.config.gr_config")

    class _GRConfig:
        def __init__(self, max_seq_len=512, max_candidates=1000):
            self.max_seq_len = max_seq_len
            self.max_candidates = max_candidates

    fake_gr_config.GRConfig = _GRConfig

    fake_input_processor = types.ModuleType("vllm_gr.engine.gr_input_processor")

    class _GRInputProcessor:
        def __init__(self, **kwargs):
            self.adapter = kwargs.get("adapter")

    fake_input_processor.GRInputProcessor = _GRInputProcessor

    fake_output_processor = types.ModuleType("vllm_gr.engine.gr_output_processor")

    class _GROutputProcessor:
        def __init__(self, **kwargs):
            self.adapter = kwargs.get("adapter")

    fake_output_processor.GROutputProcessor = _GROutputProcessor

    fake_inputs_pkg = types.ModuleType("vllm_gr.inputs")
    fake_inputs_data = types.ModuleType("vllm_gr.inputs.data")

    def _is_gr_prompt(prompt):
        return (
            isinstance(prompt, dict)
            and "user_id" in prompt
            and "user_history" in prompt
            and "candidates" in prompt
        )

    fake_inputs_data.is_gr_prompt = _is_gr_prompt
    fake_inputs_data.GRPrompt = dict
    fake_inputs_pkg.is_gr_prompt = _is_gr_prompt
    fake_inputs_pkg.GRPrompt = dict

    fake_outputs = types.ModuleType("vllm_gr.outputs")

    class _GRRequestOutput:
        pass

    fake_outputs.GRRequestOutput = _GRRequestOutput

    sys.modules.update(
        {
            "vllm": fake_vllm,
            "vllm.config": fake_vllm_config,
            "vllm.usage": fake_usage_pkg,
            "vllm.usage.usage_lib": fake_usage,
            "vllm.v1.engine.async_llm": fake_async_llm,
            "vllm.lora": fake_lora_pkg,
            "vllm.lora.request": fake_lora,
            "vllm.pooling_params": fake_pooling,
            "vllm.v1.metrics": fake_metrics_pkg,
            "vllm.v1.metrics.loggers": fake_metrics,
            "vllm_gr.config.gr_config": fake_gr_config,
            "vllm_gr.engine.gr_input_processor": fake_input_processor,
            "vllm_gr.engine.gr_output_processor": fake_output_processor,
            "vllm_gr.inputs": fake_inputs_pkg,
            "vllm_gr.inputs.data": fake_inputs_data,
            "vllm_gr.outputs": fake_outputs,
        }
    )

    # Track exactly which keys we install so we can remove them later.
    fake_keys = [
        "vllm", "vllm.config", "vllm.usage", "vllm.usage.usage_lib",
        "vllm.v1.engine.async_llm",
        "vllm.lora", "vllm.lora.request", "vllm.pooling_params",
        "vllm.v1.metrics", "vllm.v1.metrics.loggers",
        "vllm_gr.config.gr_config",
        "vllm_gr.engine.gr_input_processor",
        "vllm_gr.engine.gr_output_processor",
        "vllm_gr.inputs", "vllm_gr.inputs.data",
        "vllm_gr.outputs",
    ]

    try:
        yield
    finally:
        _restore_modules(saved)
        # Remove fakes we installed that weren't previously in sys.modules.
        for key in fake_keys:
            if key not in saved:
                sys.modules.pop(key, None)


class TestGRAsyncLLM(unittest.TestCase):
    """Test GRAsyncLLM adapter wiring and input validation."""

    @classmethod
    def setUpClass(cls):
        async_path = ROOT / "vllm_gr" / "engine" / "async_gr_engine.py"
        with _fake_async_engine_runtime():
            cls.module = _load_module(
                "async_gr_engine_test_module", async_path,
            )
        # Clean up the loaded test module to avoid polluting sys.modules
        sys.modules.pop("async_gr_engine_test_module", None)

    def test_replace_processors_creates_adapter_and_injects_into_processors(self):
        """_replace_processors() creates HSTUModelAdapter and passes it to both processors."""
        import types as _types

        fake_adapter = object()
        adapter_created = {}

        class _FakeAdapter:
            pass

        # We mock HSTUModelAdapter by intercepting the import
        saved_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __builtins__["__import__"]

        # Build a fake async_llm instance
        instance = self.module.GRAsyncLLM.__new__(self.module.GRAsyncLLM)
        instance.gr_config = self.module.GRConfig()

        fake_vllm_config = _types.SimpleNamespace(
            model_config=_types.SimpleNamespace(
                hf_config=_types.SimpleNamespace(
                    model_type="hstu",
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
                    contextual_feature_to_max_length={},
                    contextual_feature_to_min_uih_length={},
                ),
            ),
            scheduler_config=_types.SimpleNamespace(stream_interval=1),
        )
        instance.vllm_config = fake_vllm_config

        # Mock tokenizer and log_stats
        instance.tokenizer = None
        instance.log_stats = False

        # We need torchrec for the real HSTUModelAdapter.
        # Instead, verify the import path and adapter injection by checking
        # that the processors receive the same adapter instance.
        try:
            import torchrec  # noqa: F401
            _HAS_TORCHREC = True
        except ImportError:
            _HAS_TORCHREC = False

        if not _HAS_TORCHREC:
            self.skipTest("torchrec not available for full adapter wiring test")

        instance._replace_processors()

        # Both processors must have the same adapter instance
        self.assertIsNotNone(instance.input_processor.adapter)
        self.assertIsNotNone(instance.output_processor.adapter)
        self.assertIs(
            instance.input_processor.adapter,
            instance.output_processor.adapter,
        )

    def test_encode_rejects_non_gr_prompt(self):
        """encode() raises TypeError for non-GRPrompt input."""
        instance = self.module.GRAsyncLLM.__new__(self.module.GRAsyncLLM)
        instance.input_processor = None
        instance.output_processor = None

        import asyncio

        async def _run():
            params = self.module.GRAsyncLLM.__bases__[0].__dict__.get(
                "encode", None,
            )
            coro = instance.encode(
                prompt="not a gr prompt",
                pooling_params=self.module.PoolingParams(task="embed"),
                request_id="r1",
            )
            with self.assertRaises(TypeError):
                async for _ in coro:
                    pass

        # Use the GRAsyncLLM.encode method directly
        async def _test():
            coro = self.module.GRAsyncLLM.encode(
                instance,
                prompt="not a gr prompt",
                pooling_params=self.module.PoolingParams(task="embed"),
                request_id="r1",
            )
            with self.assertRaises(TypeError):
                async for _ in coro:
                    pass

        asyncio.get_event_loop().run_until_complete(_test())

    def test_encode_rejects_wrong_pooling_task(self):
        """encode() raises ValueError when PoolingParams.task != 'embed'."""
        instance = self.module.GRAsyncLLM.__new__(self.module.GRAsyncLLM)

        import asyncio

        async def _test():
            coro = self.module.GRAsyncLLM.encode(
                instance,
                prompt={
                    "user_id": "u1",
                    "user_history": [],
                    "candidates": [],
                },
                pooling_params=self.module.PoolingParams(task="classify"),
                request_id="r1",
            )
            with self.assertRaises(ValueError):
                async for _ in coro:
                    pass

        asyncio.get_event_loop().run_until_complete(_test())


if __name__ == "__main__":
    unittest.main()
