# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Unit tests for Pangu / LLM passthrough engine-arg defaults and guards."""

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _has_vllm_engine_runtime() -> bool:
    try:
        import vllm.engine.arg_utils  # noqa: F401
        import vllm.platforms  # noqa: F401
    except Exception:
        return False
    return True


def _has_transformers() -> bool:
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


_HAS_VLLM = _has_vllm_engine_runtime()
_HAS_FACTORY_DEPS = _HAS_VLLM and _has_transformers()
_SKIP = "vLLM engine/runtime submodules are unavailable in this environment"
_SKIP_FACTORY = "transformers/vLLM required for engine_factory import tests"


@contextmanager
def _stubbed_gr_npu_model_runner_import():
    """Import GRNPUModelRunner without vllm-ascend (avoids numba/numpy pin issues)."""
    fake_vllm_ascend = types.ModuleType("vllm_ascend")
    fake_vllm_ascend_worker = types.ModuleType("vllm_ascend.worker")
    fake_vllm_ascend_runner = types.ModuleType("vllm_ascend.worker.model_runner_v1")

    class _FakeNPUModelRunner:
        def __init__(self, vllm_config, device, **kwargs) -> None:
            del kwargs
            self.vllm_config = vllm_config
            self.device = device

    fake_vllm_ascend.__spec__ = importlib.machinery.ModuleSpec(
        "vllm_ascend", loader=None
    )
    fake_vllm_ascend_worker.__spec__ = importlib.machinery.ModuleSpec(
        "vllm_ascend.worker", loader=None
    )
    fake_vllm_ascend_runner.__spec__ = importlib.machinery.ModuleSpec(
        "vllm_ascend.worker.model_runner_v1", loader=None
    )
    fake_vllm_ascend_runner.NPUModelRunner = _FakeNPUModelRunner

    patched_modules = {
        "vllm_ascend": fake_vllm_ascend,
        "vllm_ascend.worker": fake_vllm_ascend_worker,
        "vllm_ascend.worker.model_runner_v1": fake_vllm_ascend_runner,
    }
    with mock.patch.dict(sys.modules, patched_modules):
        sys.modules.pop("vllm_gr.worker.gr_npu_model_runner", None)
        module = importlib.import_module("vllm_gr.worker.gr_npu_model_runner")
        try:
            yield module.GRNPUModelRunner
        finally:
            sys.modules.pop("vllm_gr.worker.gr_npu_model_runner", None)


class TestPanguLlmDefaults(unittest.TestCase):

    def test_apply_pangu_llm_defaults_sets_generate_runner(self):
        from vllm_gr.adapters.pangu_llm._defaults import apply_pangu_llm_defaults

        class _Args:
            trust_remote_code = False
            skip_tokenizer_init = True
            runner = "pooling"
            worker_cls = "vllm_gr.worker.gr_worker.GRWorker"
            dtype = "auto"
            max_model_len = None
            tokenizer_mode = "auto"
            enable_chunked_prefill = True
            enable_prefix_caching = True
            _explicit_cli_args = set()

        args = _Args()
        apply_pangu_llm_defaults(args)
        self.assertTrue(args.trust_remote_code)
        self.assertFalse(args.skip_tokenizer_init)
        self.assertEqual(args.runner, "generate")
        self.assertIsNone(args.worker_cls)
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.max_model_len, 32768)
        self.assertEqual(args.tokenizer_mode, "slow")
        self.assertFalse(args.enable_chunked_prefill)
        self.assertFalse(args.enable_prefix_caching)


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestPanguLlmEngineArgs(unittest.TestCase):

    def _write_pangu_config(self, directory: str) -> str:
        import json

        config = {
            "model_type": "PanguEmbedded",
            "architectures": ["PanguEmbeddedForCausalLM"],
            "hidden_size": 1536,
            "num_hidden_layers": 26,
        }
        path = os.path.join(directory, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        return directory

    def test_model_family_llm_alias(self):
        from vllm_gr.engine.arg_utils import GRAsyncEngineArgs

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._write_pangu_config(tmp)
            args = GRAsyncEngineArgs(model=model_dir, model_family="llm")
        self.assertEqual(args.model_category, "llm")
        self.assertEqual(args.runner, "generate")
        self.assertIsNone(args.worker_cls)
        self.assertFalse(args.skip_tokenizer_init)

    def test_auto_detect_pangu_from_config(self):
        from vllm_gr.engine.arg_utils import GRAsyncEngineArgs

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._write_pangu_config(tmp)
            args = GRAsyncEngineArgs(model=model_dir)
        self.assertEqual(args.model_family, "panguembedded")
        self.assertEqual(args.model_category, "llm")
        self.assertEqual(args.runner, "generate")
        self.assertIsNone(args.worker_cls)

    def test_hstu_family_still_uses_gr_worker(self):
        from vllm_gr.engine.arg_utils import GREngineArgs

        with mock.patch.object(
            GREngineArgs,
            "_is_ascend_platform",
            return_value=False,
        ):
            args = GREngineArgs(
                model="test-model",
                model_family="hstu",
            )
        self.assertEqual(args.model_category, "gr")
        self.assertEqual(args.worker_cls, "vllm_gr.worker.gr_worker.GRWorker")
        self.assertEqual(args.runner, "pooling")


class TestPluginStrictMode(unittest.TestCase):

    def test_initialize_runtime_lenient_does_not_raise_on_missing_gr(self):
        from vllm_gr import plugin

        with mock.patch.object(plugin, "register_gr_configs", return_value=False), \
             mock.patch.object(plugin, "register_gr_model_architectures", return_value=False), \
             mock.patch("vllm_gr.patch.re_apply_patches", return_value=0), \
             mock.patch.object(plugin, "_register_adapter_families"):
            state = plugin.initialize_runtime(strict_gr=False)
        self.assertFalse(state["config_registered"])
        self.assertFalse(state["models_registered"])

    def test_initialize_runtime_strict_raises(self):
        from vllm_gr import plugin

        with mock.patch.object(plugin, "register_gr_configs", return_value=False), \
             mock.patch.object(plugin, "register_gr_model_architectures", return_value=True), \
             mock.patch("vllm_gr.patch.re_apply_patches", return_value=0), \
             mock.patch.object(plugin, "_register_adapter_families"):
            with self.assertRaises(RuntimeError):
                plugin.initialize_runtime(strict_gr=True)


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestGrllmRejectsLlmCategory(unittest.TestCase):

    def test_grllm_init_raises_for_llm_category(self):
        from vllm_gr.entrypoints.gr import GRLLM

        fake_args = mock.Mock()
        fake_args.model_category = "llm"
        fake_args.model_family = "llm"

        with mock.patch(
            "vllm_gr.entrypoints.gr.GREngineArgs",
            return_value=fake_args,
        ):
            with self.assertRaises(ValueError) as ctx:
                GRLLM(model="/fake/pangu", model_family="llm")
        self.assertIn("GRLLM only supports", str(ctx.exception))


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestGrNpuRunnerGuard(unittest.TestCase):

    def test_gr_npu_runner_rejects_llm_category(self):
        vllm_config = mock.Mock()
        vllm_config.model_category = "llm"
        with _stubbed_gr_npu_model_runner_import() as GRNPUModelRunner:
            with self.assertRaises(RuntimeError) as ctx:
                GRNPUModelRunner(vllm_config, "npu:0")
        self.assertIn("GRNPUModelRunner must not", str(ctx.exception))


def _load_engine_factory_module():
    module_path = Path(ROOT) / "vllm_gr" / "engine" / "engine_factory.py"
    spec = importlib.util.spec_from_file_location(
        "vllm_gr_engine_factory_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(_HAS_FACTORY_DEPS, _SKIP_FACTORY)
class TestEngineFactoryRouting(unittest.TestCase):

    def test_is_llm_category_routes_async_factory(self):
        factory = _load_engine_factory_module()
        create_async_llm_from_vllm_config = (
            factory.create_async_llm_from_vllm_config
        )
        from vllm_gr.config.gr_config import GRConfig

        vllm_config = mock.Mock()
        vllm_config.model_category = "llm"

        with mock.patch(
            "vllm.v1.engine.async_llm.AsyncLLM.from_vllm_config",
            return_value="async-llm",
        ) as mock_async:
            result = create_async_llm_from_vllm_config(
                vllm_config,
                gr_config=GRConfig(),
                usage_context=mock.Mock(),
            )
        self.assertEqual(result, "async-llm")
        mock_async.assert_called_once()

    def test_is_gr_category_routes_gr_async_factory(self):
        factory = _load_engine_factory_module()
        create_async_llm_from_vllm_config = (
            factory.create_async_llm_from_vllm_config
        )
        from vllm_gr.config.gr_config import GRConfig

        vllm_config = mock.Mock()
        vllm_config.model_category = "gr"

        with mock.patch(
            "vllm_gr.engine.async_gr_engine.GRAsyncLLM.from_vllm_config",
            return_value="gr-async",
        ) as mock_gr:
            result = create_async_llm_from_vllm_config(
                vllm_config,
                gr_config=GRConfig(),
                usage_context=mock.Mock(),
            )
        self.assertEqual(result, "gr-async")
        mock_gr.assert_called_once()


if __name__ == "__main__":
    unittest.main()
