# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"

"""Unit tests for Pangu LLM passthrough routing via ``is_adapter_registered``."""

import gc
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

    fake_vllm_ascend.__spec__ = importlib.machinery.ModuleSpec("vllm_ascend", loader=None)
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


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestPanguEngineArgsRouting(unittest.TestCase):
    def _write_config(self, directory: str, config: dict) -> str:
        import json

        path = os.path.join(directory, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        return directory

    def test_unregistered_pangu_does_not_apply_gr_defaults(self):
        from vllm_gr.engine.arg_utils import GRAsyncEngineArgs

        config = {
            "model_type": "PanguEmbedded",
            "architectures": ["PanguEmbeddedForCausalLM"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._write_config(tmp, config)
            args = GRAsyncEngineArgs(model=model_dir)
        self.assertEqual(args.model_type, "panguembedded")
        self.assertNotEqual(
            getattr(args, "worker_cls", None),
            "vllm_gr.worker.gr_worker.GRWorker",
        )


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestPanguGrllmPassthrough(unittest.TestCase):
    def test_unregistered_pangu_uses_vllm_llm_init(self):
        from vllm_gr.entrypoints.gr import GRLLM

        with (
            mock.patch(
                "vllm_gr.adapters.registry.is_adapter_registered",
                return_value=False,
            ),
            mock.patch(
                "vllm.entrypoints.llm.LLM.__init__",
                return_value=None,
            ) as mock_llm_init,
            mock.patch(
                "vllm_gr.entrypoints.gr.create_sync_engine",
            ) as mock_gr_engine,
        ):
            GRLLM(model="/fake/pangu")

        mock_llm_init.assert_called_once()
        mock_gr_engine.assert_not_called()


class TestPanguPluginStrictMode(unittest.TestCase):
    def test_initialize_runtime_lenient_does_not_raise_on_missing_gr(self):
        from vllm_gr import plugin

        with (
            mock.patch.object(plugin, "register_gr_configs", return_value=False),
            mock.patch.object(plugin, "register_gr_model_architectures", return_value=False),
            mock.patch("vllm_gr.patch.re_apply_patches", return_value=0),
            mock.patch.object(plugin, "_register_adapter_families"),
        ):
            state = plugin.initialize_runtime(strict_gr=False)
        self.assertFalse(state["config_registered"])
        self.assertFalse(state["models_registered"])

    def test_initialize_runtime_strict_raises(self):
        from vllm_gr import plugin

        with (
            mock.patch.object(plugin, "register_gr_configs", return_value=False),
            mock.patch.object(plugin, "register_gr_model_architectures", return_value=True),
            mock.patch("vllm_gr.patch.re_apply_patches", return_value=0),
            mock.patch.object(plugin, "_register_adapter_families"),
        ):
            with self.assertRaises(RuntimeError):
                plugin.initialize_runtime(strict_gr=True)


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestPanguNpuRunnerGuard(unittest.TestCase):
    def test_gr_npu_runner_rejects_unregistered_pangu(self):
        vllm_config = mock.Mock()
        vllm_config.model_config.hf_config.model_type = "panguembedded"
        with _stubbed_gr_npu_model_runner_import() as GRNPUModelRunner:
            import vllm_gr.worker.gr_npu_model_runner as runner_mod

            with mock.patch.object(
                runner_mod,
                "is_adapter_registered",
                return_value=False,
            ):
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
class TestPanguEngineFactoryRouting(unittest.TestCase):
    def test_unregistered_pangu_routes_async_llm(self):
        factory = _load_engine_factory_module()
        create_async_llm = factory.create_async_llm
        from vllm_gr.config.gr_config import GRConfig

        vllm_config = mock.Mock()
        vllm_config.model_config.hf_config.model_type = "panguembedded"

        with (
            mock.patch.object(factory, "is_adapter_registered", return_value=False),
            mock.patch(
                "vllm.v1.engine.async_llm.AsyncLLM.from_vllm_config",
                return_value="async-llm",
            ) as mock_async,
        ):
            result = create_async_llm(
                vllm_config=vllm_config,
                gr_config=GRConfig(),
                usage_context=mock.Mock(),
            )
        self.assertEqual(result, "async-llm")
        mock_async.assert_called_once()


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestPanguGrllmShutdown(unittest.TestCase):
    def test_grllm_shutdown_releases_engine(self):
        from vllm_gr.entrypoints.gr import GRLLM

        llm = GRLLM.__new__(GRLLM)
        llm._shutdown_complete = False
        engine = mock.Mock()
        engine.shutdown = mock.Mock()
        llm.llm_engine = engine
        llm.default_sampling_params = mock.Mock()
        llm.io_processor = mock.Mock()

        with mock.patch(
            "vllm_gr.utils.npu_adapter.release_accelerator_memory",
            create=True,
        ):
            with mock.patch.object(gc, "collect") as mock_gc:
                GRLLM.shutdown(llm)

        engine.shutdown.assert_called_once_with()
        mock_gc.assert_called_once_with()
        self.assertIsNone(llm.llm_engine)
        self.assertTrue(llm._shutdown_complete)


if __name__ == "__main__":
    unittest.main()
