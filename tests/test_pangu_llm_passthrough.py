# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# mypy: disable-error-code="attr-defined,no-untyped-def,no-any-return,misc,arg-type,var-annotated,union-attr,return-value,func-returns-value,no-untyped-call,operator,index,valid-type,no-redef,has-type,type-arg"

"""Unit tests for the Pangu LLM passthrough path.

After removing the GR adapter / recommend runtime, ``GRLLM`` is a thin
subclass of vLLM's :class:`~vllm.entrypoints.llm.LLM` that always delegates
construction to the standard vLLM initializer. These tests verify that
Pangu-style checkpoints go through that passthrough path.
"""

import gc
import os
import sys
import unittest
from types import SimpleNamespace
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


_HAS_VLLM = _has_vllm_engine_runtime()
_SKIP = "vLLM engine/runtime submodules are unavailable in this environment"


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestPanguGrllmPassthrough(unittest.TestCase):
    def test_grllm_init_delegates_to_vllm_llm(self):
        """GRLLM should delegate construction to vLLM's LLM.__init__."""
        from vllm_gr.entrypoints.gr import GRLLM

        with mock.patch(
            "vllm.entrypoints.llm.LLM.__init__",
            return_value=None,
        ) as mock_llm_init:
            GRLLM(model="/fake/pangu")

        mock_llm_init.assert_called_once()


class TestPanguPluginInitializeRuntime(unittest.TestCase):
    def test_initialize_runtime_does_not_raise(self):
        from vllm_gr import plugin

        with (
            mock.patch.object(
                plugin,
                "initialize_process_backend",
                return_value=SimpleNamespace(device_type="cuda"),
            ),
            mock.patch("vllm_gr.patch.re_apply_patches", return_value=0),
            mock.patch("vllm_gr.patch.get_last_beam_patch_report", return_value=None),
        ):
            state = plugin.initialize_runtime()
        self.assertIn("patch_count", state)
        self.assertIn("patches", state)


class TestModelRunnerDefault(unittest.TestCase):
    def test_defaults_to_v1_when_unset(self):
        from vllm_gr._env_check import configure_model_runner_default

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
            configure_model_runner_default()
            self.assertEqual(os.environ["VLLM_USE_V2_MODEL_RUNNER"], "0")

    def test_preserves_explicit_v2_override(self):
        from vllm_gr._env_check import configure_model_runner_default

        with mock.patch.dict(
            os.environ,
            {"VLLM_USE_V2_MODEL_RUNNER": "1"},
            clear=False,
        ):
            configure_model_runner_default()
            self.assertEqual(os.environ["VLLM_USE_V2_MODEL_RUNNER"], "1")


@unittest.skipUnless(_HAS_VLLM, _SKIP)
class TestPanguGrllmShutdown(unittest.TestCase):
    def test_grllm_shutdown_releases_engine(self):
        from vllm_gr.entrypoints.gr import GRLLM

        llm = GRLLM.__new__(GRLLM)
        llm._shutdown_complete = False
        engine = mock.Mock()
        engine.shutdown = mock.Mock()
        llm.llm_engine = engine

        release_memory = mock.Mock()
        backend = mock.Mock(release_accelerator_memory=release_memory)
        with mock.patch("vllm_gr.platforms.get_active_backend", return_value=backend):
            with mock.patch.object(gc, "collect") as mock_gc:
                GRLLM.shutdown(llm)

        engine.shutdown.assert_called_once_with()
        mock_gc.assert_called_once_with()
        release_memory.assert_called_once_with()
        self.assertIsNone(llm.llm_engine)
        self.assertTrue(llm._shutdown_complete)


if __name__ == "__main__":
    unittest.main()
