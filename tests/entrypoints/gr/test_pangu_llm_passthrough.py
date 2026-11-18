# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

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

import pytest

from tests.helpers.paths import REPO_ROOT

ROOT = str(REPO_ROOT)
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

        # The mocked LLM.__init__ must leave behind the llm_engine attribute
        # GRLLM.__init__ reads after super() (vllm_config resolution + the
        # Constraint Table install).  The real vLLM initializer always sets it.
        # ``mock.patch`` on an unbound __init__ receives only the call's
        # kwargs (self is pre-bound), so the instance is captured by the
        # wrapper below instead of a side_effect argument.
        calls: list[dict] = []

        # Assign a plain function as the class __init__: when GRLLM calls
        # super().__init__(model=...), Python invokes it as a normal method
        # with the instance as the first argument.
        def bound_llm_init(llm_self, **kwargs):
            # The real vLLM initializer always leaves ``llm_engine`` behind;
            # GRLLM.__init__ reads it right after super() (vllm_config
            # resolution + the Constraint Table install).
            llm_self.llm_engine = SimpleNamespace(
                vllm_config=SimpleNamespace(additional_config={}),
                model_config=SimpleNamespace(get_vocab_size=lambda: 100),
            )
            llm_self.get_tokenizer = lambda: SimpleNamespace()
            calls.append(kwargs)

        with (
            mock.patch(
                "vllm.entrypoints.llm.LLM.__init__",
                new=bound_llm_init,
            ),
            # GRLLM.__init__ explicitly runs the full runtime initialization
            # (plugin patches + EngineCore entrypoint swap) before super();
            # stub it so this test stays focused on the delegation contract
            # and does not mutate real vLLM classes process-wide.
            mock.patch(
                "vllm_gr.plugin.initialize_runtime",
                return_value={"patch_count": 0},
            ),
        ):
            GRLLM(model="/fake/pangu")

        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs.pop("_gr_mode") is True
        assert "model" in kwargs.pop("_gr_explicit_native_fields")
        assert kwargs == {"model": "/fake/pangu"}


class TestPanguPluginInitializeRuntime(unittest.TestCase):
    def test_initialize_runtime_does_not_raise(self):
        from vllm_gr import plugin

        with (
            mock.patch("vllm_gr._env_check.check_environment"),
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


pytestmark = pytest.mark.cpu_test
