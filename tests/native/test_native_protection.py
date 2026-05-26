# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Subprocess-isolated tests that verify native vLLM path is unpolluted.

After ``register()`` is called **without** the ``_VLLM_GR_RUNTIME_SELECTION``
ENV variable, no GR-specific entries should appear in the vLLM ModelRegistry,
and no core patches / config registrations should be applied.

These tests exercise the same boundaries as the in-process
``TestRegisterMetadataOnly`` suite but in isolated subprocesses so that
``sys.modules`` / global-registry state never leaks between tests.
"""

import subprocess
import sys
import textwrap
import unittest

_SUBPROCESS_TESTS = {
    "no_patches_after_register": textwrap.dedent("""\
        import os

        os.environ.pop("_VLLM_GR_RUNTIME_SELECTION", None)

        from vllm_gr.plugin import register
        register()

        from vllm_gr.patch import is_patched, get_patch_count
        assert is_patched() is False
        assert get_patch_count() == 0, (
            f"Expected 0 patches, got {get_patch_count()}"
        )
    """),

    "no_activation_state_after_register": textwrap.dedent("""\
        import os

        os.environ.pop("_VLLM_GR_RUNTIME_SELECTION", None)

        from vllm_gr.plugin import register
        register()

        from vllm_gr.runtime.activation import _ACTIVATED, _LAST_STATE
        assert not _ACTIVATED
        assert _LAST_STATE == {}, f"Expected empty state, got {_LAST_STATE}"
    """),

    "activate_native_no_patches": textwrap.dedent("""\
        from vllm_gr._runtime_selection import RuntimeSelection
        from vllm_gr._static_descriptors import FAMILY_NATIVE
        from vllm_gr.runtime.patch_plan import build_patch_plan
        from vllm_gr.runtime.activation import activate

        selection = RuntimeSelection(family=FAMILY_NATIVE)
        plan = build_patch_plan(selection)
        state = activate(plan)

        assert state["patch_count"] == 0
        assert state["config_registered"] is True
        assert state["models_registered"] is True

        from vllm_gr.patch import is_patched
        assert is_patched() is False, (
            "activate(native) must not apply patches"
        )
    """),

    "activate_native_then_register_idempotent": textwrap.dedent("""\
        import os

        os.environ.pop("_VLLM_GR_RUNTIME_SELECTION", None)

        # activate(native) first
        from vllm_gr._runtime_selection import RuntimeSelection
        from vllm_gr._static_descriptors import FAMILY_NATIVE
        from vllm_gr.runtime.patch_plan import build_patch_plan
        from vllm_gr.runtime.activation import activate

        selection = RuntimeSelection(family=FAMILY_NATIVE)
        plan = build_patch_plan(selection)
        activate(plan)

        # Then register() — should be a no-op (already activated)
        from vllm_gr.plugin import register
        register()

        from vllm_gr.patch import is_patched, get_patch_count
        assert is_patched() is False
        assert get_patch_count() == 0
    """),

    "no_gr_model_registry_after_register": textwrap.dedent("""\
        import os

        os.environ.pop("_VLLM_GR_RUNTIME_SELECTION", None)

        from vllm_gr.plugin import register
        register()

        # GR-internal flag must not be set.
        from vllm_gr.model_executor.models.registry import is_registered
        assert not is_registered(), (
            "register() without ENV must not register GR model architectures"
        )
    """),

    "no_gr_archs_in_vllm_model_registry": textwrap.dedent("""\
        import os

        os.environ.pop("_VLLM_GR_RUNTIME_SELECTION", None)

        from vllm_gr.plugin import register
        register()

        # If vLLM is installed, verify its ModelRegistry has no GR entries.
        try:
            from vllm.model_executor.models.registry import ModelRegistry
        except ImportError:
            pass  # vLLM not installed — nothing to check.
        else:
            gr_archs = ["HSTUEmbeddingModel"]
            supported = ModelRegistry.get_supported_archs()
            leaked = [a for a in gr_archs if a in supported]
            assert not leaked, (
                f"vLLM ModelRegistry contains GR archs after register() "
                f"without ENV: {leaked}"
            )
    """),
}


class TestNativeProtection(unittest.TestCase):

    def _run_subprocess(self, script: str) -> None:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            result.returncode, 0,
            f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )

    for _name, _script in _SUBPROCESS_TESTS.items():
        locals()[f"test_{_name}"] = lambda self, s=_script: self._run_subprocess(s)


if __name__ == "__main__":
    unittest.main()
