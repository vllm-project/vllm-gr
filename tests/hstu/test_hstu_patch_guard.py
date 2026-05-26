# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Subprocess-isolated tests for HSTU patch guard.

Verifies that in a clean process, activating the ``native`` runtime selection
does not apply any patches, and that the HSTU adapter layer structure is
correctly guarded (Layer 1 always available, Layer 2 only when activated).
"""

import subprocess
import sys
import textwrap
import unittest

_SUBPROCESS_TESTS = {
    "native_activation_zero_patches": textwrap.dedent("""\
        from vllm_gr._runtime_selection import RuntimeSelection
        from vllm_gr._static_descriptors import FAMILY_NATIVE
        from vllm_gr.runtime.patch_plan import build_patch_plan
        from vllm_gr.runtime.activation import activate

        selection = RuntimeSelection(family=FAMILY_NATIVE)
        plan = build_patch_plan(selection)
        state = activate(plan)

        assert state["patch_count"] == 0, (
            f"native plan should have 0 patches, got {state['patch_count']}"
        )
        assert not plan.needs_core_patches
        assert not plan.needs_hstu_config
        assert not plan.needs_model_registry
        assert not plan.needs_adapter_layer2
    """),

    "layer1_registered_without_activation": textwrap.dedent("""\
        # After register() (no ENV), Layer 1 adapter metadata should be
        # available but Layer 2 adapter classes should not.
        import os

        os.environ.pop("_VLLM_GR_RUNTIME_SELECTION", None)

        from vllm_gr.plugin import register
        register()

        from vllm_gr.adapters.registry import get_registered_families
        families = get_registered_families()
        assert "hstu" in families, (
            f"Layer 1 should register 'hstu', got {families}"
        )
    """),

    "layer2_not_imported_without_activation": textwrap.dedent("""\
        # Layer 2 adapter classes must not be imported when no activation
        # has occurred.
        import os
        import sys

        os.environ.pop("_VLLM_GR_RUNTIME_SELECTION", None)

        from vllm_gr.plugin import register
        register()

        # adapter.py should not be in sys.modules
        adapter_mod = "vllm_gr.adapters.hstu.adapter"
        assert adapter_mod not in sys.modules, (
            "Layer 2 adapter.py should not be imported without activation"
        )
    """),

    "gr_recommend_activation_applies_patches": textwrap.dedent("""\
        from unittest.mock import MagicMock, patch
        import types
        import sys

        # Mock vllm_gr.patch to control re_apply_patches return value.
        mock_patch = types.ModuleType("vllm_gr.patch")
        mock_re_apply = MagicMock(return_value=5)
        mock_patch.re_apply_patches = mock_re_apply
        mock_patch.is_patched = MagicMock(return_value=True)
        mock_patch.get_patch_count = MagicMock(return_value=5)
        sys.modules["vllm_gr.patch"] = mock_patch

        mock_config = types.ModuleType("vllm_gr.config.hstu_vllm_registration")
        mock_config.ensure_hstu_config_registered = MagicMock(return_value=True)
        sys.modules["vllm_gr.config.hstu_vllm_registration"] = mock_config

        try:
            with (
                patch("vllm_gr.plugin.register_gr_model_architectures", return_value=True),
                patch("vllm_gr.plugin._register_adapter_families"),
            ):
                from vllm_gr._runtime_selection import RuntimeSelection
                from vllm_gr._static_descriptors import FAMILY_GR_RECOMMEND, MODEL_ARCH_HSTU
                from vllm_gr.runtime.patch_plan import build_patch_plan
                from vllm_gr.runtime.activation import activate

                selection = RuntimeSelection(
                    family=FAMILY_GR_RECOMMEND,
                    model_arch=MODEL_ARCH_HSTU,
                )
                plan = build_patch_plan(selection)
                state = activate(plan)

            assert state["patch_count"] == 5
            mock_re_apply.assert_called_once()
            assert plan.needs_core_patches
            assert plan.needs_adapter_layer2
        finally:
            sys.modules.pop("vllm_gr.patch", None)
            sys.modules.pop("vllm_gr.config.hstu_vllm_registration", None)
    """),
}


class TestHstuPatchGuard(unittest.TestCase):

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
