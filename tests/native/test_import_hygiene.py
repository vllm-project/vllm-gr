# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Subprocess-isolated tests that verify ``import vllm_gr`` stays lightweight.

Each assertion runs in a fresh process so that ``sys.modules`` state from
one test cannot leak into another — mirroring the isolation pattern used by
``test_outputs_phase2.py``.
"""

import subprocess
import sys
import textwrap
import unittest

_SUBPROCESS_TESTS = {
    "import_vllm_gr_no_heavy_imports": textwrap.dedent("""\
        import sys

        # Record modules before import.
        before = set(sys.modules)

        import vllm_gr

        after = set(sys.modules)
        new_modules = after - before

        forbidden = {"torch", "torchrec", "tensorflow", "vllm"}
        leaked = forbidden & new_modules
        assert not leaked, (
            f"import vllm_gr pulled in heavy modules: {sorted(leaked)}"
        )
    """),

    "import_vllm_gr_no_patching": textwrap.dedent("""\
        import vllm_gr

        # patch.is_patched() should be False without ever calling activate().
        from vllm_gr.patch import is_patched
        assert is_patched() is False, "import vllm_gr must not apply patches"
    """),

    "import_vllm_gr_no_activation": textwrap.dedent("""\
        import vllm_gr

        from vllm_gr.runtime.activation import _ACTIVATED
        assert not _ACTIVATED, "import vllm_gr must not trigger activation"
    """),

    "register_without_env_no_patching": textwrap.dedent("""\
        import os

        # Ensure no ENV signal.
        os.environ.pop("_VLLM_GR_RUNTIME_SELECTION", None)

        from vllm_gr.plugin import register
        register()

        from vllm_gr.patch import is_patched
        assert is_patched() is False, (
            "register() without ENV must not apply patches"
        )
    """),

    "register_without_env_no_activation": textwrap.dedent("""\
        import os

        os.environ.pop("_VLLM_GR_RUNTIME_SELECTION", None)

        from vllm_gr.plugin import register
        register()

        from vllm_gr.runtime.activation import _ACTIVATED
        assert not _ACTIVATED, (
            "register() without ENV must not trigger activation"
        )
    """),

    "register_without_env_no_heavy_imports": textwrap.dedent("""\
        import os
        import sys

        os.environ.pop("_VLLM_GR_RUNTIME_SELECTION", None)
        before = set(sys.modules)

        from vllm_gr.plugin import register
        register()

        new_modules = set(sys.modules) - before
        forbidden = {"torch", "torchrec", "tensorflow", "vllm"}
        leaked = forbidden & new_modules
        assert not leaked, (
            f"register() without ENV pulled in heavy modules: {sorted(leaked)}"
        )
    """),
}


class TestImportHygiene(unittest.TestCase):

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
