# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""The sidecar stays independent of the inference runtime and control server."""

import subprocess
import sys

import pytest

MODULES = (
    "vllm_gr.sidecar",
    "vllm_gr.sidecar.client",
    "vllm_gr.sidecar.config",
    "vllm_gr.sidecar.contracts",
    "vllm_gr.sidecar.controller",
    "vllm_gr.sidecar.filesystems",
    "vllm_gr.sidecar.modes",
    "vllm_gr.sidecar.processes",
    "vllm_gr.sidecar.validation",
)
FORBIDDEN = ("torch", "vllm", "vllm_gr.entrypoints", "fastapi", "numpy", "safetensors")
_IMPORT_CHECK = """
import importlib
import sys

importlib.import_module({module!r})
loaded = [name for name in {forbidden!r} if name in sys.modules]
assert not loaded, loaded
"""


def test_package_import_is_inference_free() -> None:
    _assert_inference_free("vllm_gr.sidecar")


@pytest.mark.parametrize("module", MODULES)  # type: ignore[misc]
def test_module_import_is_inference_free(module: str) -> None:
    _assert_inference_free(module)


def test_cli_help_is_inference_free() -> None:
    subprocess.run(
        [sys.executable, "-m", "vllm_gr.sidecar", "--help"],
        check=True,
        capture_output=True,
    )


def _assert_inference_free(module: str) -> None:
    subprocess.run(
        [sys.executable, "-c", _IMPORT_CHECK.format(module=module, forbidden=FORBIDDEN)],
        check=True,
    )


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
