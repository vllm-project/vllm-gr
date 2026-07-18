# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.metadata import PackageNotFoundError
from unittest import mock

import pytest

from vllm_gr._env_check import check_runtime_versions


def test_supported_vllm_version() -> None:
    with mock.patch(
        "vllm_gr._env_check.importlib.metadata.version",
        side_effect=lambda name: {"vllm": "0.22.1rc1", "vllm-ascend": "0.22.1rc1+npu"}[name],
    ):
        check_runtime_versions(require_ascend=True)


def test_rejects_old_vllm_version() -> None:
    with mock.patch(
        "vllm_gr._env_check.importlib.metadata.version",
        return_value="0.14.1",
    ):
        with pytest.raises(RuntimeError, match="required 0.22.1rc1"):
            check_runtime_versions()


def test_requires_ascend_distribution_on_npu() -> None:
    def version(name: str) -> str:
        if name == "vllm":
            return "0.22.1rc1"
        raise PackageNotFoundError(name)

    with mock.patch("vllm_gr._env_check.importlib.metadata.version", side_effect=version):
        with pytest.raises(RuntimeError, match="vllm-ascend"):
            check_runtime_versions(require_ascend=True)
