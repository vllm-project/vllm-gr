# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock

import pytest

from vllm_gr._env_check import check_runtime_versions


def test_supported_vllm_version() -> None:
    with mock.patch(
        "vllm_gr._env_check.importlib.metadata.version",
        return_value="0.22.1",
    ):
        assert check_runtime_versions() == {"vllm": "0.22.1"}


def test_supported_vllm_local_version() -> None:
    with mock.patch(
        "vllm_gr._env_check.importlib.metadata.version",
        return_value="0.22.1+cu130",
    ):
        check_runtime_versions()


def test_rejects_release_candidate_vllm_version() -> None:
    with (
        mock.patch(
            "vllm_gr._env_check.importlib.metadata.version",
            return_value="0.22.1rc1",
        ),
        pytest.raises(RuntimeError, match="required 0.22.1"),
    ):
        check_runtime_versions()


def test_rejects_old_vllm_version() -> None:
    with (
        mock.patch(
            "vllm_gr._env_check.importlib.metadata.version",
            return_value="0.14.1",
        ),
        pytest.raises(RuntimeError, match="required 0.22.1"),
    ):
        check_runtime_versions()
