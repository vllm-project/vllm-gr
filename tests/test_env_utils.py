# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_gr.utils.env import is_truthy_env


def test_is_truthy_env_accepts_shared_true_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("1", "true", "TRUE", "yes", "Y", "on", "ON"):
        monkeypatch.setenv("VLLM_GR_TEST_TRUTHY", value)

        assert is_truthy_env("VLLM_GR_TEST_TRUTHY")


def test_is_truthy_env_rejects_other_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in (None, "", "0", "false", "no", "off", " true "):
        if value is None:
            monkeypatch.delenv("VLLM_GR_TEST_TRUTHY", raising=False)
        else:
            monkeypatch.setenv("VLLM_GR_TEST_TRUTHY", value)

        assert not is_truthy_env("VLLM_GR_TEST_TRUTHY")
