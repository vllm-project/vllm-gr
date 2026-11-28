# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native platform identity and in-tree GR implementation selection."""

from __future__ import annotations

import pytest
import vllm.platforms

import vllm_gr.platforms as table
from tests.helpers.native_platform import native_platform
from vllm_gr.platforms.errors import UnsupportedBackendError

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize(  # type: ignore[misc]
    "enum, device", [("ROCM", "cuda"), ("CPU", "cpu"), ("OOT", "foreign")]
)
def test_only_supported_native_platforms_match(
    monkeypatch: pytest.MonkeyPatch, enum: str, device: str
) -> None:
    monkeypatch.setattr(vllm.platforms, "current_platform", native_platform(enum, device))
    with pytest.raises(UnsupportedBackendError, match="No GR integration"):
        table.get_current_gr_platform()


@pytest.mark.parametrize(  # type: ignore[misc]
    "enum, device, name", [("CUDA", "cuda", "cuda"), ("OOT", "npu", "ascend")]
)
def test_winner_inherits_native_platform(
    monkeypatch: pytest.MonkeyPatch, enum: str, device: str, name: str
) -> None:
    native = native_platform(enum, device)
    monkeypatch.setattr(vllm.platforms, "current_platform", native)
    selected = table.get_current_gr_platform()
    assert isinstance(selected, type(native))
    assert selected.backend_id == name
    assert table.get_current_gr_platform() is selected
