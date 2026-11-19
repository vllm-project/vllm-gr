# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Bind a real attention policy to host-side kernel doubles."""

from typing import Any

from vllm_gr.platforms.ascend.attention import AscendAttention
from vllm_gr.platforms.attention import MetadataProfile
from vllm_gr.platforms.cuda.attention import CudaAttention


def profile(*, cuda: Any = False, ascend: Any = False) -> Any:
    if ascend:
        return AscendAttention.metadata_profile
    if cuda:
        return CudaAttention.metadata_profile
    return MetadataProfile()


def bind_attention(
    monkeypatch: Any, facade: Any, *, cuda: Any = True, execution: Any = None
) -> Any:
    service = CudaAttention() if cuda else AscendAttention("cascade")
    if execution is not None:
        monkeypatch.setattr(service, "execution_module", execution)
        monkeypatch.setattr(service, "prefill_module", execution)
    monkeypatch.setattr(facade, "for_config", lambda *args, **kwargs: service)
    if hasattr(facade, "admission_for_config"):
        monkeypatch.setattr(facade, "admission_for_config", lambda: service)
    # A few contract tests drive forward on an object without calling __init__.
    monkeypatch.setattr(facade.BeamAttentionImpl, "_execution", service, raising=False)
    return service
