# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
from types import SimpleNamespace

import pytest

from vllm_gr.platforms.ascend.attention import AscendAttention
from vllm_gr.platforms.cuda.attention import ENV_DENSE_PREFILL, CudaAttention
from vllm_gr.v1.attention.backends import beam_attn


def _config(prefix_caching=False, chunked=False, kv_transfer=None):
    return SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=chunked),
        kv_transfer_config=kv_transfer,
    )


@pytest.mark.parametrize(
    "raw,cuda,config,expected",
    [
        (None, True, _config(), None),
        ("off", True, _config(), None),
        ("fa2", True, _config(), "fa2"),
        (" CUDNN ", True, _config(), "cudnn"),
        ("flash", True, _config(), None),
        ("cudnn", False, _config(), None),
        # Cached context lives only in the paged cache.
        ("cudnn", True, _config(prefix_caching=True), None),
        ("cudnn", True, _config(chunked=True), None),
        ("cudnn", True, _config(kv_transfer=SimpleNamespace(kv_connector="X")), None),
        ("cudnn", True, None, None),
    ],
)
def test_dense_prefill_mode_requires_fresh_prompts(monkeypatch, raw, cuda, config, expected):
    import vllm.config

    monkeypatch.setattr(beam_attn.envs, "VLLM_BATCH_INVARIANT", False)

    if raw is None:
        monkeypatch.delenv(ENV_DENSE_PREFILL, raising=False)
    else:
        monkeypatch.setenv(ENV_DENSE_PREFILL, raw)
    executor = CudaAttention() if cuda else AscendAttention("cascade")
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    assert executor.dense_prefill_mode(config) == expected


def test_dense_prefill_mode_off_when_batch_invariant(monkeypatch):
    import vllm.config

    monkeypatch.setenv(ENV_DENSE_PREFILL, "cudnn")
    executor = CudaAttention()
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", _config)
    monkeypatch.setattr(beam_attn.envs, "VLLM_BATCH_INVARIANT", True)
    assert executor.dense_prefill_mode(_config()) is None
    monkeypatch.setattr(beam_attn.envs, "VLLM_BATCH_INVARIANT", False)
    assert executor.dense_prefill_mode(_config()) == "cudnn"


@pytest.mark.parametrize(
    "changes,forwards",
    [
        ({}, True),
        ({"_dense_prefill_mode": None}, False),
        ({"kv_cache_dtype": "fp8_e4m3"}, False),
        ({"kv_cache_dtype": "fp8"}, False),
        ({"kv_sharing_target_layer_name": "layers.0.attn"}, False),
    ],
)
def test_dense_prefill_kwargs_forward_fresh_kv_only(changes, forwards):
    impl = SimpleNamespace(
        _dense_prefill_mode="fa2", kv_cache_dtype="auto", kv_sharing_target_layer_name=None
    )
    for name, value in changes.items():
        setattr(impl, name, value)
    key, value = object(), object()
    kwargs = beam_attn._dense_prefill_kwargs(impl, key, value)
    if forwards:
        assert kwargs == {"dense_key": key, "dense_value": value, "dense_mode": "fa2"}
    else:
        assert kwargs == {}
    assert beam_attn._dense_prefill_kwargs(impl, None, value) == {}


def test_fresh_prefill_capture_scope_and_replay_requirement(monkeypatch):
    from vllm_gr.v1.attention.backends import beam_dense_prefill as dense

    assert not dense.in_fresh_prefill_capture()
    with dense.fresh_prefill_capture():
        assert dense.in_fresh_prefill_capture()
        with pytest.raises(RuntimeError), dense.fresh_prefill_capture():
            raise RuntimeError
        assert dense.in_fresh_prefill_capture()
    assert not dense.in_fresh_prefill_capture()
    for raw, expected in (
        (None, False),
        ("off", False),
        ("bogus", False),
        ("FA2", True),
        ("cudnn", True),
    ):
        if raw is None:
            monkeypatch.delenv(dense.ENV_DENSE_PREFILL, raising=False)
        else:
            monkeypatch.setenv(dense.ENV_DENSE_PREFILL, raw)
        assert dense.requires_fresh_prefill_replay() is expected


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
