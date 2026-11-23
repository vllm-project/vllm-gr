# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
from types import SimpleNamespace

import pytest

from vllm_gr.v1.attention.backends import beam_attn


def _config(prefix_caching=False, chunked=False, kv_transfer=None, *, mode=None):
    return SimpleNamespace(
        additional_config={"vllm_gr": {"gr": {"dense_prefill": mode}}} if mode is not None else {},
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
        ("cudnn", True, _config(), "cudnn"),
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

    if config is not None:
        config.additional_config = _config(mode=raw).additional_config
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", cuda)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    assert beam_attn._resolve_dense_prefill_mode() == expected


def test_dense_prefill_mode_off_when_batch_invariant(monkeypatch):
    import vllm.config

    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", True)
    config = _config(mode="cudnn")
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(beam_attn.envs, "VLLM_BATCH_INVARIANT", True)
    assert beam_attn._resolve_dense_prefill_mode() is None
    monkeypatch.setattr(beam_attn.envs, "VLLM_BATCH_INVARIANT", False)
    assert beam_attn._resolve_dense_prefill_mode() == "cudnn"


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


def test_fresh_prefill_capture_scope():
    from vllm_gr.v1 import forward_context

    assert not forward_context.in_fresh_prefill_capture()
    with forward_context.fresh_prefill_capture():
        assert forward_context.in_fresh_prefill_capture()
        with pytest.raises(RuntimeError), forward_context.fresh_prefill_capture():
            raise RuntimeError
        assert forward_context.in_fresh_prefill_capture()
    assert not forward_context.in_fresh_prefill_capture()


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
