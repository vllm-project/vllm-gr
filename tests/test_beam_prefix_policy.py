# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm_gr.v1.attention.backends import beam_prefix_policy as policy
from vllm_gr.v1.worker.beam_decode_graph import BeamDecodeGraph


@pytest.mark.parametrize(
    "changes",
    [
        {"fa_version": 1},
        {"num_splits": 3},
        {"fa_version": 2, "num_splits": 4},
        {"segments": 4},
        {"segments": True},
    ],
)
def test_invalid_or_unsupported_plans_rejected(changes):
    with pytest.raises(ValueError):
        policy.PrefixPlan(**({"fa_version": 3} | changes))


def test_length_buckets_cover_live_lengths_and_respect_capacity():
    for length in (0, 1, 16, 1023, 1024, 1025, 10240, 12000):
        bucket = policy.prefix_bucket(length, 16, 12000)
        assert length <= bucket <= 12000
        assert bucket % 16 == 0
    assert policy.prefix_bucket(1024, 16, 12000) != policy.prefix_bucket(1025, 16, 12000)


def test_length_graphs_do_not_alias_and_new_storage_invalidates_all_buckets():
    graph = object.__new__(BeamDecodeGraph)
    graph.wrapper = SimpleNamespace(concrete_cudagraph_entries={})
    graph._registries = {}
    graph._bindings = {}
    graph.worker = SimpleNamespace()
    for bucket in (1024, 16384):
        graph.select(4, 128, bucket)
        graph.wrapper.concrete_cudagraph_entries[graph.descriptor] = bucket
        graph._bindings[graph.signature] = (None, None, None)
    graph.select(4, 128, 1024)
    assert graph.wrapper.concrete_cudagraph_entries[graph.descriptor] == 1024
    graph.select(4, 128, 16384)
    assert graph.wrapper.concrete_cudagraph_entries[graph.descriptor] == 16384
    graph.discard_geometry(4, 128)
    assert not graph._bindings
    assert all(not entries for entries in graph._registries.values())


def test_fixed_width_v2_captures_a_new_length_bucket(monkeypatch):
    from unittest.mock import Mock

    from vllm_gr.v1.worker.gpu_beam_stage_runner_v2 import V2StageExecutionHooks

    worker = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        _vllm_gr_config=SimpleNamespace(beam_graph_enabled=True, beam_online_batch_sizes=(1,)),
        model=SimpleNamespace(compute_logits=Mock(return_value="logits")),
    )
    graph = Mock(ready=False)
    graph.capture.side_effect = lambda *args, **kwargs: setattr(graph, "ready", True)
    hooks = V2StageExecutionHooks(worker)
    monkeypatch.setattr(hooks, "decode_graph", Mock(return_value=graph))
    owner = SimpleNamespace(
        bindings=(None,),
        inputs=SimpleNamespace(width=128),
        dynamic=None,
        metadata=SimpleNamespace(prefix_length_bucket=1024),
        _forward_hidden=Mock(),
        _forward=Mock(),
    )
    assert hooks.decode_logits(owner) == "logits"
    assert hooks.decode_logits(owner) == "logits"
    graph.select.assert_called_with(1, 128, 1024)
    graph.capture.assert_called_once_with(owner._forward_hidden, persistent_install=False)
    assert graph.replay.call_count == 2
    owner._forward.assert_not_called()


def test_metadata_buckets_only_when_a_segmented_plan_is_selected(monkeypatch):
    import torch

    from vllm_gr.v1.attention.backends.beam_attn_metadata import (
        BeamAttentionMetadataBuilder as Builder,
    )

    monkeypatch.setattr(policy, "policy_mode", lambda: "auto")
    monkeypatch.setattr(policy, "available_prefix_versions", lambda device: (2,))
    monkeypatch.setattr(policy, "device_fingerprint", lambda device: {"capability": [8, 9]})
    monkeypatch.setattr(policy, "multi_processor_count", lambda device: 92)
    context = SimpleNamespace(
        max_decode_steps=3,
        dtype=torch.bfloat16,
        pools=[SimpleNamespace(unshared_key=torch.empty(1, 2, 2, 3, 64))],
    )

    def build(length, plan):
        monkeypatch.setattr(policy, "resolve_prefix_plan", lambda shape, **kwargs: plan)
        layout = Builder.build_async_batch_layout(
            width=2,
            prompt_lengths=[length],
            decode_step=0,
            block_ids_by_request=[list(range((length + 15) // 16))],
            max_blocks=1024,
        )
        views = {name: torch.tensor(values, dtype=torch.int32) for name, values in layout.items()}
        return Builder.build_async(
            layout=layout,
            views=views,
            session=SimpleNamespace(context=context),
            mask=torch.ones(2, dtype=torch.bool),
            slots=torch.full((2,), -1),
            block_size=16,
            max_model_len=16384,
            num_heads=4,
            num_kv_heads=2,
        )

    # Tensor.item is forbidden: all bucket dispatch must use existing host mirrors.
    monkeypatch.setattr(torch.Tensor, "item", lambda *args: pytest.fail("device scalar read"))
    segmented = policy.PrefixPlan(2, 1, 4)
    short = build(1023, segmented)
    assert short.prefix_max_seq_len == short.prefix_length_bucket == 1024
    ladder = build(1025, segmented)
    assert ladder.prefix_length_bucket == 2048
    # No plan -> native single graph at model capacity.
    native = build(1025, None)
    assert native.prefix_length_bucket == 0 and native.prefix_max_seq_len == 16384


def test_bucket_ladder_is_1024_aligned_and_splittable():
    for length, expected in ((1024, 1024), (1025, 2048), (10240, 10240), (10241, 11264)):
        bucket = policy.prefix_bucket(length, 16, 16384)
        assert bucket == expected
        assert bucket % policy._LENGTH_BUCKET_STEP == 0
        # The always-available segment counts (2, 4, 8) divide any bucket.
        for segments in (2, 4, 8):
            assert (bucket // 16) % segments == 0
    # The model bound still wins when it is tighter than the ladder step.
    assert policy.prefix_bucket(12000, 16, 12000) == 12000


def test_segment_search_only_uses_exact_splits():
    calibration = policy.PrefixCalibration(
        u_ns_per_token=31.6, c_comp_us=9.3e-5, c_mem_us=6.8e-3, m_us=4.0
    )
    long = policy.PrefixShape(
        batch=1,
        width=32,
        length=10240,
        heads=16,
        kv_heads=8,
        dim=128,
        dtype="torch.bfloat16",
        block_size=16,
    )
    assert calibration.best(long, 92)[0] == 5
    # 1024 tokens is 64 pages: five segments would not divide it evenly.
    short = policy.PrefixShape(
        batch=1,
        width=32,
        length=1024,
        heads=16,
        kv_heads=8,
        dim=128,
        dtype="torch.bfloat16",
        block_size=16,
    )
    assert calibration.best(short, 92)[0] in (1, 2, 4, 8)


def test_calibration_roundtrip_and_lookup(monkeypatch, tmp_path):
    path = tmp_path / "calib.yaml"
    monkeypatch.setenv(policy.ENV_CALIBRATION_PATH, str(path))
    fingerprint = {"name": "NVIDIA L20", "capability": [8, 9], "sm_count": 92}
    geometry = {"heads": 16, "kv_heads": 8, "dim": 128, "dtype": "torch.bfloat16", "block_size": 16}
    calibration = policy.PrefixCalibration(
        u_ns_per_token=31.6, c_comp_us=9.3e-5, c_mem_us=6.8e-3, m_us=4.0
    )
    assert policy.load_calibration(fingerprint, geometry) is None
    policy.store_calibration(fingerprint, geometry, calibration)
    text = path.read_text(encoding="utf-8")
    assert "Cost model" in text and "u_ns_per_token" in text
    loaded = policy.load_calibration(fingerprint, geometry)
    assert loaded == calibration
    # A different geometry must not reuse the entry.
    assert policy.load_calibration(fingerprint, geometry | {"dim": 64}) is None


def test_explicit_fa_version_is_remembered_and_wins(monkeypatch, tmp_path):
    """Request path has no vLLM config context; the start-up choice must win."""
    monkeypatch.setenv(policy.ENV_CALIBRATION_PATH, str(tmp_path / "calib.yaml"))
    monkeypatch.setenv(policy.ENV_PREFIX_POLICY, "auto")
    shape = policy.PrefixShape(
        batch=1,
        width=32,
        length=10240,
        heads=16,
        kv_heads=8,
        dim=128,
        dtype="torch.bfloat16",
        block_size=16,
    )
    fingerprint = {"name": "H20", "capability": [9, 0], "sm_count": 92}
    geometry = {"heads": 16, "kv_heads": 8, "dim": 128, "dtype": "torch.bfloat16", "block_size": 16}
    policy.store_calibration(
        fingerprint,
        geometry,
        policy.PrefixCalibration(u_ns_per_token=31.6, c_comp_us=9.3e-5, c_mem_us=6.8e-3, m_us=4.0),
    )
    args = {
        "default_version": 2,
        "fingerprint": fingerprint,
        "supported_versions": (2, 3),
        "sm_count": 92,
    }
    policy.remember_explicit_fa_version(2)
    try:
        assert policy.resolve_prefix_plan(shape, **args) == policy.PrefixPlan(2, 1, 5)
    finally:
        policy.reset_explicit_fa_version()
    # An explicit FA3/FA4 keeps the backend's own split-K heuristic.
    policy.remember_explicit_fa_version(3)
    try:
        assert policy.resolve_prefix_plan(shape, **args) is None
    finally:
        policy.reset_explicit_fa_version()


def test_broken_optional_backend_is_treated_as_unavailable(monkeypatch):
    import contextlib
    import sys
    from types import ModuleType

    import torch

    # NPU images ship a vllm without the vllm_flash_attn package, so stub it
    # instead of importing it; the probe only needs is_fa_version_supported.
    def fake_supported(version):
        if version == 4:
            # What a broken CuTe install raises on some SM90 images.
            raise AttributeError("module 'cutlass.cute.core' is missing a required symbol")
        return version in (2, 3)

    package = ModuleType("vllm.vllm_flash_attn")
    flash = ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
    monkeypatch.setattr(flash, "is_fa_version_supported", fake_supported, raising=False)
    monkeypatch.setattr(package, "flash_attn_interface", flash, raising=False)
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", package)
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn.flash_attn_interface", flash)
    monkeypatch.setattr(torch.accelerator, "device_index", lambda index: contextlib.nullcontext())
    policy.available_prefix_versions.cache_clear()
    try:
        assert policy.available_prefix_versions(0) == (2, 3)
    finally:
        policy.available_prefix_versions.cache_clear()


def test_default_on_needs_a_measured_win_and_never_overrides_fa3(monkeypatch, tmp_path):
    monkeypatch.setenv(policy.ENV_CALIBRATION_PATH, str(tmp_path / "calib.yaml"))
    monkeypatch.setenv(policy.ENV_PREFIX_POLICY, "auto")
    shape = policy.PrefixShape(
        batch=1,
        width=32,
        length=10240,
        heads=16,
        kv_heads=8,
        dim=128,
        dtype="torch.bfloat16",
        block_size=16,
    )
    fingerprint = {"name": "NVIDIA L20", "capability": [8, 9], "sm_count": 92}
    geometry = {"heads": 16, "kv_heads": 8, "dim": 128, "dtype": "torch.bfloat16", "block_size": 16}
    args = {
        "default_version": 2,
        "fingerprint": fingerprint,
        "supported_versions": (2,),
        "sm_count": 92,
    }
    # No calibration yet -> keep the backend's native path.
    assert policy.resolve_prefix_plan(shape, **args) is None
    policy.store_calibration(
        fingerprint,
        geometry,
        policy.PrefixCalibration(u_ns_per_token=31.6, c_comp_us=9.3e-5, c_mem_us=6.8e-3, m_us=4.0),
    )
    # 10240 / B1 / W32 on 92 SMs: five segments, a measured 3.6x on L20.
    assert policy.resolve_prefix_plan(shape, **args) == policy.PrefixPlan(2, 1, 5)
    # FA3/FA4 are never overridden: their own split-K heuristic runs instead.
    assert (
        policy.resolve_prefix_plan(
            shape, **(args | {"default_version": 3, "supported_versions": (2, 3)})
        )
        is None
    )
    # Compute-bound wide beam stays native.
    wide = policy.PrefixShape(
        batch=4,
        width=256,
        length=10240,
        heads=16,
        kv_heads=8,
        dim=128,
        dtype="torch.bfloat16",
        block_size=16,
    )
    assert policy.resolve_prefix_plan(wide, **args) is None
    # Single-request wide beam has a predicted win but no headroom over the
    # compute floor: measured on L20 it actually loses 12%, so it stays native.
    narrow_wide = policy.PrefixShape(
        batch=1,
        width=256,
        length=10240,
        heads=16,
        kv_heads=8,
        dim=128,
        dtype="torch.bfloat16",
        block_size=16,
    )
    assert policy.resolve_prefix_plan(narrow_wide, **args) is None
    # The kill switch always wins.
    monkeypatch.setenv(policy.ENV_PREFIX_POLICY, "off")
    assert policy.resolve_prefix_plan(shape, **args) is None


def test_short_prefix_stays_native(monkeypatch, tmp_path):
    """A 1024-token prefix is launch bound, so the relative win is not real.

    Measured on L20 with the calibration below: the model predicts a 1.5x prefix
    win at 1024/B1/W128 and the end-to-end request came out 5% slower.
    """
    monkeypatch.setenv(policy.ENV_CALIBRATION_PATH, str(tmp_path / "calib.yaml"))
    monkeypatch.setenv(policy.ENV_PREFIX_POLICY, "auto")
    fingerprint = {"name": "NVIDIA L20", "capability": [8, 9], "sm_count": 92}
    geometry = {"heads": 16, "kv_heads": 8, "dim": 128, "dtype": "torch.bfloat16", "block_size": 16}
    policy.store_calibration(
        fingerprint,
        geometry,
        policy.PrefixCalibration(u_ns_per_token=31.6, c_comp_us=9.3e-5, c_mem_us=6.8e-3, m_us=4.0),
    )
    args = {
        "default_version": 2,
        "fingerprint": fingerprint,
        "supported_versions": (2,),
        "sm_count": 92,
    }
    short = policy.PrefixShape(
        batch=1,
        width=128,
        length=1024,
        heads=16,
        kv_heads=8,
        dim=128,
        dtype="torch.bfloat16",
        block_size=16,
    )
    assert policy.resolve_prefix_plan(short, **args) is None
    long = policy.PrefixShape(
        batch=1,
        width=128,
        length=10240,
        heads=16,
        kv_heads=8,
        dim=128,
        dtype="torch.bfloat16",
        block_size=16,
    )
    assert policy.resolve_prefix_plan(long, **args) is not None
