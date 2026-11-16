# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Measure the device-local prefix cost model once, at engine start-up.

The framework segmentation path has four device-local coefficients (serial
chain, attention FLOPs, KV bytes, per-segment overhead). They are measured
here on synthetic paged KV, then cached per device+geometry so later start-ups
only read a YAML file. Nothing here runs on a serving request, and every
failure path degrades to "no calibration" instead of failing start-up.
"""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger

from vllm_gr.v1.attention.backends.beam_prefix_policy import (
    ENV_CALIBRATION,
    PrefixCalibration,
    device_fingerprint,
    load_calibration,
    remember_calibration,
    store_calibration,
)

logger = init_logger(__name__)

# Bounded synthetic workload: one chain-dominated point, one compute-dominated
# point, one byte-dominated point and one split point. Keep it small enough to
# run before the KV pool is sized, and short enough to stay under a few seconds.
_CHAIN_WIDTH = 32
_COMPUTE_BATCH = 4
_COMPUTE_WIDTH = 256
_MEMORY_BATCH = 4
_MEMORY_WIDTH = 32
_SPLIT_SEGMENTS = 8
_MAX_CALIBRATION_TOKENS = 16384
_MIN_CHAIN_TOKENS = 2048


def _make_case(
    batch: int,
    width: int,
    length: int,
    *,
    heads: int,
    kv_heads: int,
    dim: int,
    dtype: Any,
    block_size: int,
    device: Any,
):
    import torch

    pages = (length + block_size - 1) // block_size
    query = torch.randn(batch * width, heads, dim, device=device, dtype=dtype)
    key = torch.randn(batch * pages, block_size, kv_heads, dim, device=device, dtype=dtype)
    value = torch.randn_like(key)
    meta = type("CalibrationMetadata", (), {})()
    meta.prefix_cu_seqlens_q_device = (
        torch.arange(batch + 1, device=device, dtype=torch.int32) * width
    )
    meta.prefix_seqlens_kv_device = torch.full((batch,), length, device=device, dtype=torch.int32)
    meta.prefix_max_q_len = width
    meta.prefix_max_seq_len = length
    meta.prefix_block_table = torch.arange(batch * pages, device=device, dtype=torch.int32).view(
        batch, pages
    )
    meta.prefix_scheduler_metadata = None
    meta.max_num_splits = 0
    meta.prefix_seqlens_q_cpu = [width] * batch
    return query, key, value, meta


def _measure(case, plan, *, repeats: int = 8) -> float:
    import torch

    from vllm_gr.v1.attention.backends.beam_attn_gpu import prefix_attention

    query, key, value, meta = case
    scale = query.shape[-1] ** -0.5

    def call():
        prefix_attention(
            query,
            key,
            value,
            meta,
            scale=scale,
            fa_version=2,
            sliding_window_size=(-1, -1),
            logits_soft_cap=0,
            plan=plan,
        )

    for _ in range(3):
        call()
    torch.accelerator.synchronize(query.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            call()
    for _ in range(2):
        graph.replay()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    samples = []
    for _ in range(3):
        start.record()
        for _ in range(4):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / (4 * repeats))
    return sorted(samples)[len(samples) // 2]


def calibrate_prefix_device(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: Any,
    block_size: int,
    max_model_len: int,
    device_index: int | None = None,
) -> PrefixCalibration:
    """Measure the four coefficients on this device (no serving traffic)."""
    import torch

    from vllm_gr.v1.attention.backends.beam_prefix_policy import PrefixPlan

    device = torch.device("cuda", device_index if device_index is not None else 0)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    geometry = {
        "heads": num_heads,
        "kv_heads": num_kv_heads,
        "dim": head_dim,
        "dtype": dtype,
        "block_size": block_size,
    }
    chain_length = min(_MAX_CALIBRATION_TOKENS, (max_model_len // block_size) * block_size)
    chain_length = max(chain_length, min(_MIN_CHAIN_TOKENS, max_model_len))
    other_length = min(chain_length, 8192)

    def shape(batch, width, length):
        from vllm_gr.v1.attention.backends.beam_prefix_policy import PrefixShape

        return PrefixShape(
            batch=batch,
            width=width,
            length=length,
            heads=num_heads,
            kv_heads=num_kv_heads,
            dim=head_dim,
            dtype=str(dtype),
            block_size=block_size,
        )

    def timed(batch, width, length, segments):
        case = _make_case(batch, width, length, **geometry, device=device)
        try:
            plan = PrefixPlan(2) if segments == 1 else PrefixPlan(2, 1, segments)
            return _measure(case, plan)
        finally:
            del case
            torch.accelerator.empty_cache()

    # The probe reports milliseconds; the cost model works in microseconds.
    native_chain = timed(1, _CHAIN_WIDTH, chain_length, 1) * 1000.0
    split_chain = timed(1, _CHAIN_WIDTH, chain_length, _SPLIT_SEGMENTS) * 1000.0
    native_compute = timed(_COMPUTE_BATCH, _COMPUTE_WIDTH, other_length, 1) * 1000.0
    native_memory = timed(_MEMORY_BATCH, _MEMORY_WIDTH, other_length, 1) * 1000.0

    def waves(batch, width, segments):
        ctas = batch * num_heads * ((width + 127) // 128)
        return max(1, -(-(ctas * segments) // max(1, sm_count)))

    # Serial chain: only the (1, W32) point is chain dominated, but the wave
    # count still has to be divided out.
    u_ns = native_chain * 1000.0 / (waves(1, _CHAIN_WIDTH, 1) * chain_length)
    # Compute floor: the wide-beam point is FLOP bound, so the whole measured
    # time is attributed to the attention FLOPs.
    c_comp = native_compute / (_COMPUTE_BATCH * _COMPUTE_WIDTH * other_length)
    # Byte floor: whatever the chain/no-FLOPs model does not explain.
    chain_mem = waves(_MEMORY_BATCH, _MEMORY_WIDTH, 1) * other_length * u_ns / 1000.0
    comp_mem = c_comp * _MEMORY_BATCH * _MEMORY_WIDTH * other_length
    c_mem = max(0.0, native_memory - max(chain_mem, comp_mem)) / (_MEMORY_BATCH * other_length)
    # Per-segment overhead: residual of the split point over the model floor.
    chain_split = waves(1, _CHAIN_WIDTH, _SPLIT_SEGMENTS) * (chain_length / _SPLIT_SEGMENTS)
    floor_split = max(
        chain_split * u_ns / 1000.0,
        c_comp * 1 * _CHAIN_WIDTH * chain_length,
        c_mem * 1 * chain_length,
    )
    m_us = max(0.0, split_chain - floor_split) / _SPLIT_SEGMENTS

    calibration = PrefixCalibration(
        u_ns_per_token=max(u_ns, 1e-3),
        c_comp_us=max(c_comp, 0.0),
        c_mem_us=c_mem,
        m_us=m_us,
    )
    probe = shape(1, _CHAIN_WIDTH, chain_length)
    predicted = calibration.estimate_us(probe, 1, sm_count)
    logger.info(
        "vllm_gr: prefix calibration %s u=%.2fns/token c_comp=%.3gu c_mem=%.3gu "
        "m=%.2fus (measured %.3fms, predicted %.3fms, sm=%d)",
        device,
        calibration.u_ns_per_token,
        calibration.c_comp_us,
        calibration.c_mem_us,
        calibration.m_us,
        native_chain,
        predicted,
        sm_count,
    )
    return calibration


def ensure_prefix_calibration(worker: Any) -> PrefixCalibration | None:
    """Initialize a fixed calibration snapshot once for this worker.

    Runs before any Beam graph capture and blocks start-up for the measurement
    (~1-3 s per device). Never raises: without a calibration the runtime keeps
    the native prefix path. Failed or disabled initialization is remembered too.
    """
    if not hasattr(worker, "_gr_prefix_calibration"):
        worker._gr_prefix_calibration = _initialize_prefix_calibration(worker)
    return worker._gr_prefix_calibration


def _initialize_prefix_calibration(worker: Any) -> PrefixCalibration | None:
    if (os.environ.get(ENV_CALIBRATION) or "").strip().lower() == "off":
        return None
    try:
        import torch

        device = torch.device(getattr(worker, "device", "cuda"))
        if device.type != "cuda":
            return None
        model_config = worker.model_config
        hf_config = model_config.hf_config
        heads = int(hf_config.num_attention_heads)
        kv_heads = int(getattr(hf_config, "num_key_value_heads", heads))
        dim = int(getattr(hf_config, "head_dim", hf_config.hidden_size // heads))
        dtype = getattr(model_config, "dtype", torch.bfloat16)
        cache_config = getattr(worker, "cache_config", None)
        block_size = int(getattr(cache_config, "block_size", 16))
        fingerprint = device_fingerprint(device.index)
        geometry = {
            "heads": heads,
            "kv_heads": kv_heads,
            "dim": dim,
            "dtype": str(dtype),
            "block_size": block_size,
        }
        remember_calibration(fingerprint, geometry, None)
        cached = load_calibration(fingerprint, geometry)
        if cached is not None:
            remember_calibration(fingerprint, geometry, cached)
            return cached
        calibration = calibrate_prefix_device(
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=dim,
            dtype=dtype,
            block_size=block_size,
            max_model_len=int(model_config.max_model_len),
            device_index=device.index,
        )
        # Match the persisted precision formerly read back on serving requests.
        calibration = PrefixCalibration(**calibration.as_mapping())
        store_calibration(fingerprint, geometry, calibration)
        remember_calibration(fingerprint, geometry, calibration)
        return calibration
    except Exception as error:  # noqa: BLE001 - start-up must stay alive
        logger.warning("vllm_gr: prefix calibration skipped (%s)", error)
        return None
