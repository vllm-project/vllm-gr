# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disallow-untyped-decorators=False
"""Dense references and exact FA2 comparisons for short-suffix attention."""

from collections.abc import Generator
from types import SimpleNamespace

import pytest
import torch

Case = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="session", params=[2, 3, 4])
def prefix_fa_version(request: pytest.FixtureRequest) -> int:
    from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported

    version = int(request.param)
    if not is_fa_version_supported(version):
        pytest.skip(f"FlashAttention {version} is not supported on this device")
    if version == 4:
        pytest.importorskip(
            "vllm.vllm_flash_attn.cute.interface",
            reason="FA4 requires compatible optional CuTe/Quack dependencies",
            exc_type=ImportError,
        )
    return version


@pytest.fixture(autouse=True)
def disable_tf32() -> Generator[None, None, None]:
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def make_case(
    width: int,
    heads: int,
    kv_heads: int,
    dim: int,
    capacity: int,
    dtype: torch.dtype,
    prefix_len: int = 128,
    pool_slots: int = 2,
) -> Case:
    torch.manual_seed(1947)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=torch.float32).to(dtype)

    # Nonzero slot and excess pool capacity catch accidental contiguous/slot assumptions.
    key = rand(pool_slots, width + 3, kv_heads, capacity, dim)
    value = rand(pool_slots, width + 3, kv_heads, capacity, dim)
    query = rand(width, heads, dim)
    pk, pv = rand(prefix_len, kv_heads, dim), rand(prefix_len, kv_heads, dim)
    active = torch.arange(width, device="cuda") % 3 != 1

    return query, key, value, pk, pv, active


def reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pk: torch.Tensor,
    pv: torch.Tensor,
    step: int,
    active: torch.Tensor,
    slot: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize full prefix+suffix attention in FP32, without FlashAttention."""
    width, heads, dim = query.shape
    group = heads // key.shape[2]

    q = query.float()
    pkey, pval = pk.float().repeat_interleave(group, 1), pv.float().repeat_interleave(group, 1)
    ps = torch.einsum("bhd,thd->bht", q, pkey) * dim**-0.5
    if pk.shape[0]:
        lse = torch.logsumexp(ps, -1)
        prefix = torch.einsum("bht,thd->bhd", ps.softmax(-1), pval)
    else:
        lse = torch.full((width, heads), -float("inf"), device="cuda")
        prefix = torch.zeros_like(q)

    extent = max(step + 1, 1)
    k = key[slot, :width, :, :extent].float().repeat_interleave(group, 1)
    v = value[slot, :width, :, :extent].float().repeat_interleave(group, 1)
    live = active & (step >= 0)
    k = torch.where(live[:, None, None, None], k, 0)
    v = torch.where(live[:, None, None, None], v, 0)

    ss = torch.einsum("bhd,bhtd->bht", q, k) * dim**-0.5
    scores = torch.cat((ps, ss), -1)
    weights = scores.softmax(-1)
    dense = torch.einsum("bht,thd->bhd", weights[..., : pk.shape[0]], pval)
    dense += torch.einsum("bht,bhtd->bhd", weights[..., pk.shape[0] :], v)

    return prefix, lse.T.contiguous(), dense


def launch(
    case: Case,
    prefix: torch.Tensor,
    lse: torch.Tensor,
    output: torch.Tensor,
    step: torch.Tensor,
    slot: int = 1,
) -> None:
    from vllm_gr.v1.attention.backends.beam_short_suffix import (
        merge_short_suffix_attention,
    )

    q, k, v, _, _, active = case
    merge_short_suffix_attention(
        output, q, prefix, lse, k, v, slot, step, q.shape[-1] ** -0.5, active
    )


@torch.inference_mode()
def test_runtime_slot_uses_one_compilation_and_matches_fp32_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One fresh JIT specialization must address all four distinct KV slots."""
    from triton.runtime import JITFunction

    from vllm_gr.v1.attention.backends import beam_short_suffix as short

    production_kernel = short._short_suffix_merge_kernel
    # Copy the production parameter policy, rather than imposing the desired
    # SLOT policy in the test. This fails if SLOT becomes constexpr again or
    # loses do_not_specialize, even with a warm process/disk cache.
    fresh_kernel = JITFunction(
        production_kernel.fn,
        do_not_specialize=[
            parameter.name for parameter in production_kernel.params if parameter.do_not_specialize
        ],
    )
    monkeypatch.setattr(short, "_short_suffix_merge_kernel", fresh_kernel)
    kernel_cache = fresh_kernel.device_caches[torch.accelerator.current_device_index()][0]
    assert not kernel_cache

    case = make_case(7, 8, 4, 128, 3, torch.float32, pool_slots=4)
    q, k, v, pk, pv, active = case
    # Distinct histories make incorrect reuse of a previously specialized slot
    # fail the independent attention comparison, not just the cache assertion.
    for slot in range(4):
        v[slot].add_(slot * 3)
    output = torch.empty_like(q)
    step_buffer = torch.tensor([1], dtype=torch.int32, device="cuda")
    for slot in range(4):
        prefix, lse, expected = reference(q, k, v, pk, pv, 1, active, slot=slot)
        launch(case, prefix, lse, output, step_buffer, slot=slot)
        torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
        assert len(kernel_cache) == 1, f"slot={slot} created another compiled specialization"


@pytest.mark.parametrize(
    "geometry",
    [
        (1, 4, 4, 64),
        (7, 8, 4, 128),
        (128, 16, 8, 128),
        (256, 16, 4, 256),
        (3, 8, 1, 64),
        (3, 3, 1, 64),
        (7, 8, 4, 32),
        (7, 8, 4, 96),
    ],
)
@pytest.mark.parametrize("capacity,step", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_against_fp32_dense(
    geometry: tuple[int, int, int, int], capacity: int, step: int, dtype: torch.dtype
) -> None:
    case = make_case(*geometry, capacity, dtype)
    q, k, v, pk, pv, active = case
    prefix, lse, expected = reference(*case[:5], step, active)
    # Production prefix outputs are activation dtype. Test against a full FP32
    # reference, allowing only the expected low-precision intermediate/output error.
    prefix = prefix.to(dtype)
    # Strided query/output/LSE views ensure the fused path does not assume packed tensors.
    q_storage = torch.empty((*q.shape[:-1], q.shape[-1] * 2), dtype=dtype, device="cuda")
    q_view = q_storage[..., ::2]
    q_view.copy_(q)
    case = (q_view, k, v, pk, pv, active)
    output_storage = torch.full_like(q_storage, 123)
    output = output_storage[..., ::2]
    step_buf = torch.tensor([step], dtype=torch.int32, device="cuda")
    launch(case, prefix, lse.T.contiguous().T, output, step_buf)
    atol, rtol = {
        torch.float32: (2e-6, 2e-5),
        torch.float16: (2e-3, 3e-3),
        torch.bfloat16: (1.6e-2, 2e-2),
    }[dtype]
    torch.testing.assert_close(output.float(), expected, atol=atol, rtol=rtol)
    assert torch.all(output_storage[..., 1::2] == 123)
    assert torch.isfinite(output).all()
    print(
        f"fp32_ref dtype={dtype} geometry={geometry} capacity={capacity} step={step} "
        f"max_abs={(output.float() - expected).abs().max().item():.9g}"
    )


@pytest.mark.parametrize("prefix_len", [0, 1, 128])
@pytest.mark.parametrize("magnitude", [1.0, 30.0])
@torch.inference_mode()
def test_fp32_extreme_and_empty_prefix(prefix_len: int, magnitude: float) -> None:
    case = make_case(7, 8, 4, 128, 3, torch.float32, prefix_len)
    q, k, v, pk, pv, active = case
    q.mul_(magnitude)
    for step in [-1, 0, 1, 2]:
        prefix, lse, expected = reference(q, k, v, pk, pv, step, active)
        output = torch.empty_like(q)
        launch(case, prefix, lse, output, torch.tensor([step], device="cuda", dtype=torch.int32))
        torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-4)
        assert torch.isfinite(output).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_empty_fa2_prefix_positive_infinite_lse(dtype: torch.dtype) -> None:
    """FA2 uses +inf LSE for an empty sequence; merge treats it as absent."""
    case = make_case(7, 8, 4, 128, 3, dtype, prefix_len=0)
    q, k, v, pk, pv, active = case
    # The absent prefix state must not contribute, even if its output is stale.
    prefix = torch.full_like(q, float("nan"))
    lse = torch.full((8, 7), float("inf"), device="cuda")
    for step in [0, 1, 2]:
        _, _, expected = reference(q, k, v, pk, pv, step, active)
        output = torch.empty_like(q)
        launch(case, prefix, lse, output, torch.tensor([step], device="cuda", dtype=torch.int32))
        atol, rtol = {
            torch.float32: (2e-6, 2e-5),
            torch.float16: (2e-3, 3e-3),
            torch.bfloat16: (1.6e-2, 2e-2),
        }[dtype]
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("capacity", [3, 4, 32, 128])
@pytest.mark.parametrize("slot", [0, 3])
@torch.inference_mode()
def test_graph_replay_reads_step_mask_and_pool(capacity: int, slot: int) -> None:
    case = make_case(128, 16, 8, 128, capacity, torch.float32, pool_slots=max(2, slot + 1))
    q, k, v, pk, pv, active = case
    prefix, lse, _ = reference(q, k, v, pk, pv, 0, active, slot=slot)
    output = torch.empty_like(q)
    step_buf = torch.zeros(1, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch(case, prefix, lse, output, step_buf, slot=slot)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(case, prefix, lse, output, step_buf, slot=slot)
    for step in [0, capacity - 1, capacity // 2, -1, 0, capacity - 1]:
        step_buf.fill_(step)
        active.logical_not_()
        # Reordered/updated histories at stable addresses, as after beam selection.
        k.copy_(k.roll(1, 1))
        v.copy_(v.roll(1, 1))
        _, _, expected = reference(q, k, v, pk, pv, step, active, slot=slot)
        graph.replay()
        torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("enable_short_suffix", [False, True])
@torch.inference_mode()
def test_cascade_dispatch_and_fallback(
    monkeypatch: pytest.MonkeyPatch, enable_short_suffix: bool
) -> None:
    """Exercise the real caller; FP32 fake prefix avoids FA2's FP16/BF16 restriction."""
    from vllm_gr.v1.attention.backends import beam_attn_gpu as gpu
    from vllm_gr.v1.attention.backends import beam_short_suffix as short

    expect_fallback = not enable_short_suffix
    case = make_case(7, 8, 4, 128, 3, torch.float32)
    q, k, v, pk, pv, active = case
    prefix, lse, expected = reference(q, k, v, pk, pv, 1, active)
    step = torch.tensor([1], dtype=torch.int32, device="cuda")
    pool = SimpleNamespace(
        unshared_key=k, unshared_value=v, num_kv_heads=4, head_dim=128, max_decode_steps=3
    )
    session = SimpleNamespace(
        is_active=True,
        slot=1,
        beam_width=7,
        context=SimpleNamespace(
            decode_step_buf=step, enable_short_suffix_attention=enable_short_suffix
        ),
        suffix_k_buf=torch.empty((21, 4, 128), device="cuda"),
        suffix_v_buf=torch.empty((21, 4, 128), device="cuda"),
    )
    metadata = SimpleNamespace(
        std_indices=None,
        prefix_indices=torch.arange(7, device="cuda"),
        num_prefix_indices=7,
        num_actual_tokens=7,
        prefix_cu_seqlens_q_device=None,
        prefix_seqlens_kv_device=None,
        prefix_max_q_len=7,
        prefix_max_seq_len=128,
        prefix_block_table=None,
        prefix_scheduler_metadata=None,
        max_num_splits=0,
        beam_session=session,
        is_beam_decode=True,
        beam_active_mask=active,
        suffix_max_q_len=1,
        suffix_cu_seqlens_q_device=None,
        suffix_cu_seqlens_k_device=None,
        num_heads=8,
        num_kv_heads=4,
    )
    calls: list[dict[str, object]] = []

    def flash(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(kwargs)
        if len(calls) > 1:
            raise RuntimeError("reached generic suffix")
        return prefix, lse

    monkeypatch.setattr(gpu, "flash_attn_varlen_func", flash)
    if expect_fallback:
        monkeypatch.setattr(
            short,
            "merge_short_suffix_attention",
            lambda *a, **kw: pytest.fail("unsupported config used fast path"),
        )
    output = torch.empty_like(q)
    kwargs = {
        "output": output,
        "query": q,
        "key_cache": pk,
        "value_cache": pv,
        "attn_metadata": metadata,
        "sliding_window_size": (-1, -1),
        "scale": 128**-0.5,
        "logits_soft_cap": 0.0,
        "vllm_flash_attn_version": 2,
        "beam_pool": pool,
    }
    if expect_fallback:
        with pytest.raises(RuntimeError, match="reached generic suffix"):
            gpu.cascade_attention(**kwargs)
    else:
        gpu.cascade_attention(**kwargs)
        assert len(calls) == 1
        torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("capacity", [3, 8, 32])
@pytest.mark.parametrize("prefix_len", [0, 128])
@pytest.mark.parametrize("width", [1, 7, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_real_prefix_against_fp32_dense(
    width: int, dtype: torch.dtype, prefix_len: int, capacity: int, prefix_fa_version: int
) -> None:
    """Verify actual prefix LSE layout/normalization, not just a mocked state."""
    fa_version = prefix_fa_version
    from vllm_gr.v1.attention.backends import beam_attn_gpu as gpu

    case = make_case(width, 16, 8, 128, capacity, dtype)
    q, k, v, pk, pv, active = case
    # FA2 paged prefix: 8 blocks of 16 tokens, shared by all beam queries.
    key_cache, value_cache = pk.reshape(8, 16, 8, 128), pv.reshape(8, 16, 8, 128)
    step_buf = torch.zeros(1, dtype=torch.int32, device="cuda")
    pool = SimpleNamespace(
        unshared_key=k, unshared_value=v, num_kv_heads=8, head_dim=128, max_decode_steps=capacity
    )
    session = SimpleNamespace(
        is_active=True,
        slot=1,
        beam_width=width,
        context=SimpleNamespace(decode_step_buf=step_buf, enable_short_suffix_attention=True),
    )
    md = SimpleNamespace(
        std_indices=None,
        prefix_indices=torch.arange(width, device="cuda"),
        num_prefix_indices=width,
        num_actual_tokens=width,
        prefix_cu_seqlens_q_device=torch.tensor([0, width], dtype=torch.int32, device="cuda"),
        prefix_seqlens_q_cpu=[width],
        prefix_seqlens_kv_device=torch.tensor([prefix_len], dtype=torch.int32, device="cuda"),
        prefix_max_q_len=width,
        prefix_max_seq_len=128,
        prefix_block_table=torch.arange(8, dtype=torch.int32, device="cuda").view(1, 8),
        prefix_scheduler_metadata=None,
        max_num_splits=0,
        beam_session=session,
        is_beam_decode=True,
        beam_active_mask=active,
        suffix_max_q_len=1,
    )
    output = torch.empty_like(q)
    for step in sorted({0, 1, capacity // 2, capacity - 1}):
        step_buf.fill_(step)
        gpu.cascade_attention(
            output,
            q,
            key_cache,
            value_cache,
            md,
            (-1, -1),
            128**-0.5,
            0.0,
            fa_version,
            beam_pool=pool,
        )
        _, _, expected = reference(q, k, v, pk[:prefix_len], pv[:prefix_len], step, active)
        atol = 2e-3 if dtype == torch.float16 else 1.6e-2
        torch.testing.assert_close(output.float(), expected, atol=atol, rtol=atol)


@pytest.mark.parametrize("width", [1, 7, 32])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("graph_replay", [False, True])
@torch.inference_mode()
def test_multi_request_prefix_state_and_graph(
    prefix_fa_version: int, width: int, fused: bool, graph_replay: bool
) -> None:
    """Real FA states, paged/unequal/empty prefixes and changing device inputs.

    Multiple W=1 requests expose FA2 GQA LSE permutations that B=1 cannot.
    Test both consumers: fused suffix and the generic suffix + CUDA merge.
    """
    from vllm_gr.v1.attention.backends import beam_attn_gpu as gpu
    from vllm_gr.v1.attention.backends.beam_attention_state import (
        normalize_flash_attn_state,
    )

    batch, heads, kv_heads, dim, capacity = 3, 8, 4, 64, 3
    rows = batch * width
    q, k, v, _, _, active = make_case(rows, heads, kv_heads, dim, capacity, torch.bfloat16)
    key_cache = torch.randn(24, 16, kv_heads, dim, device="cuda", dtype=q.dtype)
    value_cache = torch.randn_like(key_cache)
    table = torch.randperm(24, device="cuda", dtype=torch.int32).reshape(batch, 8)
    prefix_k = key_cache[table.long()].flatten(1, 2)
    prefix_v = value_cache[table.long()].flatten(1, 2)
    lengths = torch.tensor([32, 37, 125], dtype=torch.int32, device="cuda")
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * width
    step = torch.zeros(1, dtype=torch.int32, device="cuda")
    suffix_cu_k = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
    pool = SimpleNamespace(
        unshared_key=k,
        unshared_value=v,
        num_kv_heads=kv_heads,
        head_dim=dim,
        max_decode_steps=capacity,
    )
    session = SimpleNamespace(
        is_active=True,
        slot=1,
        beam_width=rows,
        context=SimpleNamespace(decode_step_buf=step, enable_short_suffix_attention=fused),
        suffix_k_buf=torch.zeros(rows * capacity, kv_heads, dim, device="cuda", dtype=q.dtype),
        suffix_v_buf=torch.zeros(rows * capacity, kv_heads, dim, device="cuda", dtype=q.dtype),
    )
    md = SimpleNamespace(
        std_indices=None,
        prefix_indices=torch.arange(rows, device="cuda"),
        num_prefix_indices=rows,
        num_actual_tokens=rows,
        prefix_cu_seqlens_q_device=cu_q,
        prefix_seqlens_q_cpu=[width] * batch,
        prefix_seqlens_kv_device=lengths,
        prefix_max_q_len=width,
        prefix_max_seq_len=128,
        prefix_block_table=table,
        prefix_scheduler_metadata=None,
        max_num_splits=0,
        beam_session=session,
        is_beam_decode=True,
        beam_active_mask=active,
        suffix_max_q_len=1,
        suffix_cu_seqlens_q_device=torch.arange(rows + 1, dtype=torch.int32, device="cuda"),
        suffix_cu_seqlens_k_device=suffix_cu_k,
        num_heads=heads,
        num_kv_heads=kv_heads,
    )
    output = torch.empty_like(q)

    def run() -> None:
        gpu.cascade_attention(
            output,
            q,
            key_cache,
            value_cache,
            md,
            (-1, -1),
            dim**-0.5,
            0.0,
            prefix_fa_version,
            beam_pool=pool,
        )

    # Check LSE itself, independently of either suffix consumer. This catches
    # a permutation even when prefix attention dominates the final output.
    out, lse = gpu.flash_attn_varlen_func(
        q=q,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_q,
        seqused_k=lengths,
        max_seqlen_q=width,
        max_seqlen_k=128,
        block_table=table,
        causal=False,
        softmax_scale=dim**-0.5,
        return_softmax_lse=True,
        fa_version=prefix_fa_version,
    )
    state = normalize_flash_attn_state(
        out,
        lse,
        fa_version=prefix_fa_version,
        num_kv_heads=kv_heads,
        max_query_len=width,
        paged_kv=True,
    )
    for b, length in enumerate([32, 37, 125]):
        sl = slice(b * width, (b + 1) * width)
        ref_out, ref_lse, _ = reference(
            q[sl],
            k[:, sl],
            v[:, sl],
            prefix_k[b, :length],
            prefix_v[b, :length],
            0,
            active[sl],
        )
        torch.testing.assert_close(state.lse[:, sl], ref_lse, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(state.out[sl].float(), ref_out, atol=1.6e-2, rtol=2e-2)

    run()
    graph = None
    if graph_replay:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()

    for decode_step, prefix_lengths in [(0, [32, 37, 125]), (2, [0, 88, 1]), (1, [53, 0, 96])]:
        step.fill_(decode_step)
        lengths.copy_(torch.tensor(prefix_lengths, dtype=torch.int32, device="cuda"))
        suffix_cu_k.copy_(md.suffix_cu_seqlens_q_device * (decode_step + 1))
        active.logical_not_()
        # Simulate an updated/reordered beam history between graph replays.
        v.copy_(v.roll(1, dims=1))
        if graph is None:
            run()
        else:
            graph.replay()
        for b, length in enumerate(prefix_lengths):
            sl = slice(b * width, (b + 1) * width)
            _, _, expected = reference(
                q[sl],
                k[:, sl],
                v[:, sl],
                prefix_k[b, :length],
                prefix_v[b, :length],
                decode_step,
                active[sl],
            )
            assert torch.isfinite(expected).all(), ("reference", decode_step, b)
            assert torch.isfinite(output[sl]).all(), ("output", decode_step, b)
            torch.testing.assert_close(output[sl].float(), expected, atol=1.6e-2, rtol=2e-2)


@torch.inference_mode()
def test_unmasked_suffix() -> None:
    from vllm_gr.v1.attention.backends.beam_short_suffix import (
        merge_short_suffix_attention,
    )

    case = make_case(7, 8, 4, 128, 3, torch.float32)
    q, k, v, pk, pv, active = case
    active.fill_(True)
    prefix, lse, expected = reference(q, k, v, pk, pv, 2, active)
    output = torch.empty_like(q)
    step = torch.tensor([2], dtype=torch.int32, device="cuda")
    merge_short_suffix_attention(output, q, prefix, lse, k, v, 1, step, 128**-0.5)
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dim", [32, 64, 96, 128, 256])
@pytest.mark.parametrize("step", [0, 1, 2])
@torch.inference_mode()
def test_fp32_large_values_and_cancellation(dim: int, step: int) -> None:
    """Preserve small FP32 residuals when large prefix/suffix values cancel.

    Q/K and prefix LSE are zero: all valid states have equal weight and every
    exp is exactly one. This isolates FP32 value accumulation from dot-product
    and exp approximation. The independent CPU FP64 reference is an average.
    """
    from vllm_gr.v1.attention.backends.beam_short_suffix import (
        merge_short_suffix_attention,
    )

    width, heads, kv_heads, capacity, slot = 7, 8, 4, 3, 1
    extent = step + 1
    query = torch.zeros((width, heads, dim), device="cuda", dtype=torch.float32)
    key = torch.zeros((2, width + 3, kv_heads, capacity, dim), device="cuda", dtype=torch.float32)
    value = torch.zeros_like(key)
    # Base and prefix values are exactly representable even in BF16. The
    # 2^-12 suffix residual is representable in FP32 at this magnitude, but
    # disappears if an intermediate V/attention state is reduced to BF16.
    base = (
        (torch.arange(width * kv_heads * dim, device="cuda").reshape(width, kv_heads, dim) % 18)
        + 13
    ).float() * 8.0
    value[slot, :width] = (base + 2.0**-12).unsqueeze(2).expand(-1, -1, capacity, -1)
    prefix = -extent * base.repeat_interleave(heads // kv_heads, dim=1)
    prefix_lse = torch.zeros((heads, width), dtype=torch.float32, device="cuda")
    active = torch.arange(width, device="cuda") % 3 != 1
    output = torch.empty_like(query)
    step_buffer = torch.tensor([step], device="cuda", dtype=torch.int32)
    merge_short_suffix_attention(
        output,
        query,
        prefix,
        prefix_lse,
        key,
        value,
        slot,
        step_buffer,
        dim**-0.5,
        active,
    )

    # Inactive beams retain extent zero-KV states, so their denominator is
    # still extent+1. Compute entirely in CPU FP64 from the stored FP32 inputs.
    prefix64 = prefix.cpu().double()
    suffix64 = value[slot, :width, :, :extent].cpu().double()
    suffix64 = suffix64.repeat_interleave(heads // kv_heads, dim=1)
    active_cpu = active.cpu()
    suffix64 = torch.where(active_cpu[:, None, None, None], suffix64, 0.0)
    expected = (prefix64 + suffix64.sum(dim=2)) / (extent + 1)
    weighted_absolute_value = (prefix64.abs() + suffix64.abs().sum(dim=2)) / (extent + 1)
    actual = output.cpu().double()
    error = (actual - expected).abs()

    # Prespecified FP32 addition/division budget scaled by the values being
    # combined, which remains meaningful when the signed result cancels.
    # No empirical tolerance tuning or exp error assumption is needed here.
    unit_roundoff = 2.0**-24
    operations = extent + 4  # Sum plus a conservative final-division allowance.
    gamma = operations * unit_roundoff / (1.0 - operations * unit_roundoff)
    budget = gamma * weighted_absolute_value
    assert torch.isfinite(actual).all()
    assert torch.all(error <= budget)
    # The residual is deliberately larger than the roundoff budget: an
    # accidental BF16 intermediate that erases it must fail this regression.
    assert torch.all(expected[active_cpu] > budget[active_cpu])
    assert torch.all(actual[active_cpu] > 0)
    tight_failures = (error > 2e-6 + 2e-5 * expected.abs()).sum().item()
    print(
        f"fp64_large_value dim={dim} step={step} "
        f"max_abs={error.max().item():.9g} "
        f"max_error_to_budget={(error / budget).max().item():.9g} "
        f"tight_fp32_failures={tight_failures}"
    )


@pytest.mark.parametrize("geometry", [(7, 8, 4, 64), (128, 16, 8, 128), (3, 8, 1, 256)])
@pytest.mark.parametrize("capacity", [4, 8, 16, 32, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("prefix_len", [0, 128])
@torch.inference_mode()
def test_long_suffix_against_fp32_dense(
    geometry: tuple[int, int, int, int], capacity: int, dtype: torch.dtype, prefix_len: int
) -> None:
    """Long pools exercise early, middle, full and masked bootstrap extents."""
    case = make_case(*geometry, capacity, dtype, prefix_len)
    q, k, v, pk, pv, active = case
    output = torch.empty_like(q)
    step_buffer = torch.zeros(1, dtype=torch.int32, device="cuda")
    atol, rtol = {
        torch.float32: (2e-6, 2e-5),
        torch.float16: (2e-3, 3e-3),
        torch.bfloat16: (1.6e-2, 2e-2),
    }[dtype]
    for step in sorted({-1, 0, 1, capacity // 2, capacity - 1}):
        prefix, lse, expected = reference(q, k, v, pk, pv, step, active)
        step_buffer.fill_(step)
        launch(case, prefix.to(dtype), lse, output, step_buffer)
        torch.testing.assert_close(output.float(), expected, atol=atol, rtol=rtol)
        assert torch.isfinite(output).all()
        print(
            f"long_suffix_ref dtype={dtype} geometry={geometry} capacity={capacity} "
            f"prefix_len={prefix_len} step={step} "
            f"max_abs={(output.float() - expected).abs().max().item():.9g}"
        )


def packed_fa2_reference(
    query: torch.Tensor,
    prefix: torch.Tensor,
    prefix_lse: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot: int,
    step: torch.Tensor,
    active: torch.Tensor | None,
) -> torch.Tensor:
    """Run the real disabled path, including FA2's GQA LSE conversion."""
    from vllm_gr.v1.attention.backends import beam_attn_gpu as gpu
    from vllm_gr.v1.attention.backends.beam_graph_kernels import copy_suffix_kv_graph

    width, heads, dim = query.shape
    kv_heads, capacity = key.shape[2:4]
    extent = max(int(step.item()) + 1, 1)

    packed_key = torch.empty((width * capacity, kv_heads, dim), dtype=query.dtype, device="cuda")
    packed_value = torch.empty_like(packed_key)
    copy_suffix_kv_graph(key, value, packed_key, packed_value, slot, step, width, active)

    cu_q = torch.arange(width + 1, dtype=torch.int32, device="cuda")
    suffix, lse = gpu.flash_attn_varlen_func(
        q=query.contiguous(),
        k=packed_key,
        v=packed_value,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_q * extent,
        max_seqlen_q=1,
        max_seqlen_k=capacity,
        softmax_scale=dim**-0.5,
        causal=True,
        return_softmax_lse=True,
        fa_version=2,
        num_splits=0,
    )

    group = heads // kv_heads
    if group > 1:
        lse = lse.view(kv_heads, group, -1).transpose(1, 2).contiguous().view(heads, -1)

    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    gpu.merge_attn_states(output, prefix.contiguous(), prefix_lse.contiguous(), suffix, lse)

    return output


@pytest.mark.parametrize(
    "geometry",
    [
        (1, 4, 4, 64),
        (7, 8, 4, 128),
        (128, 16, 8, 128),
        (256, 16, 4, 256),
        (3, 8, 1, 64),
        (3, 3, 1, 64),
        (7, 16, 1, 128),
        (512, 16, 8, 128),
    ],
)
@pytest.mark.parametrize("capacity,decode_step", [(1, 0), (2, 1), (3, -1), (3, 0), (3, 1), (3, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_matches_packed_fa2_bitwise(
    geometry: tuple[int, int, int, int], capacity: int, decode_step: int, dtype: torch.dtype
) -> None:
    from vllm_gr.v1.attention.backends.beam_short_suffix import (
        merge_short_suffix_attention,
    )

    torch.manual_seed(7819)
    width, heads, kv_heads, dim = geometry
    query_storage = torch.randn((width, heads, dim * 2), dtype=dtype, device="cuda")
    query = query_storage[..., ::2]
    key = torch.randn((4, width + 3, kv_heads, capacity, dim), dtype=dtype, device="cuda")
    value = torch.randn_like(key)
    prefix = torch.randn_like(query)
    lse = torch.randn((width, heads), device="cuda").T
    active = torch.arange(width, device="cuda") % 3 != 1
    step = torch.tensor([decode_step], dtype=torch.int32, device="cuda")

    expected = packed_fa2_reference(query, prefix, lse, key, value, 3, step, active)
    output_storage = torch.full_like(query_storage, 123)
    output = output_storage[..., ::2]

    merge_short_suffix_attention(output, query, prefix, lse, key, value, 3, step, dim**-0.5, active)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.all(output_storage[..., 1::2] == 123)


@pytest.mark.parametrize("slot", [0, 3])
@pytest.mark.parametrize("masked", [False, True])
@torch.inference_mode()
def test_fa2_graph_replay_reads_step_mask_and_pool(slot: int, masked: bool) -> None:
    from vllm_gr.v1.attention.backends.beam_short_suffix import (
        merge_short_suffix_attention,
    )

    torch.manual_seed(7820)
    width, heads, kv_heads, dim = 256, 16, 8, 128
    query = torch.randn((width, heads, dim), dtype=torch.bfloat16, device="cuda")
    key = torch.randn((4, width, kv_heads, 3, dim), dtype=query.dtype, device="cuda")
    value = torch.randn_like(key)
    prefix = torch.randn_like(query)
    lse = torch.randn((heads, width), device="cuda")
    active = torch.arange(width, device="cuda") % 3 != 1 if masked else None
    step = torch.zeros(1, dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)

    def launch() -> None:
        merge_short_suffix_attention(
            output, query, prefix, lse, key, value, slot, step, dim**-0.5, active
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    # Negative bootstrap steps are defined only for the masked packed-copy
    # path; an unmasked copy requires a positive suffix extent.
    for decode_step in [0, 2, 1, -1, 0, 2] if masked else [0, 2, 1, 0, 2]:
        step.fill_(decode_step)
        if active is not None:
            active.logical_not_()
        key.copy_(key.roll(1, 1))
        value.copy_(value.roll(1, 1))
        expected = packed_fa2_reference(query, prefix, lse, key, value, slot, step, active)

        graph.replay()

        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_fa2_runtime_slot_reuses_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_gr.v1.attention.backends import beam_short_suffix_fa2 as short
    from vllm_gr.v1.attention.backends.beam_short_suffix import (
        merge_short_suffix_attention,
    )

    production_kernel = short._short_suffix_fa2_merge_kernel
    fresh_kernel = type(production_kernel)(
        production_kernel.fn,
        do_not_specialize=[
            parameter.name for parameter in production_kernel.params if parameter.do_not_specialize
        ],
    )
    monkeypatch.setattr(short, "_short_suffix_fa2_merge_kernel", fresh_kernel)
    cache = fresh_kernel.device_caches[torch.accelerator.current_device_index()][0]
    assert not cache

    torch.manual_seed(7821)
    query = torch.randn((7, 8, 128), dtype=torch.bfloat16, device="cuda")
    key = torch.randn((4, 10, 4, 3, 128), dtype=query.dtype, device="cuda")
    value = torch.randn_like(key)
    prefix = torch.randn_like(query)
    lse = torch.randn((8, 7), device="cuda")
    step = torch.ones(1, dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)

    for slot in range(4):
        expected = packed_fa2_reference(query, prefix, lse, key, value, slot, step, None)
        merge_short_suffix_attention(output, query, prefix, lse, key, value, slot, step, 128**-0.5)

        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert len(cache) == 1, f"slot={slot} created another compiled specialization"
