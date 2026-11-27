# mypy: disable-error-code="misc"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import pytest
import torch

from vllm_gr.v1.worker.cuda.gpu_iid_sampling import draw_children


def draw(
    weights: list[list[float]],
    ids: list[list[int]],
    counts: list[int],
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    parents, tokens, child_counts, logprobs = draw_children(
        torch.tensor(ids),
        torch.tensor(weights).log(),
        torch.tensor(counts),
        sum(counts),
        temperature,
        top_k,
        top_p,
        torch.Generator().manual_seed(19),
    )
    return parents, tokens, child_counts, logprobs


@pytest.mark.cpu_test
@pytest.mark.l1
def test_parent_multiplicities_generate_independent_conditional_draws() -> None:
    parents, tokens, counts, logprobs = draw(
        [[1.0, 3.0], [3.0, 1.0]], [[1, 2], [1, 3]], [80000, 20000]
    )
    observed = {
        (p, t): (n, lp)
        for p, t, n, lp in zip(
            parents.tolist(), tokens.tolist(), counts.tolist(), logprobs.exp().tolist(), strict=True
        )
    }
    assert set(observed) == {(0, 1), (0, 2), (1, 1), (1, 3)}
    for parent, token, n, p in [
        (0, 1, 80000, 0.25),
        (0, 2, 80000, 0.75),
        (1, 1, 20000, 0.75),
        (1, 3, 20000, 0.25),
    ]:
        count, prob = observed[parent, token]
        assert count / n == pytest.approx(p, abs=0.012)
        assert prob == pytest.approx(p)
    assert counts.sum().item() == 100000


@pytest.mark.cpu_test
@pytest.mark.l1
def test_top_k_then_nucleus_renormalizes_and_keeps_boundary_ties() -> None:
    _, tokens, counts, _ = draw(
        [[1.0, 2.0, 2.0, 5.0, 0.0]], [[0, 1, 2, 3, 9]], [20000], top_k=2, top_p=1.0
    )
    assert set(tokens.tolist()) == {1, 2, 3}
    _, tokens, counts, _ = draw(
        [[1.0, 2.0, 2.0, 5.0, 0.0]], [[0, 1, 2, 3, 9]], [20000], temperature=0.6, top_k=2, top_p=0.6
    )
    assert tokens.tolist() == [3] and counts.tolist() == [20000]


@pytest.mark.cpu_test
@pytest.mark.l1
def test_seed_and_single_draw() -> None:
    args = ([[1.0, 3.0]], [[4, 7]], [1])
    lhs = draw(*args)
    rhs = draw(*args)
    assert all(torch.equal(a, b) for a, b in zip(lhs, rhs, strict=True))
    assert lhs[2].tolist() == [1]


@pytest.mark.gpu_test
@pytest.mark.l2
@pytest.mark.parametrize("width", [1, 7, 128, 1249])
def test_indexed_cuda_cdf_matches_reference(width: int) -> None:
    from vllm.platforms import current_platform

    if current_platform.device_type != "cuda":
        pytest.skip("requires CUDA")
    from vllm_gr.ops.cuda.iid_sampling import search_parent_cdfs

    generator = torch.Generator().manual_seed(246)
    probs = torch.rand((5, width), generator=generator)
    if width > 1:
        probs[:, 0] = 0
    cdf = probs.cumsum(1)
    cdf /= cdf[:, -1:].clone()
    parents = torch.arange(317) % 5
    uniforms = torch.rand(317, generator=generator)
    uniforms[:3] = torch.tensor([0, cdf[1, width // 2] if width > 1 else 0.5, 1 - 2**-24])
    expected = torch.searchsorted(cdf[parents], uniforms[:, None], right=True).flatten()
    actual = search_parent_cdfs(cdf.to("cuda"), parents.to("cuda"), uniforms.to("cuda"))
    assert actual.cpu().tolist() == expected.tolist()


@pytest.mark.gpu_test
@pytest.mark.l2
def test_iid_catalog_gathers_each_members_trie() -> None:
    from vllm.platforms import current_platform

    if current_platform.device_type != "cuda":
        pytest.skip("requires CUDA")
    from tests.constraints.test_multi_trie import SIDS, make_tries
    from vllm_gr.constraints.loader import _upload_cuda_backend

    backend, _ = _upload_cuda_backend(
        make_tries(), device=torch.device("cuda"), max_top_k=4, max_rows=2
    )
    backend._mark_ready()
    # The two tries share (1, 2) and give it different legal children.
    roots = torch.tensor([1, 0], device="cuda")
    prefixes = torch.tensor([[1, 2, 0], [1, 2, 0]], device="cuda")
    scores = torch.arange(6, device="cuda").float().expand(2, -1)
    for step in range(3):
        ids, allowed = backend.iid_candidates(scores, prefixes, step, roots)
        for i, trie in enumerate([1, 0]):
            want = {sid[step] for sid in SIDS[trie] if sid[:step] == (1, 2)[:step]}
            finite = allowed[i].isfinite()
            assert set(ids[i, finite].tolist()) == want
            assert torch.equal(allowed[i, finite], ids[i, finite].float())
