# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"

"""Real NPU multi-Trie selection against an independent legal-set reference."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.constraints.test_npu_multi_trie import TRIPLES, make_npu_tries
from vllm_gr.config import ConstraintTableConfig
from vllm_gr.config.models import MULTI_TRIE_CONSTRAINT_TABLE_FORMAT
from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.backend import INVALID_LOGIT, ConstraintMetadata
from vllm_gr.constraints.loader import load_constraint_backend
from vllm_gr.constraints.npu_backend import NPUConstraintBackend
from vllm_gr.ops.npu.rec_constrained_topk import is_available

pytestmark = [pytest.mark.npu_test, pytest.mark.l2]

try:
    import torch_npu  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"torch-npu is not installed: {exc}", allow_module_level=True)

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Ascend NPU is unavailable", allow_module_level=True)
if not is_available():
    pytest.skip("real NPU rec_constrained_topk is unavailable", allow_module_level=True)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_mixed_tries_match_independent_legal_set_reference(
    tmp_path: Path,
    compact: bool,
    step: int,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_GR_NPU_STAGE_LM_HEAD", "1")
    path = tmp_path / "multi.safetensors"
    digest = save_constraint_table_artifact(make_npu_tries(), path)
    backend = load_constraint_backend(
        ConstraintTableConfig(
            enabled=True,
            backend="npu",
            format=MULTI_TRIE_CONSTRAINT_TABLE_FORMAT,
            path=str(path),
            artifact_digest=digest,
            tokenizer_digest="sha256:" + "b" * 64,
            max_top_k=3,
        ),
        device=torch.device("npu"),
        vocab_size=512,
        dtype=dtype,
        max_rows=4,
        enable_npu_stage_vocab=compact,
    )
    assert isinstance(backend, NPUConstraintBackend)
    roots = (0, 1, 2, 1)
    prefixes = [(11, 7, 0), (11, 7, 0), (237, 399, 0), (53, 137, 0)]
    temperatures = torch.tensor([0.0, 2.0, 0.5, 1.0])
    host_logits = torch.randn(4, 512, generator=torch.Generator().manual_seed(725)).to(dtype)
    logits = host_logits.to("npu")
    try:
        if compact:
            vocab = backend.stage_vocab
            assert vocab is not None
            selected = logits[:, vocab.stage_token_ids[step]]
            logits = torch.nn.functional.pad(
                selected, (0, vocab.vocab_size - selected.shape[1]), value=-float("inf")
            ).contiguous()
        metadata = ConstraintMetadata(
            torch.tensor(prefixes, dtype=torch.int32, device="npu"),
            temperatures.to("npu"),
            step,
            3,
            torch.tensor(roots, dtype=torch.int32, device="npu"),
        )
        output = backend.select_for_rows(logits, metadata, roots)
        checked = backend.select(logits, metadata)
        torch.npu.synchronize()
        for name in ("token_ids", "logprobs", "valid_mask"):
            torch.testing.assert_close(
                getattr(output, name), getattr(checked, name), rtol=0, atol=0
            )
        ids, scores, masks = output.token_ids.cpu(), output.logprobs.cpu(), output.valid_mask.cpu()
        for row, root in enumerate(roots):
            legal = sorted(
                {triple[step] for triple in TRIPLES[root] if triple[:step] == prefixes[row][:step]}
            )
            reference = host_logits[row, legal].float() / (float(temperatures[row]) or 1.0)
            count = min(3, len(legal))
            order = torch.argsort(reference, descending=True, stable=True)[:count]
            assert masks[row].tolist() == [index < count for index in range(3)]
            assert ids[row, :count].tolist() == [legal[index] for index in order.tolist()]
            torch.testing.assert_close(
                scores[row, :count],
                reference[order] - torch.logsumexp(reference, dim=0),
                rtol=1e-4,
                atol=1e-3,
            )
            assert scores[row, count:].eq(INVALID_LOGIT).all()
    finally:
        backend.close()
