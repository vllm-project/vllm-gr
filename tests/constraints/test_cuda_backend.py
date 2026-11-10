# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.constraints.test_table import make_table
from vllm_gr.config import ConstraintTableBackendEnum, ConstraintTableConfig
from vllm_gr.constraints import loader
from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.backend import ConstraintMetadata
from vllm_gr.constraints.loader import load_constraint_backend
from vllm_gr.v1.beam.beam_decision import (
    BeamPostprocessWorkspace,
    BeamPostprocessWorkspacePool,
)
from vllm_gr.v1.worker import constraint_hook

pytestmark = pytest.mark.gpu


def test_workspace_execution_adapter_preserves_sequence_group_geometry() -> None:
    """The allocation-free adapter path must keep the legacy operator ABI."""

    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=2,
        beam_width=4,
        top_k=4,
        max_decode_steps=3,
        device="cuda",
        logits_dtype=torch.bfloat16,
    )
    pool = BeamPostprocessWorkspacePool()
    pool.add(workspace)
    runner = type(
        "Runner",
        (),
        {
            "device": torch.device("cuda"),
            "model_config": type("ModelConfig", (), {"dtype": torch.bfloat16})(),
            "_beam_postprocess_workspaces": pool,
            "input_batch": type(
                "InputBatch",
                (),
                {
                    "sampling_metadata": type(
                        "SamplingMetadata",
                        (),
                        {"temperature": torch.tensor([0.6], device="cuda")},
                    )()
                },
            )(),
        },
    )()

    class RecordingExecution:
        def __init__(self) -> None:
            self.sequence_groups: list[torch.Tensor] = []

        def run_out(
            self,
            logits: torch.Tensor,
            sequence_group: torch.Tensor,
            temperatures: torch.Tensor,
            current_step: int,
            top_k: int,
            *,
            out_token_ids: torch.Tensor,
            out_logprobs: torch.Tensor,
            out_valid_mask: torch.Tensor,
            out_valid_counts: torch.Tensor,
            out_wire_logprobs: torch.Tensor,
        ) -> None:
            del logits, temperatures, current_step
            assert sequence_group.dtype == torch.int32
            assert sequence_group.is_contiguous()
            self.sequence_groups.append(sequence_group.cpu())
            out_token_ids.zero_()
            out_logprobs.zero_()
            out_valid_mask.fill_(True)
            out_valid_counts.fill_(top_k)
            out_wire_logprobs.zero_()

    execution = RecordingExecution()
    prefill_rows = [
        ("a", 0, 1, {"constraint_step": 0, "beam_width": 4}),
        ("b", 1, 1, {"constraint_step": 0, "beam_width": 4}),
    ]
    prefill = constraint_hook._try_sample_into_postprocess_workspace(
        runner,
        torch.zeros((2, 6), dtype=torch.bfloat16, device="cuda"),
        prefill_rows,
        4,
        None,
        execution,
    )
    assert prefill is not None
    assert tuple(execution.sequence_groups[-1].shape) == (2, 1, 1)

    decode_rows = [
        (
            "a",
            0,
            4,
            {
                "constraint_step": 1,
                "beam_width": 4,
                "beam_suffix": [[10], [11], [12], [13]],
            },
        ),
        (
            "b",
            4,
            4,
            {
                "constraint_step": 1,
                "beam_width": 4,
                "beam_suffix": [[20], [21], [22], [23]],
            },
        ),
    ]
    decode = constraint_hook._try_sample_into_postprocess_workspace(
        runner,
        torch.zeros((8, 6), dtype=torch.bfloat16, device="cuda"),
        decode_rows,
        4,
        None,
        execution,
    )
    assert decode is not None
    sequence_group = execution.sequence_groups[-1]
    assert tuple(sequence_group.shape) == (2, 4, 3)
    assert sequence_group[:, :, 0].tolist() == [
        [10, 11, 12, 13],
        [20, 21, 22, 23],
    ]


def test_cuda_loader_readiness_select_and_close(tmp_path: Path) -> None:
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)
    backend = load_constraint_backend(
        ConstraintTableConfig(
            enabled=True,
            backend=ConstraintTableBackendEnum.CUDA,
            path=str(path),
            artifact_digest=digest,
            tokenizer_digest="sha256:" + "b" * 64,
            max_top_k=4,
        ),
        device=torch.device("cuda"),
        vocab_size=6,
        dtype=torch.bfloat16,
        max_rows=2,
    )
    assert backend is not None
    assert backend.info.backend is ConstraintTableBackendEnum.CUDA
    assert backend.info.host_memory_bytes == 0
    assert backend.info.device_memory_bytes == make_table().total_bytes
    output = backend.select(
        torch.tensor([[-8, -5, 1, 7, 3, 9]], dtype=torch.bfloat16, device="cuda"),
        ConstraintMetadata(
            prefix_token_ids=torch.tensor([[1, 0, 0]], dtype=torch.int32, device="cuda"),
            temperatures=torch.ones(1, dtype=torch.float32, device="cuda"),
            current_step=1,
            top_k=4,
        ),
    )
    assert output.token_ids.dtype == torch.int32
    assert output.logprobs.dtype == torch.float32
    assert output.valid_mask.cpu().tolist() == [[True, True, False, False]]
    assert set(output.token_ids[0][output.valid_mask[0]].cpu().tolist()) == {2, 4}

    fixed_outputs = (
        torch.empty((1, 4), dtype=torch.int32, device="cuda"),
        torch.empty((1, 4), dtype=torch.float32, device="cuda"),
        torch.empty((1, 4), dtype=torch.bool, device="cuda"),
        torch.empty((1,), dtype=torch.int32, device="cuda"),
    )
    fixed_addresses = tuple(tensor.data_ptr() for tensor in fixed_outputs)
    fixed = backend.select_out(
        torch.tensor([[-8, -5, 1, 7, 3, 9]], dtype=torch.bfloat16, device="cuda"),
        ConstraintMetadata(
            prefix_token_ids=torch.tensor([[1, 0, 0]], dtype=torch.int32, device="cuda"),
            temperatures=torch.ones(1, dtype=torch.float32, device="cuda"),
            current_step=1,
            top_k=4,
        ),
        out_token_ids=fixed_outputs[0],
        out_logprobs=fixed_outputs[1],
        out_valid_mask=fixed_outputs[2],
        out_valid_counts=fixed_outputs[3],
    )
    assert tuple(tensor.data_ptr() for tensor in fixed_outputs) == fixed_addresses
    assert torch.equal(fixed.token_ids, output.token_ids)
    torch.testing.assert_close(fixed.logprobs, output.logprobs, rtol=0, atol=0)
    assert torch.equal(fixed.valid_mask, output.valid_mask)
    assert fixed_outputs[3].cpu().tolist() == [2]

    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=4,
        top_k=4,
        max_decode_steps=3,
        device="cuda",
        logits_dtype=torch.bfloat16,
    )
    pool = BeamPostprocessWorkspacePool()
    pool.add(workspace)
    runner = type(
        "Runner",
        (),
        {
            "device": torch.device("cuda"),
            "model_config": type("ModelConfig", (), {"dtype": torch.bfloat16})(),
            "_beam_postprocess_workspaces": pool,
            "input_batch": type(
                "InputBatch",
                (),
                {
                    "sampling_metadata": type(
                        "SamplingMetadata",
                        (),
                        {"temperature": torch.tensor([0.6], device="cuda")},
                    )()
                },
            )(),
        },
    )()
    workspace_addresses = tuple(tensor.data_ptr() for tensor in workspace.device_tensors())
    sampler_output = constraint_hook._try_sample_into_postprocess_workspace(
        runner,
        torch.tensor([[-8, -5, 1, 7, 3, 9]], dtype=torch.bfloat16, device="cuda"),
        [("r", 0, 1, {"constraint_step": 0, "beam_width": 4})],
        4,
        backend,
        None,
    )
    assert sampler_output is not None
    assert (
        sampler_output.logprobs_tensors.logprob_token_ids.data_ptr()
        == workspace.topk_token_ids.data_ptr()
    )
    assert sampler_output._gr_constraint_logprobs.data_ptr() == workspace.topk_scores.data_ptr()
    assert (
        sampler_output._gr_constraint_valid_mask.data_ptr() == workspace.topk_valid_mask.data_ptr()
    )
    assert tuple(tensor.data_ptr() for tensor in workspace.device_tensors()) == workspace_addresses

    status = workspace.prefill_staging[:1]
    payload = workspace.prefill_staging[workspace.max_batch :]
    backend.select_out_unchecked(
        torch.tensor([[-8, -5, 1, 7, 3, 9]], dtype=torch.bfloat16, device="cuda"),
        ConstraintMetadata(
            prefix_token_ids=workspace.prefix_token_ids.view(-1)[:1].view(1, 1),
            temperatures=torch.ones(1, dtype=torch.float32, device="cuda"),
            current_step=0,
            top_k=4,
        ),
        out_token_ids=workspace.topk_token_ids.view(-1, 4)[:1],
        out_logprobs=workspace.topk_scores.view(-1, 4)[:1],
        out_valid_mask=workspace.topk_valid_mask.view(-1, 4)[:1],
        out_valid_counts=workspace.topk_valid_counts.view(-1)[:1],
        out_wire_logprobs=workspace.wire_topk_scores.view(-1, 4)[:1],
        out_prefill_status=status,
        out_prefill_payload=payload,
    )
    torch.accelerator.synchronize()
    assert status.cpu().tolist() == [0]
    count = int(workspace.topk_valid_counts.view(-1)[0].item())
    assert torch.equal(payload[:count], workspace.topk_token_ids.view(-1, 4)[0, :count])
    assert payload[count:4].cpu().tolist() == [0] * (4 - count)
    assert payload[4:8].cpu().tolist() == [0, 0, 0, 0]
    assert torch.equal(payload[8:12].view(torch.float32), workspace.wire_topk_scores.view(-1, 4)[0])
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError, match="not ready"):
        backend.select(
            torch.zeros((1, 6), device="cuda"),
            ConstraintMetadata(
                prefix_token_ids=torch.zeros((1, 3), dtype=torch.int32, device="cuda"),
                temperatures=torch.ones(1, device="cuda"),
                current_step=0,
                top_k=1,
            ),
        )


def test_cuda_readiness_failure_closes_unpublished_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)
    uploaded: list[object] = []
    original_upload = loader._upload_cuda_backend  # noqa: SLF001

    def record_upload(*args: object, **kwargs: object) -> object:
        result = original_upload(*args, **kwargs)
        uploaded.append(result[0])
        return result

    monkeypatch.setattr(loader, "_upload_cuda_backend", record_upload)
    monkeypatch.setattr(
        loader,
        "_run_cuda_readiness",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("readiness failed")),
    )
    with pytest.raises(RuntimeError, match="readiness failed"):
        load_constraint_backend(
            ConstraintTableConfig(
                enabled=True,
                backend=ConstraintTableBackendEnum.CUDA,
                path=str(path),
                artifact_digest=digest,
                tokenizer_digest="sha256:" + "b" * 64,
                max_top_k=4,
            ),
            device=torch.device("cuda"),
            vocab_size=6,
            dtype=torch.float16,
            max_rows=2,
        )
    assert len(uploaded) == 1
    with pytest.raises(RuntimeError, match="closed"):
        uploaded[0]._tables()  # type: ignore[attr-defined]  # noqa: SLF001


def test_cuda_loader_rejects_fp32_logits_before_upload(tmp_path: Path) -> None:
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)
    with pytest.raises(ValueError, match="fp16/bf16"):
        load_constraint_backend(
            ConstraintTableConfig(
                enabled=True,
                backend=ConstraintTableBackendEnum.CUDA,
                path=str(path),
                artifact_digest=digest,
                tokenizer_digest="sha256:" + "b" * 64,
                max_top_k=4,
            ),
            device=torch.device("cuda"),
            vocab_size=6,
            dtype=torch.float32,
            max_rows=2,
        )


def test_cuda_loader_rejects_unsupported_table_degree_before_upload() -> None:
    table = make_table()
    oversized = replace(
        table,
        metadata=replace(table.metadata, max_prefix1_degree=65537),
    )

    with pytest.raises(ValueError, match="above supported 65536"):
        loader._upload_cuda_backend(  # noqa: SLF001
            oversized,
            device=torch.device("cuda"),
            max_top_k=4,
            max_rows=1,
        )


@pytest.mark.parametrize("offset", [0, 129])  # type: ignore[misc]
@pytest.mark.parametrize("multi_trie", [False, True])  # type: ignore[misc]
def test_canonical_worker_enables_vocab_slice_before_first_forward(
    tmp_path: Path, offset: int, multi_trie: bool
) -> None:
    from unittest.mock import MagicMock

    from vllm_gr.config import GRConfig
    from vllm_gr.config.parser import gr_config_to_mapping
    from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
    from vllm_gr.constraints.json_compiler import compile_relations_json
    from vllm_gr.ops.sliced_logits import (
        get_vocab_slice_config,
        is_vocab_slice_enabled,
    )
    from vllm_gr.v1.worker.model_runner_patch import (
        make_load_model_with_constraint_backend,
        make_shutdown_with_constraint_backend,
    )

    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {str(i): i for i in range(256)}
    tokenizer.convert_tokens_to_ids.side_effect = lambda values: [int(v) for v in values]
    triples = [[str(offset + i) for i in row] for row in [(1, 3, 0), (1, 3, 8), (2, 4, 7)]]
    compiler = ConstraintTableArtifactCompiler(
        tokenizer, 256, tokenizer_digest="sha256:" + "b" * 64
    )
    document = compile_relations_json(triples)
    if multi_trie:
        second = compile_relations_json(
            [[str(int(token) + 16) for token in row] for row in triples]
        )
        table = compiler.compile_tries([document, second])
    else:
        table = compiler.compile(document)
    slice_size = 25 if multi_trie else 9
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(table, path)
    config = GRConfig(
        beam_max_width=2,
        constraint_table=ConstraintTableConfig(
            enabled=True,
            backend=ConstraintTableBackendEnum.CUDA,
            path=str(path),
            artifact_digest=digest,
            tokenizer_digest=table.metadata.tokenizer_digest,
            format=table.metadata.format,
            max_top_k=2,
        ),
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config={"vllm_gr": gr_config_to_mapping(config)}),
        model_config=SimpleNamespace(dtype=torch.float16, get_vocab_size=lambda: 256),
        device=torch.device("cuda"),
    )

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    from vllm_gr.ops.sliced_logits_patch import apply_sliced_logits_patch

    apply_sliced_logits_patch()

    def load_weights(self: SimpleNamespace) -> None:
        assert not is_vocab_slice_enabled()
        self.model_memory_usage = 0
        # The native model loader supplies the current config during construction.
        with set_current_vllm_config(VllmConfig()):
            self.model = torch.nn.Sequential(LogitsProcessor(256))
        self.model[0]._gather_logits = lambda logits: logits

    load = make_load_model_with_constraint_backend(load_weights)
    shutdown = make_shutdown_with_constraint_backend()
    try:
        load(runner)
        assert runner.model[0]._gr_vocab_slice == (offset, slice_size)
        assert get_vocab_slice_config()[:2] == (None, None)
        # Exercise the existing projection with the worker-derived configuration.
        torch.manual_seed(0)
        head = torch.nn.Linear(64, 256, bias=False, device="cuda", dtype=torch.float16)
        hidden = torch.randn(2, 64, device="cuda", dtype=torch.float16)
        full_logits = head(hidden)
        head.quant_method = type(
            "UnquantizedEmbeddingMethod",
            (),
            {
                "apply": lambda self, layer, hidden, bias=None: torch.nn.functional.linear(
                    hidden, layer.weight, bias
                )
            },
        )()
        sliced_logits = runner.model[0]._get_logits(hidden, head, None)
        assert sliced_logits.shape == (2, slice_size)
        torch.testing.assert_close(
            sliced_logits, full_logits[:, offset : offset + slice_size], atol=2e-3, rtol=2e-3
        )
        # All three steps must preserve global token IDs and the validity mask.
        # The scoring contract normalizes over the logits window that was
        # actually computed: full_logits uses the full window, sliced_logits
        # uses the shared slice window. The two normalize over different
        # domains, so their absolute logprobs legitimately differ; each must
        # match the shared full-window implementation over its own window.
        from vllm_gr.constraints.scoring import score_selected_tokens

        for step in range(3):
            metadata = ConstraintMetadata(
                prefix_token_ids=torch.tensor(
                    [
                        [offset + 1, offset + 3, 0],
                        [offset + (17 if multi_trie else 1), offset + (19 if multi_trie else 3), 0],
                    ],
                    dtype=torch.int32,
                    device="cuda",
                ),
                temperatures=torch.ones(1, device="cuda"),
                current_step=step,
                top_k=2,
                trie_ids=(
                    torch.tensor([0, 1], dtype=torch.int32, device="cuda") if multi_trie else None
                ),
            )
            expected = runner.constraint_backend.select(full_logits, metadata)
            actual = runner.constraint_backend.select(sliced_logits, metadata)
            assert torch.equal(actual.token_ids, expected.token_ids)
            assert torch.equal(actual.valid_mask, expected.valid_mask)
            full_ref = score_selected_tokens(
                full_logits,
                expected.token_ids,
                expected.valid_mask,
                torch.ones(1, device="cuda"),
            )
            slice_ref = score_selected_tokens(
                sliced_logits,
                actual.token_ids,
                actual.valid_mask,
                torch.ones(1, device="cuda"),
                vocab_offset=offset,
            )
            torch.testing.assert_close(actual.logprobs, slice_ref, atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(expected.logprobs, full_ref, atol=2e-3, rtol=2e-3)
        # A reload must clear the old slice before constructing the next model.
        load(runner)
        assert runner.model[0]._gr_vocab_slice == (offset, slice_size)
        assert get_vocab_slice_config()[:2] == (None, None)
    finally:
        shutdown(runner)
    assert not is_vocab_slice_enabled()


def test_cuda_backend_scores_over_full_window_not_candidate_set(
    tmp_path: Path,
) -> None:
    """The CUDA backend must score over the full logits window.

    A row whose legal candidate set is a single token would normalize to
    logprob zero under the legacy per-candidate-set domain, erasing the
    model's preference across parent rows.  The scoring contract requires
    the full-window domain; verify ``backend.select`` returns the
    full-window logprob, not the candidate-set zero.
    """
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)
    backend = load_constraint_backend(
        ConstraintTableConfig(
            enabled=True,
            backend=ConstraintTableBackendEnum.CUDA,
            path=str(path),
            artifact_digest=digest,
            tokenizer_digest="sha256:" + "b" * 64,
            max_top_k=4,
        ),
        device=torch.device("cuda"),
        vocab_size=6,
        dtype=torch.bfloat16,
        max_rows=2,
    )
    assert backend is not None

    # vocab=6, candidate set for step0 is first_token_ids=[1,3].  Row 0 gets
    # logit 5 on token 3, logit 1 on token 1, -inf elsewhere.  Under the
    # legacy candidate-set domain the selected logprobs would normalize over
    # {1,3} only; the scoring contract requires the full-window domain.
    logits = torch.tensor(
        [[-1e20, 1.0, -1e20, 5.0, -1e20, -1e20]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = backend.select(
        logits,
        ConstraintMetadata(
            prefix_token_ids=torch.tensor([[0, 0, 0]], dtype=torch.int32, device="cuda"),
            temperatures=torch.ones(1, dtype=torch.float32, device="cuda"),
            current_step=0,
            top_k=4,
        ),
    )
    assert output.token_ids.dtype == torch.int32
    assert output.logprobs.dtype == torch.float32
    valid = output.valid_mask[0]
    tokens = output.token_ids[0][valid]
    logprobs = output.logprobs[0][valid]
    assert tokens.cpu().tolist() == [3, 1]
    # Full-window domain: each logprob must reflect the model score minus the
    # whole-vocab normalizer (logsumexp over all 6 positions).  The legacy
    # candidate-set domain would instead normalize over {1,3}, so token 3
    # would sit at ~0.0 regardless of the -inf background.
    full_lse = torch.logsumexp(logits[0].float(), dim=0)
    expected_3 = 5.0 - full_lse.item()
    assert logprobs[0].item() == pytest.approx(expected_3, rel=1e-3)
    # It must match the shared full-window reference implementation.
    from vllm_gr.constraints.scoring import score_selected_tokens

    expected = score_selected_tokens(
        logits,
        output.token_ids,
        output.valid_mask,
        torch.ones(1, dtype=torch.float32, device="cuda"),
    )
    torch.testing.assert_close(output.logprobs, expected, rtol=2e-3, atol=2e-3)
