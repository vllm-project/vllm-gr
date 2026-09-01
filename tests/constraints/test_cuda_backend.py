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

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


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


def test_prelaunched_prefill_orders_prepare_launch_copy_and_event_across_streams() -> None:
    width = 4
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=1,
        beam_width=width,
        top_k=width,
        max_decode_steps=3,
        device="cuda",
        logits_dtype=torch.float16,
    )
    prepare_stream = torch.cuda.Stream()
    launch_stream = torch.cuda.Stream()
    prefixes = workspace.prefix_token_ids.view(-1)[:1].view(1, 1, 1)
    temperatures = workspace.temperatures.view(-1)[:1]
    copy_words = workspace.max_batch + 3 * width
    recorded_streams: list[torch.cuda.Stream] = []

    class RecordingBackend:
        def select_out_unchecked(
            self,
            logits: torch.Tensor,
            metadata: ConstraintMetadata,
            **outputs: torch.Tensor,
        ) -> None:
            del logits
            recorded_streams.append(torch.cuda.current_stream())
            value = metadata.prefix_token_ids.reshape(-1)[0]
            outputs["out_token_ids"].fill_(value)
            outputs["out_logprobs"].zero_()
            outputs["out_wire_logprobs"].zero_()
            outputs["out_valid_mask"].fill_(True)
            outputs["out_valid_counts"].fill_(width)
            outputs["out_prefill_status"].zero_()
            outputs["out_prefill_payload"].fill_(value)

    with torch.cuda.stream(prepare_stream):
        torch.cuda._sleep(20_000_000)
        prefixes.fill_(77)

    capacity_rows = workspace.max_batch * width
    prepared = constraint_hook.PreparedCudaPrefillConstraint(
        scheduler_output=object(),
        rows=(),
        width=width,
        workspace=workspace,
        backend=RecordingBackend(),
        execution_adapter=None,
        prefixes=prefixes,
        temperatures=temperatures,
        metadata=ConstraintMetadata(
            prefix_token_ids=prefixes,
            temperatures=temperatures,
            current_step=0,
            top_k=width,
        ),
        sequence_group=prefixes,
        out_token_ids=workspace.topk_token_ids.view(capacity_rows, width)[:1],
        out_logprobs=workspace.topk_scores.view(capacity_rows, width)[:1],
        out_valid_mask=workspace.topk_valid_mask.view(capacity_rows, width)[:1],
        out_valid_counts=workspace.topk_valid_counts.view(-1)[:1],
        out_wire_logprobs=workspace.wire_topk_scores.view(capacity_rows, width)[:1],
        out_status=workspace.prefill_staging[:1],
        out_payload=workspace.prefill_staging[workspace.max_batch :],
        device_copy=workspace.prefill_staging[:copy_words],
        host_copy=workspace.host_prefill_staging[:copy_words],
        copy_event=workspace.prefill_copy_event,
        stream=prepare_stream,
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(update_async_output_token_ids=lambda: None),
        _gr_prepared_cuda_prefill_constraint=prepared,
        _gr_prelaunched_cuda_prefill=None,
    )
    logits = torch.zeros((1, 6), dtype=torch.float16, device="cuda")

    with torch.cuda.stream(launch_stream):
        assert constraint_hook.launch_prepared_cuda_prefill_constraint(runner, logits)
    assert workspace.prefill_copy_event is not None
    workspace.prefill_copy_event.synchronize()

    assert recorded_streams == [launch_stream]
    assert workspace.host_prefill_staging[workspace.max_batch].item() == 77
