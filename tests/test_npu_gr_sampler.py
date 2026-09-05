# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the persistent NPU constrained sampler boundary.

The injected Backend below exercises the real ``ConstraintOutput`` and vLLM
``SamplerOutput`` ABIs on CPU tensors.  These tests do not exercise an NPU
kernel, stream ordering, graph replay, or device-residency performance.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.config.models import ConstraintTableBackendEnum
from vllm_gr.constraints.backend import (
    ConstraintMetadata,
    ConstraintOutput,
)
from vllm_gr.v1.worker import npu_gr_sampler
from vllm_gr.v1.worker.npu_gr_sampler import (
    NPUPersistentConstraintSampler,
    NPU_SAMPLER_STATUS_EMPTY,
    NPU_SAMPLER_STATUS_INVALID_NUMERIC,
    NPU_SAMPLER_STATUS_OK,
    NPU_SAMPLER_STATUS_VALIDITY_HOLE,
)


class _Backend:
    def __init__(self, *, width: int = 4, vocab_size: int = 32) -> None:
        self.info = SimpleNamespace(
            max_rows=width,
            max_top_k=width,
            backend=ConstraintTableBackendEnum.NPU,
            table_metadata=SimpleNamespace(vocab_size=vocab_size),
        )
        self.outputs: list[ConstraintOutput | BaseException | object] = []
        self.calls: list[tuple[torch.Tensor, ConstraintMetadata]] = []
        self.prefix_values: list[torch.Tensor] = []
        self.temperature_values: list[torch.Tensor] = []

    def select(
        self,
        logits: torch.Tensor,
        metadata: ConstraintMetadata,
    ) -> ConstraintOutput:
        assert type(metadata) is ConstraintMetadata
        self.calls.append((logits, metadata))
        self.prefix_values.append(metadata.prefix_token_ids.clone())
        self.temperature_values.append(metadata.temperatures.clone())
        output = self.outputs.pop(0) if self.outputs else _output(logits.shape[0], metadata.top_k)
        if isinstance(output, BaseException):
            raise output
        return output  # type: ignore[return-value]


def _output(
    rows: int,
    width: int,
    *,
    token_ids: torch.Tensor | None = None,
    logprobs: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> ConstraintOutput:
    if token_ids is None:
        token_ids = torch.arange(rows * width, dtype=torch.int32).reshape(rows, width)
        token_ids.add_(10)
    if logprobs is None:
        logprobs = -torch.arange(rows * width, dtype=torch.float32).reshape(rows, width)
        logprobs.div_(10)
    if valid_mask is None:
        valid_mask = torch.ones((rows, width), dtype=torch.bool)
    return ConstraintOutput(token_ids, logprobs, valid_mask)


def _sampler(
    backend: Any | None = None,
    *,
    width: int = 4,
    steps: int = 3,
    temperature: float = 1.0,
) -> tuple[NPUPersistentConstraintSampler, Any]:
    backend = backend if backend is not None else _Backend(width=width)
    return (
        NPUPersistentConstraintSampler(
            "gr",
            width,
            steps,
            temperature,
            torch.device("cpu"),
            backend,
            cpu_test_backend=True,
        ),
        backend,
    )


def _sequence(width: int = 4, steps: int = 3) -> torch.Tensor:
    return torch.zeros((1, width, steps), dtype=torch.int32)


def _logits(rows: int, vocab: int = 32) -> torch.Tensor:
    return torch.arange(rows * vocab, dtype=torch.float32).reshape(rows, vocab)


def test_three_steps_use_real_metadata_and_fixed_canonical_device_prefix() -> None:
    sampler, backend = _sampler(temperature=0.25)
    sequence = _sequence()
    prefix_ptr = sampler.prefix_token_ids.data_ptr()
    temperature_ptr = sampler.temperatures.data_ptr()

    pointers: list[tuple[int, ...]] = []
    for step, rows in ((0, 1), (1, 4), (2, 4)):
        if step == 1:
            sequence[0, :, 0] = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
        elif step == 2:
            sequence[0, :, 1] = torch.tensor([21, 22, 23, 24], dtype=torch.int32)
        result = sampler.select(_logits(rows), sequence, expected_device_step=step)
        wire = result.sampler_output.logprobs_tensors
        assert wire is not None
        pointers.append(
            (
                result.valid_counts.data_ptr(),
                result.status.data_ptr(),
                result.sampler_output.sampled_token_ids.data_ptr(),
                wire.logprobs.data_ptr(),
                wire.selected_token_ranks.data_ptr(),
            )
        )
        assert result.request_id == "gr"
        assert result.device_step == step
        assert backend.calls[-1][0].shape == (rows, 32)
        metadata = backend.calls[-1][1]
        assert type(metadata) is ConstraintMetadata
        assert metadata.current_step == step
        assert metadata.top_k == 4
        assert metadata.prefix_token_ids.data_ptr() == prefix_ptr
        assert metadata.temperatures.data_ptr() == temperature_ptr
        assert metadata.prefix_token_ids.dtype is torch.int32
        assert metadata.temperatures.dtype is torch.float32
        sampler.consume(result)

    assert backend.prefix_values[0].tolist() == [[0, 0, 0]]
    assert backend.prefix_values[1].tolist() == [
        [11, 0, 0],
        [12, 0, 0],
        [13, 0, 0],
        [14, 0, 0],
    ]
    assert backend.prefix_values[2].tolist() == [
        [11, 21, 0],
        [12, 22, 0],
        [13, 23, 0],
        [14, 24, 0],
    ]
    assert all(value.tolist() == [0.25] for value in backend.temperature_values)
    assert pointers[0] == pointers[1] == pointers[2]


@pytest.mark.parametrize("temperature", [0.0, 0.125, 2.0])
def test_zero_and_nonunit_temperature_use_fixed_request_tensor(
    temperature: float,
) -> None:
    sampler, backend = _sampler(temperature=temperature, steps=1)
    pointer = sampler.temperatures.data_ptr()
    result = sampler.select(_logits(1), _sequence(), expected_device_step=0)
    metadata = backend.calls[0][1]
    assert metadata.temperatures.data_ptr() == pointer
    assert metadata.temperatures.tolist() == [temperature]
    sampler.consume(result)


@pytest.mark.parametrize("temperature", [-0.1, float("nan"), float("inf"), True, "1"])
def test_invalid_temperature_fails_before_backend_construction(
    temperature: Any,
) -> None:
    backend = _Backend()
    with pytest.raises(RuntimeError, match="temperature"):
        NPUPersistentConstraintSampler(
            "gr",
            4,
            3,
            temperature,
            torch.device("cpu"),
            backend,
            cpu_test_backend=True,
        )
    assert backend.calls == []


def test_prefill_partial_candidates_count_structural_mask_not_scores() -> None:
    backend = _Backend()
    raw = _output(
        1,
        4,
        logprobs=torch.tensor([[-0.1, -0.2, 99.0, float("nan")]]),
        valid_mask=torch.tensor([[True, True, False, False]]),
    )
    backend.outputs.append(raw)
    sampler, _ = _sampler(backend, steps=1)
    result = sampler.select(_logits(1), _sequence(), expected_device_step=0)

    wire = result.sampler_output.logprobs_tensors
    assert wire is not None
    assert result.constraint_output.token_ids is raw.token_ids
    assert result.constraint_output.logprobs is raw.logprobs
    assert result.constraint_output.valid_mask is raw.valid_mask
    assert result.valid_counts.tolist() == [2]
    assert result.status.tolist() == [NPU_SAMPLER_STATUS_OK]
    assert result.constraint_output.logprobs.tolist()[0][:3] == pytest.approx([-0.1, -0.2, 99.0])
    assert torch.isnan(result.constraint_output.logprobs[0, 3])
    assert wire.logprobs[0, :2].tolist() == pytest.approx([-0.1, -0.2])
    assert torch.isneginf(wire.logprobs[0, 2:]).all()
    assert wire.logprob_token_ids is raw.token_ids
    assert wire.selected_token_ranks.dtype is torch.int64
    assert wire.selected_token_ranks.tolist() == [0]
    assert result.sampler_output.sampled_token_ids.tolist() == [[10]]
    assert result.sampler_output._gr_constraint_logprobs is raw.logprobs
    assert result.sampler_output._gr_constraint_valid_mask is raw.valid_mask
    assert result.sampler_output._gr_constraint_valid_counts is result.valid_counts
    assert result.sampler_output._gr_npu_sampler_status is result.status


def test_structurally_valid_negative_infinity_is_not_used_to_derive_count() -> None:
    backend = _Backend()
    backend.outputs.append(
        _output(
            1,
            4,
            logprobs=torch.tensor([[float("-inf"), -2.0, -3.0, -4.0]]),
            valid_mask=torch.tensor([[True, False, False, False]]),
        )
    )
    sampler, _ = _sampler(backend, steps=1)
    result = sampler.select(_logits(1), _sequence(), expected_device_step=0)
    assert result.valid_counts.tolist() == [1]
    assert result.status.tolist() == [NPU_SAMPLER_STATUS_OK]


def test_empty_holey_and_nonfinite_rows_defer_device_status_and_safe_token() -> None:
    masks = torch.tensor(
        [
            [False, False, False, False],
            [True, False, True, False],
            [True, True, False, False],
            [True, False, False, False],
        ]
    )
    scores = torch.tensor(
        [
            [9.0, 8.0, 7.0, 6.0],
            [-1.0, -2.0, -3.0, -4.0],
            [-1.0, float("nan"), -3.0, -4.0],
            [-1.0, 5.0, float("nan"), 6.0],
        ]
    )
    backend = _Backend()
    backend.outputs.extend([_output(1, 4), _output(4, 4, logprobs=scores, valid_mask=masks)])
    sampler, _ = _sampler(backend, steps=2)
    sequence = _sequence()
    prefill = sampler.select(_logits(1), sequence, expected_device_step=0)
    sampler.consume(prefill)
    result = sampler.select(_logits(4), sequence, expected_device_step=1)
    wire = result.sampler_output.logprobs_tensors
    assert wire is not None

    assert result.valid_counts.tolist() == [0, 2, 2, 1]
    assert result.status.tolist() == [
        NPU_SAMPLER_STATUS_EMPTY,
        NPU_SAMPLER_STATUS_VALIDITY_HOLE,
        NPU_SAMPLER_STATUS_INVALID_NUMERIC,
        NPU_SAMPLER_STATUS_OK,
    ]
    assert result.sampler_output.sampled_token_ids[:, 0].tolist() == [0, 14, 18, 22]
    assert torch.isneginf(wire.logprobs[0]).all()
    assert torch.isneginf(wire.logprobs[1, 1])
    assert wire.logprobs[1, 2].item() == -3.0
    assert torch.isnan(result.constraint_output.logprobs[2, 1])
    assert torch.isneginf(wire.logprobs[3, 1:]).all()


def test_select_path_does_not_call_host_tensor_materialization(monkeypatch) -> None:
    backend = _Backend()
    backend.outputs.append(_output(1, 4, valid_mask=torch.zeros((1, 4), dtype=torch.bool)))
    sampler, _ = _sampler(backend, steps=1)
    sequence = _sequence()
    logits = _logits(1)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("persistent sampler must not materialize tensor values on Host")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "tolist", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        result = sampler.select(logits, sequence, expected_device_step=0)
    assert result.device_step == 0
    assert result.valid_counts.tolist() == [0]
    assert result.status.tolist() == [NPU_SAMPLER_STATUS_EMPTY]
    assert result.sampler_output.sampled_token_ids.tolist() == [[0]]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda raw: SimpleNamespace(), "wrong output type"),
        (
            lambda raw: ConstraintOutput(raw.token_ids[:, :-1], raw.logprobs, raw.valid_mask),
            "returned shape",
        ),
        (
            lambda raw: ConstraintOutput(
                raw.token_ids.to(torch.int64), raw.logprobs, raw.valid_mask
            ),
            "token_ids must be int32",
        ),
        (
            lambda raw: ConstraintOutput(
                raw.token_ids, raw.logprobs.to(torch.float16), raw.valid_mask
            ),
            "logprobs must be float32",
        ),
        (
            lambda raw: ConstraintOutput(
                raw.token_ids, raw.logprobs, raw.valid_mask.to(torch.int32)
            ),
            "valid_mask must be bool",
        ),
        (
            lambda raw: ConstraintOutput(
                torch.arange(8, dtype=torch.int32).reshape(1, 8)[:, ::2],
                raw.logprobs,
                raw.valid_mask,
            ),
            "token_ids must be contiguous",
        ),
    ],
)
def test_invalid_backend_output_poison_fails_closed(
    mutate,
    match: str,
) -> None:
    backend = _Backend()
    backend.outputs.append(mutate(_output(1, 4)))
    sampler, _ = _sampler(backend, steps=1)
    with pytest.raises(RuntimeError, match=match):
        sampler.select(_logits(1), _sequence(), expected_device_step=0)
    assert sampler.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        sampler.select(_logits(1), _sequence(), expected_device_step=0)
    assert len(backend.calls) == 1


def test_backend_exception_identity_is_preserved_and_never_falls_back() -> None:
    backend = _Backend()
    error = LookupError("fused NPU operator failed")
    backend.outputs.append(error)
    sampler, _ = _sampler(backend, steps=1)
    with pytest.raises(LookupError) as caught:
        sampler.select(_logits(1), _sequence(), expected_device_step=0)
    assert caught.value is error
    assert sampler.poisoned
    with pytest.raises(RuntimeError, match="poisoned") as poisoned:
        sampler.select(_logits(1), _sequence(), expected_device_step=0)
    assert poisoned.value.__cause__ is error
    assert len(backend.calls) == 1


@pytest.mark.parametrize(
    "logits,sequence,step",
    [
        (torch.zeros((2, 32)), _sequence(), 0),
        (torch.zeros((1, 31)), _sequence(), 0),
        (torch.zeros((1, 32), dtype=torch.int32), _sequence(), 0),
        (torch.zeros((1, 32)).t(), _sequence(), 0),
        (torch.zeros((1, 32)), torch.zeros((1, 3, 3), dtype=torch.int32), 0),
        (torch.zeros((1, 32)), torch.zeros((1, 4, 2), dtype=torch.int32), 0),
        (torch.zeros((1, 32)), torch.zeros((1, 4, 3), dtype=torch.int64), 0),
        (torch.zeros((1, 32)), _sequence().transpose(1, 2), 0),
    ],
)
def test_structural_input_failures_do_not_call_or_poison_backend(
    logits: torch.Tensor,
    sequence: torch.Tensor,
    step: int,
) -> None:
    sampler, backend = _sampler()
    with pytest.raises(RuntimeError):
        sampler.select(logits, sequence, expected_device_step=step)
    assert not sampler.poisoned
    assert backend.calls == []
    canonical_was_bound = (
        tuple(sequence.shape) == (1, 4, 3)
        and sequence.dtype is torch.int32
        and sequence.is_contiguous()
    )
    retry_sequence = sequence if canonical_was_bound else _sequence()
    result = sampler.select(_logits(1), retry_sequence, expected_device_step=0)
    assert result.device_step == 0


def test_device_mismatch_and_production_cpu_backend_are_rejected() -> None:
    backend = _Backend()
    with pytest.raises(RuntimeError, match="NPU-resident"):
        NPUPersistentConstraintSampler("gr", 4, 3, 1.0, torch.device("cpu"), backend)
    sampler, _ = _sampler(backend)
    with pytest.raises(RuntimeError, match="logits ABI mismatch"):
        sampler.select(
            torch.empty((1, 32), device="meta"),
            _sequence(),
            expected_device_step=0,
        )


def test_pending_replay_out_of_order_and_foreign_consume_are_rejected() -> None:
    sampler, backend = _sampler(steps=2)
    sequence = _sequence()
    first = sampler.select(_logits(1), sequence, expected_device_step=0)
    with pytest.raises(RuntimeError, match="awaiting decision consume"):
        sampler.select(_logits(1), sequence, expected_device_step=0)
    with pytest.raises(RuntimeError, match="stale or foreign"):
        sampler.consume(object())  # type: ignore[arg-type]
    sampler.consume(first)
    with pytest.raises(RuntimeError, match="stale or out of range"):
        sampler.select(_logits(1), sequence, expected_device_step=0)
    with pytest.raises(RuntimeError, match="stale or out of range"):
        sampler.select(_logits(4), sequence, expected_device_step=2)
    second = sampler.select(_logits(4), sequence, expected_device_step=1)
    sampler.consume(second)
    with pytest.raises(RuntimeError, match="stale or foreign"):
        sampler.consume(second)
    assert len(backend.calls) == 2


def test_consume_none_without_pending_is_never_accepted() -> None:
    sampler, _ = _sampler(steps=1)
    with pytest.raises(RuntimeError, match="stale or foreign"):
        sampler.consume(None)  # type: ignore[arg-type]


def test_resolved_cpu_device_alias_owns_all_fixed_buffers() -> None:
    backend = _Backend()
    sampler = NPUPersistentConstraintSampler(
        "gr",
        4,
        1,
        0.0,
        torch.device("cpu:0"),
        backend,
        cpu_test_backend=True,
    )
    assert sampler.prefix_token_ids.device == torch.device("cpu")
    assert sampler.temperatures.device == torch.device("cpu")
    result = sampler.select(_logits(1), _sequence(), expected_device_step=0)
    assert result.valid_counts.device == torch.device("cpu")
    assert result.status.device == torch.device("cpu")


def test_fixed_workspace_layout_change_is_rejected_before_backend() -> None:
    sampler, backend = _sampler(steps=1)
    sampler.prefix_token_ids.resize_(2, 6)
    with pytest.raises(RuntimeError, match="fixed tensor binding changed"):
        sampler.select(_logits(1), _sequence(), expected_device_step=0)
    assert backend.calls == []


def test_positive_infinite_valid_logprob_sets_invalid_numeric_status() -> None:
    backend = _Backend()
    backend.outputs.append(
        _output(
            1,
            4,
            logprobs=torch.tensor([[float("inf"), -2.0, -3.0, -4.0]]),
            valid_mask=torch.tensor([[True, False, False, False]]),
        )
    )
    sampler, _ = _sampler(backend, steps=1)
    result = sampler.select(_logits(1), _sequence(), expected_device_step=0)
    assert result.valid_counts.tolist() == [1]
    assert result.status.tolist() == [NPU_SAMPLER_STATUS_INVALID_NUMERIC]
    assert torch.isposinf(result.constraint_output.logprobs[0, 0])


def test_canonical_object_and_storage_are_bound_across_steps() -> None:
    sampler, backend = _sampler(steps=2)
    sequence = _sequence()
    first = sampler.select(_logits(1), sequence, expected_device_step=0)
    sampler.consume(first)
    replacement = sequence.clone()
    with pytest.raises(RuntimeError, match="binding changed"):
        sampler.select(_logits(4), replacement, expected_device_step=1)
    assert len(backend.calls) == 1


def test_canonical_history_cannot_alias_sampler_prefix_scratch() -> None:
    sampler, backend = _sampler()
    sampler.prefix_token_ids.fill_(3)
    alias = sampler.prefix_token_ids.unsqueeze(0)
    with pytest.raises(RuntimeError, match="canonical sequence aliases workspace"):
        sampler.select(_logits(1), alias, expected_device_step=0)
    assert backend.calls == []
    assert not sampler.poisoned
    assert torch.equal(alias, torch.full_like(alias, 3))
    assert sampler.select(_logits(1), _sequence(), expected_device_step=0).device_step == 0


def test_backend_info_identity_change_is_rejected_before_second_dispatch() -> None:
    sampler, backend = _sampler(steps=2)
    sequence = _sequence()
    first = sampler.select(_logits(1), sequence, expected_device_step=0)
    sampler.consume(first)
    backend.info = SimpleNamespace(**vars(backend.info))
    with pytest.raises(RuntimeError, match="ownership changed"):
        sampler.select(_logits(4), sequence, expected_device_step=1)
    assert len(backend.calls) == 1


def test_release_is_request_scoped_idempotent_and_blocks_pending_work() -> None:
    sampler, _ = _sampler(steps=1)
    pending = sampler.select(_logits(1), _sequence(), expected_device_step=0)
    with pytest.raises(RuntimeError, match="stale or foreign"):
        sampler.release("other")
    assert sampler.release("gr") is True
    assert sampler.release("gr") is False
    with pytest.raises(RuntimeError, match="released"):
        sampler.select(_logits(1), _sequence(), expected_device_step=0)
    with pytest.raises(RuntimeError, match="stale or foreign"):
        sampler.consume(pending)


@pytest.mark.parametrize(
    "request_id,width,steps,cpu_test,match",
    [
        ("", 4, 3, True, "request_id"),
        ("gr", True, 3, True, "Beam width"),
        ("gr", 0, 3, True, "Beam width"),
        ("gr", 4, 0, True, "generation steps"),
        ("gr", 4, 4, True, "steps 0/1/2"),
        ("gr", 4, 3, 1, "cpu_test_backend"),
    ],
)
def test_constructor_request_width_step_and_mode_gates(
    request_id: Any,
    width: Any,
    steps: Any,
    cpu_test: Any,
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        NPUPersistentConstraintSampler(
            request_id,
            width,
            steps,
            1.0,
            torch.device("cpu"),
            _Backend(),
            cpu_test_backend=cpu_test,
        )


def test_backend_capacity_and_contract_are_fail_fast() -> None:
    too_small = _Backend(width=4)
    too_small.info.max_rows = 3
    with pytest.raises(RuntimeError, match="capacity"):
        _sampler(too_small)
    no_select = SimpleNamespace(info=_Backend().info)
    with pytest.raises(RuntimeError, match="Backend is invalid"):
        _sampler(no_select)


def test_source_contains_no_host_materialization_or_torch_fallback() -> None:
    source = inspect.getsource(npu_gr_sampler)
    tree = ast.parse(source)
    forbidden_methods = {"cpu", "tolist", "item", "numpy"}
    used = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert used.isdisjoint(forbidden_methods)
    assert "vllm_gr.ops.onerec_decode" not in source
    assert "rec_constrained_topk_torch" not in source
