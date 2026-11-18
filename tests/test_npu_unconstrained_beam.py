# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""CPU references for catalog-free NPU Beam admission and candidate scoring."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.constraints.backend import ConstraintOutput
from vllm_gr.ops.npu.unconstrained_topk import select_unconstrained_candidates
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchItem,
    BeamBatchParams,
    BeamBatchRequest,
    BeamOutputOptions,
)
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
from vllm_gr.v1.worker.npu_beam_stage_adapter import (
    NPUEagerStagePlan,
    validate_npu_eager_capabilities,
)
from vllm_gr.v1.worker.npu_beam_stage_runner import NPUEagerBeamStageRunner


def _scalar_reference(
    row: list[float], temperature: float, top_k: int
) -> tuple[list[int], list[float]]:
    """Enumerate the full finite vocabulary, independently of device TopK."""
    temperature = temperature if temperature > 0.0 else 1.0
    finite = [(index, value) for index, value in enumerate(row) if math.isfinite(value)]
    if not finite:
        return [], []
    finite.sort(key=lambda pair: (-pair[1], pair[0]))
    maximum = max(value for _, value in finite)
    centered = [(index, (value - maximum) / temperature) for index, value in finite]
    normalizer = math.log(math.fsum(math.exp(value) for _, value in centered))
    finite = [(index, value - normalizer) for index, value in centered]
    finite = [(index, value) for index, value in finite if value >= -torch.finfo(torch.float32).max]
    ordered = finite[:top_k]
    return (
        [index for index, _ in ordered],
        [value for _, value in ordered],
    )


def _assert_matches_reference(
    result: ConstraintOutput, logits: torch.Tensor, temperatures: torch.Tensor, top_k: int
) -> None:
    assert (
        result.token_ids.shape
        == result.logprobs.shape
        == result.valid_mask.shape
        == (
            logits.shape[0],
            top_k,
        )
    )
    assert result.token_ids.dtype == torch.int32
    assert result.logprobs.dtype == torch.float32
    assert result.valid_mask.dtype == torch.bool
    assert (
        result.token_ids.device
        == result.logprobs.device
        == result.valid_mask.device
        == logits.device
    )
    rows = logits.float().cpu().tolist()
    scales = temperatures.float().cpu().reshape(-1).tolist()
    for row, values in enumerate(rows):
        ids, scores = _scalar_reference(values, scales[0 if len(scales) == 1 else row], top_k)
        live = len(ids)
        assert result.token_ids[row, :live].cpu().tolist() == ids
        torch.testing.assert_close(
            result.logprobs[row, :live].cpu(), torch.tensor(scores), rtol=1e-5, atol=2e-5
        )
        assert result.valid_mask[row].cpu().tolist() == [True] * live + [False] * (top_k - live)
        assert torch.isneginf(result.logprobs[row, live:]).all()
        assert (result.token_ids[row, live:] == 0).all()
        assert not torch.isnan(result.logprobs).any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab_size,top_k", [(17, 1), (17, 17), (17, 32), (1031, 3)])
@pytest.mark.parametrize("broadcast_temperature", [False, True])
def test_full_vocab_scores_match_scalar_enumeration(
    dtype: torch.dtype, vocab_size: int, top_k: int, broadcast_temperature: bool
) -> None:
    logits = torch.randn((3, vocab_size), generator=torch.Generator().manual_seed(408)).to(dtype)
    # Exact ties, including a Top1 cutoff, must have reproducible token ordering.
    logits[0].zero_()
    temperatures = torch.tensor([0.75] if broadcast_temperature else [0.0, 0.5, 2.0])
    result = select_unconstrained_candidates(logits, temperatures, top_k)
    _assert_matches_reference(result, logits, temperatures, top_k)


def test_scores_are_normalized_before_candidate_truncation() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [-1.0, -2.0, -3.0, -4.0]])
    temperatures = torch.ones(2)
    narrow = select_unconstrained_candidates(logits, temperatures, 1)
    wide = select_unconstrained_candidates(logits, temperatures, 4)
    _assert_matches_reference(narrow, logits, temperatures, 1)
    torch.testing.assert_close(narrow.logprobs[:, 0], wide.logprobs[:, 0], rtol=0, atol=0)
    assert (narrow.logprobs < 0).all(), "Top1 must retain the full-vocabulary denominator"


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 2.0])
def test_ordinary_scores_remain_close_to_direct_fp32_normalization(temperature: float) -> None:
    logits = torch.randn((4, 257), generator=torch.Generator().manual_seed(953))
    result = select_unconstrained_candidates(logits, torch.tensor([temperature]), 17)
    direct = logits / (temperature or 1.0)
    expected_scores, expected_ids = torch.sort(direct, dim=-1, descending=True, stable=True)
    expected = expected_scores[:, :17] - torch.logsumexp(direct, dim=-1, keepdim=True)
    assert torch.equal(result.token_ids, expected_ids[:, :17].to(torch.int32))
    torch.testing.assert_close(result.logprobs, expected, rtol=1e-5, atol=2e-5)
    zero = select_unconstrained_candidates(logits, torch.zeros(1), 17)
    one = select_unconstrained_candidates(logits, torch.ones(1), 17)
    torch.testing.assert_close(zero.logprobs, one.logprobs, rtol=0, atol=0)


@pytest.mark.parametrize(
    "values,temperature",
    [
        ([2.0, 1.0], 1e-40),
        ([-2.0, -1.0], 1e-40),
        ([2.0, 2.0, 1.0], 1e-40),
        ([float("nan"), float("inf"), -float("inf"), 2.0, 1.0], 1e-40),
        ([float("nan"), float("inf"), -float("inf")], 1e-40),
        ([2e-40, 1e-40], 1e-40),
        ([3e38, 3e38, 2e38], 0.0),
        ([-3e38, -3e38, -3.1e38], 0.5),
        ([-3e38, 3e38], 3e38),
        ([-3e38, 3e38], 1.0),
        ([1.0, 2.0], 3e38),
    ],
)
def test_extreme_temperatures_and_logits_match_double_precision_reference(
    values: list[float], temperature: float
) -> None:
    logits, temperatures = torch.tensor([values]), torch.tensor([temperature])
    top_k = len(values) + 2
    _assert_matches_reference(
        select_unconstrained_candidates(logits, temperatures, top_k), logits, temperatures, top_k
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_temperature_dtype_matches_native_npu_arithmetic(dtype: torch.dtype) -> None:
    result = select_unconstrained_candidates(
        torch.tensor([[2.0, 1.0]]), torch.ones(1, dtype=dtype), 2
    )
    assert result.valid_mask.tolist() == [[True, True]]


@pytest.mark.parametrize("temperature", [1.0, 1e-50, 1e40])
def test_float64_temperature_is_rejected_before_silent_fp32_conversion(temperature: float) -> None:
    with pytest.raises(TypeError, match="temperatures.*float16, bfloat16, or float32"):
        select_unconstrained_candidates(
            torch.tensor([[2.0, 1.0]]), torch.tensor([temperature], dtype=torch.float64), 2
        )


@pytest.mark.parametrize("top_k", [1, 4, 8])
def test_nonfinite_logits_never_become_candidates_or_poison_finite_rows(top_k: int) -> None:
    logits = torch.tensor(
        [
            [float("nan"), float("inf"), -float("inf"), 1.0, 0.0],
            [-float("inf"), float("nan"), float("inf"), -float("inf"), float("nan")],
            [0.0, -float("inf"), -float("inf"), -float("inf"), -float("inf")],
        ]
    )
    temperatures = torch.tensor([0.0, 1.0, 0.5])
    _assert_matches_reference(
        select_unconstrained_candidates(logits, temperatures, top_k), logits, temperatures, top_k
    )


def test_candidate_selection_accepts_row_padding_without_host_tensor_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = torch.randn((4, 23), generator=torch.Generator().manual_seed(1708))
    logits, temperatures = storage[:, :17], torch.tensor([0.0, 0.5, 1.0, 2.0])
    original = storage.clone()

    def forbid_host_read(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("candidate selection must not read tensor values on the host")

    with monkeypatch.context() as guard:
        for name in ("cpu", "item", "tolist", "__bool__"):
            guard.setattr(torch.Tensor, name, forbid_host_read)
        result = select_unconstrained_candidates(logits, temperatures, 8)
    _assert_matches_reference(result, logits, temperatures, 8)
    torch.testing.assert_close(storage, original, rtol=0, atol=0)


def test_selection_preserves_caller_owned_fp32_nonfinite_logits() -> None:
    storage = torch.tensor([[2.0, float("nan"), float("inf"), -float("inf"), 1.0, 17.0]])
    logits = storage[:, :5]
    original_bits = storage.view(torch.int32).clone()
    result = select_unconstrained_candidates(logits, torch.tensor([1e-40]), 7)
    assert torch.equal(storage.view(torch.int32), original_bits)
    assert result.token_ids.tolist() == [[0] * 7]
    assert result.valid_mask.tolist() == [[True] + [False] * 6]
    assert result.logprobs[0, 0] == 0


def _prefill_runner(batch_size: int, constrained: bool) -> NPUEagerBeamStageRunner:
    execution = BeamBatchExecutionView(
        session_id="catalog-free-session",
        dispatch_id=0,
        stage_index=0,
        kind="prefill",
        decode_step=None,
        owner_generation=1,
        request_ids=tuple(f"request-{index}" for index in range(batch_size)),
        num_prompt_tokens_by_request=(2,) * batch_size,
        beam_width=2,
        produces_output=True,
    )
    plan = NPUEagerStagePlan(
        execution,
        (2,) * batch_size,
        batch_size * 2,
        tuple(2 * index + 1 for index in range(batch_size)),
        tuple(range(batch_size)),
        None,
    )
    metadata = SimpleNamespace(
        beam_params=SimpleNamespace(temperature=0.5), is_last_stage=False, output_device_state=None
    )
    worker = SimpleNamespace(
        _vllm_gr_config=SimpleNamespace(constrained_generation=constrained),
        _current_scheduler_output=SimpleNamespace(
            gr_stage_metadata={execution.session_id: metadata}
        ),
        execute_model_state=object(),
        kv_connector_output=object(),
    )
    stage = object.__new__(NPUEagerBeamStageRunner)
    stage.worker, stage.pending_plan, stage.execution_error = worker, plan, None
    stage.logits = torch.arange(batch_size * 7, dtype=torch.float32).view(batch_size, 7) / 10
    stage.attention_batch = None
    return stage


@pytest.mark.parametrize("batch_size", [1, 4])
def test_catalog_free_prefill_uses_full_vocab_without_constraint_state(
    batch_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = _prefill_runner(batch_size, constrained=False)
    assert stage.logits is not None
    assert stage.pending_plan is not None
    execution = stage.pending_plan.execution
    logits = stage.logits.clone()
    # A leftover backend must not override explicit unconstrained generation.
    stage.worker.constraint_backend = SimpleNamespace(
        select=Mock(side_effect=AssertionError("catalog-free selection queried the backend"))
    )
    monkeypatch.setattr(
        stage, "_constraint_prefixes", Mock(side_effect=AssertionError("unexpected prefix access"))
    )
    output = stage.sample()
    assert stage.prefill_candidates is not None
    _assert_matches_reference(stage.prefill_candidates, logits, torch.full((batch_size,), 0.5), 2)
    assert stage.latest_token_ids is stage.prefill_candidates.token_ids
    assert stage.previous is execution
    receipt = output.gr_worker_results["catalog-free-session"]
    assert not receipt.finished and not receipt.resources_released
    assert receipt.owner_generation == 1 and receipt.terminal_result is None
    stage.worker.constraint_backend.select.assert_not_called()
    assert stage.pending_plan is stage.logits is None


@pytest.mark.parametrize("batch_size", [1, 4])
def test_constrained_prefill_keeps_backend_scores_and_prefix_metadata(batch_size: int) -> None:
    stage = _prefill_runner(batch_size, constrained=True)
    ids = torch.tensor([[5, 1]] * batch_size, dtype=torch.int32)
    candidates = ConstraintOutput(
        ids, torch.full((batch_size, 2), -2.5), torch.ones_like(ids).bool()
    )
    backend = SimpleNamespace(
        info=SimpleNamespace(max_rows=batch_size), select=Mock(return_value=candidates)
    )
    stage.worker.constraint_backend = backend
    stage.sample()
    backend.select.assert_called_once()
    metadata = backend.select.call_args.args[1]
    assert metadata.prefix_token_ids.shape == (batch_size, 3)
    assert metadata.current_step == 0 and metadata.top_k == 2
    torch.testing.assert_close(metadata.temperatures, torch.full((batch_size,), 0.5))
    assert stage.prefill_candidates is candidates


@pytest.mark.parametrize(
    "eager,steps,expected", [(True, 3, False), (False, 1, False), (False, 3, True)]
)
def test_unconstrained_startup_selects_the_existing_npu_graph_adapter(
    eager: bool, steps: int, expected: bool
) -> None:
    from vllm_gr.v1.worker.gr_startup import _capture_enabled

    worker = SimpleNamespace(
        device="npu",
        model_config=SimpleNamespace(enforce_eager=eager),
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=True)),
        _vllm_gr_config=SimpleNamespace(
            constrained_generation=False,
            beam_worker_decision=True,
            beam_max_width=2,
            beam_max_decode_steps=steps,
            beam_graph_enabled=True,
            constraint_table=None,
        ),
    )
    assert _capture_enabled(worker) is expected


@pytest.mark.parametrize("capacity", [3, 4, 5, 6, 7])
def test_npu_generation_capacity_is_checked_before_worker_state_is_allocated(
    capacity: int,
) -> None:
    from vllm_gr.config import GRConfig

    config = SimpleNamespace(
        additional_config={},
        device_config=SimpleNamespace(device="npu"),
        model_config=SimpleNamespace(
            get_num_kv_heads=lambda parallel: 8, get_head_size=lambda: 128
        ),
    )
    GRConfig(
        constrained_generation=False, beam_max_width=2, beam_max_decode_steps=capacity
    ).write_to_vllm_config(config)
    worker = SimpleNamespace(
        device="npu",
        use_async_scheduling=True,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        parallel_config=None,
        vllm_config=config,
    )
    if capacity <= 6:
        validate_npu_eager_capabilities(worker)
    else:
        with pytest.raises(ValueError, match=r"NPU unconstrained.*1\.\.6 generated tokens"):
            validate_npu_eager_capabilities(worker)


def _generation_request(
    steps: int, *, dynamic_beam_widths: tuple[int, ...] | None = None
) -> BeamBatchRequest:
    native = EngineCoreRequest(
        request_id="native-generation-budget",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=steps, temperature=0.0, ignore_eos=True),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    return BeamBatchRequest(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id=native.request_id,
        items=(BeamBatchItem(0, native),),
        beam_params=BeamBatchParams(
            2, steps, 0.0, (), True, dynamic_beam_widths=dynamic_beam_widths
        ),
        output_options=BeamOutputOptions(2, True, True, False),
    )


@pytest.mark.l1
@pytest.mark.parametrize(
    "device,capacity,steps,error",
    [
        ("npu", 6, 1, None),
        ("npu", 6, 2, None),
        ("npu", 6, 3, None),
        ("npu", 4, 4, None),
        ("npu", 6, 4, None),
        ("npu", 6, 5, None),
        ("npu", 6, 6, None),
        ("npu", 3, 4, "runtime limit 3"),
        ("npu", 5, 6, "runtime limit 5"),
        ("npu", 6, 7, r"NPU unconstrained.*1\.\.6 generated tokens"),
        ("npu", 7, 6, r"NPU unconstrained.*1\.\.6 generated tokens"),
        ("cuda", 4, 4, None),
        ("cuda", 5, 5, None),
        ("cuda", 6, 6, None),
        ("cuda", 8, 8, None),
    ],
)
def test_generation_horizon_checks_actual_request_and_capacity(
    device: str, capacity: int, steps: int, error: str | None
) -> None:
    from vllm_gr.config import GRConfig
    from vllm_gr.v1.engine.beam_search_v1 import (
        validate_gr_device_config,
        validate_gr_request,
    )

    config = SimpleNamespace(
        additional_config={},
        device_config=SimpleNamespace(device=device),
        model_config=SimpleNamespace(
            get_num_kv_heads=lambda parallel: 8, get_head_size=lambda: 128
        ),
    )
    GRConfig(
        constrained_generation=False, beam_max_width=2, beam_max_decode_steps=capacity
    ).write_to_vllm_config(config)
    request = _generation_request(steps)

    def validate() -> None:
        validate_gr_device_config(config, request)
        validate_gr_request(request, config=config)

    if error is None:
        validate()
    else:
        with pytest.raises(ValueError, match=error):
            validate()


@pytest.mark.l1
@pytest.mark.parametrize("entry", ["startup", "request", "worker"])
@pytest.mark.parametrize(
    "capacity,heads,head_size,accepted",
    [
        (4, 8, 128, True),
        (5, 8, 128, True),
        (6, 8, 128, True),
        (6, 63, 128, True),
        (6, 64, 128, False),
        (5, 64, 128, True),
        (5, 76, 128, True),
        (5, 77, 128, False),
        (4, 95, 128, True),
        (4, 96, 128, False),
        (6, 31, 256, True),
        (6, 32, 256, False),
        (4, 48, 256, False),
        (6, 1024, 128, False),
        (6, 1, 65536, False),
        (3, 64, 128, True),
    ],
)
def test_extended_generation_fits_native_kv_reorder_buffer(
    entry: str, capacity: int, heads: int, head_size: int, accepted: bool
) -> None:
    from vllm_gr.config import GRConfig
    from vllm_gr.v1.engine.beam_search_v1 import validate_gr_device_config

    parallel = SimpleNamespace()
    get_heads = Mock(return_value=heads)
    model = SimpleNamespace(
        dtype=torch.bfloat16, get_num_kv_heads=get_heads, get_head_size=lambda: head_size
    )
    config = SimpleNamespace(
        additional_config={},
        device_config=SimpleNamespace(device="npu"),
        model_config=model,
        parallel_config=parallel,
    )
    GRConfig(
        constrained_generation=False, beam_max_width=2, beam_max_decode_steps=capacity
    ).write_to_vllm_config(config)

    def validate() -> None:
        if entry == "worker":
            validate_npu_eager_capabilities(
                SimpleNamespace(
                    device="npu",
                    use_async_scheduling=True,
                    model_config=model,
                    parallel_config=parallel,
                    vllm_config=config,
                )
            )
        else:
            # A shorter request still uses the configured pool capacity.
            request = _generation_request(3) if entry == "request" else None
            validate_gr_device_config(config, request)

    if accepted:
        validate()
    else:
        with pytest.raises(ValueError, match="SelectUnsharedKV 95 KiB buffer"):
            validate()
    if capacity > 3:
        get_heads.assert_called_once_with(parallel)
    else:
        get_heads.assert_not_called()


@pytest.mark.l1
@pytest.mark.parametrize(
    "heads,head_size", [(0, 128), (-1, 128), (True, 128), (8, 0), (8, None), (8, 128.0)]
)
def test_extended_generation_rejects_invalid_model_geometry(heads: Any, head_size: Any) -> None:
    from vllm_gr.config import GRConfig
    from vllm_gr.v1.engine.beam_search_v1 import validate_gr_device_config

    config = SimpleNamespace(
        additional_config={},
        device_config=SimpleNamespace(device="npu"),
        model_config=SimpleNamespace(
            get_num_kv_heads=lambda parallel: heads, get_head_size=lambda: head_size
        ),
    )
    GRConfig(constrained_generation=False, beam_max_decode_steps=6).write_to_vllm_config(config)
    with pytest.raises(ValueError, match="positive integer KV-head geometry"):
        validate_gr_device_config(config)


@pytest.mark.l1
@pytest.mark.parametrize(
    "device,capacity,rejected", [("npu", 6, True), ("npu", 3, False), ("cuda", 6, False)]
)
@pytest.mark.parametrize("missing_model", [False, True])
def test_extended_generation_fails_closed_without_model_geometry(
    device: str, capacity: int, rejected: bool, missing_model: bool
) -> None:
    from vllm_gr.config import GRConfig
    from vllm_gr.v1.engine.beam_search_v1 import validate_gr_device_config

    config = SimpleNamespace(additional_config={}, device_config=SimpleNamespace(device=device))
    if not missing_model:
        config.model_config = SimpleNamespace()
    GRConfig(constrained_generation=False, beam_max_decode_steps=capacity).write_to_vllm_config(
        config
    )
    if rejected:
        with pytest.raises(ValueError, match="above three tokens requires model KV-head geometry"):
            validate_gr_device_config(config)
    else:
        validate_gr_device_config(config)


@pytest.mark.l1
@pytest.mark.parametrize("steps", [4, 5, 6])
def test_bounded_generic_support_preserves_sid_stage_capacity(steps: int) -> None:
    from vllm_gr.config import GRConfig
    from vllm_gr.v1.engine.beam_search_v1 import (
        validate_gr_device_config,
        validate_gr_request,
    )

    with pytest.raises(ValueError, match=r"beam\.max_decode_steps <= 3"):
        GRConfig(
            constrained_generation=False,
            beam_execution_mode="v1",
            beam_worker_decision=True,
            beam_max_width=2,
            beam_max_decode_steps=steps,
            beam_sid_stage_vocab=True,
        )
    config = SimpleNamespace(additional_config={}, device_config=SimpleNamespace(device="npu"))
    GRConfig(
        constrained_generation=False,
        beam_execution_mode="v1",
        beam_worker_decision=True,
        beam_max_width=2,
        beam_max_decode_steps=3,
        beam_sid_stage_vocab=True,
    ).write_to_vllm_config(config)
    request = _generation_request(steps)
    validate_gr_device_config(config, request)
    with pytest.raises(ValueError, match="runtime limit 3"):
        validate_gr_request(request, config=config)


@pytest.mark.l1
@pytest.mark.parametrize("steps", [4, 5, 6])
def test_bounded_generic_support_preserves_catalog_depth(steps: int) -> None:
    from vllm_gr.config import (
        ConstraintTableBackendEnum,
        ConstraintTableConfig,
        GRConfig,
        write_canonical_gr_config,
    )
    from vllm_gr.v1.engine.beam_search_v1 import validate_gr_request

    config = SimpleNamespace(additional_config={}, device_config=SimpleNamespace(device="npu"))
    gr_config = GRConfig(
        constrained_generation=True,
        beam_max_width=2,
        beam_max_decode_steps=6,
        constraint_table=ConstraintTableConfig(
            enabled=True,
            backend=ConstraintTableBackendEnum.NPU,
            path="/tmp/test-constraint-table.safetensors",
            format="constraint_table_v1",
            artifact_digest="sha256:" + "a" * 64,
            tokenizer_digest="sha256:" + "b" * 64,
            max_top_k=2,
        ),
    )
    write_canonical_gr_config(config, gr_config)
    table = GRConfig.from_vllm_config(config).constraint_table
    assert table is not None and table.enabled and table.format == "constraint_table_v1"
    validate_gr_request(_generation_request(3), config=config)
    with pytest.raises(
        ValueError,
        match=rf"^effective generation length {steps} must equal constraint_table_v1 depth \(3\)$",
    ):
        validate_gr_request(_generation_request(steps), config=config)


@pytest.mark.l1
@pytest.mark.parametrize("entry", ["public", "wire"])
@pytest.mark.parametrize("steps", [4, 5, 6])
def test_bounded_generic_support_preserves_dynamic_schedule_contract(
    entry: str, steps: int
) -> None:
    from vllm.exceptions import VLLMValidationError

    from vllm_gr.sampling_params import BeamSearchParams
    from vllm_gr.v1.engine.beam_search_v1 import validate_gr_request

    error_type = VLLMValidationError if entry == "public" else ValueError
    with pytest.raises(error_type, match="dynamic_beam_widths"):
        if entry == "public":
            BeamSearchParams(beam_width=2, max_tokens=steps, dynamic_beam_widths=[2] * steps)
        else:
            validate_gr_request(_generation_request(steps, dynamic_beam_widths=(2,) * steps))


@pytest.mark.parametrize("device", ["cuda", "npu"])
@pytest.mark.parametrize("candidate_width", [None, 2, 4])
@pytest.mark.parametrize("widths", [None, (1, 2, 1)])
def test_native_request_candidate_width_is_checked_at_device_admission(
    device: str, candidate_width: int | None, widths: tuple[int, ...] | None
) -> None:
    from vllm_gr.config import GRConfig
    from vllm_gr.v1.engine.beam_search_v1 import (
        validate_gr_device_config,
        validate_gr_request,
    )

    config = SimpleNamespace(additional_config={}, device_config=SimpleNamespace(device=device))
    GRConfig(
        constrained_generation=False, beam_max_width=2, beam_max_decode_steps=3
    ).write_to_vllm_config(config)
    native = EngineCoreRequest(
        request_id="native-candidate-width",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    request = BeamBatchRequest(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id=native.request_id,
        items=(BeamBatchItem(0, native),),
        beam_params=BeamBatchParams(
            beam_width=2,
            max_tokens=3,
            temperature=0.0,
            stop_token_ids=(),
            ignore_stop_tokens=True,
            candidate_width=candidate_width,
            dynamic_beam_widths=widths,
        ),
        output_options=BeamOutputOptions(1, True, True, False),
    )
    # The wire payload is structurally valid even when it carries an NPU-only
    # unsupported K. Native submission must reject it before a Worker starts.
    validate_gr_request(request, config=config)
    unsupported = candidate_width is not None and (candidate_width != 2 or widths is not None)
    if device == "npu" and unsupported:
        with pytest.raises(ValueError, match="candidate_width.*beam width"):
            validate_gr_device_config(config, request)
    else:
        validate_gr_device_config(config, request)


def test_npu_device_validation_accepts_partial_non_beam_config() -> None:
    from vllm_gr.v1.engine.beam_search_v1 import validate_gr_device_config

    validate_gr_device_config(SimpleNamespace(device_config=SimpleNamespace(device="npu")))


pytestmark = pytest.mark.cpu_test
