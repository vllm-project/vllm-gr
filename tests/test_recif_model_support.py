# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import shutil
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import load_file
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.outputs import CompletionOutput, RequestOutput

import vllm_gr.entrypoints.recif.completion_handler as recif_completion_handler
from examples.offline_inference.beam_search.recif_utils import (
    build_prefix,
    generated_tokens_to_sid,
    levels_to_sid,
    sid_to_levels,
)
from tools.convert_recif_checkpoint import (
    InsufficientDiskSpaceError,
    _load_checkpoint,
    architecture_from_config,
    convert_checkpoint,
    merge_recif_heads,
)
from vllm_gr.entrypoints.beam_search_config import (
    build_beam_search_steps as resolve_beam_search_steps,
)
from vllm_gr.entrypoints.recif.completion_handler import (
    _build_recif_completion_response,
    _create_recif_completion,
    _is_recif_serving,
)
from vllm_gr.models.recif import (
    FULL_VOCAB,
    NUM_SID_LEVELS,
    RECIF_MAX_BEAM_WIDTH,
    VOCAB,
    RecifForCausalLM,
    build_beam_search_steps,
    register_recif_model,
)


@pytest.mark.parametrize(  # type: ignore[misc]
    "sid",
    [0, levels_to_sid(8191, 8191, 8191), 598080194427, 755993681678],
)
def test_sid_round_trip(sid: int) -> None:
    assert levels_to_sid(*sid_to_levels(sid)) == sid


def test_build_prefix_uses_level_offsets() -> None:
    sid = levels_to_sid(1, 2, 3)
    assert build_prefix([sid]) == [0, 1, VOCAB + 2, 2 * VOCAB + 3, 0]


def test_generated_tokens_to_sid_removes_level_offsets() -> None:
    # Raw level-1/level-2 token IDs carry VOCAB / 2 * VOCAB offsets that must
    # be removed before packing: [1915, 8558, 18612] -> levels (1915, 366, 2228).
    assert generated_tokens_to_sid([1915, 8558, 18612]) == 598080194427
    sid = levels_to_sid(1, 2, 3)
    assert generated_tokens_to_sid([1, VOCAB + 2, 2 * VOCAB + 3]) == sid


def test_recif_hierarchical_search_policy_is_model_owned() -> None:
    steps = build_beam_search_steps(NUM_SID_LEVELS, 64)
    assert len(steps) == NUM_SID_LEVELS
    assert [step.expand_width for step in steps] == [8, 8, 8]
    assert [step.keep_width for step in steps] == [8, 64, 64]
    assert [step.allowed_token_ids[0] for step in steps] == [0, VOCAB, 2 * VOCAB]
    assert [step.allowed_token_ids[-1] for step in steps] == [
        VOCAB - 1,
        2 * VOCAB - 1,
        FULL_VOCAB - 1,
    ]


@pytest.mark.parametrize("beam_width", [0, -1, 513])  # type: ignore[misc]
def test_recif_rejects_beam_width_outside_hierarchy(beam_width: int) -> None:
    with pytest.raises(ValueError, match="beam_width must be between 1 and 512"):
        build_beam_search_steps(NUM_SID_LEVELS, beam_width)


def test_recif_accepts_maximum_beam_width() -> None:
    steps = build_beam_search_steps(NUM_SID_LEVELS, RECIF_MAX_BEAM_WIDTH)
    assert [step.keep_width for step in steps] == [8, 64, 512]


def test_recif_requires_processed_logprobs_at_model_initialization() -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            logprobs_mode="raw_logprobs",
            hf_text_config=SimpleNamespace(
                vocab_size=FULL_VOCAB,
                tie_word_embeddings=False,
            ),
        )
    )
    with pytest.raises(ValueError, match="logprobs_mode='processed_logprobs'"):
        RecifForCausalLM(vllm_config=vllm_config)


def test_recif_startup_branch_override_hook_is_not_request_state() -> None:
    steps = build_beam_search_steps(NUM_SID_LEVELS, 64, branch_factors=(2, 3, 4))
    assert [step.expand_width for step in steps] == [2, 3, 4]
    assert [step.keep_width for step in steps] == [2, 6, 24]


def test_one_rec_fixed_width_policy_is_unchanged() -> None:
    steps = resolve_beam_search_steps(["Qwen3ForCausalLM"], 3, 16)
    assert [(step.expand_width, step.keep_width) for step in steps] == [
        (16, 16),
        (16, 16),
        (16, 16),
    ]
    register_recif_model()
    recif_steps = resolve_beam_search_steps(["RecifForCausalLM"], 3, 64)
    assert [step.expand_width for step in recif_steps] == [8, 8, 8]


def test_recif_completion_dispatch_is_model_scoped() -> None:
    recif_serving = SimpleNamespace(
        model_config=SimpleNamespace(architectures=["RecifForCausalLM"])
    )
    one_rec_serving = SimpleNamespace(
        model_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"])
    )
    assert _is_recif_serving(recif_serving)
    assert not _is_recif_serving(one_rec_serving)


class _FakeRecifServing:
    """Minimal stand-in for OpenAIServingCompletion in RECIF handler tests."""

    def __init__(self) -> None:
        self.model_config = SimpleNamespace(architectures=["RecifForCausalLM"])
        self.default_sampling_params: dict[str, Any] = {}
        self.system_fingerprint = "fp-test"
        self.models = SimpleNamespace(model_name=lambda _lora_request: "recif")
        self.engine_client = SimpleNamespace()

    async def render_completion_request(
        self,
        request: CompletionRequest,
    ) -> list[dict[str, Any]]:
        return [{"type": "token", "prompt_token_ids": [0, 1, 8194, 16387, 0]}]

    def _base_request_id(self, raw_request: Any, request_id: str) -> str:
        return "base"

    def _maybe_get_adapters(self, request: CompletionRequest) -> None:
        return None

    async def _get_trace_headers(self, headers: Any) -> None:
        return None

    def _get_data_parallel_rank(self, raw_request: Any) -> None:
        return None

    def _log_inputs(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int | HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        code = status_code.value if isinstance(status_code, HTTPStatus) else int(status_code)
        return ErrorResponse(
            error=ErrorInfo(message=message, type=err_type, code=code, param=param)
        )


def _recif_completion_request() -> CompletionRequest:
    return CompletionRequest(
        model="recif",
        prompt=[0, 1, 8194, 16387, 0],
        use_beam_search=True,
        n=64,
        max_tokens=3,
        return_token_ids=True,
    )


def _recif_request_output() -> RequestOutput:
    """Long-prompt RECIF output with 64 generated three-token choices."""

    return RequestOutput(
        request_id="cmpl-test",
        prompt="",
        prompt_token_ids=[0, 1, 8194, 16387, 0] * 2048,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=idx,
                text="",
                token_ids=[idx, VOCAB + idx, 2 * VOCAB + idx],
                cumulative_logprob=float(-idx),
                logprobs=None,
                finish_reason="length",
            )
            for idx in range(64)
        ],
        finished=True,
    )


def test_recif_response_omits_prompt_token_ids_per_choice() -> None:
    response = _build_recif_completion_response(
        _FakeRecifServing(),
        _recif_request_output(),
        request_id="cmpl-test",
        created_time=123,
        model_name="recif",
        request_metadata=RequestResponseMetadata(request_id="cmpl-test"),
    )
    assert len(response.choices) == 64
    assert all(choice.prompt_token_ids is None for choice in response.choices)
    assert [choice.token_ids for choice in response.choices[:2]] == [
        [0, VOCAB, 2 * VOCAB],
        [1, VOCAB + 1, 2 * VOCAB + 1],
    ]
    assert response.usage.prompt_tokens == 10240
    assert response.usage.completion_tokens == 192
    assert response.usage.total_tokens == 10240 + 192


def test_recif_completion_returns_generated_ids_without_prompt_duplication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_beam_search(*_args: Any, **_kwargs: Any) -> RequestOutput:
        return _recif_request_output()

    monkeypatch.setattr(
        recif_completion_handler,
        "recif_beam_search",
        fake_beam_search,
    )
    response = asyncio.run(
        _create_recif_completion(_FakeRecifServing(), _recif_completion_request(), None)
    )
    assert isinstance(response, CompletionResponse)
    assert len(response.choices) == 64
    assert all(choice.prompt_token_ids is None for choice in response.choices)
    assert [choice.token_ids for choice in response.choices[:2]] == [
        [0, VOCAB, 2 * VOCAB],
        [1, VOCAB + 1, 2 * VOCAB + 1],
    ]


def test_recif_runtime_failure_becomes_structured_5xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("Beam-search engine step failed")

    monkeypatch.setattr(
        recif_completion_handler,
        "recif_beam_search",
        fail,
    )
    response = asyncio.run(
        _create_recif_completion(_FakeRecifServing(), _recif_completion_request(), None)
    )
    assert isinstance(response, ErrorResponse)
    assert response.error.type == "InternalServerError"
    assert response.error.code == HTTPStatus.INTERNAL_SERVER_ERROR.value


def test_recif_user_validation_failure_stays_4xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("n must be at most 512")

    monkeypatch.setattr(
        recif_completion_handler,
        "recif_beam_search",
        fail,
    )
    response = asyncio.run(
        _create_recif_completion(_FakeRecifServing(), _recif_completion_request(), None)
    )
    assert isinstance(response, ErrorResponse)
    assert response.error.type == "BadRequestError"
    assert response.error.code == HTTPStatus.BAD_REQUEST.value


def test_architecture_from_config_reports_missing_fields() -> None:
    with pytest.raises(ValueError, match="num_hidden_layers"):
        architecture_from_config({})


def _head(value: float, hidden_size: int = 2) -> torch.Tensor:
    return torch.full((VOCAB, hidden_size), value)


def test_converter_merges_three_recif_heads() -> None:
    merged = merge_recif_heads(
        {
            "lm_heads": {
                f"heads.{level}.weight": _head(float(level)) for level in range(NUM_SID_LEVELS)
            }
        },
        hidden_size=2,
    )
    assert tuple(merged.shape) == (FULL_VOCAB, 2)
    assert [merged[level * VOCAB, 0].item() for level in range(3)] == [0, 1, 2]


def test_converter_rejects_invalid_recif_heads() -> None:
    with pytest.raises(RuntimeError, match="heads.2.weight"):
        merge_recif_heads(
            {"lm_heads": {"heads.0.weight": _head(0), "heads.1.weight": _head(1)}},
            hidden_size=2,
        )
    with pytest.raises(ValueError, match="Unexpected heads.0.weight shape"):
        merge_recif_heads(
            {
                "lm_heads": {
                    "heads.0.weight": torch.zeros(1, 2),
                    "heads.1.weight": _head(1),
                    "heads.2.weight": _head(2),
                }
            },
            hidden_size=2,
        )


def test_checkpoint_loader_does_not_fallback_to_unsafe_pickle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[bool] = []

    def fake_load(*_args: object, weights_only: bool, **_kwargs: object) -> None:
        calls.append(weights_only)
        raise RuntimeError("safe load failed")

    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.raises(RuntimeError, match="safe load failed"):
        _load_checkpoint(tmp_path / "checkpoint.pt")
    assert calls == [True]


def test_checkpoint_loader_unsafe_fallback_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[bool] = []

    def fake_load(*_args: object, weights_only: bool, **_kwargs: object) -> dict[str, torch.Tensor]:
        calls.append(weights_only)
        if weights_only:
            raise RuntimeError("legacy pickle")
        return {"weight": torch.zeros(1)}

    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.warns(RuntimeWarning, match="execute arbitrary code"):
        loaded = _load_checkpoint(
            tmp_path / "checkpoint.pt",
            allow_unsafe_pickle=True,
        )
    assert list(loaded) == ["weight"]
    assert calls == [True, False]


def _write_minimal_checkpoint(input_dir: Path) -> None:
    """Write the smallest valid RECIF checkpoint (zero transformer layers)."""

    input_dir.mkdir()
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "vocab_size": 151936,
        "tie_word_embeddings": True,
        "num_hidden_layers": 0,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "hidden_size": 2,
        "num_experts": 0,
    }
    (input_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.save(
        {
            "embedding.word_embeddings.weight": torch.zeros(FULL_VOCAB + 1, 2),
            "decoder.final_layernorm.weight": torch.zeros(2),
        },
        input_dir / "_model_rank0.pt",
    )
    torch.save(
        {
            "lm_heads": {
                f"heads.{level}.weight": _head(float(level)) for level in range(NUM_SID_LEVELS)
            }
        },
        input_dir / "external_rank0.pt",
    )


def test_converter_writes_recif_runtime_config(tmp_path: Path) -> None:
    input_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "checkpoint_hf"
    _write_minimal_checkpoint(input_dir)
    # Stale artifacts from a previous failed run must be cleaned first.
    output_dir.mkdir()
    (output_dir / "model.safetensors").write_bytes(b"stale partial output")

    convert_checkpoint(input_dir, output_dir)

    converted_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert converted_config["architectures"] == ["RecifForCausalLM"]
    assert converted_config["vocab_size"] == FULL_VOCAB
    assert converted_config["tie_word_embeddings"] is False
    weights = load_file(output_dir / "model.safetensors")
    assert tuple(weights["model.embed_tokens.weight"].shape) == (FULL_VOCAB, 2)
    assert tuple(weights["lm_head.weight"].shape) == (FULL_VOCAB, 2)


def test_converter_rejects_insufficient_disk_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "checkpoint_hf"
    _write_minimal_checkpoint(input_dir)
    stale = output_dir / "model.safetensors"
    output_dir.mkdir()
    stale.write_bytes(b"stale partial output")

    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )
    with pytest.raises(InsufficientDiskSpaceError, match="Insufficient disk space"):
        convert_checkpoint(input_dir, output_dir)

    # Stale artifacts are cleaned before the space check, and a failed
    # conversion must not leave any safetensors behind.
    assert not stale.exists()
    assert list(output_dir.iterdir()) == []
