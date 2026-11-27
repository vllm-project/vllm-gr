# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# ruff: noqa: E402 -- Optional gRPC dependencies must be checked before imports.
"""L1 CPU tests for gRPC prompt truncation with generation budget reservation."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("grpc", minversion="1.84.0", reason="install vllm-gr[grpc]")
pytest.importorskip("google.protobuf", minversion="7.35.1", reason="install vllm-gr[grpc]")
pytest.importorskip("smg_grpc_servicer", reason="install vllm-gr[grpc]")

from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2 as pb
from vllm_gr.entrypoints.grpc.serving import BeamVllmEngineServicer

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def _make_engine(
    *,
    max_model_len: int = 8,
    truncation_side: str | None = "left",
    vocab_size: int = 128,
) -> SimpleNamespace:
    tokenizer = (
        None
        if truncation_side is None
        else SimpleNamespace(max_token_id=vocab_size - 1, truncation_side=truncation_side)
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=max_model_len, get_vocab_size=lambda: vocab_size
        ),
        input_processor=SimpleNamespace(tokenizer=tokenizer),
        errored=False,
    )


def _make_request(
    tokens: list[int],
    *,
    max_tokens: int = 3,
    truncate_prompt_tokens: int | None = None,
    binary: bool = True,
) -> Any:
    sampling_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "n": 2,
    }
    if truncate_prompt_tokens is not None:
        sampling_kwargs["truncate_prompt_tokens"] = truncate_prompt_tokens
    request = pb.GenerateRequest(
        request_id="req-0",
        tokenized=pb.TokenizedInput(original_text=""),
        sampling_params=pb.SamplingParams(**sampling_kwargs),
        beam_search=pb.BeamSearchOptions(beam_width=2),
    )
    if binary:
        request.tokenized.input_ids_i64_le = np.array(tokens, dtype="<i8").tobytes()
    else:
        request.tokenized.input_ids.extend(tokens)
    return request


@pytest.mark.parametrize("binary", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("side", ["left", "right", None])  # type: ignore[misc]
def test_auto_truncation_reserves_max_tokens(binary: bool, side: str | None) -> None:
    servicer = BeamVllmEngineServicer(_make_engine(max_model_len=8, truncation_side=side), 0.0)
    tokens = [10, 11, 12, 13, 14, 15, 16]
    request = _make_request(tokens, max_tokens=3, truncate_prompt_tokens=-1, binary=binary)

    batch, _ = servicer.build_batch([request])
    expected = tokens[-5:] if side == "left" else tokens[:5]
    np.testing.assert_array_equal(batch.items[0].request.prompt_token_ids, expected)


@pytest.mark.parametrize("tokens", [[10, 11, 12, 13, 14], [10, 11, 12]])  # type: ignore[misc]
def test_auto_truncation_keeps_prompts_within_budget(tokens: list[int]) -> None:
    servicer = BeamVllmEngineServicer(_make_engine(max_model_len=8, truncation_side="left"), 0.0)
    request = _make_request(tokens, max_tokens=3, truncate_prompt_tokens=-1)

    batch, _ = servicer.build_batch([request])
    np.testing.assert_array_equal(batch.items[0].request.prompt_token_ids, tokens)


@pytest.mark.parametrize("max_tokens", [8, 9])  # type: ignore[misc]
def test_auto_truncation_rejects_when_max_tokens_leaves_no_budget(max_tokens: int) -> None:
    servicer = BeamVllmEngineServicer(_make_engine(max_model_len=8), 0.0)
    request = _make_request([1, 2, 3], max_tokens=max_tokens, truncate_prompt_tokens=-1)

    with pytest.raises(ValueError, match="prompt plus generation exceeds model length"):
        servicer.build_batch([request])


@pytest.mark.parametrize("side", ["left", "right"])  # type: ignore[misc]
def test_auto_truncation_validates_vocab_after_truncating(side: str) -> None:
    servicer = BeamVllmEngineServicer(
        _make_engine(max_model_len=8, truncation_side=side, vocab_size=128), 0.0
    )
    discarded_oov = [999, 999, 1, 2, 3, 4, 5] if side == "left" else [1, 2, 3, 4, 5, 999, 999]
    batch, _ = servicer.build_batch(
        [_make_request(discarded_oov, max_tokens=3, truncate_prompt_tokens=-1)]
    )
    np.testing.assert_array_equal(batch.items[0].request.prompt_token_ids, [1, 2, 3, 4, 5])

    kept_oov = [1, 2, 999, 4, 5, 6, 7] if side == "left" else [1, 2, 3, 4, 999, 6, 7]
    with pytest.raises(ValueError, match="token IDs are out of vocabulary"):
        servicer.build_batch([_make_request(kept_oov, max_tokens=3, truncate_prompt_tokens=-1)])


@pytest.mark.parametrize("invalid_truncate", [0, -2])  # type: ignore[misc]
def test_invalid_truncate_prompt_tokens_rejected(invalid_truncate: int) -> None:
    servicer = BeamVllmEngineServicer(_make_engine(max_model_len=8), 0.0)
    request = _make_request([1, 2, 3], max_tokens=3, truncate_prompt_tokens=invalid_truncate)

    with pytest.raises(ValueError, match="truncate_prompt_tokens must be positive or -1"):
        servicer.build_batch([request])


def test_positive_truncate_prompt_tokens_regression() -> None:
    servicer = BeamVllmEngineServicer(_make_engine(max_model_len=8, truncation_side="left"), 0.0)
    ok_req = _make_request([10, 11, 12, 13, 14, 15], max_tokens=3, truncate_prompt_tokens=5)
    batch, _ = servicer.build_batch([ok_req])
    np.testing.assert_array_equal(batch.items[0].request.prompt_token_ids, [11, 12, 13, 14, 15])

    too_long = _make_request([10, 11, 12, 13, 14, 15], max_tokens=3, truncate_prompt_tokens=6)
    with pytest.raises(ValueError, match="prompt plus generation exceeds model length"):
        servicer.build_batch([too_long])
