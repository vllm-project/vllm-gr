# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Catalog-free SID maps share the existing runner projection lifecycle."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, NoReturn, Protocol, TypeVar, cast

import pytest
import torch

from vllm_gr.config import GRConfig
from vllm_gr.v1.worker.model_runner_patch import (
    make_load_model_with_constraint_backend,
    make_shutdown_with_constraint_backend,
)
from vllm_gr.v1.worker.npu.npu_model_runner_patch import (
    make_patched_load_model,
    make_patched_shutdown,
)

_TestFunction = TypeVar("_TestFunction", bound=Callable[..., None])


class _Parametrize(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Callable[[_TestFunction], _TestFunction]: ...


_parametrize = cast(_Parametrize, pytest.mark.parametrize)


def _runner() -> SimpleNamespace:
    config = {
        "schema_version": 1,
        "attention_backend": "CUSTOM",
        "constrained_generation": False,
        "beam": {
            "execution_mode": "v1",
            "worker_decision": True,
            "max_width": 4,
            "max_decode_steps": 3,
            "sid_stage_vocab": True,
        },
        "constraint_table": {"enabled": False, "backend": "npu", "max_top_k": 4},
    }
    return SimpleNamespace(
        model=torch.nn.Module(),
        vllm_config=SimpleNamespace(additional_config={"vllm_gr": config}),
        model_config=SimpleNamespace(get_vocab_size=lambda: 40),
        device=torch.device("npu:0") if hasattr(torch, "npu") else "npu:0",
        model_memory_usage=100,
    )


@_parametrize("project", [False, True])
def test_sid_maps_load_without_catalog_and_close(
    monkeypatch: pytest.MonkeyPatch, project: bool
) -> None:
    import vllm.tokenizers

    from vllm_gr.ops import npu_stage_logits
    from vllm_gr.ops.npu import sid_stage_vocab

    runner = _runner()
    tokenizer = object()
    maps = SimpleNamespace(
        device_memory_bytes=48,
        stage_token_ids=tuple(torch.tensor(ids) for ids in ([2, 5], [8, 9], [20, 21])),
    )
    closed: list[bool] = []
    projection = SimpleNamespace(device_memory_bytes=96, close=lambda: closed.append(True))
    monkeypatch.setattr(vllm.tokenizers, "cached_tokenizer_from_config", lambda cfg: tokenizer)

    def build(actual_tokenizer: Any, vocab_size: int, device: Any) -> SimpleNamespace:
        assert actual_tokenizer is tokenizer
        assert vocab_size == 40 and str(device).startswith("npu")
        return maps

    monkeypatch.setattr(sid_stage_vocab, "build_sid_stage_vocab", build)
    monkeypatch.setattr(
        npu_stage_logits,
        "install_npu_sid_stage_projection",
        lambda actual, vocab: projection if project else None,
    )
    native_calls: list[bool] = []

    def native(actual: Any) -> str:
        assert actual is runner
        native_calls.append(True)
        return "loaded"

    wrapped = make_patched_load_model(make_load_model_with_constraint_backend(native))
    assert wrapped(runner) == "loaded"
    assert native_calls == [True]
    assert runner.constraint_backend is None
    assert runner._gr_npu_sid_stage_vocab is maps
    assert runner._gr_npu_stage_projection is (projection if project else None)
    assert runner.model_memory_usage == 148 + (96 if project else 0)
    shutdown = make_patched_shutdown(make_shutdown_with_constraint_backend())
    shutdown(runner)
    shutdown(runner)
    assert runner._gr_npu_sid_stage_vocab is None
    assert closed == ([True] if project else [])


@_parametrize("failure", ["tokenizer", "mapping", "projection"])
def test_sid_load_failure_does_not_publish_partial_runtime(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import vllm.tokenizers

    from vllm_gr.ops import npu_stage_logits
    from vllm_gr.ops.npu import sid_stage_vocab

    runner = _runner()
    monkeypatch.setattr(
        vllm.tokenizers,
        "cached_tokenizer_from_config",
        lambda cfg: None if failure == "tokenizer" else object(),
    )

    def fail(*args: Any) -> NoReturn:
        raise ValueError("bad SID model")

    monkeypatch.setattr(
        sid_stage_vocab,
        "build_sid_stage_vocab",
        fail if failure == "mapping" else lambda *args: SimpleNamespace(),
    )
    monkeypatch.setattr(npu_stage_logits, "install_npu_sid_stage_projection", fail)
    with pytest.raises(ValueError):
        make_patched_load_model(make_load_model_with_constraint_backend(lambda actual: None))(
            runner
        )
    assert runner.constraint_backend is None
    assert runner._gr_npu_sid_stage_vocab is None
    assert getattr(runner, "_gr_npu_stage_projection", None) is None
    assert runner.model_memory_usage == 100


def test_generic_unconstrained_load_does_not_initialize_sid_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.tokenizers

    runner = _runner()
    runner.vllm_config.additional_config["vllm_gr"]["beam"]["sid_stage_vocab"] = False

    def forbidden(*args: Any) -> NoReturn:
        raise AssertionError("generic generation must not load SID tokenizer")

    monkeypatch.setattr(vllm.tokenizers, "cached_tokenizer_from_config", forbidden)
    assert GRConfig.from_vllm_config(runner.vllm_config).beam_sid_stage_vocab is False
    wrapped = make_patched_load_model(
        make_load_model_with_constraint_backend(lambda actual: "loaded")
    )
    assert wrapped(runner) == "loaded"
    assert runner._gr_npu_sid_stage_vocab is None


def test_shared_cuda_loader_does_not_create_npu_sid_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.tokenizers

    runner = _runner()
    runner.device = "cuda:0"
    runner.vllm_config.additional_config["vllm_gr"]["beam"]["sid_stage_vocab"] = False

    def forbidden(*args: Any) -> NoReturn:
        raise AssertionError("CUDA loading must not initialize NPU SID state")

    monkeypatch.setattr(vllm.tokenizers, "cached_tokenizer_from_config", forbidden)
    assert make_load_model_with_constraint_backend(lambda actual: "loaded")(runner) == "loaded"
    assert not hasattr(runner, "_gr_npu_sid_stage_vocab")


def test_npu_shutdown_clears_sid_maps_before_native_failure() -> None:
    runner = _runner()
    runner._gr_npu_sid_stage_vocab = object()
    closed: list[bool] = []
    runner._gr_npu_stage_projection = SimpleNamespace(close=lambda: closed.append(True))

    def native(actual: Any) -> NoReturn:
        assert actual._gr_npu_sid_stage_vocab is None
        raise RuntimeError("native shutdown failed")

    shutdown = make_patched_shutdown(make_shutdown_with_constraint_backend(native))
    with pytest.raises(RuntimeError, match="native shutdown failed"):
        shutdown(runner)
    assert runner._gr_npu_sid_stage_vocab is None
    assert runner._gr_npu_stage_projection is None
    assert closed == [True]


pytestmark = pytest.mark.cpu_test
