# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vllm_gr.config import ConstraintTableBackend, ConstraintTableConfig
from vllm_gr.constraints.artifact import write_constraint_table_artifact
from vllm_gr.constraints.backend import ConstraintMetadata
from vllm_gr.constraints.loader import load_constraint_backend
from vllm_gr.constraints.npu_backend import NPUConstraintBackend
from vllm_gr.constraints.table import TABLE_NAMES
from vllm_gr.ops.rec_constrained_topk import rec_constrained_topk_torch

from .helpers import TOKENIZER_DIGEST, make_backend_info, make_table


def _oracle_op(*args):
    tokens, logprobs = rec_constrained_topk_torch(*args)
    return tokens.to(torch.int32), logprobs


def _patch_cpu_npu_loader(monkeypatch: pytest.MonkeyPatch, op=_oracle_op) -> list[int]:
    import vllm_gr.constraints.loader as loader
    import vllm_gr.constraints.npu_backend as npu_backend

    sync_steps: list[int] = []
    memory = iter([100, 148])
    monkeypatch.setattr(loader, "resolve_npu_constraint_op", lambda: op)
    monkeypatch.setattr(loader, "_synchronize_npu", lambda device: sync_steps.append(1))
    monkeypatch.setattr(loader, "_npu_memory_allocated", lambda device: next(memory))
    monkeypatch.setattr(npu_backend, "_is_npu_device", lambda device: True)
    return sync_steps


def _constructor_kwargs(op=_oracle_op) -> dict[str, object]:
    table = make_table()
    kwargs = {name: torch.tensor(getattr(table, name)) for name in TABLE_NAMES}
    kwargs.update(info=make_backend_info(table), op=op)
    return kwargs


def test_constructor_rejects_invalid_callable_and_table_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.constraints.npu_backend as npu_backend

    kwargs = _constructor_kwargs(op=None)
    with pytest.raises(TypeError, match="op must be callable"):
        NPUConstraintBackend(**kwargs)  # type: ignore[arg-type]

    kwargs = _constructor_kwargs()
    with pytest.raises(ValueError, match="must be on an NPU device"):
        NPUConstraintBackend(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(npu_backend, "_is_npu_device", lambda device: True)
    kwargs = _constructor_kwargs()
    kwargs["prefix1_pair_keys"] = kwargs["prefix1_pair_keys"].to(torch.int32)
    with pytest.raises(ValueError, match=r"prefix1_pair_keys.*torch.int64"):
        NPUConstraintBackend(**kwargs)  # type: ignore[arg-type]


def test_ready_check_executes_steps_in_strict_order(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_gr.constraints.loader as loader

    steps: list[int] = []

    def recording_op(*args):
        steps.append(args[9])
        return _oracle_op(*args)

    syncs = _patch_cpu_npu_loader(monkeypatch, recording_op)
    backend = loader._create_npu_constraint_backend(
        make_table(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_top_k=4,
    )

    assert steps == [0, 1, 2]
    # One synchronization after upload plus one after each real launch.
    assert len(syncs) == 4
    assert backend.info.device_memory_bytes == 48
    backend.close()


def test_callable_that_fails_real_launch_cannot_be_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.constraints.loader as loader

    def failing_op(*args):
        raise RuntimeError("launch failed")

    _patch_cpu_npu_loader(monkeypatch, failing_op)
    with pytest.raises(
        RuntimeError,
        match=r"step=0.*dtype=torch.float32.*top_k=4.*artifact_digest",
    ):
        loader._create_npu_constraint_backend(
            make_table(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_top_k=4,
        )


def test_npu_backend_retains_only_device_tables_info_and_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.constraints.loader as loader

    _patch_cpu_npu_loader(monkeypatch)
    backend = loader._create_npu_constraint_backend(
        make_table(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_top_k=4,
    )
    assert set(NPUConstraintBackend.__slots__) == {
        "_first_token_ids",
        "_prefix1_offsets",
        "_prefix1_values",
        "_prefix1_pair_keys",
        "_prefix2_value_offsets",
        "_prefix2_values",
        "_info",
        "_op",
    }
    assert not hasattr(backend, "_table")
    backend.close()
    backend.close()


@pytest.mark.parametrize(
    ("prefixes", "step", "error"),
    [
        (torch.zeros((1, 3), dtype=torch.int64), 2, "must be int32"),
        (torch.zeros((1, 2), dtype=torch.int32), 2, "width 2 must be greater"),
        (
            torch.arange(6, dtype=torch.int32).reshape(1, 6)[:, ::2],
            2,
            "must be contiguous",
        ),
    ],
)
def test_npu_select_rejects_inputs_that_require_implicit_prefix_fixup(
    monkeypatch: pytest.MonkeyPatch,
    prefixes: torch.Tensor,
    step: int,
    error: str,
) -> None:
    import vllm_gr.constraints.loader as loader

    _patch_cpu_npu_loader(monkeypatch)
    backend = loader._create_npu_constraint_backend(
        make_table(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_top_k=4,
    )
    metadata = ConstraintMetadata(
        prefix_token_ids=prefixes,
        temperatures=torch.ones(1, dtype=torch.float32),
        current_step=step,
        top_k=2,
    )
    with pytest.raises(ValueError, match=error):
        backend.select(torch.zeros((1, 8)), metadata)
    backend.close()


@pytest.mark.parametrize("input_name", ["logits", "temperatures"])
def test_npu_select_rejects_noncontiguous_runtime_inputs(
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
) -> None:
    import vllm_gr.constraints.loader as loader

    _patch_cpu_npu_loader(monkeypatch)
    backend = loader._create_npu_constraint_backend(
        make_table(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_top_k=4,
    )
    logits = torch.zeros((2, 16))[:, ::2]
    temperatures = torch.ones((2, 2), dtype=torch.float32)[:, 0]
    if input_name != "logits":
        logits = logits.contiguous()
    if input_name != "temperatures":
        temperatures = temperatures.contiguous()
    metadata = ConstraintMetadata(
        prefix_token_ids=torch.zeros((2, 3), dtype=torch.int32),
        temperatures=temperatures,
        current_step=2,
        top_k=2,
    )
    with pytest.raises(ValueError, match=rf"NPU .*{input_name}.*contiguous"):
        backend.select(logits, metadata)
    backend.close()


def test_npu_select_does_not_make_operator_outputs_contiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_gr.constraints.loader as loader

    _patch_cpu_npu_loader(monkeypatch)
    backend = loader._create_npu_constraint_backend(
        make_table(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_top_k=4,
    )

    def noncontiguous_output(*args):
        tokens, logprobs = _oracle_op(*args)
        token_storage = torch.stack((tokens, tokens), dim=-1)
        logprob_storage = torch.stack((logprobs, logprobs), dim=-1)
        return token_storage[..., 0], logprob_storage[..., 0]

    backend._op = noncontiguous_output
    metadata = ConstraintMetadata(
        prefix_token_ids=torch.zeros((2, 3), dtype=torch.int32),
        temperatures=torch.ones(2, dtype=torch.float32),
        current_step=0,
        top_k=2,
    )
    with pytest.raises(RuntimeError, match="must be contiguous"):
        backend.select(torch.zeros((2, 8)), metadata)
    backend.close()


def test_npu_upload_failure_closes_partial_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_gr.constraints.loader as loader

    closed: list[bool] = []
    original_close = NPUConstraintBackend.close

    def tracked_close(self):
        closed.append(True)
        original_close(self)

    def bad_output(*args):
        tokens, probs = _oracle_op(*args)
        return tokens[:, :1], probs[:, :1]

    _patch_cpu_npu_loader(monkeypatch, bad_output)
    monkeypatch.setattr(NPUConstraintBackend, "close", tracked_close)
    with pytest.raises(RuntimeError, match="readiness failed"):
        loader._create_npu_constraint_backend(
            make_table(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_top_k=4,
        )
    assert closed


@pytest.mark.npu_integration
def test_real_npu_loader_runs_readiness(tmp_path: Path) -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    from vllm_gr.ops.npu.rec_constrained_topk import is_available

    if not is_available():
        pytest.skip("NPU rec_constrained_topk unavailable")

    path = tmp_path / "constraint_table.safetensors"
    digest = write_constraint_table_artifact(make_table(), path)
    config = ConstraintTableConfig(
        enabled=True,
        backend=ConstraintTableBackend.NPU,
        path=str(path),
        artifact_digest=digest,
        tokenizer_digest=TOKENIZER_DIGEST,
        max_top_k=4,
    )
    backend = load_constraint_backend(
        config=config,
        device=torch.device("npu"),
        vocab_size=8,
        dtype=torch.float16,
    )
    assert backend is not None
    assert backend.info.device_memory_bytes > 0
    backend.close()
