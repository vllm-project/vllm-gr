# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Issue #398 Agent A validation: startup beam-width sweep against the CPU reference.

Covers W=1..256 on the native GPU stage pipeline with a synthetic constraint
table whose legal candidate universe exceeds the beam width at every stage,
so full-width outputs are exercised. Compares the compact terminal result
(tokens, parents, cumulative FP32 scores) against the independent pure-CPU
reference, checks the single-transfer byte count, and survives live-state
mutation before consumption.
"""

from __future__ import annotations

from itertools import product
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import msgspec
import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.constraints.artifact_compiler import ConstraintTableArtifactCompiler
from vllm_gr.constraints.json_compiler import compile_relations_json
from vllm_gr.constraints.loader import _upload_cuda_backend
from vllm_gr.constraints.table import ConstraintTable
from vllm_gr.v1.beam.batch_contracts import (
    BEAM_BATCH_PROTOCOL_VERSION,
    BeamBatchItem,
    BeamBatchParams,
    BeamBatchRequest,
    BeamOutputOptions,
)
from vllm_gr.v1.beam.beam_attention_context import BeamAttentionContext
from vllm_gr.v1.beam.beam_decision import BeamPostprocessWorkspace
from vllm_gr.v1.beam.beam_search_session import BeamSearchSession
from vllm_gr.v1.beam.beam_state import BeamSearchState
from vllm_gr.v1.engine.beam_search_v1 import (
    GRDeviceStateRef,
    GRStageKind,
    GRStageMetadata,
)
from vllm_gr.v1.engine.gr_async_scheduler import (
    GR_STAGE_OUTPUT_ATTR,
    GR_WORKER_RESULT_ATTR,
)
from vllm_gr.v1.worker.beam_search_runner import BeamSearchRunner
from vllm_gr.v1.worker.gpu_beam_stage_runner import (
    BeamRequestBinding,
    GPUBeamStageRunner,
)
from vllm_gr.v1.worker.gr_inputs import GRInputBuffers

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")

# pytest decorators are untyped in the mypy-only pre-commit environments.
# mypy: disable-error-code="misc,no-untyped-def,no-untyped-call,no-any-return"

_TOKENIZER_DIGEST = "sha256:" + "a" * 64


def _request(max_tokens: int, width: int) -> BeamBatchRequest:
    request = EngineCoreRequest(
        request_id="session-0",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=1.0, ignore_eos=True),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    return BeamBatchRequest(
        protocol_version=BEAM_BATCH_PROTOCOL_VERSION,
        batch_id="session-0",
        items=(BeamBatchItem(item_index=0, request=request),),
        beam_params=BeamBatchParams(
            beam_width=width,
            max_tokens=max_tokens,
            temperature=1.0,
            stop_token_ids=(2,),
            ignore_stop_tokens=True,
        ),
        output_options=BeamOutputOptions(
            num_return_sequences=width,
            include_stop_token=True,
            include_cumulative_logprob=True,
            include_parent_beam_indices=True,
        ),
    )


def _metadata(stage: int, n: int, width: int) -> GRStageMetadata:
    request = _request(n, width)
    return GRStageMetadata(
        session_id=request.batch_id,
        dispatch_id=stage,
        stage_index=stage,
        kind=GRStageKind.PREFILL if stage == 0 else GRStageKind.DECODE,
        decode_step=stage - 1 if stage else None,
        is_last_stage=stage == n - 1,
        input_device_state=GRDeviceStateRef(request.batch_id, stage - 1) if stage else None,
        output_device_state=GRDeviceStateRef(request.batch_id, stage),
        beam_params=msgspec.structs.replace(request.beam_params, stop_token_ids=()),
        output_options=request.output_options,
        num_prompt_tokens=7,
        produces_output=True,
    )


def _dispatch(m: GRStageMetadata) -> Any:
    result = SimpleNamespace(num_scheduled_tokens={m.session_id: 7 if m.stage_index == 0 else 1})
    setattr(result, GR_STAGE_OUTPUT_ATTR, {m.session_id: m})
    return result


def _table(width: int, universe: int, vocab: int) -> ConstraintTable:
    """Compile a triple table whose every stage keeps >= width legal candidates.

    Stage 1 offers ``universe`` tokens per prompt row; stages 2/3 offer four
    tokens per parent, so a width-W beam always has W*4 decode candidates.
    """
    triples = [
        [f"<t{i}>", f"<t{universe + b}>", f"<t{universe + 4 + c}>"]
        for i, b, c in product(range(universe), range(4), range(4))
    ]
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {f"<t{i}>": i for i in range(vocab)}
    tokenizer.convert_tokens_to_ids.side_effect = lambda values: [
        int(v.removeprefix("<t").removesuffix(">")) for v in values
    ]
    table = ConstraintTableArtifactCompiler(
        tokenizer, vocab, tokenizer_digest=_TOKENIZER_DIGEST
    ).compile(compile_relations_json(triples))
    return table


def _owner(n: int, width: int, universe: int, vocab: int, dtype: torch.dtype) -> Any:
    context = BeamAttentionContext(3, width + 2, 1, n + 2, 8, 2, torch.device("cuda"), dtype)
    state = BeamSearchState(
        context.decision,
        beam_width=width,
        max_new_tokens=n,
        stop_token_ids=(),
        ignore_stop_tokens=True,
    )
    slot = context.registry.allocate("session-0", width, 7)
    session = BeamSearchSession("session-0", slot, context)
    session.begin_session(width, 7, decision_index=2)
    workspace = BeamPostprocessWorkspace.allocate(
        max_batch=3,
        beam_width=width,
        top_k=width,
        max_decode_steps=n + 2,
        device="cuda",
        logits_dtype=dtype,
    )
    beam = BeamSearchRunner(session, state, workspace, state_index=2, workspace_index=1)
    backend, _ = _upload_cuda_backend(
        _table(width, universe, vocab),
        device=torch.device("cuda"),
        max_top_k=width,
        max_rows=width,
    )
    backend._mark_ready()
    worker = SimpleNamespace(
        _beam_sessions={"session-0": session},
        execute_model_state=None,
        _update_states=Mock(),
    )
    owner = GPUBeamStageRunner(
        worker, beam, BeamRequestBinding("session-0", 0, "session-0", 2, 0, 1), backend
    )
    owner.metadata = SimpleNamespace(
        suffix_cu_seqlens_q_device=torch.arange(width + 1, device="cuda", dtype=torch.int32),
        suffix_seqlens_kv_device=torch.empty(width, device="cuda", dtype=torch.int32),
        suffix_cu_seqlens_k_device=torch.empty(width + 1, device="cuda", dtype=torch.int32),
    )
    owner.temperatures.fill_(1)
    owner.prefixes.fill_(-1)
    return owner


def _prepare_binding(owner: Any, entry: Any) -> None:
    if owner.inputs is None:
        owner.inputs = GRInputBuffers(owner.beam.state.beam_width, 64, 16, owner.beam.state.device)
    slot = owner.pending[1] if owner.pending is not None else 0
    owner.prepared = owner.inputs.prepare_cpu(
        entry.metadata, entry.binding, block_ids=[[1]], slot=slot
    )
    owner.device_positions = owner.inputs.upload(owner.prepared)["positions"]
    owner.prepare_decode(entry)


def _run_width(n: int, width: int, dtype: torch.dtype = torch.float16) -> dict[str, Any]:
    from vllm_gr.v1.beam.cpu_reference import (
        normalize_constraint_candidates,
        select_final_sequences,
        select_initial_beams,
        select_next_beams,
    )

    universe = width + 2
    vocab = universe + 8
    owner = _owner(n, width, universe, vocab, dtype)
    receipts: list[Any] = []
    candidate_history = []
    for stage in range(n):
        if stage == 2:
            receipts[0].get_output()
        execution = owner.admit(_dispatch(_metadata(stage, n, width)))
        if stage:
            _prepare_binding(owner, execution.entries[0])
        rows = 1 if stage == 0 else width
        # Values stay < 1000 so every entry is exact in FP16 and per-row
        # distinct; row offsets rotate each parent's preference order.
        owner.logits = (
            (torch.arange(rows * vocab, device="cuda") % 1000).reshape(rows, vocab).to(dtype)
        )
        receipts.append(owner.sample())
        candidate_history.append(
            tuple(
                value[:rows].cpu().numpy().copy()
                for value in (owner.tokens, owner.scores, owner.valid)
            )
        )
    final = receipts[-1]
    owner.beam.state.history_token_ids.fill_(999)
    owner.tokens.fill_(999)
    owner.beam.state.completed_scores.fill_(-999)
    result = getattr(final.get_output(), GR_WORKER_RESULT_ATTR)["session-0"].terminal_result
    assert result.failure is None

    ref: Any = None
    expected: Any = None
    for stage in range(n):
        rows = 1 if stage == 0 else ref.items[0].num_active_beams
        candidates = (
            normalize_constraint_candidates(
                item_index=0,
                parent_beam_indices=np.arange(rows),
                token_ids=candidate_history[stage][0][:rows],
                logprobs=candidate_history[stage][1][:rows],
                valid_mask=candidate_history[stage][2][:rows],
            ),
        )
        if stage == n - 1:
            expected = select_final_sequences(
                ref,
                candidates,
                beam_width=width,
                num_return_sequences=width,
                ignore_stop_tokens=True,
            )
        elif ref is None:
            ref = select_initial_beams(candidates, beam_width=width, ignore_stop_tokens=True)
        else:
            ref = select_next_beams(ref, candidates, ignore_stop_tokens=True)

    actual_sequences = result.output.items[0].sequences
    assert len(actual_sequences) == width, "full-width output expected by construction"
    assert len(expected.items[0].sequences) == width
    token_mismatch = score_mismatch = parent_mismatch = 0
    max_score_delta = 0.0
    for index, (actual, wanted) in enumerate(
        zip(actual_sequences, expected.items[0].sequences, strict=True)
    ):
        if actual.token_ids != wanted.token_ids:
            token_mismatch += 1
            if token_mismatch == 1:
                pytest.fail(
                    f"W={width} n={n} rank {index}: tokens {actual.token_ids[:8]}... != "
                    f"{wanted.token_ids[:8]}..."
                )
        if actual.parent_beam_indices != wanted.parent_beam_indices:
            parent_mismatch += 1
        if actual.cumulative_logprob != wanted.cumulative_logprob:
            score_mismatch += 1
            max_score_delta = max(
                max_score_delta, abs(actual.cumulative_logprob - wanted.cumulative_logprob)
            )
    assert parent_mismatch == 0, f"W={width} n={n}: {parent_mismatch} parent mismatches"
    assert score_mismatch == 0, (
        f"W={width} n={n}: {score_mismatch} score mismatches, max delta {max_score_delta}"
    )
    assert owner.final_output.final_d2h_count == 1
    assert owner.final_output.final_d2h_bytes == 4 * (4 + width * (2 * n + 4))
    assert not hasattr(owner, "candidate_tokens")
    for receipt in receipts:
        receipt.get_output()
    # Draining the final consumer must retire the session without a separate
    # release call, including when the terminal receipt is consumed first.
    assert owner.closed
    assert not owner.beam.session.context.registry.has_session("session-0")
    assert "session-0" not in owner.worker._beam_sessions
    owner.worker._update_states.assert_called_once()
    cleanup = owner.worker._update_states.call_args.args[0]
    assert cleanup.finished_req_ids == {owner.binding.native_request_id}
    assert cleanup.total_num_scheduled_tokens == 0
    return {"width": width, "n": n, "sequences": len(actual_sequences)}


REPRESENTATIVE_WIDTHS = [1, 4, 16, 64, 127, 128, 129, 255, 256]


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("width", REPRESENTATIVE_WIDTHS)
def test_representative_widths_match_reference(width: int, n: int) -> None:
    _run_width(n, width)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_boundary_widths_dtypes(n: int, dtype) -> None:
    _run_width(n, 128, dtype=dtype)


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_full_width_sweep_1_to_256() -> None:
    results = [_run_width(2, width) for width in range(1, 257)]
    assert all(r["sequences"] == r["width"] for r in results)
