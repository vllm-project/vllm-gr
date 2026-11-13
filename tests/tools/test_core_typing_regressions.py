# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise nullable boundaries exposed by checking the complete core dirs."""

from types import SimpleNamespace

import pytest
import torch

from tests.test_beam_decision import _make_context, _make_session, _sampler_output
from tests.test_gr_async_scheduler import _multi_batch_request, _scheduler
from vllm_gr.v1.beam.beam_decision import run_beam_decision
from vllm_gr.v1.engine.gr_async_scheduler import GRAdmissionError, add_gr_session

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_missing_beam_metadata_returns_no_worker_decision() -> None:
    session = _make_session(_make_context(max_beams=2))
    session.begin_session(beam_width=2, prefix_len=8)
    sampler = _sampler_output(
        torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
        torch.tensor([[-4.0, -6.0], [-0.1, -1.0]], dtype=torch.float32),
    )
    runner = SimpleNamespace(device=torch.device("cpu"))
    assert run_beam_decision(runner, session, sampler, is_last_step=False) is None


def test_null_native_token_budget_rejects_admission() -> None:
    scheduler = _scheduler()
    scheduler.scheduler_config.max_num_seqs = 2
    scheduler.scheduler_config.max_num_batched_tokens = None
    scheduler.max_num_scheduled_tokens = None
    with pytest.raises(GRAdmissionError, match="token budget is not configured"):
        add_gr_session(scheduler, _multi_batch_request())
    assert not getattr(scheduler, "_gr_v1_sessions", {})
