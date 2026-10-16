# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An unconstrained request rejection must retain its wire version."""

from queue import Queue
from types import SimpleNamespace

import msgspec
import pytest

from tests.test_gr_session_client_full_vocab import _request
from vllm_gr.v1.beam.batch_contracts import admission_rejected
from vllm_gr.v1.engine.core import (
    _admit_gr_session,
    _handle_submit_gr_session,
    send_gr_rejection,
)
from vllm_gr.v1.engine.gr_session_client import _terminal_result


@pytest.mark.parametrize("version", [1, 2])
def test_admission_failure_retains_explicit_version(version):
    assert (
        admission_rejected("batch", "reason", protocol_version=version).protocol_version == version
    )


@pytest.mark.parametrize("handler", [_admit_gr_session, _handle_submit_gr_session])
def test_full_vocab_rejection_before_native_preprocessing(handler):
    request = _request()
    request = msgspec.structs.replace(
        request, beam_params=msgspec.structs.replace(request.beam_params, temperature=1.0)
    )
    core = SimpleNamespace(output_queue=Queue())
    handler(core, request)
    _, outputs = core.output_queue.get_nowait()
    result = _terminal_result(request, outputs.outputs[0])
    assert result.protocol_version == 2
    assert result.failure.code == "admission_rejected"
    assert core.output_queue.empty()


def test_native_add_rejection_derives_version_from_batch_owner():
    core = SimpleNamespace(output_queue=Queue())
    native = SimpleNamespace(
        request_id="batch", client_index=0, gr_batch_request=SimpleNamespace(protocol_version=2)
    )
    send_gr_rejection(core, native, "capacity exceeded")
    _, outputs = core.output_queue.get_nowait()
    assert outputs.outputs[0].kv_transfer_params["gr_batch_result"].protocol_version == 2
