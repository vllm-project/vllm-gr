# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from typing import Any

import pytest

pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
def test_grpc_client_accepts_a_short_catalog_result_and_checks_its_shape() -> None:
    pytest.importorskip("grpc", minversion="1.84.0")
    pytest.importorskip("google.protobuf", minversion="7.35.1")
    from examples.online_serving.grpc.client import generate
    from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2 as pb

    class Stub:
        offsets: tuple[int, ...] = (0, 3)
        scores: tuple[float, ...] = (-1.0,)
        flags: tuple[bool, ...] = (True,)

        def GenerateBatch(self, request: Any, *, timeout: float) -> Any:
            assert timeout > 0
            assert request.requests[0].beam_search.beam_width == 128
            return pb.BatchGenerateResponse(
                batch_id=request.batch_id,
                items=[
                    pb.BatchGenerateItem(
                        request_id=request.requests[0].request_id,
                        beams=pb.BeamBatchOutput(
                            output_ids=[4, 5, 6],
                            output_offsets=self.offsets,
                            cumulative_logprobs=self.scores,
                            has_cumulative_logprob=self.flags,
                        ),
                    )
                ],
            )

    stub = Stub()
    assert generate(stub, [[1]], width=128)["items"] == [
        {"token_ids": [[4, 5, 6]], "scores": [-1.0]}
    ]
    for offsets in ((1, 3), (0, 2), (0, 3, 2, 3)):
        stub.offsets = offsets
        with pytest.raises(ValueError, match="incomplete beam output"):
            generate(stub, [[1]], width=128)
    stub.offsets = (0, 3)
    stub.scores = ()
    with pytest.raises(ValueError, match="incomplete beam output"):
        generate(stub, [[1]], width=128)
    stub.scores = (-1.0,)
    stub.flags = ()
    with pytest.raises(ValueError, match="incomplete beam output"):
        generate(stub, [[1]], width=128)
