# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Geometry-only tests; multi-group cases do not enable B>1 execution."""

import pytest
import torch

from vllm_gr.v1.beam.decode_geometry import BeamDecodeGeometry


@pytest.mark.parametrize(
    "widths,prefixes,indices,offsets",
    [
        ((1,), (0,), [0], [0, 1]),
        ((4,), (7,), [0, 0, 0, 0], [0, 4]),
        ((2, 3), (7, 11), [0, 0, 1, 1, 1], [0, 2, 5]),
        ((2, 2, 2), (9, 9, 9), [0, 0, 1, 1, 2, 2], [0, 2, 4, 6]),
    ],
)
def test_writes_static_layout_without_host_tensor_upload(
    monkeypatch, widths, prefixes, indices, offsets
):
    geometry = BeamDecodeGeometry(widths, prefixes)
    req_indices = torch.empty(geometry.num_rows, dtype=torch.int64)
    prefix_kv = torch.empty(geometry.num_requests, dtype=torch.int32)
    prefix_cu_q = torch.empty(geometry.num_requests + 1, dtype=torch.int32)
    suffix_cu_q = torch.empty(geometry.num_rows + 1, dtype=torch.int32)
    outputs = (req_indices, prefix_kv, prefix_cu_q, suffix_cu_q)
    addresses = [output.data_ptr() for output in outputs]

    def forbid_host_tensor(*args, **kwargs):
        pytest.fail("static geometry must not construct a Host tensor for upload")

    monkeypatch.setattr(torch, "tensor", forbid_host_tensor)
    monkeypatch.setattr(torch, "as_tensor", forbid_host_tensor)
    monkeypatch.setattr(torch, "from_numpy", forbid_host_tensor)
    for _ in range(2):
        geometry.write_request_indices(req_indices)
        geometry.write_static_seqlens(prefix_kv, prefix_cu_q, suffix_cu_q)
        assert req_indices.tolist() == indices
        assert prefix_kv.tolist() == list(prefixes)
        assert prefix_cu_q.tolist() == offsets
        assert suffix_cu_q.tolist() == list(range(geometry.num_rows + 1))
        assert [output.data_ptr() for output in outputs] == addresses
    assert geometry.matches_request_indices(indices)
    assert not geometry.matches_request_indices(indices + [0])
    assert not geometry.matches_request_indices([99] * geometry.num_rows)


@pytest.mark.parametrize(
    "widths,prefixes",
    [((), ()), ((2,), ()), ((0,), (7,)), ((-2,), (7,)), ((2,), (-1,)), ((True,), (1,))],
)
def test_invalid_geometry_fails_before_any_device_work(widths, prefixes):
    with pytest.raises(ValueError, match="Decode geometry"):
        BeamDecodeGeometry(widths, prefixes)


@pytest.mark.parametrize("invalid_output", [0, 1, 2])
def test_invalid_destination_does_not_partially_update(invalid_output):
    geometry = BeamDecodeGeometry((2, 3), (7, 11))
    outputs = [torch.full((size,), -1, dtype=torch.int32) for size in (2, 3, 6)]
    outputs[invalid_output] = outputs[invalid_output][:-1]
    with pytest.raises(ValueError, match="integer vector"):
        geometry.write_static_seqlens(*outputs)
    assert all((output == -1).all() for output in outputs)


def test_prefix_dtype_overflow_fails_before_writing():
    geometry = BeamDecodeGeometry((1,), (2**31,))
    outputs = [torch.full((size,), -1, dtype=torch.int32) for size in (1, 2, 2)]
    with pytest.raises(ValueError, match="dtype capacity"):
        geometry.write_static_seqlens(*outputs)
    assert all((output == -1).all() for output in outputs)


def test_request_index_validation_rejects_tensor_without_scalar_reads(monkeypatch):
    geometry = BeamDecodeGeometry((1,), (0,))
    indices = torch.zeros(1, dtype=torch.int64)

    def forbid_scalar(*args, **kwargs):
        pytest.fail("index validation must not read a device scalar")

    monkeypatch.setattr(torch.Tensor, "__bool__", forbid_scalar)
    with pytest.raises(TypeError, match="Host integer sequence"):
        geometry.matches_request_indices(indices)
