# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="no-untyped-def,no-untyped-call,no-any-return,misc"
"""Array wire, native state, cache identity and actual native staging regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_gr.v1.beam.batch_contracts import (
    BeamBatchItem,
    BeamEngineCoreRequest,
)
from vllm_gr.v1.beam.token_buffer import (
    NumpyTokenBuffer,
    initialize_numpy_request,
    install_numpy_request_factory,
    install_numpy_scheduler_data,
)
from vllm_gr.v1.worker.numpy_staging import (
    NumpyStagedContents,
    array_h2d,
    install_numpy_staging,
)


def native(tokens):
    return EngineCoreRequest(
        request_id="id",
        prompt_token_ids=tokens,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=3, ignore_eos=True, detokenize=False),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


@pytest.mark.parametrize("binary", [False, True])
def test_batch_keeps_canonical_arrays_without_copying_or_changing_flags(binary):
    source = np.array([1, 2, 3], dtype=np.int64)
    if binary:
        source = np.frombuffer(source.tobytes(), dtype="<i8")
    writable = source.flags.writeable
    native_request = native(source)
    values = {
        field: getattr(native_request, field) for field in EngineCoreRequest.__struct_fields__
    }
    item = BeamBatchItem(0, BeamEngineCoreRequest(**values))
    assert item.request.prompt_token_ids is source
    assert source.flags.writeable == writable


def test_legacy_list_input_converts_to_array_at_batch_boundary():
    request = BeamBatchItem(0, native([1, 2, 3])).request
    assert isinstance(request, BeamEngineCoreRequest)
    assert request.prompt_token_ids.dtype == np.int64
    np.testing.assert_array_equal(request.prompt_token_ids, [1, 2, 3])
    assert request.external_req_id == "id"


@pytest.mark.parametrize(
    "value",
    [
        None,
        np.array([], dtype=np.int64),
        np.ones((2, 2), dtype=np.int64),
        np.array([1.5]),
        np.array([True]),
        np.array([-1], dtype=np.int64),
        np.array([2**63], dtype=np.uint64),
        np.arange(8, dtype=np.int64)[::2],
    ],
)
def test_contract_still_rejects_invalid_prompt_arrays(value):
    from tests.test_gr_async_scheduler import _batch_request
    from vllm_gr.v1.beam.batch_contracts import validate_beam_batch_request

    request = _batch_request()
    request.items[0].request.prompt_token_ids = value
    with pytest.raises(ValueError, match="contiguous nonnegative int64"):
        validate_beam_batch_request(request)


def test_native_request_and_scheduler_data_stay_array_backed():
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    install_numpy_request_factory()
    install_numpy_scheduler_data()
    item = BeamBatchItem(0, native([1, 2, 3]))
    request = Request.from_engine_core_request(item.request, None)
    assert isinstance(request._all_token_ids, NumpyTokenBuffer)
    request.append_output_token_ids(4)
    request.append_output_token_ids([5, 6])
    np.testing.assert_array_equal(request.all_token_ids.copy(), [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(item.request.prompt_token_ids, [1, 2, 3])
    with pytest.raises(ValueError, match="capacity"):
        request.append_output_token_ids(7)
    scheduled = NewRequestData.from_request(
        request, ([1],), request._all_token_ids, uses_mrope=False
    )
    assert isinstance(scheduled.prompt_token_ids, np.ndarray)
    assert isinstance(scheduled.prefill_token_ids, np.ndarray)
    np.testing.assert_array_equal(scheduled.prefill_token_ids, [1, 2, 3, 4, 5, 6])
    # Native non-GR request construction is not changed by the scoped hook.
    ordinary = Request.from_engine_core_request(native([1, 2]), None)
    assert type(ordinary._all_token_ids) is list


@pytest.mark.parametrize("uses_mrope", [False, True])
def test_scheduler_data_preserves_native_mrope_metadata(uses_mrope):
    from vllm.multimodal.inputs import (
        MultiModalBatchedField,
        MultiModalFeatureSpec,
        MultiModalFieldElem,
        MultiModalKwargsItem,
        PlaceholderRange,
    )
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    install_numpy_scheduler_data()
    request = Request.from_engine_core_request(native([1, 2, 3]), None)
    grid = MultiModalFieldElem(torch.tensor([1, 2, 2]), MultiModalBatchedField(keep_on_cpu=True))
    data = MultiModalKwargsItem(
        grid=grid,
        pixels=MultiModalFieldElem(torch.ones(1), MultiModalBatchedField()),
    )
    request.mm_features = [
        MultiModalFeatureSpec(data, "image", "cached-image", PlaceholderRange(0, 1))
    ]
    request.num_computed_tokens = 1
    scheduled = NewRequestData.from_request(request, ([1],), uses_mrope=uses_mrope)
    if uses_mrope:
        assert scheduled.mm_features[0].data == {"grid": grid}
    else:
        assert scheduled.mm_features[0].data is None
    assert request.mm_features[0].data is data


@pytest.mark.parametrize("salt", [None, "different-model-features"])
def test_numpy_hashes_match_full_recompute_for_cold_warm_and_incremental(salt):
    from vllm.utils.hashing import sha256
    from vllm.v1.core import kv_cache_utils as kv

    from vllm_gr.v1.engine.engine_core_patch import apply_kv_hash_memo_patch

    if getattr(kv, "NONE_HASH", None) is None:
        kv.init_none_hash(sha256)
    apply_kv_hash_memo_patch()
    hasher = kv.get_request_block_hasher(4, sha256)

    def make_request(tokens):
        request = native(tokens)
        request.cache_salt = salt
        return initialize_numpy_request(BeamBatchItem(0, request).request, hasher)

    cold = make_request(list(range(10)))
    warm = make_request(list(range(10)))
    assert cold.block_hashes == warm.block_hashes
    cold.append_output_token_ids([10, 11])
    complete = make_request(list(range(12)))
    assert cold.block_hashes == complete.block_hashes
    assert len(cold.block_hashes) == 3

    changed = make_request([99, *range(1, 12)])
    assert all(a != b for a, b in zip(changed.block_hashes, complete.block_hashes))


def test_numpy_block_hash_uses_bytes_without_python_iteration():
    from vllm.utils.hashing import sha256
    from vllm.v1.core import kv_cache_utils as kv

    from vllm_gr.v1.engine.engine_core_patch import apply_kv_hash_memo_patch

    class NoIteration(np.ndarray):
        def __iter__(self):
            raise AssertionError("must not materialize Python token integers")

    apply_kv_hash_memo_patch()
    tokens = np.arange(4, dtype=np.int64).view(NoIteration)
    result = kv.hash_block_tokens(sha256, None, tokens)
    assert result == kv.hash_block_tokens(sha256, None, tokens.copy())
    assert result != kv.hash_block_tokens(sha256, None, tokens, ("salt",))
    assert result != kv.hash_block_tokens(sha256, result, tokens)


def test_worker_detaches_arrays_without_python_token_lists():
    from vllm_gr.v1.worker.gr_execution import worker_input

    prompt = np.array([1, 2, 3], dtype=np.int64)
    source = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                prompt_token_ids=prompt,
                prefill_token_ids=prompt,
                block_ids=([1],),
                sampling_params=None,
            )
        ],
        num_scheduled_tokens={"id": 3},
        finished_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
    )
    result = worker_input(source).scheduled_new_reqs[0]
    assert isinstance(result.prompt_token_ids, np.ndarray)
    result.prompt_token_ids[0] = 99
    result.prefill_token_ids[0] = 98
    np.testing.assert_array_equal(prompt, [1, 2, 3])


def _cpu_uva_pool(monkeypatch):
    from vllm.v1.worker.gpu import buffer_utils

    def buffer(size, dtype):
        cpu = torch.zeros(size, dtype=dtype)
        return SimpleNamespace(cpu=cpu, np=cpu.numpy(), uva=lambda n: cpu[:n])

    # Exercise the native pool's NumPy assignment and growth without a device.
    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: True)
    monkeypatch.setattr(buffer_utils, "UvaBuffer", buffer)
    return buffer_utils.UvaBufferPool(1, dtype=torch.int32, max_concurrency=2)


@pytest.mark.parametrize("array_backed", [False, True])
@pytest.mark.parametrize("uva_contents", [False, True])
def test_native_v2_staging_batches_arrays_and_keeps_kernel_write_order(
    monkeypatch, array_backed, uva_contents
):
    from vllm.v1.worker.gpu import buffer_utils

    install_numpy_staging()
    staged = object.__new__(buffer_utils.StagedWriteTensor)
    staged.dtype, staged.device = torch.int32, torch.device("cpu")
    staged.gpu = torch.zeros((2, 8), dtype=torch.int32)
    staged._staged_write_indices = []
    staged._staged_write_starts = []
    staged._staged_write_contents = []
    staged._staged_write_cu_lens = []
    pool = SimpleNamespace(copy_to_uva=lambda values: torch.tensor(values, dtype=torch.int32))
    staged.write_indices = staged.write_starts = staged.write_cu_lens = pool
    staged.write_contents = _cpu_uva_pool(monkeypatch) if uva_contents else None

    class Kernel:
        def __getitem__(self, grid):
            def apply(output, stride, indices, starts, contents, lengths, group_ids, **kwargs):
                assert group_ids is None and not kwargs["MULTI_GROUP"]
                previous = 0
                for index, start, stop in zip(indices, starts, lengths):
                    count = int(stop) - previous
                    output[index, start : start + count] = contents[previous : int(stop)]
                    previous = int(stop)

            return apply

    monkeypatch.setattr(buffer_utils, "_apply_write_kernel", Kernel())
    first = np.array([1, 2, 3], dtype=np.int64) if array_backed else [1, 2, 3]
    second = np.array([4, 5], dtype=np.int64) if array_backed else [4, 5]
    staged.stage_write(0, 0, first)
    staged.stage_write(1, 2, second)
    staged.stage_write(0, 1, [9])  # Native write after an array write.
    assert isinstance(staged._staged_write_contents, NumpyStagedContents) == array_backed
    staged.apply_write()
    torch.testing.assert_close(staged.gpu[0, :3], torch.tensor([1, 9, 3], dtype=torch.int32))
    torch.testing.assert_close(staged.gpu[1, 2:4], torch.tensor([4, 5], dtype=torch.int32))
    assert len(staged._staged_write_contents) == 0
    np.testing.assert_array_equal(
        array_h2d(np.array([7, 8], dtype=np.int64), torch.int32, "cpu").numpy(), [7, 8]
    )


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("readonly", [False, True])
def test_native_v2_uva_contents_use_one_bulk_conversion(monkeypatch, batch, readonly):
    pool = _cpu_uva_pool(monkeypatch)
    contents = NumpyStagedContents(np.int32)
    original_array = contents.array
    calls = []

    def observed_array():
        calls.append(len(contents))
        return original_array()

    monkeypatch.setattr(contents, "array", observed_array)
    for iteration in range(3):
        chunks = []
        for member in range(batch):
            source = np.arange(1024, dtype=np.int32) + (iteration * batch + member) * 1024
            if readonly:
                source = np.frombuffer(source.tobytes(), dtype=source.dtype)
            chunks.append(source)
            contents.extend(source)
        calls.clear()
        output = pool.copy_to_uva(contents)
        np.testing.assert_array_equal(output.numpy(), np.concatenate(chunks))
        assert calls == [batch * 1024]
        assert all(chunk.flags.writeable != readonly for chunk in chunks)
        contents.clear()


@pytest.mark.parametrize("chunks", [0, 1, 4])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("copy", [None, False, True])
def test_staged_contents_array_protocol_dtype_and_copy(chunks, dtype, copy):
    source = np.arange(chunks * 3, dtype=np.int32)
    contents = NumpyStagedContents(source.dtype)
    for index in range(chunks):
        contents.extend(source[index * 3 : (index + 1) * 3])
    if copy is False and (chunks != 1 or dtype != source.dtype):
        with pytest.raises(ValueError, match="copy"):
            contents.__array__(dtype=dtype, copy=copy)
        return
    # Call the protocol directly so copy=False is also tested on NumPy 1.x,
    # whose np.array does not forward the copy argument to __array__.
    result = contents.__array__(dtype=dtype, copy=copy)
    np.testing.assert_array_equal(result, source)
    assert result.dtype == dtype
    if chunks == 1 and dtype == source.dtype:
        assert np.shares_memory(result, source) == (copy is not True)


@pytest.mark.parametrize("readonly", [False, True])
def test_native_v2_array_transfer_preserves_output_buffer(readonly):
    from vllm.v1.worker.gpu import buffer_utils

    install_numpy_staging()
    source = np.array([3, 7], dtype=np.int32)
    if readonly:
        source = np.frombuffer(source.tobytes(), dtype=source.dtype)
    contents = NumpyStagedContents(source.dtype)
    contents.extend(source)
    output = torch.empty(2, dtype=torch.int64)
    result = buffer_utils.async_tensor_h2d(contents, "cpu", out=output)
    assert result is output
    torch.testing.assert_close(output, torch.tensor([3, 7], dtype=torch.int64))
    assert source.flags.writeable != readonly


def test_native_v1_input_batch_copies_arrays_in_bulk():
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

    batch = InputBatch(
        max_num_reqs=2,
        max_model_len=32,
        max_num_batched_tokens=32,
        device=torch.device("cpu"),
        vocab_size=64,
        block_sizes=[16],
        kernel_block_sizes=[16],
        max_num_blocks_per_req=[2],
    )
    source = np.array([1, 2, 3, 4], dtype=np.int64)
    state = CachedRequestState(
        req_id="id",
        prompt_token_ids=source,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=3),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=0,
        output_token_ids=[],
    )
    index = batch.add_request(state)
    assert state.prompt_token_ids is source
    np.testing.assert_array_equal(batch.token_ids_cpu[index, :4], source)
    assert batch.num_prompt_tokens[index] == 4
    assert batch.remove_request("id") is not None
