# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Tests for the driver/worker Constraint Tables resource injection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_gr.constraints.compiler import (
    ConstraintTableCompiler,
    ConstraintTableResource,
    from_wire,
    to_wire,
)
from vllm_gr.constraints.driver_inject import inject_constraint_resources
from vllm_gr.constraints.execution import valid_candidate_mask
from vllm_gr.constraints.json_compiler import compile_relations_json
from vllm_gr.constraints.resource_manager import ConstraintResourceManager
from vllm_gr.constraints.worker_inject import (
    install_worker_resource_hook,
    set_gr_constraint_resources,
)


@pytest.fixture  # type: ignore[misc]
def tokenizer() -> MagicMock:
    result = MagicMock()
    result.convert_tokens_to_ids.side_effect = lambda values: [int(value) for value in values]
    result.get_vocab.return_value = {str(i): i for i in range(10000)}
    return result


def _build_resource(
    tokenizer: MagicMock, vocab_size: int = 1000, n_triples: int = 5
) -> ConstraintTableResource:
    """Build a resource whose catalog tokens all sit within vocab_size.

    Catalog uses ``[i, i + n, i + 2n]`` so the max token is ``3n - 1``; the
    default vocab_size of 1000 covers the largest default catalog (3*50=149).
    """
    catalog = [[str(i), str(i + n_triples), str(i + 2 * n_triples)] for i in range(n_triples)]
    document = compile_relations_json(catalog)
    return ConstraintTableCompiler(tokenizer, vocab_size).compile(document)


def test_wire_roundtrip_preserves_content_and_immutability(tokenizer: MagicMock) -> None:
    """to_wire -> from_wire must keep content, digest, and read-only arrays."""
    resource = _build_resource(tokenizer)
    wire = to_wire(resource)

    # Wire payload is base types only (survives untyped msgpack decode).
    assert isinstance(wire["array_fields"]["prefix1_values"]["data"], bytes)
    assert isinstance(wire["array_fields"]["prefix1_values"]["dtype"], str)
    assert isinstance(wire["array_fields"]["prefix1_values"]["shape"], list)

    rebuilt = from_wire(wire)
    assert rebuilt.digest == resource.digest
    assert np.array_equal(rebuilt.prefix1_values, resource.prefix1_values)
    assert np.array_equal(rebuilt.prefix1_pair_keys, resource.prefix1_pair_keys)
    assert not rebuilt.prefix1_values.flags.writeable
    # from_wire must re-freeze arrays even after pickle-style transport.
    with pytest.raises(ValueError):
        rebuilt.prefix1_values.setflags(write=True)


def test_wire_roundtrip_detects_digest_mismatch(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer)
    wire = to_wire(resource)
    wire["digest"] = "deadbeef"

    with pytest.raises(ValueError, match="digest mismatch"):
        from_wire(wire)


def test_wire_rejects_malformed_payload(tokenizer: MagicMock) -> None:
    with pytest.raises(ValueError, match="malformed"):
        from_wire({"nope": 1})
    with pytest.raises(ValueError, match="malformed"):
        from_wire({})


def test_set_gr_constraint_resources_idempotent_same_digest(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer)
    wire = to_wire(resource)
    manager = ConstraintResourceManager()
    status1 = set_gr_constraint_resources(object(), wire, manager)
    status2 = set_gr_constraint_resources(object(), wire, manager)

    assert status1["cached"] is True
    assert status2["cached"] is False  # same digest -> reused
    assert manager.current_digest == resource.digest


def test_set_gr_constraint_resources_rejects_different_digest(tokenizer: MagicMock) -> None:
    resource_a = _build_resource(tokenizer, vocab_size=100)
    resource_b = _build_resource(tokenizer, vocab_size=200)
    manager = ConstraintResourceManager()

    set_gr_constraint_resources(object(), to_wire(resource_a), manager)
    with pytest.raises(ValueError, match="cannot replace loaded"):
        set_gr_constraint_resources(object(), to_wire(resource_b), manager)


def test_set_gr_constraint_resources_fails_when_runner_lacks_sample_hook(
    tokenizer: MagicMock,
) -> None:
    """A Constraint Table worker without the raw-logits hook must fail startup."""
    from vllm_gr.v1.worker.model_runner_contracts import (
        _WORKER_SAMPLE_PATCH_MARKER,
    )

    resource = _build_resource(tokenizer)
    wire = to_wire(resource)

    class FakeWorker:
        model_runner = object()  # a runner class without _sample

    manager = ConstraintResourceManager()
    with pytest.raises(RuntimeError, match="Constraint Tables cannot start"):
        set_gr_constraint_resources(FakeWorker(), wire, manager)

    class PatchedRunner:
        pass

    def patched_sample(self: Any, *args: Any, **kwargs: Any) -> Any:
        return object()

    setattr(patched_sample, _WORKER_SAMPLE_PATCH_MARKER, True)
    setattr(PatchedRunner, "_sample", patched_sample)

    class PatchedWorker:
        model_runner = PatchedRunner()

    manager = ConstraintResourceManager()
    status = set_gr_constraint_resources(PatchedWorker(), wire, manager)
    assert status["cached"] is True


def test_set_gr_constraint_resources_accepts_v2_stage_hooks(
    tokenizer: MagicMock,
) -> None:
    """The V2 stage execute/sample_tokens pair satisfies the resource gate."""
    from vllm_gr.v1.worker.model_runner_contracts import (
        _WORKER_EXECUTE_PATCH_MARKER,
        _WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER,
    )

    resource = _build_resource(tokenizer)
    wire = to_wire(resource)

    class PatchedV2Runner:
        pass

    def patched_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
        return object()

    def patched_sample_tokens(self: Any, *args: Any, **kwargs: Any) -> Any:
        return object()

    setattr(patched_execute, _WORKER_EXECUTE_PATCH_MARKER, True)
    setattr(patched_sample_tokens, _WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER, True)
    setattr(PatchedV2Runner, "execute_model", patched_execute)
    setattr(PatchedV2Runner, "sample_tokens", patched_sample_tokens)

    class PatchedV2Worker:
        model_runner = PatchedV2Runner()

    manager = ConstraintResourceManager()
    status = set_gr_constraint_resources(PatchedV2Worker(), wire, manager)
    assert status["cached"] is True


@pytest.mark.parametrize(  # type: ignore[misc]
    ("patch_execute", "patch_sample_tokens"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_inject_constraint_resources_requires_complete_cuda_v1_stage_hooks(
    tokenizer: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    patch_execute: bool,
    patch_sample_tokens: bool,
) -> None:
    """Real V1 wrappers permit injection/preload without patching native _sample."""
    from vllm_gr.constraints import worker_inject
    from vllm_gr.v1.worker.gpu_model_runner_patch import make_patched_sample_tokens_gpu
    from vllm_gr.v1.worker.model_runner_common import (
        make_patched_execute_model,
    )
    from vllm_gr.v1.worker.model_runner_contracts import (
        _WORKER_SAMPLE_PATCH_MARKER,
    )

    class FakeRunner:
        device = torch.device("cuda")
        _gr_constraint_execution: Any = None

        def _sample(self, *args: Any, **kwargs: Any) -> None:
            pass

        def execute_model(self, *args: Any, **kwargs: Any) -> None:
            pass

        def sample_tokens(self, *args: Any, **kwargs: Any) -> None:
            pass

    native_sample = FakeRunner._sample
    if patch_execute:
        setattr(FakeRunner, "execute_model", make_patched_execute_model(FakeRunner.execute_model))
    if patch_sample_tokens:
        setattr(
            FakeRunner, "sample_tokens", make_patched_sample_tokens_gpu(FakeRunner.sample_tokens)
        )

    class FakeWorker:
        model_runner = FakeRunner()

    install_worker_resource_hook(FakeWorker)
    worker = FakeWorker()

    def collective_rpc(method: str, *, args: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
        return [getattr(worker, method)(*args)]

    engine = SimpleNamespace(engine_core=SimpleNamespace(collective_rpc=collective_rpc))
    resource = _build_resource(tokenizer)
    manager = ConstraintResourceManager()
    monkeypatch.setattr(worker_inject, "_WORKER_MANAGER", manager)
    adapter = MagicMock()
    create_adapter = MagicMock(return_value=adapter)
    monkeypatch.setattr("vllm_gr.constraints.startup.create_constraint_execution", create_adapter)

    if not (patch_execute and patch_sample_tokens):
        with pytest.raises(RuntimeError, match="Constraint Tables cannot start"):
            inject_constraint_resources(engine, resource)
        create_adapter.assert_not_called()
        assert worker.model_runner._gr_constraint_execution is None
        return

    assert inject_constraint_resources(engine, resource) == [
        {"digest": resource.digest, "cached": True}
    ]
    assert inject_constraint_resources(engine, resource) == [
        {"digest": resource.digest, "cached": False}
    ]
    assert manager.current_digest == resource.digest
    create_adapter.assert_called_once_with(device=FakeRunner.device)
    adapter.load_tables.assert_called_once_with(manager.require(resource.digest))
    assert worker.model_runner._gr_constraint_execution is adapter
    assert FakeRunner._sample is native_sample
    assert not getattr(FakeRunner._sample, _WORKER_SAMPLE_PATCH_MARKER, False)


def test_install_worker_resource_hook_idempotent(tokenizer: MagicMock) -> None:
    class FakeWorker:
        pass

    first = install_worker_resource_hook(FakeWorker)
    second = install_worker_resource_hook(FakeWorker)

    assert first == 1
    assert second == 0
    assert hasattr(FakeWorker, "set_gr_constraint_resources")

    # The mounted method works on a worker instance (self is ignored).  Pass an
    # explicit manager to avoid leaking state through the process singleton.
    resource = _build_resource(tokenizer)
    manager = ConstraintResourceManager()
    status = FakeWorker.set_gr_constraint_resources(FakeWorker(), to_wire(resource), manager)
    assert status["cached"] is True


def test_inject_constraint_resources_forwards_to_collective_rpc(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer)
    engine_core = MagicMock()
    engine_core.collective_rpc.return_value = [{"digest": resource.digest, "cached": True}]
    engine = SimpleNamespace(engine_core=engine_core)

    statuses = inject_constraint_resources(engine, resource)

    engine_core.collective_rpc.assert_called_once_with(
        "set_gr_constraint_resources", args=(to_wire(resource),)
    )
    assert statuses == [{"digest": resource.digest, "cached": True}]


def test_inject_requires_collective_rpc(tokenizer: MagicMock) -> None:
    resource = _build_resource(tokenizer)
    engine = SimpleNamespace(engine_core=SimpleNamespace())  # no collective_rpc
    with pytest.raises(TypeError, match="collective_rpc"):
        inject_constraint_resources(engine, resource)


def test_large_resource_roundtrips_via_msgpack_and_is_re_frozen(
    tokenizer: MagicMock,
) -> None:
    """A realistic-sized resource must survive the wire transport re-frozen.

    vLLM v1 utility RPC decodes payloads with an untyped MsgpackDecoder, so the
    decoded arrays arrive writable; from_wire must re-freeze them or a worker
    could mutate a table without the digest changing.
    """
    import msgspec

    resource = _build_resource(tokenizer, vocab_size=2000, n_triples=50)
    encoded = msgspec.msgpack.encode(to_wire(resource))
    restored = from_wire(msgspec.msgpack.decode(encoded))

    assert restored.digest == resource.digest
    assert np.array_equal(restored.prefix1_values, resource.prefix1_values)
    assert np.array_equal(restored.prefix1_pair_keys, resource.prefix1_pair_keys)
    assert not restored.prefix1_values.flags.writeable
    assert not restored.prefix1_pair_keys.flags.writeable


def test_valid_candidate_mask_marks_padding() -> None:
    counts = torch.tensor([2, 1], dtype=torch.int64)
    mask = valid_candidate_mask(counts, 3)

    assert mask.dtype == torch.bool
    assert mask.shape == (2, 3)
    assert mask.tolist() == [
        [True, True, False],
        [True, False, False],
    ]


def test_load_constraint_tables_on_worker_preloads_to_device(
    tokenizer: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Injected + device-backed runner preloads tables at weight-load time."""
    from vllm_gr.constraints import worker_inject
    from vllm_gr.constraints.worker_inject import (
        load_constraint_tables_on_worker,
    )

    resource = _build_resource(tokenizer)
    manager = ConstraintResourceManager()
    manager.ensure(resource)
    monkeypatch.setattr(worker_inject, "get_worker_manager", lambda: manager)

    adapter = MagicMock()
    monkeypatch.setattr(
        "vllm_gr.constraints.startup.create_constraint_execution",
        lambda **kw: adapter,
    )

    class FakeRunner:
        device = "npu:0"
        # Set by load_constraint_tables_on_worker; declared here so mypy can
        # see the dynamic attribute, matching how the real runner exposes it.
        _gr_constraint_execution: Any = None

    runner = FakeRunner()
    loaded = load_constraint_tables_on_worker(runner)

    assert loaded is True
    adapter.load_tables.assert_called_once()
    assert runner._gr_constraint_execution is adapter
    # Idempotent: second call reuses the cached adapter, no rebuild.
    adapter.reset_mock()
    assert load_constraint_tables_on_worker(runner) is True
    adapter.load_tables.assert_not_called()


def test_load_constraint_tables_on_worker_no_device_returns_false(
    tokenizer: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner without a device (mocked) keeps the lazy runtime path."""
    from vllm_gr.constraints import worker_inject
    from vllm_gr.constraints.worker_inject import load_constraint_tables_on_worker

    resource = _build_resource(tokenizer)
    manager = ConstraintResourceManager()
    manager.ensure(resource)
    monkeypatch.setattr(worker_inject, "get_worker_manager", lambda: manager)

    class FakeRunner:
        device = None

    runner = FakeRunner()
    assert load_constraint_tables_on_worker(runner) is False
    assert not hasattr(runner, "_gr_constraint_execution")


def test_load_constraint_tables_on_worker_nothing_injected_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.constraints import worker_inject
    from vllm_gr.constraints.worker_inject import load_constraint_tables_on_worker

    monkeypatch.setattr(worker_inject, "get_worker_manager", lambda: ConstraintResourceManager())

    class FakeRunner:
        device = "npu:0"

    runner = FakeRunner()
    assert load_constraint_tables_on_worker(runner) is False


@pytest.mark.asyncio  # type: ignore[misc]
async def test_inject_constraint_resources_async_forwards_to_collective_rpc_async(
    tokenizer: MagicMock,
) -> None:
    resource = _build_resource(tokenizer)
    engine_core = MagicMock()

    async def fake_async(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"digest": resource.digest, "cached": True}]

    engine_core.collective_rpc_async.side_effect = fake_async
    engine = SimpleNamespace(engine_core=engine_core)

    from vllm_gr.constraints.driver_inject import inject_constraint_resources_async

    statuses = await inject_constraint_resources_async(engine, resource)

    engine_core.collective_rpc_async.assert_called_once_with(
        "set_gr_constraint_resources", args=(to_wire(resource),)
    )
    assert statuses == [{"digest": resource.digest, "cached": True}]


def test_wire_survives_untyped_msgpack_roundtrip(tokenizer: MagicMock) -> None:
    """Simulate vLLM v1's untyped MsgpackDecoder on the wire payload.

    vLLM encodes utility payloads with MsgpackEncoder and decodes them with an
    untyped MsgpackDecoder (no type hint), so a dataclass would arrive as a
    plain dict with nested lists.  A to_wire() payload must survive that and
    still rebuild via from_wire().
    """
    import msgspec

    resource = _build_resource(tokenizer)
    wire = to_wire(resource)

    # Encode + untyped decode (no type arg) — mirrors EngineCore generic decoder.
    encoded = msgspec.msgpack.encode(wire)
    decoded = msgspec.msgpack.decode(encoded)

    # The decoded object must be plain base types (dict/bytes/list/int/str).
    assert isinstance(decoded, dict)
    assert isinstance(decoded["array_fields"]["prefix1_values"]["data"], bytes)
    assert isinstance(decoded["array_fields"]["prefix1_values"]["shape"], list)

    rebuilt = from_wire(decoded)
    assert rebuilt.digest == resource.digest
    assert np.array_equal(rebuilt.prefix1_values, resource.prefix1_values)
    assert not rebuilt.prefix1_values.flags.writeable


def test_inject_verifies_all_worker_acks(tokenizer: MagicMock) -> None:
    """The driver must fail when a worker ack is missing/mismatched."""
    from vllm_gr.constraints.driver_inject import _verify_injected

    resource = _build_resource(tokenizer)
    ok = [{"digest": resource.digest, "cached": True}]

    _verify_injected(resource, ok)  # must not raise

    with pytest.raises(RuntimeError, match="no worker"):
        _verify_injected(resource, [])
    with pytest.raises(RuntimeError, match="non-dict"):
        _verify_injected(resource, ["x"])
    with pytest.raises(RuntimeError, match="expected"):
        _verify_injected(resource, [{"digest": "deadbeef"}])
    with pytest.raises(RuntimeError, match="expected"):
        _verify_injected(resource, [{"cached": True}])  # missing digest


pytestmark = pytest.mark.cpu_test
