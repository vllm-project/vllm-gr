# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm_gr.constraints.execution import INVALID_LOGIT, valid_candidate_mask
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
    from vllm_gr.v1.worker.model_runner_common import _WORKER_SAMPLE_PATCH_MARKER

    resource = _build_resource(tokenizer)
    wire = to_wire(resource)

    class FakeWorker:
        model_runner = object()  # a runner class without _sample

    manager = ConstraintResourceManager()
    with pytest.raises(RuntimeError, match="raw-logits hook"):
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
    logprobs = torch.tensor(
        [[-0.5, INVALID_LOGIT, -2.0], [INVALID_LOGIT, INVALID_LOGIT, -1.0]],
        dtype=torch.float32,
    )
    mask = valid_candidate_mask(logprobs)

    assert mask.dtype == torch.bool
    assert mask.shape == logprobs.shape
    assert mask.tolist() == [
        [True, False, True],
        [False, False, True],
    ]


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


def test_try_import_npu_worker_prefers_worker_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    """_try_import_npu_worker must resolve worker_v1.NPUWorker over worker.NPUWorker.

    Simulates both modules being importable by patching builtins.__import__ and
    asserts the v1 path wins (and that only v1 is attempted when it succeeds).
    """
    import builtins

    import vllm_gr.v1.ascend_compat as ascend_compat

    class V1Worker:
        pass

    class LegacyWorker:
        pass

    v1_module = SimpleNamespace(NPUWorker=V1Worker)
    legacy_module = SimpleNamespace(NPUWorker=LegacyWorker)
    calls: list[str] = []
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "vllm_ascend.worker.worker_v1":
            calls.append(name)
            return v1_module
        if name == "vllm_ascend.worker.worker":
            calls.append(name)
            return legacy_module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cls = ascend_compat._try_import_npu_worker()

    assert cls is V1Worker
    assert calls == ["vllm_ascend.worker.worker_v1"]


def test_try_import_npu_worker_falls_back_to_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """When worker_v1 is missing, the legacy worker path must be attempted."""
    import builtins

    import vllm_gr.v1.ascend_compat as ascend_compat

    class LegacyWorker:
        pass

    legacy_module = SimpleNamespace(NPUWorker=LegacyWorker)
    calls: list[str] = []
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "vllm_ascend.worker.worker_v1":
            calls.append(name)
            raise ImportError("no v1")
        if name == "vllm_ascend.worker.worker":
            calls.append(name)
            return legacy_module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cls = ascend_compat._try_import_npu_worker()

    assert cls is LegacyWorker
    assert calls == ["vllm_ascend.worker.worker_v1", "vllm_ascend.worker.worker"]


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
