# SPDX-License-Identifier: Apache-2.0
"""Real native ticket tests with a CPU-only fake registration API, never CUDA."""

import ctypes
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from tests.test_gr_completion_build import built_completion  # noqa: F401
from vllm_gr.v1.worker import cuda_completion as completion


@pytest.fixture
def api_factory(built_completion):  # noqa: F811 - imported pytest fixture
    library = ctypes.CDLL(str(built_completion))
    library.gr_ticket_callback.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    library.gr_ticket_callback.restype = None
    registrar_type = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint
    )

    def create(*, failure=0, immediate=False):
        pending = []

        def register(stream, callback, pointer, flags):
            # Only registration is simulated in Python, on this CPU test's
            # calling thread. The completion trampoline itself is native.
            if failure:
                return failure
            if immediate:
                library.gr_ticket_callback(stream, 0, pointer)
            else:
                pending.append((stream, pointer))
            return 0

        runtime = SimpleNamespace(cudaStreamAddCallback=registrar_type(register))
        api = completion.NativeCompletionAPI(library, runtime)

        def finish(status=0):
            stream, pointer = pending.pop(0)
            library.gr_ticket_callback(stream, status, pointer)

        return api, pending, finish

    yield create
    assert library.gr_ticket_live() == 0, "native ticket ownership leaked"


def test_timeout_cancel_and_late_completion_do_not_release_dma(api_factory):
    api, pending, finish = api_factory()
    ticket = api.new_ticket(23)
    try:
        ticket.register(0)  # Default stream pointer is valid.
        assert not ticket.wait_device(0)
        ticket.cancel_delivery()
        ticket.cancel_delivery()
        with pytest.raises(CancelledError):
            ticket.wait_delivery(0)
        assert not ticket.wait_device(0)
        assert not ticket.snapshot().device_succeeded
        finish()
        assert ticket.wait_device(0)
        with pytest.raises(CancelledError):
            ticket.wait_delivery(0)
        assert not pending
    finally:
        ticket.close()


def test_immediate_callback_and_claimed_delivery_are_stable(api_factory):
    api, pending, _ = api_factory(immediate=True)
    ticket = api.new_ticket((1 << 64) - 1)
    try:
        ticket.register(1234)
        assert ticket.wait_device(0)
        assert ticket.wait_delivery(0)
        ticket.cancel_delivery()
        assert ticket.wait_delivery(0)
        snapshot = ticket.snapshot()
        assert snapshot.device_succeeded and snapshot.delivery_claimed
        assert not snapshot.delivery_cancelled and not pending
        with pytest.raises(completion.CompletionRegistrationError, match="-2"):
            ticket.register(1234)
        assert ticket.wait_device(0)  # Duplicate rejection did not corrupt registration.
    finally:
        assert ticket.close()
        assert not ticket.close()


@pytest.mark.parametrize("cancel", [False, True])
def test_unbounded_notification_wait_preserves_completion_and_cancellation(api_factory, monkeypatch, cancel):
    api, _, finish = api_factory()
    ticket = api.new_ticket(99)
    ticket.register(0)
    entered = threading.Event()
    original = api.lib.gr_ticket_wait

    def wait(pointer, timeout, device):
        assert timeout == -1 and device == 0
        entered.set()
        return original(pointer, timeout, device)

    monkeypatch.setattr(api.lib, "gr_ticket_wait", wait)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(ticket.wait_delivery, None)
            assert entered.wait(5)
            if cancel:
                ticket.cancel_delivery()
                with pytest.raises(CancelledError):
                    result.result(timeout=5)
                finish()
            else:
                finish()
                assert result.result(timeout=5)
    finally:
        ticket.close()


@pytest.mark.parametrize("kind", ["registration", "device"])
def test_failed_notification_never_returns_success(api_factory, kind):
    api, pending, finish = api_factory(failure=801 if kind == "registration" else 0)
    ticket = api.new_ticket(17)
    try:
        if kind == "registration":
            error = completion.CompletionRegistrationError
            with pytest.raises(error, match="801"):
                ticket.register(0)
        else:
            error = completion.CompletionDeviceError
            ticket.register(0)
            finish(719)
        for wait in (ticket.wait_device, ticket.wait_delivery):
            with pytest.raises(error):
                wait(0)
        assert not ticket.snapshot().device_succeeded and not pending
    finally:
        ticket.close()


def test_close_owner_preserves_pending_callback_reference(api_factory):
    api, _, finish = api_factory()
    ticket = api.new_ticket(42)
    ticket.register(0)
    assert ticket.close()
    assert api.lib.gr_ticket_live() == 1
    with pytest.raises(RuntimeError, match="closed"):
        ticket.wait_device(0)
    finish()
    assert api.lib.gr_ticket_live() == 0


def test_close_callback_and_waiter_have_independent_native_refs(api_factory, monkeypatch):
    api, _, finish = api_factory()
    ticket = api.new_ticket(7)
    ticket.register(0)
    entering_wait = threading.Event()
    real_wait = api.lib.gr_ticket_wait

    def wait(*args):
        entering_wait.set()  # Ticket._borrow has already acquired its own reference.
        return real_wait(*args)

    monkeypatch.setattr(api.lib, "gr_ticket_wait", wait)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(ticket.wait_device, 5000)
        assert entering_wait.wait(5)
        ticket.close()
        finish()
        assert result.result(timeout=5) is True
    assert api.lib.gr_ticket_live() == 0


@pytest.mark.parametrize("value", [-1, True, None, 1 << 64])
def test_generation_cannot_wrap_native_uint64(api_factory, value):
    api, _, _ = api_factory()
    with pytest.raises(ValueError, match="generation"):
        api.new_ticket(value)


@pytest.mark.parametrize("value", [-1, True, 1 << 31])
def test_timeout_cannot_wrap_native_int(api_factory, value):
    api, _, _ = api_factory()
    ticket = api.new_ticket(0)
    try:
        with pytest.raises(ValueError, match="timeout_ms"):
            ticket.wait_device(value)
    finally:
        ticket.close()


@pytest.mark.parametrize("value", [-1, True, None, 1 << (ctypes.sizeof(ctypes.c_void_p) * 8)])
def test_stream_pointer_validation_precedes_registration(api_factory, value):
    api, pending, _ = api_factory()
    ticket = api.new_ticket(0)
    try:
        with pytest.raises(ValueError, match="stream pointer"):
            ticket.register(value)
        assert not pending
    finally:
        ticket.close()


def _loader_fixture(monkeypatch, api):
    imports, handles = [], []
    extension = SimpleNamespace(API_VERSION=2, __file__="/native-completion.so")
    torch = SimpleNamespace(version=SimpleNamespace(cuda="13.0"),
                            _C=SimpleNamespace(__file__="/torch-linked-runtime.so"))

    def load_module(name):
        imports.append(name)
        return torch if name == "torch" else extension

    def load_library(path):
        handles.append(path)
        return api.runtime if path == torch._C.__file__ else api.lib

    monkeypatch.setattr(completion, "_loaded_api", None)
    monkeypatch.setattr(completion, "import_module", load_module)
    monkeypatch.setattr(completion.ctypes, "CDLL", load_library)
    return torch, extension, imports, handles


def test_loader_is_lazy_cached_and_uses_torch_dependency_handle(api_factory, monkeypatch):
    api, _, _ = api_factory()
    _, _, imports, handles = _loader_fixture(monkeypatch, api)
    first = completion.load_native_completion_api()
    assert completion.load_native_completion_api() is first
    assert imports == ["torch", "vllm_gr.ops.cuda._completion"]
    assert handles == ["/native-completion.so", "/torch-linked-runtime.so"]


@pytest.mark.parametrize("kind", ["cpu_torch", "module_version", "c_version", "missing_symbol"])
def test_loader_rejects_incompatible_capability_without_fallback(api_factory, monkeypatch, kind):
    api, _, _ = api_factory()
    torch, extension, _, handles = _loader_fixture(monkeypatch, api)
    if kind == "cpu_torch":
        torch.version.cuda = None
    elif kind == "module_version":
        extension.API_VERSION = True  # bool is not an integer protocol version.
    elif kind == "c_version":
        monkeypatch.setattr(api.lib, "gr_ticket_api_version", lambda: 1)
    else:
        del api.runtime.cudaStreamAddCallback
    with pytest.raises(RuntimeError):
        completion.load_native_completion_api()
    assert completion._loaded_api is None
    if kind in ("cpu_torch", "module_version"):
        assert not handles


def test_missing_installed_extension_has_actionable_error(monkeypatch):
    monkeypatch.setattr(completion, "_loaded_api", None)

    def load(name):
        if name == "torch":
            return SimpleNamespace(version=SimpleNamespace(cuda="13.0"))
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(completion, "import_module", load)
    with pytest.raises(RuntimeError, match="build/install"):
        completion.load_native_completion_api()
