# SPDX-License-Identifier: Apache-2.0
"""Experimental native notification lifetimes, not ModelRunner integration.

Build the companion C++ file as a shared library first. Run with exactly one
visible GPU. This success/cancellation test does not inject a real CUDA fault.
No callback executes Python, calls CUDA or owns a torch tensor.
"""

import argparse
from contextlib import contextmanager
import ctypes
import json
from pathlib import Path
import threading

import torch


class NativeAPI:
    def __init__(self, library, runtime):
        self.lib = ctypes.CDLL(str(library))
        self.runtime = ctypes.CDLL(str(runtime))
        signatures = {
            "gr_ticket_create": ([ctypes.c_uint64], ctypes.c_void_p),
            "gr_ticket_retain": ([ctypes.c_void_p], None),
            "gr_ticket_release": ([ctypes.c_void_p], None),
            "gr_ticket_live": ([], ctypes.c_int),
            "gr_ticket_register": ([ctypes.c_void_p] * 3, ctypes.c_int),
            "gr_ticket_cancel_delivery": ([ctypes.c_void_p], None),
            "gr_ticket_wait": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int),
            "gr_ticket_snapshot": (
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int)],
                ctypes.c_int,
            ),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        self.register_address = ctypes.cast(self.runtime.cudaStreamAddCallback, ctypes.c_void_p)


class Ticket:
    def __init__(self, api, generation):
        self.api = api  # Keep the native library loaded, including while waiting.
        self._lock = threading.Lock()
        self._pointer = api.lib.gr_ticket_create(generation)
        if not self._pointer:
            raise MemoryError("native completion ticket allocation failed")

    @contextmanager
    def borrow(self):
        # Reserve a per-call reference while the owner is still protected.
        # close() may then run concurrently without destroying a waiter's CV.
        with self._lock:
            pointer = self._pointer
            if pointer is None:
                raise RuntimeError("ticket owner already closed")
            self.api.lib.gr_ticket_retain(pointer)
        try:
            yield pointer
        finally:
            self.api.lib.gr_ticket_release(pointer)

    def register(self, stream):
        with self.borrow() as pointer:
            return self.api.lib.gr_ticket_register(pointer, stream, self.api.register_address)

    def wait(self, timeout_ms, *, device=False):
        if timeout_ms < 0:
            raise ValueError("wait timeout must be nonnegative")
        with self.borrow() as pointer:
            return self.api.lib.gr_ticket_wait(pointer, timeout_ms, int(device))

    def cancel_delivery(self):
        with self.borrow() as pointer:
            self.api.lib.gr_ticket_cancel_delivery(pointer)

    def snapshot(self):
        generation, status = ctypes.c_uint64(), ctypes.c_int()
        with self.borrow() as pointer:
            flags = self.api.lib.gr_ticket_snapshot(
                pointer, ctypes.byref(generation), ctypes.byref(status)
            )
        return {
            "generation": generation.value,
            "status": status.value,
            "device_done": bool(flags & 1),
            "delivery_cancelled": bool(flags & 2),
            "registration_failed": bool(flags & 4),
            "delivery_consumed": bool(flags & 8),
        }

    def close(self):
        with self._lock:
            pointer, self._pointer = self._pointer, None
        if pointer is None:
            return False
        self.api.lib.gr_ticket_release(pointer)
        return True


def main():
    if not __debug__:
        raise RuntimeError("this diagnostic requires assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    api = NativeAPI(args.library, args.runtime)
    assert api.lib.gr_ticket_live() == 0
    compute, copy = torch.cuda.Stream(), torch.cuda.Stream()
    buffers = [torch.empty(1, device="cuda", dtype=torch.int32) for _ in range(2)]
    hosts = [torch.empty(1, pin_memory=True, dtype=torch.int32) for _ in range(2)]
    # This toy pending list models two leases only. It is not installed as a
    # new EngineCore queue, nor does it replace native BatchQueue/Future.
    pending, all_tickets, reports = [], [], []
    owners = [None, None]
    events = []
    timeout_observed = False
    premature_reuse_rejections = 0
    torch.cuda.synchronize()

    def drain():
        stage, slot, ticket = pending.pop(0)
        assert ticket.wait(10000, device=True) == 0
        snapshot = ticket.snapshot()
        assert snapshot["device_done"] and snapshot["status"] == 0
        assert snapshot["generation"] == stage and owners[slot] == stage
        observed = ctypes.c_int32.from_address(hosts[slot].data_ptr()).value
        assert observed == stage
        assert ticket.wait(0) == (2 if snapshot["delivery_cancelled"] else 0)
        snapshot = ticket.snapshot()
        # Host has now consumed or explicitly discarded the delivery, and
        # device completion is proven. Only now may this slot be reused.
        reports.append({"stage": stage, "slot": slot, "control": observed, **snapshot})
        owners[slot] = None
        assert ticket.close() and not ticket.close()

    try:
        for stage in range(8):
            slot = stage % 2
            if len(pending) == 2:
                assert owners[slot] is not None
                premature_reuse_rejections += 1
                drain()
            assert owners[slot] is None
            owners[slot] = stage
            event = torch.cuda.Event()
            events.append(event)
            with torch.cuda.stream(compute):
                torch.cuda._sleep(100_000_000 if stage == 1 else 2_000_000)
                buffers[slot].fill_(stage)
                event.record()
            with torch.cuda.stream(copy):
                copy.wait_event(event)
                hosts[slot].copy_(buffers[slot], non_blocking=True)
            ticket = Ticket(api, stage)
            all_tickets.append(ticket)
            status = ticket.register(copy.cuda_stream)
            if status:
                raise RuntimeError(f"callback registration failed: {status}; buffers require drain")
            assert ticket.register(copy.cuda_stream) == -2  # No duplicate CUDA registration.
            pending.append((stage, slot, ticket))
            if stage == 1:
                timeout_observed = ticket.wait(0, device=True) == 1
                assert timeout_observed, (
                    "device completed before timeout probe; timeout case unverified"
                )
                ticket.cancel_delivery()
                ticket.cancel_delivery()
                assert ticket.wait(0) == 2
                assert not ticket.snapshot()["device_done"]
                assert owners[slot] == stage  # Cancellation has NOT released DMA ownership.
        while pending:
            drain()
    finally:
        # Success/ordinary Python failure: retain library, tickets and tensor
        # owners through the final drain. A real CUDA/context failure remains
        # outside this experiment; it needs a separate process-level protocol.
        torch.cuda.synchronize()
        for ticket in all_tickets:
            ticket.close()
    assert api.lib.gr_ticket_live() == 0 and owners == [None, None]
    print(
        json.dumps(
            {
                "scope": "native completion success/cancel/timeout/two-slot reuse; not V1 integration or performance",
                "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda,
                "tickets_created": len(all_tickets),
                "tickets_live_after_drain": api.lib.gr_ticket_live(),
                "real_timeout_observed": timeout_observed,
                "premature_reuse_rejections": premature_reuse_rejections,
                "premature_slot_reuse": 0,
                "wrong_generation_updates": 0,
                "reports": reports,
                "limitations": [
                    "no real CUDA fault injection",
                    "no input staging leases",
                    "shutdown/context-failure drain not accepted",
                    "Host wait remains",
                    "runtime callback API scheduled for eventual removal",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
