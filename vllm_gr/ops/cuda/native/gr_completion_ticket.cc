// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// CUDA-private completion primitive; no ModelRunner integration here.
// The ticket/callback contains no CUDA/Python calls and owns no tensor resources.
// Every caller must hold a strong reference throughout a native API call.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <new>

namespace {
std::atomic<int> live{0};
struct Ticket {
  std::atomic<int> refs{1};  // Owner; registration reserves callback ownership.
  std::mutex mutex;
  std::condition_variable changed;
  const uint64_t generation;
  bool device_done = false;
  bool cancelled = false;
  bool delivered = false;
  bool registration_failed = false;
  bool registration_started = false;
  int status = 0;
  explicit Ticket(uint64_t value) : generation(value) { ++live; }
  ~Ticket() { --live; }
};
void release(Ticket* ticket) {
  if (ticket->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete ticket;
}
}  // namespace

extern "C" {
int gr_ticket_api_version() { return 2; }
void* gr_ticket_create(uint64_t generation) {
  // nothrow handles allocation failure, not exceptions from e.g. the
  // condition_variable constructor. Never unwind through the ctypes C ABI.
  try {
    return new (std::nothrow) Ticket(generation);
  } catch (...) {
    return nullptr;
  }
}
void gr_ticket_retain(void* pointer) {
  static_cast<Ticket*>(pointer)->refs.fetch_add(1, std::memory_order_relaxed);
}
void gr_ticket_release(void* pointer) { release(static_cast<Ticket*>(pointer)); }
int gr_ticket_live() { return live.load(); }

// Registration is single-shot. Caller holds its own ref until the API returns.
// Reserving before the external API call also protects immediate callbacks.
int gr_ticket_begin_registration(void* pointer) {
  auto* ticket = static_cast<Ticket*>(pointer);
  std::lock_guard<std::mutex> guard(ticket->mutex);
  if (ticket->registration_started) return -2;
  ticket->registration_started = true;
  ticket->refs.fetch_add(1, std::memory_order_relaxed);
  return 0;
}

// ABI-compatible cudaStreamCallback_t. This function must be called exactly
// once for successful registration, and never for failed registration.
void gr_ticket_callback(void*, int status, void* pointer) {
  auto* ticket = static_cast<Ticket*>(pointer);
  {
    std::lock_guard<std::mutex> guard(ticket->mutex);
    ticket->status = status;
    ticket->device_done = true;
  }
  ticket->changed.notify_all();
  release(ticket);  // Last access, after unlocking and notifying waiters.
}

void gr_ticket_registration_failed(void* pointer, int status) {
  auto* ticket = static_cast<Ticket*>(pointer);
  {
    std::lock_guard<std::mutex> guard(ticket->mutex);
    ticket->status = status;
    ticket->registration_failed = true;
    // Registration failure does NOT establish that prior H2D/D2H is done.
  }
  ticket->changed.notify_all();
  release(ticket);  // Drop the unused callback reference exactly once.
}

// Pass the loaded runtime's cudaStreamAddCallback address; no CUDA headers,
// Python callback or CUDA call is needed inside the completion trampoline.
using StreamCallback = void (*)(void*, int, void*);
using RegisterCallback = int (*)(void*, StreamCallback, void*, unsigned);
int gr_ticket_register(void* pointer, void* stream, RegisterCallback register_fn) {
  int begin = gr_ticket_begin_registration(pointer);
  if (begin != 0) return begin;
  int status = register_fn ? register_fn(stream, gr_ticket_callback, pointer, 0) : 801;
  if (status != 0) gr_ticket_registration_failed(pointer, status);
  return status;
}

void gr_ticket_cancel_delivery(void* pointer) {
  auto* ticket = static_cast<Ticket*>(pointer);
  {
    std::lock_guard<std::mutex> guard(ticket->mutex);
    // Once a consumer has observed completion, delivery is immutable. A
    // cancellation before consumption still does not imply device completion.
    if (!ticket->delivered) ticket->cancelled = true;
  }
  ticket->changed.notify_all();
}

// 0: device completed; 1: timeout; 2: delivery cancelled; 3: registration failed.
// Device status is read separately. Device wait ignores delivery cancellation.
int gr_ticket_wait(void* pointer, int timeout_ms, int wait_device) {
  auto* ticket = static_cast<Ticket*>(pointer);
  std::unique_lock<std::mutex> lock(ticket->mutex);
  auto ready = [&] {
    return ticket->device_done || ticket->registration_failed ||
           (!wait_device && ticket->cancelled);
  };
  if (timeout_ms == -1) {
    ticket->changed.wait(lock, ready);
  } else if (!ticket->changed.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready)) {
    return 1;
  }
  if (ticket->registration_failed) return 3;
  if (!wait_device && ticket->cancelled) return 2;
  if (!wait_device) ticket->delivered = true;
  return 0;
}

// Snapshot atomically: done=1, cancelled=2, reg-failed=4, delivery consumed=8.
int gr_ticket_snapshot(void* pointer, uint64_t* generation, int* status) {
  auto* ticket = static_cast<Ticket*>(pointer);
  std::lock_guard<std::mutex> guard(ticket->mutex);
  *generation = ticket->generation;
  *status = ticket->status;
  return int(ticket->device_done) | (int(ticket->cancelled) << 1) |
         (int(ticket->registration_failed) << 2) | (int(ticket->delivered) << 3);
}
}  // extern "C"


#ifndef GR_COMPLETION_STANDALONE
// A real importable CPython module, while ctypes consumers use the C ABI.
// No PyTorch/CUDA imports, CUDA headers, NVCC, or pybind dependency.
#include <Python.h>
namespace {
PyModuleDef completion_module = {
    PyModuleDef_HEAD_INIT,
    "_completion",
    "CUDA-private native completion ticket ABI.",
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};
}
PyMODINIT_FUNC PyInit__completion() {
  PyObject* module = PyModule_Create(&completion_module);
  if (module == nullptr) return nullptr;
  if (PyModule_AddIntConstant(module, "API_VERSION", gr_ticket_api_version()) < 0) {
    Py_DECREF(module);
    return nullptr;
  }
  return module;
}
#endif
