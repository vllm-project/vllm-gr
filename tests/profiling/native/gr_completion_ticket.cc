// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Standalone diagnostic/ASan entrypoint; production is the sole ABI source.
#define GR_COMPLETION_STANDALONE 1
#include "../../../vllm_gr/ops/cuda/native/gr_completion_ticket.cc"

#ifdef GR_COMPLETION_TEST_MAIN
#include <cassert>
#include <future>
#include <iostream>
#include <thread>

int main() {
  // Cancellation/timeout never grants device completion or permits slot reuse.
  auto* cancelled = gr_ticket_create(17);
  assert(gr_ticket_begin_registration(cancelled) == 0);
  assert(gr_ticket_begin_registration(cancelled) == -2);
  assert(gr_ticket_wait(cancelled, 0, 0) == 1);
  gr_ticket_cancel_delivery(cancelled);
  gr_ticket_cancel_delivery(cancelled);
  assert(gr_ticket_wait(cancelled, 0, 0) == 2);
  assert(gr_ticket_wait(cancelled, 0, 1) == 1);
  uint64_t generation;
  int status;
  assert(gr_ticket_snapshot(cancelled, &generation, &status) == 2);
  assert(generation == 17);
  gr_ticket_callback(nullptr, 0, cancelled);
  assert(gr_ticket_wait(cancelled, 0, 1) == 0);
  assert(gr_ticket_snapshot(cancelled, &generation, &status) == 3);
  gr_ticket_release(cancelled);

  auto* failed = gr_ticket_create(18);
  assert(gr_ticket_register(failed, nullptr, nullptr) == 801);
  assert(gr_ticket_wait(failed, 0, 1) == 3);
  assert(gr_ticket_snapshot(failed, &generation, &status) == 4);
  assert(status == 801);  // No device_done bit despite the registration error.
  gr_ticket_release(failed);

  auto* immediate = gr_ticket_create(19);
  auto inline_register = +[](void* stream, StreamCallback callback, void* pointer, unsigned) {
    callback(stream, 0, pointer);
    return 0;
  };
  assert(gr_ticket_register(immediate, nullptr, inline_register) == 0);
  assert(gr_ticket_register(immediate, nullptr, inline_register) == -2);
  assert(gr_ticket_wait(immediate, 0, 1) == 0);
  assert(gr_ticket_wait(immediate, 0, 0) == 0);
  gr_ticket_cancel_delivery(immediate);
  assert(gr_ticket_wait(immediate, 0, 0) == 0);  // Consumed terminal result cannot change.
  assert(gr_ticket_snapshot(immediate, &generation, &status) == 9);
  gr_ticket_release(immediate);

  auto* never_registered = gr_ticket_create(20);
  gr_ticket_release(never_registered);  // No reserved callback ref to leak.

  // Deterministic close-during-registration: registrar borrows independently,
  // owner closes after reservation, then completion is permitted to run.
  auto* closing = gr_ticket_create(21);
  gr_ticket_retain(closing);
  std::promise<void> reserved, permit_callback;
  auto reservation = reserved.get_future();
  auto permission = permit_callback.get_future();
  std::thread registrar([&] {
    assert(gr_ticket_begin_registration(closing) == 0);
    reserved.set_value();
    permission.wait();
    gr_ticket_callback(nullptr, 0, closing);
    assert(gr_ticket_wait(closing, 0, 1) == 0);
    gr_ticket_release(closing);
  });
  reservation.wait();
  gr_ticket_release(closing);
  permit_callback.set_value();
  registrar.join();

  // A waiter has its OWN strong reference; owner close and callback completion
  // may race without destroying its condition_variable or mutex underneath it.
  for (int i = 0; i < 1000; ++i) {
    auto* ticket = gr_ticket_create(i);
    assert(gr_ticket_begin_registration(ticket) == 0);
    gr_ticket_retain(ticket);  // Reserve waiter's reference before spawning.
    std::promise<void> started;
    auto signal = started.get_future();
    std::thread waiter([&] {
      started.set_value();
      assert(gr_ticket_wait(ticket, 5000, 1) == 0);
      uint64_t observed_generation;
      int observed_status;
      assert(gr_ticket_snapshot(ticket, &observed_generation, &observed_status) == 1);
      assert(observed_generation == uint64_t(i) && observed_status == 719);
      gr_ticket_release(ticket);
    });
    signal.wait();
    gr_ticket_release(ticket);  // Owner no longer keeps ticket alive.
    gr_ticket_callback(nullptr, 719, ticket);  // Simulated device error notification.
    waiter.join();
  }
  assert(gr_ticket_live() == 0);
  std::cout << "PASS: cancellation, timeout, registration failure, 1000 waiter/owner/callback races; live=0\n";
}
#endif
