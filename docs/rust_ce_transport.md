# Native CE transport

The optional native bridge moves CE PUSH/PULL socket ownership, polling, send
retries, and the bounded TX queue into Rust. The scheduler submits metadata and
numeric buffers directly; Rust expands grouped rows, packs arrays and encodes v6
messages. The existing multipart protocol is retained. Native Rust CE is
currently activated only by `VLLM_GR_CE_BP_TRANSPORT=sharedmem`, which uses
shared memory in both directions between CE and BP. The default
`VLLM_GR_CE_BP_TRANSPORT=zmq` path continues to use Python CE threads; it does
not use the native Rust bridge. See the
[shared-memory transport](sharedmem_ce_transport.md) documentation.

CE drains the bounded native RX inbox on its scheduler thread. Rust RX never
calls Python on this path. Decoding, plan application, and child admission run
on the scheduler; native replies bypass the Python PB queue. The PB queue is
retained for compatibility with other producers and the Python transport.
The Rust CE scheduler uses `os.sched_yield()` when idle, checking both AS input
and BP replies. This may consume an entire CPU core while idle. Python CE mode
retains its 5 microsecond scheduler sleep. AS communication, including final
results sent directly from BP to AS, continues to use ZMQ.

Build from the checkout:

```bash
cargo build --release --lib --bin beam-proc --locked --features iceoryx --manifest-path rust/Cargo.toml
```

Set `VLLM_GR_CE_BP_TRANSPORT=sharedmem` on the existing BeamProc server command
to enable Rust CE workers and shared-memory IPC together. The default `zmq`
uses Python CE threads. There is no separate thread-mode selector.
The library loads automatically. `VLLM_GR_CE_TRANSPORT_LIBRARY` can select an
explicit shared library; otherwise the loader checks package `bin` and the local
Rust release directory. Wheel builds include the native library with BeamProc.

The workspace pins iceoryx2 to exactly `0.9.3`. `beam-proc` and
`libce_transport.so` communicate through iceoryx2 services with shared layout
and protocol assumptions, so both artifacts must be built from the same Cargo
workspace and lockfile. Do not combine artifacts compiled against different
iceoryx2 versions.

Each socket is created, used, and destroyed on its native owner thread. Sending
copies input frames into Rust-owned memory before returning, so the Python buffer
pool can reuse its arrays immediately. Receiving copies frames into Python-owned
bytes before releasing native storage, so Python requests/array views can retain
them. Blocking native calls release the GIL; Python callbacks reacquire it.

Worker failures reach the CE loop through health checks. Shutdown stops native
workers and joins socket owners; a full RX inbox does not prevent shutdown.
An absent peer or a full send queue fails after a bounded timeout. Create the
bridge after process startup; using it after fork is rejected. The existing
ABI 3 pull API is reused, so this change does not require an ABI change.
Optional callbacks remain available for diagnostics/comparisons and retain
strong ownership until close; production CE does not enable them.

CPU validation uses real local IPC peers in `tests/test_native_ce_transport.py`,
including multipart ownership/order, backpressure, concurrent shutdown, pool
reuse, final cleanup, and a live v5 plan through the existing application path.
Synthetic CPU timings include queue handoffs but are not OneRec serving latency.
Serving performance must be measured with the intended workload and matched
server settings. Callback lifecycle tests remain in
`tests/test_native_ce_callback.py`.
