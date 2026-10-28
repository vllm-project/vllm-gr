# Experimental shared-memory CE/BP transport

Normal package installation builds and bundles both native components with
shared-memory support. Python uses the bundled library through `ctypes`; no
separate Python iceoryx package is required. Source installations require
Rust/Cargo, while installing a prebuilt wheel does not.

For a manual native build:

```bash
cargo build --release --locked --features iceoryx --manifest-path rust/Cargo.toml
```

Set these variables on the server so both CE and the spawned BeamProc inherit
the choice:

```bash
export VLLM_GR_CE_BP_TRANSPORT=sharedmem
```

This selects Rust CE TX/RX workers and automatically loads the native library.
The loader checks package `bin` before the checkout's `rust/target/release`.
Optional `VLLM_GR_CE_TRANSPORT_LIBRARY` and `VLLM_GR_BEAM_PROC_BINARY` path
overrides can select freshly built components when older packaged files exist.
The `sharedmem` option uses iceoryx2. The default link transport remains `zmq`.
A shared-memory request on a binary built
without the feature fails explicitly. The default `zmq` uses Python CE threads.
The separate `VLLM_GR_CE_TRANSPORT` selector is no longer used.
Ordinary wheel builds enable the `iceoryx` Cargo feature for both components.
`VLLM_GR_SKIP_RUST_BUILD=1` explicitly skips native compilation and does not
provide shared-memory support in a fresh installation.

The workspace pins iceoryx2 to exactly `0.9.3`. Both `beam-proc` and
`libce_transport.so` must be compiled from the same Cargo workspace and
lockfile because the two processes depend on identical iceoryx2 shared-memory
layout and protocol behavior. Mixing native artifacts built against different
iceoryx2 versions is unsupported.

Both CE -> BP and BP -> CE use separate iceoryx2 publish/subscribe services.
In Rust CE mode, CE and BP retain their Rust TX and RX workers. CE submits to its bounded
native TX queue, and Rust prepares batches and sends them. Returning replies
enter the bounded native inbox, then the CE scheduler decodes/applies them and
admits children. APIServer communication continues to use ZMQ.

The internal Python shared-memory adapter remains available for CPU comparisons;
it is not selected by server environment settings. In that adapter, the CE
scheduler enqueues work in a bounded Python TX queue.
The Python TX thread prepares batches and calls synchronous iceoryx2 send via
the native library. The Python RX thread calls synchronous receive, decodes
replies, and enqueues admission work for the scheduler in the bounded Python
reply queue. The synchronous ports create no Rust CE workers or native queues;
ctypes releases the GIL during transport calls. BP still uses its Rust workers.
Shutdown interrupts pending port operations and joins the Python threads;
worker failures propagate through the scheduler's transport health check.

Comparing these modes measures the complete CE preparation and reply-handling
paths, including their queues. It does not isolate the cost of thread ownership
alone: Rust mode also prepares batches in Rust and decodes replies on the scheduler.

BeamProc reports `num_processed_tokens` with its final result. This is the
cumulative number of candidate token entries considered across beam steps,
after sampled-token deduplication and before EOS/finite-score filtering and
beam selection. It includes masked candidates, as in the Python beam-search
counter; it is not the number of generated tokens returned to the user.
The API publishes it as `vllm_gr_beam_search_processed_tokens_total`. Both CE
thread modes use this BP accounting. If a server path does not report the
metric, the benchmark displays `N/A` rather than claiming zero or an outdated
server version.

This prototype follows the upstream
[dynamic-data example](https://github.com/eclipse-iceoryx/iceoryx2/tree/main/examples/rust/publish_subscribe_dynamic_data)
using a shared-memory byte slice. Each sample carries a versioned envelope with
frame lengths and the existing multipart payload. Numeric arrays, MessagePack,
and beam processing are unchanged. Copies into/out of shared memory remain;
this is not an end-to-end zero-copy implementation.

Each service allows one publisher and one subscriber, disables history and
safe overflow, and uses the existing ZMQ HWM value as its bounded sample
capacity. A send to zero recipients is retried until the existing send timeout
or shutdown. With one subscriber this retries absent/full receivers without
partial fan-out or duplicate delivery. Incoming malformed framing and messages
larger than 16 MiB fail explicitly. Each direction also has an iceoryx2 event
service: the sender publishes the sample, then notifies the receiver. Receivers
drain queued data and block on notifications when empty, instead of sleeping
50 us between polls. They drain stale notifications and recheck the data queue
before blocking to avoid lost wakeups. The listener is registered before the
subscriber becomes visible. A 10 ms interruptible wait bounds shutdown and
health-check latency; it is not a fixed message delay. Full/absent receiver send
retries still sleep 50 us. Envelope copies remain; CE scheduler waiting is
configured separately as described below.
Both native components must be rebuilt together for notification wakeups.

The Rust CE scheduler uses `os.sched_yield()` when idle. CE/BP shared-memory
transport notifications wake the receive workers. Python CE thread mode keeps its 5 us
scheduler idle sleep. Yielding may consume an entire CPU core while idle.

Service names, runtime roots, and resource prefixes are derived with a stable
hash of the per-run endpoint strings. Each direction uses a short private
`/tmp/vllm-gr-iox2-<hash>` runtime root, preventing persistent CI jobs or
different users from colliding through iceoryx2's global `/tmp/iceoryx2`
default. The two directions and different server sessions have separate names.
Both processes must share the host's shared-memory and IPC namespace and
permissions. Existing process signal handling is retained. Ordinary endpoint
destruction uses iceoryx2 resource lifetime management; no global shared-memory
cleanup is performed. This prototype supports a single CE per BP endpoint.

CPU integration tests use a separate Rust echo process, including both transport
directions, data ownership, FIFO order, restart, full/absent peers, peer death,
backpressure recovery, and bounded close:

```bash
cargo build --release --locked --features iceoryx --lib --bin beam-proc \
  --example iceoryx-peer --manifest-path rust/Cargo.toml
CUDA_VISIBLE_DEVICES='' VLLM_GR_CE_BP_TRANSPORT=sharedmem \
  VLLM_GR_CE_TRANSPORT_LIBRARY="$PWD/rust/target/release/libce_transport.so" \
  python -m pytest tests/test_native_ce_iceoryx.py -q --timeout=25
```

Run IPC tests in an environment permitting shared-memory IPC. OneRec serving
validation uses the explicitly requested prompt count; current latency
comparisons use 5 warmup prompts followed by 1,000 measured prompts at C1/BW128.
