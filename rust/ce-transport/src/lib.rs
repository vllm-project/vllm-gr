//! CE-side ZMQ ownership. The private C ABI copies submitted frames before
//! returning. RX can synchronously invoke an explicitly retained C callback.
//! Handles are registry IDs so close can race with a blocked send/receive.

use crossbeam_channel::{Receiver, Sender, bounded};
use std::collections::HashMap;
use std::ffi::{CStr, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

mod batch;

#[cfg(feature = "iceoryx")]
pub mod iceoryx;

/// Select the transport for both directions of the CE <-> BP link.
pub fn ce_bp_iceoryx_enabled() -> Result<bool, String> {
    match std::env::var("VLLM_GR_CE_BP_TRANSPORT").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("zmq") => Ok(false),
        Ok("sharedmem") if cfg!(feature = "iceoryx") => Ok(true),
        Ok("sharedmem") => {
            Err("rebuild both native CE and BeamProc with --features iceoryx".into())
        }
        _ => Err("VLLM_GR_CE_BP_TRANSPORT must be zmq or sharedmem".into()),
    }
}

type Frames = Vec<Vec<u8>>;
const POLL_MS: i64 = 10;
const TICK: Duration = Duration::from_millis(POLL_MS as u64);
const MAX_FRAMES: usize = 64;
const MAX_BYTES: usize = 256 * 1024 * 1024;

enum Outbound {
    Frames(Frames),
    Batch { metadata: Vec<u8>, buffers: Frames },
}

#[derive(Default)]
struct State {
    stop: AtomicBool,
    receive_enabled: AtomicBool,
    error: Mutex<Option<String>>,
}

impl State {
    fn fail(&self, error: impl ToString) {
        let mut slot = self.error.lock().unwrap();
        if slot.is_none() {
            *slot = Some(error.to_string());
        }
        self.stop.store(true, Ordering::Release);
    }

    fn check(&self) -> Result<(), String> {
        if let Some(error) = self.error.lock().unwrap().as_ref() {
            return Err(error.clone());
        }
        if self.stop.load(Ordering::Acquire) {
            return Err("CE transport closed".into());
        }
        Ok(())
    }
}

struct Bridge {
    state: Arc<State>,
    tx: Option<Sender<Outbound>>,
    rx: Option<Receiver<Frames>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    timeout: Duration,
    #[cfg(feature = "iceoryx")]
    direct: Option<DirectPorts>,
}

#[cfg(feature = "iceoryx")]
struct DirectPorts {
    tx: Mutex<Option<iceoryx::FrameSender>>,
    rx: Mutex<Option<iceoryx::FrameReceiver>>,
}

impl Bridge {
    fn start(
        tx_addr: String,
        rx_addr: String,
        capacity: usize,
        hwm: i32,
        timeout: Duration,
        callback: Option<RxCallback>,
    ) -> Result<Arc<Self>, String> {
        let use_iceoryx = ce_bp_iceoryx_enabled()?;
        let state = Arc::new(State::default());
        let (tx, outbound) = bounded(capacity);
        // Callback mode has no native RX queue: dispatch is on the socket owner.
        let (inbound, rx) = if let Some(callback) = callback {
            (RxDelivery::Callback(callback), None)
        } else {
            let (inbound, rx) = bounded(capacity);
            state.receive_enabled.store(true, Ordering::Release);
            (RxDelivery::Queue(inbound), Some(rx))
        };
        let (ready, initialized) = bounded(2);
        let bridge = Arc::new(Self {
            state: state.clone(),
            tx: Some(tx),
            rx,
            threads: Mutex::new(Vec::new()),
            timeout,
            #[cfg(feature = "iceoryx")]
            direct: None,
        });
        let context = zmq::Context::new();
        for (name, kind, address) in [
            ("ce-zmq-tx", zmq::PUSH, tx_addr),
            ("ce-zmq-rx", zmq::PULL, rx_addr),
        ] {
            let context = context.clone();
            let state = state.clone();
            let ready = ready.clone();
            let outbound = outbound.clone();
            let inbound = inbound.clone();
            let worker = thread::Builder::new().name(name.into()).spawn(move || {
                let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
                    #[cfg(feature = "iceoryx")]
                    if kind == zmq::PUSH && use_iceoryx {
                        let sender = iceoryx::FrameSender::new(&address, hwm as usize)?;
                        ready.send(()).map_err(|e| e.to_string())?;
                        while !state.stop.load(Ordering::Acquire) {
                            let message = match outbound.recv_timeout(TICK) {
                                Ok(message) => message,
                                Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                                Err(_) => return Ok(()),
                            };
                            let frames = match message {
                                Outbound::Frames(frames) => frames,
                                Outbound::Batch { metadata, buffers } => {
                                    batch::encode(&metadata, buffers)?
                                }
                            };
                            sender.send(&frames, timeout, || state.stop.load(Ordering::Acquire))?;
                        }
                        return Ok(());
                    }
                    let _ = use_iceoryx;
                    #[cfg(feature = "iceoryx")]
                    if kind == zmq::PULL && use_iceoryx {
                        let receiver = iceoryx::FrameReceiver::new(&address, hwm as usize)?;
                        ready.send(()).map_err(|e| e.to_string())?;
                        while !state.stop.load(Ordering::Acquire) {
                            if !state.receive_enabled.load(Ordering::Acquire) {
                                thread::sleep(TICK);
                                continue;
                            }
                            if let Some(frames) = receiver.receive_wait(iceoryx::RECEIVE_WAIT)? {
                                deliver_rx(frames, &inbound, &state)?;
                            }
                        }
                        return Ok(());
                    }
                    // Socket construction, use and destruction all stay on its owner thread.
                    let socket = context.socket(kind).map_err(|e| e.to_string())?;
                    socket.set_linger(0).map_err(|e| e.to_string())?;
                    socket.set_sndhwm(hwm).map_err(|e| e.to_string())?;
                    socket.set_rcvhwm(hwm).map_err(|e| e.to_string())?;
                    socket
                        .set_maxmsgsize(MAX_BYTES as i64)
                        .map_err(|e| e.to_string())?;
                    if kind == zmq::PUSH {
                        socket.set_sndbuf(i32::MAX).map_err(|e| e.to_string())?;
                    }
                    socket.connect(&address).map_err(|e| e.to_string())?;
                    ready.send(()).map_err(|e| e.to_string())?;
                    if kind == zmq::PUSH {
                        tx_loop(&socket, &outbound, &state, timeout)
                    } else {
                        rx_loop(&socket, &inbound, &state)
                    }
                }));
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => state.fail(format!("{name}: {error}")),
                    Err(_) => state.fail(format!("{name}: worker panicked")),
                }
            });
            match worker {
                Ok(worker) => bridge.threads.lock().unwrap().push(worker),
                Err(error) => {
                    bridge.close();
                    return Err(error.to_string());
                }
            }
        }
        let deadline = Instant::now() + timeout;
        for _ in 0..2 {
            loop {
                if let Err(error) = state.check() {
                    bridge.close();
                    return Err(error);
                }
                if initialized.recv_timeout(TICK).is_ok() {
                    break;
                }
                if Instant::now() >= deadline {
                    bridge.close();
                    return Err("CE transport startup timed out".into());
                }
            }
        }
        Ok(bridge)
    }

    fn close(&self) {
        self.state.stop.store(true, Ordering::Release);
        #[cfg(feature = "iceoryx")]
        if let Some(ports) = &self.direct {
            // Stop is visible before waiting for in-flight synchronous calls.
            ports.tx.lock().unwrap().take();
            ports.rx.lock().unwrap().take();
        }
        for worker in self.threads.lock().unwrap().drain(..) {
            let _ = worker.join();
        }
    }

    fn send(&self, mut message: Outbound) -> Result<(), String> {
        #[cfg(feature = "iceoryx")]
        if let Some(ports) = &self.direct {
            self.state.check()?;
            let Outbound::Frames(frames) = message else {
                return Err("caller-owned transport requires prepared frames".into());
            };
            let sender = ports.tx.lock().unwrap();
            let result =
                sender
                    .as_ref()
                    .ok_or("CE transport closed")?
                    .send(&frames, self.timeout, || {
                        self.state.stop.load(Ordering::Acquire)
                    });
            if let Err(error) = &result {
                self.state.fail(error);
            }
            return result;
        }
        let deadline = Instant::now() + self.timeout;
        loop {
            self.state.check()?;
            match self
                .tx
                .as_ref()
                .ok_or("CE TX unavailable")?
                .send_timeout(message, TICK)
            {
                Ok(()) => return Ok(()),
                Err(crossbeam_channel::SendTimeoutError::Timeout(value)) => message = value,
                Err(_) => return Err("CE TX worker disconnected".into()),
            }
            if Instant::now() >= deadline {
                self.state.fail("CE outbound queue remained full");
                return self.state.check();
            }
        }
    }

    fn receive(&self, timeout: Duration) -> Result<Option<Frames>, String> {
        #[cfg(feature = "iceoryx")]
        if let Some(ports) = &self.direct {
            let receiver = ports.rx.lock().unwrap();
            let receiver = receiver.as_ref().ok_or("CE transport closed")?;
            let deadline = Instant::now() + timeout;
            loop {
                self.state.check()?;
                match receiver.receive_wait(
                    iceoryx::RECEIVE_WAIT.min(deadline.saturating_duration_since(Instant::now())),
                ) {
                    Ok(Some(frames)) => return Ok(Some(frames)),
                    Err(error) => {
                        self.state.fail(&error);
                        return Err(error);
                    }
                    Ok(None) => {}
                }
                if Instant::now() >= deadline {
                    return Ok(None);
                }
            }
        }
        let rx = self
            .rx
            .as_ref()
            .ok_or("receive is unavailable in callback mode")?;
        let deadline = Instant::now() + timeout;
        loop {
            self.state.check()?;
            match rx.recv_timeout(TICK.min(deadline.saturating_duration_since(Instant::now()))) {
                Ok(frames) => return Ok(Some(frames)),
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // Close or worker failure may race with the blocking receive.
                    // Preserve that cause instead of reporting a generic disconnect.
                    self.state.check()?;
                    return Err("CE RX worker disconnected".into());
                }
                Err(_) => {}
            }
            if Instant::now() >= deadline {
                return Ok(None);
            }
        }
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        self.close();
    }
}

fn tx_loop(
    socket: &zmq::Socket,
    outbound: &Receiver<Outbound>,
    state: &State,
    timeout: Duration,
) -> Result<(), String> {
    while !state.stop.load(Ordering::Acquire) {
        let message = match outbound.recv_timeout(TICK) {
            Ok(message) => message,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(_) => return Ok(()),
        };
        // Preparation runs on the socket owner, never on the submitting CE thread.
        let frames = match message {
            Outbound::Frames(frames) => frames,
            Outbound::Batch { metadata, buffers } => batch::encode(&metadata, buffers)?,
        };
        let deadline = Instant::now() + timeout;
        for (index, frame) in frames.iter().enumerate() {
            let flags = zmq::DONTWAIT
                | if index + 1 < frames.len() {
                    zmq::SNDMORE
                } else {
                    0
                };
            loop {
                if state.stop.load(Ordering::Acquire) {
                    return Ok(());
                }
                match socket.send(frame.as_slice(), flags) {
                    Ok(()) => break,
                    Err(zmq::Error::EAGAIN) | Err(zmq::Error::EINTR) => {
                        if Instant::now() >= deadline {
                            return Err("ZMQ send timed out".into());
                        }
                        match socket.poll(zmq::POLLOUT, POLL_MS) {
                            Ok(_) | Err(zmq::Error::EINTR) => {}
                            Err(e) => return Err(e.to_string()),
                        }
                    }
                    Err(e) => return Err(e.to_string()),
                }
            }
        }
    }
    Ok(())
}

pub type RxCallback = unsafe extern "C" fn(*const FrameView, usize) -> i32;

#[derive(Clone)]
enum RxDelivery {
    Queue(Sender<Frames>),
    Callback(RxCallback),
}

thread_local! {
    static IN_RX_CALLBACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn rx_loop(socket: &zmq::Socket, inbound: &RxDelivery, state: &State) -> Result<(), String> {
    while !state.stop.load(Ordering::Acquire) {
        if !state.receive_enabled.load(Ordering::Acquire) {
            thread::sleep(TICK);
            continue;
        }
        match socket.poll(zmq::POLLIN, POLL_MS) {
            Ok(0) | Err(zmq::Error::EINTR) => continue,
            Err(e) => return Err(e.to_string()),
            Ok(_) => {}
        }
        let mut frames = Vec::new();
        let mut bytes = 0usize;
        loop {
            let frame = match socket.recv_bytes(zmq::DONTWAIT) {
                Ok(frame) => frame,
                Err(zmq::Error::EINTR) => continue,
                Err(e) => return Err(e.to_string()),
            };
            bytes = bytes
                .checked_add(frame.len())
                .ok_or("multipart size overflow")?;
            if frames.len() == MAX_FRAMES || bytes > MAX_BYTES {
                return Err("multipart message exceeds CE transport limits".into());
            }
            frames.push(frame);
            if !socket.get_rcvmore().map_err(|e| e.to_string())? {
                break;
            }
        }
        deliver_rx(frames, inbound, state)?;
    }
    Ok(())
}

fn deliver_rx(mut frames: Frames, inbound: &RxDelivery, state: &State) -> Result<(), String> {
    if let RxDelivery::Callback(callback) = inbound {
        if state.stop.load(Ordering::Acquire) {
            return Ok(());
        }
        let views: Vec<_> = frames
            .iter()
            .map(|frame| FrameView {
                data: frame.as_ptr(),
                len: frame.len(),
            })
            .collect();
        IN_RX_CALLBACK.with(|active| active.set(true));
        // The caller retains the function pointer until close has joined RX.
        // The frames are borrowed only for this synchronous call.
        let result = unsafe { callback(views.as_ptr(), views.len()) };
        IN_RX_CALLBACK.with(|active| active.set(false));
        if result != 0 {
            return Err("CE RX callback failed".into());
        }
        return Ok(());
    }
    let RxDelivery::Queue(inbound) = inbound else {
        unreachable!()
    };
    loop {
        if state.stop.load(Ordering::Acquire) {
            return Ok(());
        }
        match inbound.send_timeout(frames, TICK) {
            Ok(()) => break,
            Err(crossbeam_channel::SendTimeoutError::Timeout(value)) => frames = value,
            Err(_) => return Ok(()),
        }
    }
    Ok(())
}

static BRIDGES: OnceLock<Mutex<HashMap<u64, Arc<Bridge>>>> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
fn registry() -> &'static Mutex<HashMap<u64, Arc<Bridge>>> {
    BRIDGES.get_or_init(|| Mutex::new(HashMap::new()))
}
fn bridge(id: u64) -> Result<Arc<Bridge>, String> {
    registry()
        .lock()
        .unwrap()
        .get(&id)
        .cloned()
        .ok_or("CE transport closed or invalid handle".into())
}

// Errors are thread-local: ctypes calls read the error on the same Python thread.
thread_local! { static LAST_ERROR: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) }; }
fn ffi<T: Copy>(failure: T, f: impl FnOnce() -> Result<T, String>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        result => {
            let message = match result {
                Ok(Err(e)) => e,
                _ => "CE transport panic".into(),
            };
            LAST_ERROR.with(|e| *e.borrow_mut() = message);
            failure
        }
    }
}

#[repr(C)]
pub struct FrameView {
    data: *const u8,
    len: usize,
}

#[unsafe(no_mangle)]
pub extern "C" fn ce_transport_abi() -> u32 {
    3
}

/// Open synchronous shared-memory ports without spawning native workers.
/// Send and receive run on their callers (normally Python TX/RX threads).
/// # Safety
/// Addresses must be valid NUL-terminated strings for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ce_transport_open_direct(
    tx: *const c_char,
    rx: *const c_char,
    hwm: i32,
    timeout_ms: u32,
) -> u64 {
    ffi(0, || {
        if tx.is_null() || rx.is_null() || hwm <= 0 || timeout_ms == 0 {
            return Err("invalid CE transport configuration".into());
        }
        #[cfg(not(feature = "iceoryx"))]
        return Err("rebuild native CE with --features iceoryx".into());
        #[cfg(feature = "iceoryx")]
        {
            if !ce_bp_iceoryx_enabled()? {
                return Err(
                    "caller-owned transport requires VLLM_GR_CE_BP_TRANSPORT=sharedmem".into(),
                );
            }
            let tx = unsafe { CStr::from_ptr(tx) }
                .to_str()
                .map_err(|e| e.to_string())?;
            let rx = unsafe { CStr::from_ptr(rx) }
                .to_str()
                .map_err(|e| e.to_string())?;
            let value = Arc::new(Bridge {
                state: Arc::new(State::default()),
                tx: None,
                rx: None,
                threads: Mutex::new(Vec::new()),
                timeout: Duration::from_millis(timeout_ms.into()),
                direct: Some(DirectPorts {
                    tx: Mutex::new(Some(iceoryx::FrameSender::new(tx, hwm as usize)?)),
                    rx: Mutex::new(Some(iceoryx::FrameReceiver::new(rx, hwm as usize)?)),
                }),
            });
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            registry().lock().unwrap().insert(id, value);
            Ok(id)
        }
    })
}

/// # Safety
/// Addresses must be valid NUL-terminated strings for this call. If present,
/// callback must remain callable until close returns. It must not unwind, close
/// a bridge, or retain borrowed frame pointers. Callback delivery is disabled
/// until start_receive, so the caller can first install its ownership guards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ce_transport_open(
    tx: *const c_char,
    rx: *const c_char,
    capacity: usize,
    hwm: i32,
    timeout_ms: u32,
    callback: Option<RxCallback>,
) -> u64 {
    ffi(0, || {
        if tx.is_null()
            || rx.is_null()
            || capacity == 0
            || capacity > 65536
            || hwm <= 0
            || timeout_ms == 0
        {
            return Err("invalid CE transport configuration".into());
        }
        let tx = unsafe { CStr::from_ptr(tx) }
            .to_str()
            .map_err(|e| e.to_string())?
            .to_owned();
        let rx = unsafe { CStr::from_ptr(rx) }
            .to_str()
            .map_err(|e| e.to_string())?
            .to_owned();
        let value = Bridge::start(
            tx,
            rx,
            capacity,
            hwm,
            Duration::from_millis(timeout_ms.into()),
            callback,
        )?;
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        registry().lock().unwrap().insert(id, value);
        Ok(id)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ce_transport_start_receive(id: u64) -> i32 {
    ffi(-1, || {
        let bridge = bridge(id)?;
        bridge.state.check()?;
        bridge.state.receive_enabled.store(true, Ordering::Release);
        Ok(0)
    })
}

/// # Safety
/// `views` points to `count` valid descriptors. Each nonempty frame points to
/// readable storage, kept alive and unmodified until this function returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ce_transport_send(id: u64, views: *const FrameView, count: usize) -> i32 {
    ffi(-1, || {
        let bridge = bridge(id)?;
        if views.is_null() || count == 0 || count > MAX_FRAMES {
            return Err("invalid frame count".into());
        }
        let views = unsafe { std::slice::from_raw_parts(views, count) };
        let mut total = 0usize;
        for view in views {
            total = total.checked_add(view.len).ok_or("frame length overflow")?;
            if total > MAX_BYTES || (view.len > 0 && view.data.is_null()) {
                return Err("invalid frame size/pointer".into());
            }
        }
        let frames = views
            .iter()
            .map(|v| {
                if v.len == 0 {
                    Vec::new()
                } else {
                    unsafe { std::slice::from_raw_parts(v.data, v.len) }.to_vec()
                }
            })
            .collect();
        bridge.send(Outbound::Frames(frames))?;
        Ok(0)
    })
}

/// Submit a metadata snapshot plus four numeric buffers per physical output.
/// The final wire header and packed arrays are prepared by the native TX worker.
/// # Safety
/// `views` points to `count` valid descriptors, first metadata then buffers.
/// Nonempty buffers must remain readable and unmodified until this call returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ce_transport_send_batch(
    id: u64,
    views: *const FrameView,
    count: usize,
) -> i32 {
    ffi(-1, || {
        let bridge = bridge(id)?;
        bridge.state.check()?;
        if views.is_null()
            || !(5..=1 + batch::MAX_OUTPUTS * 4).contains(&count)
            || !(count - 1).is_multiple_of(4)
        {
            return Err("invalid batch buffer count".into());
        }
        let views = unsafe { std::slice::from_raw_parts(views, count) };
        let mut total = 0usize;
        for view in views {
            total = total.checked_add(view.len).ok_or("batch size overflow")?;
            if total > MAX_BYTES || (view.len > 0 && view.data.is_null()) {
                return Err("invalid batch size/pointer".into());
            }
        }
        let mut copied = views.iter().map(|v| {
            if v.len == 0 {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(v.data, v.len) }.to_vec()
            }
        });
        let metadata = copied.next().unwrap();
        bridge.send(Outbound::Batch {
            metadata,
            buffers: copied.collect(),
        })?;
        Ok(0)
    })
}

/// # Safety
/// `out` is writable. On success caller owns the message until message_free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ce_transport_receive(
    id: u64,
    timeout_ms: u32,
    out: *mut *mut Frames,
) -> i32 {
    ffi(-1, || {
        if out.is_null() {
            return Err("null output pointer".into());
        }
        unsafe {
            *out = std::ptr::null_mut();
        }
        match bridge(id)?.receive(Duration::from_millis(timeout_ms.into()))? {
            Some(frames) => {
                unsafe {
                    *out = Box::into_raw(Box::new(frames));
                }
                Ok(1)
            }
            None => Ok(0),
        }
    })
}

/// # Safety
/// Message must be a live result of receive, and len a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ce_transport_frame(
    message: *const Frames,
    index: usize,
    len: *mut usize,
) -> *const u8 {
    if message.is_null() || len.is_null() {
        return std::ptr::null();
    }
    match unsafe { &*message }.get(index) {
        Some(frame) => {
            unsafe {
                *len = frame.len();
            }
            frame.as_ptr()
        }
        None => std::ptr::null(),
    }
}

/// # Safety
/// Message must be a live result of receive; may be freed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ce_transport_message_free(message: *mut Frames) {
    if !message.is_null() {
        drop(unsafe { Box::from_raw(message) });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ce_transport_check(id: u64) -> i32 {
    ffi(-1, || {
        bridge(id)?.state.check()?;
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ce_transport_close(id: u64) -> i32 {
    ffi(-1, || {
        if IN_RX_CALLBACK.with(|active| active.get()) {
            return Err("cannot close CE transport from an RX callback".into());
        }
        let value = registry().lock().unwrap().get(&id).cloned();
        if let Some(value) = value {
            value.close();
            // Keep the entry visible until workers have joined, so concurrent
            // close calls cannot return while a callback is still executing.
            registry().lock().unwrap().remove(&id);
        }
        Ok(0)
    })
}

/// # Safety
/// Buffer must be writable for `capacity` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ce_transport_error(buffer: *mut u8, capacity: usize) -> usize {
    LAST_ERROR.with(|e| {
        let error = e.borrow();
        let len = error.len().min(capacity);
        if !buffer.is_null() && len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(error.as_ptr(), buffer, len);
            }
        }
        len
    })
}
