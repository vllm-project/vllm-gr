//! Opt-in, same-host CE <-> BP byte transport. One subscriber, no overwrites.
//! Retains multipart framing; payload copies deliberately remain for this trial.
use iceoryx2::port::{
    listener::Listener, notifier::Notifier, publisher::Publisher, subscriber::Subscriber,
};
use iceoryx2::prelude::*;
use iceoryx2::service::ipc_threadsafe;
use std::time::{Duration, Instant};

type Service = ipc_threadsafe::Service;
type Frames = Vec<Vec<u8>>;
const MAX_FRAMES: usize = 64;
const MAX_PAYLOAD: usize = 16 * 1024 * 1024;
pub const IDLE_WAIT: Duration = Duration::from_micros(50);
// Only a shutdown/health-check bound; notifications interrupt this wait immediately.
pub const RECEIVE_WAIT: Duration = Duration::from_millis(10);
const MAGIC: &[u8; 8] = b"CEBP0001";

fn endpoint_hash(endpoint: &str) -> u64 {
    // Stable across processes and independent of Rust's randomized hash state.
    endpoint.bytes().fold(0xcbf29ce484222325_u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

fn service_name(endpoint: &str) -> Result<ServiceName, String> {
    format!("vllm-gr/ce-bp/v1/{:016x}", endpoint_hash(endpoint))
        .as_str()
        .try_into()
        .map_err(|e| format!("iceoryx service name: {e:?}"))
}

fn node(endpoint: &str) -> Result<Node<Service>, String> {
    if !endpoint.starts_with("ipc://") {
        return Err(format!("iceoryx endpoint must use ipc://: {endpoint}"));
    }
    // The default /tmp/iceoryx2 root is shared by every user and persistent CI
    // job on a host. Derive a short private root from the already unique IPC
    // endpoint. Keeping it short also stays below Unix-domain socket limits.
    let hash = endpoint_hash(endpoint);
    let root = format!("/tmp/vllm-gr-iox2-{hash:016x}");
    let root = Path::new(root.as_bytes()).map_err(|e| format!("iceoryx runtime path: {e:?}"))?;
    let mut config = Config::default();
    config.global.set_root_path(&root);
    config.global.prefix = FileName::new(format!("vgr_{hash:016x}_").as_bytes())
        .map_err(|e| format!("iceoryx runtime prefix: {e:?}"))?;
    NodeBuilder::new()
        .signal_handling_mode(SignalHandlingMode::Disabled)
        .config(&config)
        .create::<Service>()
        .map_err(|e| format!("iceoryx node: {e:?}"))
}

fn service(
    node: &Node<Service>,
    endpoint: &str,
    capacity: usize,
) -> Result<
    iceoryx2::service::port_factory::publish_subscribe::PortFactory<Service, [u8], ()>,
    String,
> {
    if !(1..=4096).contains(&capacity) {
        return Err("iceoryx capacity must be 1..=4096".into());
    }
    node.service_builder(&service_name(endpoint)?)
        .publish_subscribe::<[u8]>()
        .max_publishers(1)
        .max_subscribers(1)
        .subscriber_max_buffer_size(capacity)
        .subscriber_max_borrowed_samples(1)
        .enable_safe_overflow(false)
        .history_size(0)
        .open_or_create()
        .map_err(|e| format!("iceoryx service: {e:?}"))
}

pub struct FrameSender {
    publisher: Publisher<Service, [u8], ()>,
    notifier: Notifier<Service>,
    _node: Node<Service>,
}

impl FrameSender {
    pub fn new(endpoint: &str, capacity: usize) -> Result<Self, String> {
        let node = node(endpoint)?;
        let event = node
            .service_builder(&service_name(&format!("{endpoint}/ready-v1"))?)
            .event()
            .max_notifiers(1)
            .max_listeners(1)
            .open_or_create()
            .map_err(|e| format!("iceoryx event service: {e:?}"))?;
        let notifier = event
            .notifier_builder()
            .create()
            .map_err(|e| format!("iceoryx notifier: {e:?}"))?;
        let service = service(&node, endpoint, capacity)?;
        let publisher = service
            .publisher_builder()
            .initial_max_slice_len(64 * 1024)
            .allocation_strategy(AllocationStrategy::PowerOfTwo)
            .max_loaned_samples(1)
            // A zero-recipient send is retried below. With exactly one subscriber,
            // this never partially delivers or duplicates a message.
            .backpressure_strategy(BackpressureStrategy::DiscardData)
            .create()
            .map_err(|e| format!("iceoryx publisher: {e:?}"))?;
        Ok(Self {
            publisher,
            notifier,
            _node: node,
        })
    }

    pub fn send(
        &self,
        frames: &[Vec<u8>],
        timeout: Duration,
        stopped: impl Fn() -> bool,
    ) -> Result<(), String> {
        let bytes = encode(frames)?;
        let deadline = Instant::now() + timeout;
        loop {
            if stopped() {
                return Err("iceoryx send stopped".into());
            }
            if Instant::now() >= deadline {
                return Err("iceoryx send timed out: subscriber absent or full".into());
            }
            let sample = self
                .publisher
                .loan_slice_uninit(bytes.len())
                .map_err(|e| format!("iceoryx loan: {e:?}"))?
                .write_from_slice(&bytes);
            match sample.send().map_err(|e| format!("iceoryx send: {e:?}"))? {
                1 => {
                    // Publish before notifying. Never retry an already-delivered payload.
                    self.notifier
                        .notify()
                        .map_err(|e| format!("iceoryx notify: {e:?}"))?;
                    return Ok(());
                }
                0 => std::thread::sleep(IDLE_WAIT),
                _ => return Err("iceoryx unexpected multiple subscribers".into()),
            }
        }
    }
}

pub struct FrameReceiver {
    subscriber: Subscriber<Service, [u8], ()>,
    listener: Listener<Service>,
    _node: Node<Service>,
}

impl FrameReceiver {
    pub fn new(endpoint: &str, capacity: usize) -> Result<Self, String> {
        let node = node(endpoint)?;
        // Register notification reception before making the data subscriber visible.
        let event = node
            .service_builder(&service_name(&format!("{endpoint}/ready-v1"))?)
            .event()
            .max_notifiers(1)
            .max_listeners(1)
            .open_or_create()
            .map_err(|e| format!("iceoryx event service: {e:?}"))?;
        let listener = event
            .listener_builder()
            .create()
            .map_err(|e| format!("iceoryx listener: {e:?}"))?;
        let service = service(&node, endpoint, capacity)?;
        let subscriber = service
            .subscriber_builder()
            .buffer_size(capacity)
            .create()
            .map_err(|e| format!("iceoryx subscriber: {e:?}"))?;
        Ok(Self {
            subscriber,
            listener,
            _node: node,
        })
    }

    pub fn receive(&self) -> Result<Option<Frames>, String> {
        self.subscriber
            .receive()
            .map_err(|e| format!("iceoryx receive: {e:?}"))?
            .map(|sample| decode(sample.payload()))
            .transpose()
    }

    pub fn receive_wait(&self, timeout: Duration) -> Result<Option<Frames>, String> {
        if let Some(frames) = self.receive()? {
            return Ok(Some(frames));
        }
        if timeout.is_zero() {
            return Ok(None);
        }
        // Notifications are hints, not message counts. Drain stale hints, then
        // recheck the data queue so a concurrent publish cannot lose its wakeup.
        self.listener
            .try_wait_all(|_| {})
            .map_err(|e| format!("iceoryx notification drain: {e:?}"))?;
        if let Some(frames) = self.receive()? {
            return Ok(Some(frames));
        }
        self.listener
            .timed_wait_all(|_| {}, timeout)
            .map_err(|e| format!("iceoryx notification wait: {e:?}"))?;
        self.receive()
    }
}

fn encode(frames: &[Vec<u8>]) -> Result<Vec<u8>, String> {
    if frames.is_empty() || frames.len() > MAX_FRAMES {
        return Err("iceoryx invalid frame count".into());
    }
    let size = frames
        .iter()
        .try_fold(12 + frames.len() * 4, |sum, f| sum.checked_add(f.len()))
        .filter(|n| *n <= MAX_PAYLOAD)
        .ok_or("iceoryx message exceeds 16 MiB")?;
    let mut bytes = Vec::with_capacity(size);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    for frame in frames {
        bytes.extend_from_slice(&(frame.len() as u32).to_le_bytes());
    }
    for frame in frames {
        bytes.extend_from_slice(frame);
    }
    Ok(bytes)
}

fn decode(bytes: &[u8]) -> Result<Frames, String> {
    if bytes.len() < 12 || bytes.len() > MAX_PAYLOAD || &bytes[..8] != MAGIC {
        return Err("iceoryx invalid envelope".into());
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if count == 0 || count > MAX_FRAMES || bytes.len() < 12 + count * 4 {
        return Err("iceoryx invalid frame table".into());
    }
    let mut offset = 12 + count * 4;
    let mut frames = Vec::with_capacity(count);
    for length in bytes[12..offset].chunks_exact(4) {
        let len = u32::from_le_bytes(length.try_into().unwrap()) as usize;
        let end = offset
            .checked_add(len)
            .filter(|end| *end <= bytes.len())
            .ok_or("iceoryx truncated frame")?;
        frames.push(bytes[offset..end].to_vec());
        offset = end;
    }
    if offset != bytes.len() {
        return Err("iceoryx trailing bytes".into());
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires shared-memory IPC permissions"]
    fn notification_wakes_waiter_and_preserves_queued_messages() {
        let endpoint = format!("notification-test-{}", std::process::id());
        let receiver = FrameReceiver::new(&endpoint, 16).unwrap();
        let sender = FrameSender::new(&endpoint, 16).unwrap();
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(40));
            for i in 0..8_u8 {
                sender
                    .send(&[vec![i]], Duration::from_secs(2), || false)
                    .unwrap();
            }
            // Keep the publisher alive until every sample is consumed.
            sender
        });
        let start = Instant::now();
        for i in 0..8_u8 {
            assert_eq!(
                receiver.receive_wait(Duration::from_secs(2)).unwrap(),
                Some(vec![vec![i]])
            );
        }
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "notification did not interrupt wait"
        );
        let _sender = worker.join().unwrap();
        // A leftover notification may wake early; it must not fabricate data.
        assert!(
            receiver
                .receive_wait(Duration::from_millis(10))
                .unwrap()
                .is_none()
        );
        let start = Instant::now();
        assert!(
            receiver
                .receive_wait(Duration::from_millis(20))
                .unwrap()
                .is_none()
        );
        assert!(start.elapsed() >= Duration::from_millis(15));
    }

    #[test]
    fn envelope_validates_lengths_and_owns_frames() {
        let frames = vec![b"ENGINE_OUTPUT_BATCH".to_vec(), vec![], vec![255; 66560]];
        let bytes = encode(&frames).unwrap();
        assert_eq!(decode(&bytes).unwrap(), frames);
        for cut in [0, 8, 11, 16, bytes.len() - 1] {
            assert!(decode(&bytes[..cut]).is_err());
        }
        let mut corrupt = bytes.clone();
        corrupt[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode(&corrupt).is_err());
        let mut extra = bytes;
        extra.push(0);
        assert!(decode(&extra).is_err());
        assert!(encode(&[]).is_err());
        assert!(encode(&vec![vec![]; 65]).is_err());
        assert!(encode(&[vec![0; MAX_PAYLOAD]]).is_err());
    }
}
